import argparse
import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from agent.application.use_cases import run_fitness_workflow, run_profile_onboarding_workflow
from agent.llm import build_chat_model
from agent.models import FitnessRequest, UserProfile, WorkflowContext, WorkflowEvent
from agent.repositories.profile_repository import load_profile


LIST_ARTIFACT_KEYS = {"food_lookup", "exercise_lookup", "meal_plan", "workout_plan", "knowledge_lookup"}
ARTIFACT_LIMITS = {
    "meal_plan": 3,
    "workout_plan": 3,
    "food_lookup": 10,
    "exercise_lookup": 10,
    "knowledge_lookup": 6,
    "final_answer": 4,
    "user_profile": 4,
}
DEFAULT_MAX_MESSAGE_TOKENS = 128_000
DEFAULT_COMPRESSION_RATIO = 0.75
DEFAULT_RECENT_TURN_WINDOW = 12


def _format_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _is_artifact_entry(value: Any) -> bool:
    return isinstance(value, dict) and "value" in value and "summary" in value


def _artifact_entry_value(value: Any) -> Any:
    if _is_artifact_entry(value):
        return value.get("value")
    return value


def _artifact_entry_summary(value: Any) -> str:
    if _is_artifact_entry(value):
        return str(value.get("summary") or "")
    return ""


def _build_event_handler():
    def handle_event(event: WorkflowEvent) -> None:
        event_type = event.event_type
        phase = event.phase
        node = event.node
        summary = event.summary
        payload = event.data

        if phase and node and summary:
            print(f"\n[{phase}:{node}]")
            print(f"- summary: {summary}")

        if event_type == "onboarding_inspect":
            print(f"- missing_fields: {payload.get('missing_fields')}")
        elif event_type == "onboarding_review":
            print("- reviewing existing profile")
        elif event_type == "onboarding_review_decision":
            print(f"- should_modify_existing: {payload.get('should_modify_existing')}")
            print(f"- next_node: {payload.get('next_node')}")
        elif event_type == "onboarding_select_fields":
            print(f"- selected_fields: {payload.get('selected_fields')}")
        elif event_type == "onboarding_question":
            print(f"- field_name: {payload.get('field_name')}")
            print(f"- question: {payload.get('question')}")
        elif event_type == "onboarding_answer":
            print(f"- field_name: {payload.get('field_name')}")
            print(f"- is_valid: {payload.get('is_valid')}")
        elif event_type == "onboarding_update":
            print(f"- field_name: {payload.get('field_name')}")
            print(f"- remaining_missing_fields: {payload.get('missing_fields')}")
        elif event_type == "onboarding_retry":
            print(f"- field_name: \n{payload.get('field_name')}")
        elif event_type == "onboarding_complete":
            print(f"- profile: {_format_json(payload.get('profile'))}")
            print(f"- missing_fields: {payload.get('missing_fields')}")
        elif event_type == "intent":
            print(f"- primary_goal: {payload.get('primary_goal')}")
        elif event_type == "memory_update":
            print(f"- current profile: \n{_format_json(payload.get('profile') or {})}")
            print(f"- memory_update: \n{_format_json(payload.get('memory_update') or {})}")
        elif event_type == "plan":
            active_step = payload.get("active_step") or {}
            print(f"- reasoning: {payload.get('reasoning')}")
            print(f"- next tool: {active_step.get('tool_name', 'none')}")
            if active_step.get("objective"):
                print(f"- objective: {active_step.get('objective')}")
        elif event_type == "tool_start":
            print(f"- iteration: {payload.get('iteration')}")
            print(f"- tool_name: {payload.get('tool_name')}")
        elif event_type == "tool_result":
            print(f"- iteration: {payload.get('iteration')}")
            print(f"- tool_name: {payload.get('tool_name')}")
            print(f"- observation: {payload.get('observation')}")
            print(f"- artifact_keys: {payload.get('artifact_keys')}")
        elif event_type == "tool_error":
            print(f"- iteration: {payload.get('iteration')}")
            print(f"- tool_name: {payload.get('tool_name')}")
            print(f"- error: {payload.get('error')}")
        elif event_type == "observation":
            print(f"- iteration: {payload.get('iteration')}")
            print(f"- {payload.get('observation')}")
        elif event_type == "decision":
            print(f"- iteration: {payload.get('iteration')}")
            print(f"- decision: {payload.get('decision')}")
            print(f"- reasoning: {payload.get('reasoning')}")
            if payload.get("next_node"):
                print(f"- next_node: {payload.get('next_node')}")
        elif event_type == "final_answer":
            print(f"- iterations: {payload.get('iterations')}")
            if payload.get("errors"):
                print(f"- errors: {payload.get('errors')}")

    return handle_event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Command line chat for the fitness assistant")
    parser.add_argument("--user-id", default="cli-user")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    parser.add_argument("--onboarding-model", default=os.getenv("ONBOARDING_MODEL_NAME", "qwen3.5-plus"))
    parser.add_argument("--fitness-model", default=os.getenv("FITNESS_MODEL_NAME", "qwen3.5-plus"))
    parser.add_argument("--fitness-strong-model", default=None)
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--max-message-tokens", type=int, default=DEFAULT_MAX_MESSAGE_TOKENS)
    parser.add_argument("--message-compression-ratio", type=float, default=DEFAULT_COMPRESSION_RATIO)
    parser.add_argument("--recent-turn-window", type=int, default=DEFAULT_RECENT_TURN_WINDOW)
    parser.add_argument("--use-full-database", action="store_true")
    parser.add_argument("--age", type=int, default=None)
    parser.add_argument("--weight", type=float, default=None)
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument("--gender", choices=["male", "female"], default=None)
    parser.add_argument("--activity-level", default=None)
    parser.add_argument("--fitness-goal", default=None)
    parser.add_argument("--workout-frequency", type=int, default=None)
    parser.add_argument("--workout-duration", type=int, default=None)
    return parser.parse_args()


def build_default_profile(args: argparse.Namespace) -> UserProfile:
    return UserProfile(
        user_id=args.user_id,
        age=args.age,
        weight=args.weight,
        height=args.height,
        gender=args.gender,
        activity_level=args.activity_level,
        fitness_goal=args.fitness_goal,
        workout_frequency=args.workout_frequency,
        workout_duration=args.workout_duration,
    )


async def _prompt_user(question: str) -> str:
    if question:
        print(f"\nAssistant>\n{question}")
    while True:
        try:
            answer = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            raise SystemExit(0)

        if not answer:
            print("\nAssistant>\n我还在这儿，随时可以继续。")
            continue
        if answer.lower() in {"exit", "quit"}:
            print("Bye.")
            raise SystemExit(0)
        return answer


def _notify_user(message: str) -> None:
    print(f"\nAssistant>\n{message}")


def _estimate_token_count(text: str) -> int:
    return max(1, len(text) // 4)


def _estimate_messages_token_count(messages: List[Dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        total += _estimate_token_count(str(message.get("role", "")))
        total += _estimate_token_count(str(message.get("content", "")))
        total += 8
    return total


def _turn_ids(messages: List[Dict[str, Any]]) -> List[int]:
    return sorted({int(message.get("turn", 0)) for message in messages})


def _messages_for_turns(messages: List[Dict[str, Any]], turn_ids: List[int]) -> List[Dict[str, Any]]:
    selected_turns = set(turn_ids)
    return [message for message in messages if int(message.get("turn", 0)) in selected_turns]


def _select_recent_messages_with_backoff(
    messages: List[Dict[str, Any]],
    max_tokens: int,
    initial_turn_window: int,
) -> List[Dict[str, Any]]:
    turns = _turn_ids(messages)
    if not turns:
        return []

    window = min(initial_turn_window, len(turns))
    while True:
        recent_turns = turns[-window:]
        recent_messages = _messages_for_turns(messages, recent_turns)
        if _estimate_messages_token_count(recent_messages) <= max_tokens or window <= 1:
            return recent_messages
        window = max(1, window // 2)


async def _compress_message_chunk(
    *,
    base_url: str,
    model_name: str,
    existing_summary: Optional[str],
    messages_to_compress: List[Dict[str, Any]],
) -> str:
    if not messages_to_compress:
        return existing_summary or ""

    llm = build_chat_model(base_url=base_url, model_name=model_name, temperature=0.1)
    response = await llm.ainvoke(
        [
            SystemMessage(
                content=(
                    "You are compressing older conversation history for a fitness assistant. "
                    "Write a concise Chinese summary focused on the active task, stable preferences, "
                    "temporary constraints, plan revisions, unresolved needs, and prior results that still matter. "
                    "Do not preserve chit-chat or repeated wording."
                )
            ),
            HumanMessage(
                content=(
                    f"Existing summary:\n{existing_summary or 'None'}\n\n"
                    f"Older messages to compress:\n{json.dumps(messages_to_compress, ensure_ascii=False, indent=2)}\n\n"
                    "Return only the updated compressed summary."
                )
            ),
        ]
    )
    content = response.content
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


async def _compress_session_messages_if_needed(
    *,
    session_state: Dict[str, Any],
    base_url: str,
    model_name: str,
    max_tokens: int,
    compression_ratio: float,
    recent_turn_window: int,
) -> None:
    messages = session_state["messages"]
    if _estimate_messages_token_count(messages) <= max_tokens:
        return

    while _estimate_messages_token_count(messages) > max_tokens:
        turns = _turn_ids(messages)
        if len(turns) <= 1:
            break

        turns_to_compress = max(1, int(len(turns) * compression_ratio))
        turns_to_compress = min(turns_to_compress, len(turns) - 1)
        older_turns = turns[:turns_to_compress]
        older_messages = _messages_for_turns(messages, older_turns)
        session_state["conversation_summary"] = await _compress_message_chunk(
            base_url=base_url,
            model_name=model_name,
            existing_summary=session_state.get("conversation_summary"),
            messages_to_compress=older_messages,
        )
        messages = [message for message in messages if int(message.get("turn", 0)) not in set(older_turns)]
        session_state["messages"] = messages

        if _estimate_messages_token_count(messages) <= max_tokens:
            break

        recent_messages = _select_recent_messages_with_backoff(messages, max_tokens, recent_turn_window)
        if len(recent_messages) < len(messages):
            dropped_turns = sorted(set(_turn_ids(messages)) - set(_turn_ids(recent_messages)))
            dropped_messages = _messages_for_turns(messages, dropped_turns)
            session_state["conversation_summary"] = await _compress_message_chunk(
                base_url=base_url,
                model_name=model_name,
                existing_summary=session_state.get("conversation_summary"),
                messages_to_compress=dropped_messages,
            )
            session_state["messages"] = recent_messages
            messages = recent_messages


def _add_session_message(session_state: Dict[str, Any], *, role: str, content: str, turn: int) -> None:
    session_state["messages"].append(
        {
            "role": role,
            "content": content,
            "turn": turn,
            "created_at": datetime.now().isoformat(),
        }
    )


def _prune_session_artifacts(session_artifacts: Dict[str, List[Dict[str, Any]]]) -> None:
    for key, items in list(session_artifacts.items()):
        limit = ARTIFACT_LIMITS.get(key)
        if limit is not None and len(items) > limit:
            session_artifacts[key] = items[-limit:]


def _merge_session_artifacts(
    session_artifacts: Dict[str, List[Dict[str, Any]]],
    artifacts: Dict[str, Any],
    prior_artifacts: Dict[str, Any],
    turn: int,
) -> None:
    created_at = datetime.now().isoformat()
    for key, value in artifacts.items():
        if key == "prior_artifacts":
            continue
        bucket = session_artifacts.setdefault(key, [])
        if key in LIST_ARTIFACT_KEYS and isinstance(value, list):
            prior_values = prior_artifacts.get(key)
            prior_count = len(prior_values) if isinstance(prior_values, list) else 0
            values = value[prior_count:]
        else:
            values = [value]
        for item in values:
            artifact_value = _artifact_entry_value(item)
            artifact_summary = _artifact_entry_summary(item)
            bucket.append(
                {
                    "value": artifact_value,
                    "summary": artifact_summary,
                    "turn": turn,
                    "created_at": created_at,
                }
            )
    _prune_session_artifacts(session_artifacts)


def _unwrap_prior_artifacts(session_artifacts: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    raw_prior_artifacts: Dict[str, Any] = {}
    for key, items in session_artifacts.items():
        if key == "final_answer":
            continue
        values = [item.get("value") for item in items if "value" in item]
        if not values:
            continue
        raw_prior_artifacts[key] = values if key in LIST_ARTIFACT_KEYS else values[-1]
    return raw_prior_artifacts


def _latest_profile_from_session(
    session_artifacts: Dict[str, List[Dict[str, Any]]],
    fallback_profile: UserProfile,
) -> UserProfile:
    latest_user_profile = _latest_session_artifact_value(session_artifacts, "user_profile")
    if isinstance(latest_user_profile, dict):
        return UserProfile(**latest_user_profile)
    return fallback_profile


def _latest_session_artifact_value(
    session_artifacts: Dict[str, List[Dict[str, Any]]],
    key: str,
) -> Any:
    items = session_artifacts.get(key) or []
    if not items:
        return None
    return items[-1].get("value")


async def interactive_chat(args: argparse.Namespace) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    event_handler = _build_event_handler()

    print("Fitness Assistant CLI")
    print("Type 'exit' or 'quit' to stop.")

    profile = load_profile(args.user_id) or build_default_profile(args)
    profile = await run_profile_onboarding_workflow(
        profile=profile,
        base_url=args.base_url,
        model_name=args.onboarding_model,
        prompt_user=_prompt_user,
        notify_user=_notify_user,
        event_handler=event_handler,
    )

    session_state: Dict[str, Any] = {
        "turn_index": 0,
        "messages": [],
        "conversation_summary": None,
        "artifacts": {},
    }

    while True:
        user_input = await _prompt_user("")
        session_state["turn_index"] += 1
        current_turn = session_state["turn_index"]
        _add_session_message(session_state, role="user", content=user_input, turn=current_turn)

        await _compress_session_messages_if_needed(
            session_state=session_state,
            base_url=args.base_url,
            model_name=args.fitness_strong_model or args.fitness_model,
            max_tokens=args.max_message_tokens,
            compression_ratio=args.message_compression_ratio,
            recent_turn_window=args.recent_turn_window,
        )

        profile = _latest_profile_from_session(session_state["artifacts"], profile)
        prior_artifacts = _unwrap_prior_artifacts(session_state["artifacts"])
        workflow_context = WorkflowContext(
            summary=session_state["conversation_summary"],
            conversation_summary=session_state["conversation_summary"],
            recent_messages=session_state["messages"],
            prior_final_answer=_latest_session_artifact_value(session_state["artifacts"], "final_answer"),
            prior_artifacts=prior_artifacts,
        )
        request = FitnessRequest(
            user_input=user_input,
            user_profile=profile,
            context=workflow_context,
            max_iterations=args.max_iterations,
        )

        result = await run_fitness_workflow(
            fitness_request=request,
            profile=profile,
            base_url=args.base_url,
            model_name=args.fitness_model,
            strong_model_name=args.fitness_strong_model or args.fitness_model,
            prompt_user=_prompt_user,
            notify_user=_notify_user,
            event_handler=event_handler,
            max_iterations=args.max_iterations,
        )

        final_answer = str(result.get("final_answer") or "")
        if final_answer:
            _add_session_message(session_state, role="assistant", content=final_answer, turn=current_turn)
        _merge_session_artifacts(
            session_state["artifacts"],
            result.get("artifacts") or {},
            prior_artifacts,
            current_turn,
        )
        profile = _latest_profile_from_session(session_state["artifacts"], profile)


def main() -> None:
    load_dotenv()
    args = parse_args()
    asyncio.run(interactive_chat(args))


if __name__ == "__main__":
    main()
