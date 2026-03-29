import argparse
import asyncio
import json
import os
from typing import Any, Dict, Optional
from dotenv import load_dotenv

from agent.application.use_cases import run_fitness_workflow, run_profile_onboarding_workflow
from agent.models import UserProfile, WorkflowEvent
from agent.services.profile_service import (
    load_profile,
)


def _format_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


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
            print(f"- field_name: {payload.get('field_name')}")
            
        elif event_type == "onboarding_complete":
            print(f"- profile: {_format_json(payload.get('profile'))}")
            print(f"- missing_fields: {payload.get('missing_fields')}")
        
        
        elif event_type == "intent":
            print(f"- primary_goal: {payload.get('primary_goal')}")
            
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
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "qwen3.5-plus"))
    parser.add_argument("--planner-model", default=None)
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--use-full-database", action="store_true")
    parser.add_argument("--age", type=int, default=None)
    parser.add_argument("--weight", type=float, default=None)
    parser.add_argument("--height", type=float, default=None)
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
            print("\nAssistant>\n鎴戣繕娌℃敹鍒颁綘鐨勫洖绛旓紝鍙互鐩存帴鎸夎嚜鐒惰瑷€鍥炲銆?")
            continue
        if answer.lower() in {"exit", "quit"}:
            print("Bye.")
            raise SystemExit(0)
        return answer


def _notify_user(message: str) -> None:
    print(f"\nAssistant>\n{message}")


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
        model_name=args.model,
        prompt_user=_prompt_user,
        notify_user=_notify_user,
        event_handler=event_handler,
    )

    # await run_fitness_workflow(
    #     profile=profile,
    #     base_url=args.base_url,
    #     model_name=args.model,
    #     planner_model_name=args.planner_model or args.model,
    #     prompt_user=_prompt_user,
    #     notify_user=_notify_user,
    #     event_handler=event_handler,
    #     max_iterations=args.max_iterations
    # )


def main() -> None:
    load_dotenv()
    args = parse_args()
    asyncio.run(interactive_chat(args))


if __name__ == "__main__":
    main()
