from typing import Any, Awaitable, Callable, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated

from ..agents import DecisionAgent, IntentInterpreterAgent, MemoryAgent, PlannerAgent, SummaryAgent
from ..models import ActionRecord, FitnessRequest, IntentAnalysis, UserProfile, WorkflowEvent
from ..repositories.profile_repository import upsert_profile
from ..services.profile_service import apply_profile_memory_update
from ..tools import build_tool_registry


class FitnessState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    request: FitnessRequest
    intent: Optional[Dict[str, Any]]
    memory_update: Optional[Dict[str, Any]]
    active_step: Optional[Dict[str, Any]]
    remaining_steps: List[Dict[str, Any]]
    executed_steps: List[Dict[str, Any]]
    latest_observation: str
    artifacts: Dict[str, Any]
    errors: List[str]
    iterations: int
    done: bool
    final_answer: str
    next_node: str


class FitnessGraph:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        prompt_user: Callable[[str], Awaitable[str]],
        notify_user: Callable[[str], None],
        event_handler: Optional[Callable[[WorkflowEvent], None]] = None,
        planner_model_name: Optional[str] = None,
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.planner_model_name = planner_model_name or model_name
        self.prompt_user = prompt_user
        self.notify_user = notify_user
        self.intent_agent = IntentInterpreterAgent(base_url=base_url, model_name=model_name)
        self.memory_agent = MemoryAgent(base_url=base_url, model_name=model_name)
        self.planner_agent = PlannerAgent(base_url=base_url, model_name=self.planner_model_name)
        self.decision_agent = DecisionAgent(base_url=base_url, model_name=self.planner_model_name)
        self.summary_agent = SummaryAgent(base_url=base_url, model_name=model_name)
        self.tool_registry = build_tool_registry()
        self.event_handler = event_handler
        self.graph = self._build_graph().compile()

    def _emit_event(self, event: WorkflowEvent) -> None:
        if self.event_handler is None:
            return
        self.event_handler(event)

    def _emit_workflow_event(self, event_type: str, node: str, summary: str, **payload: Any) -> None:
        self._emit_event(
            WorkflowEvent(
                event_type=event_type,
                phase="main_workflow",
                node=node,
                summary=summary,
                data=payload,
            )
        )

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(FitnessState)
        workflow.add_node("intent_interpreter", self._interpret_intent)
        workflow.add_node("memory_update", self._update_memory)
        workflow.add_node("planner", self._plan)
        workflow.add_node("action", self._act)
        workflow.add_node("observation", self._observe)
        workflow.add_node("decision", self._decide)
        workflow.add_node("finalize", self._finalize)

        workflow.set_entry_point("intent_interpreter")
        workflow.add_conditional_edges(
            "intent_interpreter",
            self._route_after_intent,
            {
                "memory_update": "memory_update",
                "planner": "planner",
            },
        )
        workflow.add_edge("memory_update", "planner")
        workflow.add_conditional_edges(
            "planner",
            self._route_after_plan,
            {"action": "action", "decision": "decision"},
        )
        workflow.add_edge("action", "observation")
        workflow.add_edge("observation", "decision")
        workflow.add_conditional_edges(
            "decision",
            self._route_after_decision,
            {"planner": "planner", "action": "action", "finalize": "finalize"},
        )
        workflow.add_edge("finalize", END)
        return workflow

    async def _interpret_intent(self, state: FitnessState) -> FitnessState:
        intent = await self.intent_agent.run(state["request"])
        state["intent"] = intent.model_dump()
        state["next_node"] = "memory_update" if intent.needs_profile_update else "planner"
        self._emit_workflow_event(
            "intent",
            "intent_interpreter",
            f"Interpreted intent: {intent.primary_goal}",
            primary_goal=intent.primary_goal,
            intent=intent.model_dump(),
            next_node=state["next_node"],
        )
        state["messages"].append(SystemMessage(content=f"Intent interpreted: {intent.primary_goal}"))
        return state

    async def _update_memory(self, state: FitnessState) -> FitnessState:
        request = state["request"]
        memory_update = await self.memory_agent.run(request)
        state["memory_update"] = memory_update.model_dump()

        if memory_update.should_update_profile and request.user_profile is not None:
            request.user_profile = apply_profile_memory_update(request.user_profile, memory_update)
            request.user_profile = upsert_profile(request.user_profile)
            state["artifacts"]["user_profile"] = request.user_profile.model_dump()
            self._emit_workflow_event(
                "memory_update",
                "memory_update",
                "Updated profile memory before planning",
                memory_update=memory_update.model_dump(),
                profile=request.user_profile.model_dump(),
            )
            if memory_update.acknowledgement:
                self.notify_user(memory_update.acknowledgement)
        else:
            self._emit_workflow_event(
                "memory_update",
                "memory_update",
                "No long-term profile memory changes applied",
                memory_update=memory_update.model_dump(),
            )
        return state

    async def _plan(self, state: FitnessState) -> FitnessState:
        intent = IntentAnalysis(**(state["intent"] or {"primary_goal": "fitness planning"}))
        planner_output = await self.planner_agent.run(
            state["request"],
            intent,
            state["executed_steps"],
            state["artifacts"],
            list(self.tool_registry.values()),
        )
        next_step = planner_output.next_step

        state["active_step"] = next_step.model_dump() if next_step is not None else None
        state["remaining_steps"] = [step.model_dump() for step in planner_output.remaining_steps]

        self._emit_workflow_event(
            "plan",
            "planner",
            f"Planned next tool: {(state['active_step'] or {}).get('tool_name', 'none')}",
            reasoning=planner_output.reasoning,
            active_step=state["active_step"],
            remaining_steps=state["remaining_steps"],
        )
        state["messages"].append(SystemMessage(content=f"Planner: {planner_output.reasoning}"))
        return state

    def _build_tool_payload(self, tool_name: str, state: FitnessState) -> Dict[str, Any]:
        request = state["request"]
        artifacts = state["artifacts"]
        profile = request.user_profile

        if tool_name == "search_food_candidates":
            return {
                "user_input": request.user_input,
                "use_full_database": request.use_full_database,
                "user_profile": artifacts.get("user_profile") or (profile.model_dump() if profile else None),
            }
        if tool_name == "generate_meal_plan":
            return {
                "user_input": request.user_input,
                "user_profile": artifacts["user_profile"],
                "meal_preferences": {},
                "food_candidates": artifacts.get("food_candidates", []),
                "base_url": self.base_url,
                "model_name": self.model_name,
            }
        if tool_name == "generate_workout_plan":
            return {
                "user_input": request.user_input,
                "user_profile": artifacts["user_profile"],
                "workout_preferences": {},
                "base_url": self.base_url,
                "model_name": self.model_name,
            }
        raise ValueError(f"Unknown tool: {tool_name}")

    def _set_active_step_status(self, state: FitnessState, status: str) -> None:
        active_step = state.get("active_step")
        if active_step is None:
            return
        active_step["status"] = status

    async def _act(self, state: FitnessState) -> FitnessState:
        step = state.get("active_step") or {}
        tool_name = step.get("tool_name")

        try:
            if not tool_name or tool_name not in self.tool_registry:
                raise ValueError(f"Tool '{tool_name}' is not registered.")

            tool = self.tool_registry[tool_name]
            tool_payload = self._build_tool_payload(tool_name, state)
            self._emit_workflow_event(
                "tool_start",
                "action",
                f"Starting tool: {tool_name}",
                tool_name=tool_name,
                step=step,
                payload=tool_payload,
                iteration=state["iterations"] + 1,
            )

            tool_result = await tool.ainvoke(tool_payload)
            new_artifacts = dict(tool_result)
            observation = new_artifacts.pop("observation", f"Tool {tool_name} completed.")
            state["artifacts"].update(new_artifacts)
            state["latest_observation"] = observation
            self._set_active_step_status(state, "completed")
            self._emit_workflow_event(
                "tool_result",
                "action",
                f"Completed tool: {tool_name}",
                tool_name=tool_name,
                iteration=state["iterations"] + 1,
                observation=observation,
                artifact_keys=list(new_artifacts.keys()),
            )

            record = ActionRecord(
                iteration=state["iterations"] + 1,
                step_id=step.get("id", tool_name),
                tool_name=tool_name,
                objective=step.get("objective", ""),
                status="completed",
                observation=observation,
                artifact_keys=list(new_artifacts.keys()),
            )
            state["executed_steps"].append(record.model_dump())
        except Exception as exc:
            error_message = f"{tool_name} failed: {exc}"
            state["errors"].append(error_message)
            state["latest_observation"] = error_message
            self._set_active_step_status(state, "failed")
            self._emit_workflow_event(
                "tool_error",
                "action",
                f"Tool failed: {tool_name or 'unknown'}",
                tool_name=tool_name or "unknown",
                iteration=state["iterations"] + 1,
                error=error_message,
            )
            record = ActionRecord(
                iteration=state["iterations"] + 1,
                step_id=step.get("id", tool_name or "unknown"),
                tool_name=tool_name or "unknown",
                objective=step.get("objective", ""),
                status="failed",
                observation=error_message,
                artifact_keys=[],
            )
            state["executed_steps"].append(record.model_dump())

        state["iterations"] += 1
        state["messages"].append(SystemMessage(content=f"Action: {state['latest_observation']}"))
        return state

    async def _observe(self, state: FitnessState) -> FitnessState:
        self._emit_workflow_event(
            "observation",
            "observation",
            "Recorded tool observation",
            iteration=state["iterations"],
            observation=state["latest_observation"],
        )
        state["messages"].append(SystemMessage(content=f"Observation: {state['latest_observation']}"))
        return state

    async def _decide(self, state: FitnessState) -> FitnessState:
        request = state["request"]
        if state["iterations"] >= request.max_iterations:
            state["done"] = True
            state["next_node"] = "finalize"
            self._emit_workflow_event(
                "decision",
                "decision",
                "Reached max iterations; finishing workflow",
                decision="finish",
                reasoning="Reached max iterations.",
                iteration=state["iterations"],
                next_node="finalize",
            )
            return state

        decision = await self.decision_agent.run(
            request=request,
            artifacts=state["artifacts"],
            latest_observation=state["latest_observation"],
            remaining_steps=state["remaining_steps"],
            iteration=state["iterations"],
            max_iterations=request.max_iterations,
        )

        if decision.should_finish or decision.decision == "finish":
            state["done"] = True
            state["next_node"] = "finalize"
        elif decision.decision == "replan":
            state["next_node"] = "planner"
        elif state["remaining_steps"]:
            state["active_step"] = state["remaining_steps"].pop(0)
            state["next_node"] = "action"
        else:
            state["next_node"] = "planner"

        self._emit_workflow_event(
            "decision",
            "decision",
            f"Workflow decision: {decision.decision}",
            decision=decision.decision,
            reasoning=decision.reasoning,
            should_finish=decision.should_finish,
            iteration=state["iterations"],
            next_node=state["next_node"],
        )
        state["messages"].append(SystemMessage(content=f"Decision: {decision.reasoning}"))
        return state

    def _route_after_decision(self, state: FitnessState) -> str:
        return state.get("next_node", "finalize")

    def _route_after_intent(self, state: FitnessState) -> str:
        return state.get("next_node", "planner")

    def _route_after_plan(self, state: FitnessState) -> str:
        return "action" if state.get("active_step") else "decision"

    async def _finalize(self, state: FitnessState) -> FitnessState:
        if not state["artifacts"].get("final_answer"):
            self._emit_workflow_event(
                "summary_start",
                "finalize",
                "Starting summary agent",
                iteration=state["iterations"],
                artifact_keys=list(state["artifacts"].keys()),
            )
            final_result = await self.summary_agent.run(
                user_input=state["request"].user_input,
                artifacts=state["artifacts"],
            )
            final_artifact = dict(final_result)
            observation = final_artifact.pop("observation", "Summary agent completed.")
            state["artifacts"].update(final_artifact)
            state["latest_observation"] = observation
            self._emit_workflow_event(
                "summary_result",
                "finalize",
                "Completed summary agent",
                iteration=state["iterations"],
                observation=observation,
                artifact_keys=list(final_artifact.keys()),
            )

        state["final_answer"] = state["artifacts"].get(
            "final_answer", "The workflow completed without a final answer."
        )
        self._emit_workflow_event(
            "final_answer",
            "finalize",
            "Main workflow completed",
            final_answer=state["final_answer"],
            iterations=state["iterations"],
            errors=state["errors"],
        )
        return state

    async def run(self, request: FitnessRequest) -> Dict[str, Any]:
        initial_state: FitnessState = {
            "messages": [HumanMessage(content=request.user_input)],
            "request": request,
            "intent": None,
            "memory_update": None,
            "active_step": None,
            "remaining_steps": [],
            "executed_steps": [],
            "latest_observation": "",
            "artifacts": {},
            "errors": [],
            "iterations": 0,
            "done": False,
            "final_answer": "",
            "next_node": "planner",
        }
        result = await self.graph.ainvoke(initial_state)

        if self.notify_user is not None and result.get("final_answer"):
            self.notify_user(result["final_answer"])

        if self.notify_user is not None and result.get("errors"):
            self.notify_user("[Errors]\n" + "\n".join(f"- {error}" for error in result["errors"]))

        return result
