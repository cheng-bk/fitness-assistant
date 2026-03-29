from typing import Any, Awaitable, Callable, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from ..agents import ProfileCollectionAgent
from ..models import UserProfile, WorkflowEvent
from ..services.profile_service import (
    apply_profile_interpretation,
    clear_profile_fields,
    format_profile_summary,
    get_missing_profile_fields,
    parse_profile_modification_request,
    update_profile,
)


class OnboardingState(TypedDict):
    profile: UserProfile
    missing_fields: List[str]
    current_question_field: Optional[str]
    current_question_text: str
    latest_answer: str
    should_modify_existing: bool
    profile_modified: bool
    done: bool
    next_node: str


class ProfileOnboardingGraph:

    def __init__(
        self,
        base_url: str,
        model_name: str,
        prompt_user: Callable[[str], Awaitable[str]],
        notify_user: Callable[[str], None],
        event_handler: Optional[Callable[[WorkflowEvent], None]] = None,
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.prompt_user = prompt_user
        self.notify_user = notify_user
        self.event_handler = event_handler
        self.collector = ProfileCollectionAgent(base_url=base_url, model_name=model_name)
        self.graph = self._build_graph().compile()

    def _emit_event(self, event: WorkflowEvent) -> None:
        if self.event_handler is None:
            return
        self.event_handler(event)

    def _emit_onboarding_event(self, event_type: str, node: str, summary: str, **payload: Any) -> None:
        self._emit_event(
            WorkflowEvent(
                event_type=event_type,
                phase="onboarding",
                node=node,
                summary=summary,
                data=payload,
            )
        )

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(OnboardingState)
        workflow.add_node("inspect_profile", self._inspect_profile)
        workflow.add_node("review_existing_profile", self._review_existing_profile)
        workflow.add_node("select_modification_fields", self._select_modification_fields)
        workflow.add_node("ask_missing_field", self._ask_missing_field)
        workflow.add_node("parse_missing_field_answer", self._parse_missing_field_answer)
        workflow.add_node("finalize_profile", self._finalize_profile)

        workflow.set_entry_point("inspect_profile")
        workflow.add_conditional_edges(
            "inspect_profile",
            self._route_after_inspection,
            {
                "review_existing_profile": "review_existing_profile",
                "ask_missing_field": "ask_missing_field",
                "finalize_profile": "finalize_profile",
            },
        )
        workflow.add_conditional_edges(
            "review_existing_profile",
            self._route_after_review,
            {
                "select_modification_fields": "select_modification_fields",
                "finalize_profile": "finalize_profile",
            },
        )
        workflow.add_conditional_edges(
            "select_modification_fields",
            self._route_after_field_selection,
            {
                "ask_missing_field": "ask_missing_field",
                "finalize_profile": "finalize_profile",
            },
        )
        workflow.add_edge("ask_missing_field", "parse_missing_field_answer")
        workflow.add_conditional_edges(
            "parse_missing_field_answer",
            self._route_after_answer_parse,
            {
                "ask_missing_field": "ask_missing_field",
                "finalize_profile": "finalize_profile",
            },
        )
        workflow.add_edge("finalize_profile", END)
        return workflow

    async def _inspect_profile(self, state: OnboardingState) -> OnboardingState:
        state["missing_fields"] = get_missing_profile_fields(state["profile"])
        self._emit_onboarding_event(
            "onboarding_inspect",
            "inspect_profile",
            "Inspected current profile completeness",
            missing_fields=state["missing_fields"],
            profile=state["profile"].model_dump(),
        )
        if state["missing_fields"]:
            self.notify_user("先把你的基础信息补齐，这样后面的饮食和训练建议会更准确。")
            state["next_node"] = "ask_missing_field"
        else:
            state["next_node"] = "review_existing_profile"
        return state

    async def _review_existing_profile(self, state: OnboardingState) -> OnboardingState:
        self._emit_onboarding_event(
            "onboarding_review",
            "review_existing_profile",
            "Reviewed existing profile with the user",
            profile=state["profile"].model_dump(),
        )
        self.notify_user(format_profile_summary(state["profile"]))
        answer = await self.prompt_user(
            "如果这些信息需要修改，请回复“要修改”；如果不用修改，请回复“继续”。"
        )
        normalized = answer.lower()
        if any(token in normalized for token in ["继续", "不用", "不需要", "否", "no", "n"]):
            state["should_modify_existing"] = False
            state["next_node"] = "finalize_profile"
        else:
            state["should_modify_existing"] = True
            state["next_node"] = "select_modification_fields"
        self._emit_onboarding_event(
            "onboarding_review_decision",
            "review_existing_profile",
            "Decided whether the existing profile should be modified",
            answer=answer,
            should_modify_existing=state["should_modify_existing"],
            next_node=state["next_node"],
        )
        return state

    async def _select_modification_fields(self, state: OnboardingState) -> OnboardingState:
        answer = await self.prompt_user(
            "你想修改哪些信息？可以直接回复：年龄、体重、身高、性别、活动水平、健身目标、每周训练次数、单次训练时长；多个项目一起说也可以。"
        )
        selected_fields = parse_profile_modification_request(answer)
        self._emit_onboarding_event(
            "onboarding_select_fields",
            "select_modification_fields",
            "Identified profile fields selected for modification",
            answer=answer,
            selected_fields=selected_fields,
        )
        if not selected_fields:
            self.notify_user("我先保留当前信息不变，后面如果你想修改，也可以重新开始对话时调整。")
            state["next_node"] = "finalize_profile"
            return state

        state["profile"] = clear_profile_fields(state["profile"], selected_fields)
        state["profile_modified"] = True
        state["missing_fields"] = get_missing_profile_fields(state["profile"])
        self.notify_user("好的，我们来更新这些信息。")
        state["next_node"] = "ask_missing_field" if state["missing_fields"] else "finalize_profile"
        return state

    async def _ask_missing_field(self, state: OnboardingState) -> OnboardingState:
        question_output = await self.collector.next_question(
            state["profile"].model_dump(),
            state["missing_fields"],
        )
        state["current_question_field"] = question_output.field_name
        state["current_question_text"] = question_output.question.strip()
        self._emit_onboarding_event(
            "onboarding_question",
            "ask_missing_field",
            f"Asking for field: {state['current_question_field']}",
            field_name=state["current_question_field"],
            question=state["current_question_text"],
            missing_fields=state["missing_fields"],
        )
        state["latest_answer"] = await self.prompt_user(state["current_question_text"])
        return state

    async def _parse_missing_field_answer(self, state: OnboardingState) -> OnboardingState:
        interpretation = await self.collector.parse_answer(
            field_name=state["current_question_field"] or "age",
            question=state["current_question_text"],
            answer=state["latest_answer"],
            profile=state["profile"].model_dump(),
        )
        self._emit_onboarding_event(
            "onboarding_answer",
            "parse_missing_field_answer",
            f"Parsed answer for field: {state['current_question_field']}",
            field_name=state["current_question_field"],
            answer=state["latest_answer"],
            is_valid=interpretation.is_valid,
        )
        if interpretation.is_valid:
            state["profile"] = apply_profile_interpretation(state["profile"], interpretation)
            state["profile_modified"] = True
            self.notify_user(interpretation.acknowledgement)
            state["missing_fields"] = get_missing_profile_fields(state["profile"])
            state["next_node"] = "finalize_profile" if not state["missing_fields"] else "ask_missing_field"
            self._emit_onboarding_event(
                "onboarding_update",
                "parse_missing_field_answer",
                f"Updated field: {state['current_question_field']}",
                field_name=state["current_question_field"],
                profile=state["profile"].model_dump(),
                missing_fields=state["missing_fields"],
                next_node=state["next_node"],
            )
        else:
            self.notify_user(
                interpretation.follow_up_question
            )
            state["next_node"] = "ask_missing_field"
            self._emit_onboarding_event(
                "onboarding_retry",
                "parse_missing_field_answer",
                f"Retrying field: {state['current_question_field']}",
                field_name=state["current_question_field"],
                follow_up_question=interpretation.follow_up_question,
            )
        return state

    async def _finalize_profile(self, state: OnboardingState) -> OnboardingState:
        if state["profile_modified"]:
            state["profile"] = update_profile(state["profile"])
        if state["missing_fields"]:
            self.notify_user("还有信息未补齐，不过我先保留当前资料。")
        else:
            self.notify_user("基础信息已经准备好了，我们开始正式聊天吧。")
        state["done"] = True
        self._emit_onboarding_event(
            "onboarding_complete",
            "finalize_profile",
            "Onboarding workflow completed",
            profile=state["profile"].model_dump(),
            missing_fields=state["missing_fields"],
        )
        return state

    def _route_after_inspection(self, state: OnboardingState) -> str:
        return state["next_node"]

    def _route_after_review(self, state: OnboardingState) -> str:
        return state["next_node"]

    def _route_after_field_selection(self, state: OnboardingState) -> str:
        return state["next_node"]

    def _route_after_answer_parse(self, state: OnboardingState) -> str:
        return state["next_node"]

    async def run(self, profile: UserProfile) -> UserProfile:
        initial_state: OnboardingState = {
            "profile": profile,
            "missing_fields": [],
            "current_question_field": None,
            "current_question_text": "",
            "latest_answer": "",
            "should_modify_existing": False,
            "profile_modified": False,
            "done": False,
            "next_node": "inspect_profile",
        }
        final_state = await self.graph.ainvoke(initial_state)
        return final_state["profile"]
