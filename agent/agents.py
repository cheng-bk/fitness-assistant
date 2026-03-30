from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from .llm import invoke_structured_with_retry, build_chat_model, build_structured_output_instruction
from .models import Decision, FitnessRequest, IntentAnalysis, PlanList
from .models import ProfileAnswerInterpretation, ProfileMemoryUpdate, ProfileQuestion
from .prompts import (
    DECISION_SYSTEM_PROMPT,
    INTENT_SYSTEM_PROMPT,
    MEMORY_UPDATE_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    PROFILE_ANSWER_PARSE_SYSTEM_PROMPT,
    PROFILE_COLLECTION_SYSTEM_PROMPT,
    build_memory_update_user_prompt,
    build_decision_user_prompt,
    build_intent_user_prompt,
    build_planner_user_prompt,
    build_profile_answer_parse_user_prompt,
    build_profile_collection_user_prompt,
)
from .services.profile_service import (
    calculate_bmr,
    calculate_macros,
    calculate_tdee,
    enrich_profile,
)


class ProfileCollectionAgent:
    def __init__(self, base_url: str, model_name: str):
        self.llm = build_chat_model(
            base_url=base_url,
            model_name=model_name,
            temperature=0.2,
        )

    async def next_question(
        self,
        profile: Dict[str, Any],
        missing_fields: List[str],
    ) -> ProfileQuestion:
        return await invoke_structured_with_retry(
            self.llm,
            ProfileQuestion,
            [
                SystemMessage(
                    content=PROFILE_COLLECTION_SYSTEM_PROMPT
                    + "\n\n"
                    + build_structured_output_instruction(ProfileQuestion)
                ),
                HumanMessage(content=build_profile_collection_user_prompt(profile, missing_fields)),
            ],
        )

    async def parse_answer(
        self,
        field_name: str,
        question: str,
        answer: str,
        profile: Dict[str, Any],
    ) -> ProfileAnswerInterpretation:
        return await invoke_structured_with_retry(
            self.llm,
            ProfileAnswerInterpretation,
            [
                SystemMessage(
                    content=PROFILE_ANSWER_PARSE_SYSTEM_PROMPT
                    + "\n\n"
                    + build_structured_output_instruction(ProfileAnswerInterpretation)
                ),
                HumanMessage(
                    content=build_profile_answer_parse_user_prompt(
                        field_name=field_name,
                        question=question,
                        answer=answer,
                        profile=profile,
                    )
                ),
            ],
        )


class IntentInterpreterAgent:
    def __init__(self, base_url: str, model_name: str):
        self.llm = build_chat_model(
            base_url=base_url,
            model_name=model_name,
            temperature=0.2,
        )

    async def run(self, request: FitnessRequest) -> IntentAnalysis:
        return await invoke_structured_with_retry(
            self.llm,
            IntentAnalysis,
            [
                SystemMessage(
                    content=INTENT_SYSTEM_PROMPT
                    + "\n\n"
                    + build_structured_output_instruction(IntentAnalysis)
                ),
                HumanMessage(
                    content=build_intent_user_prompt(
                        request.user_input,
                        request.user_profile.model_dump() if request.user_profile else {},
                    )
                ),
            ],
        )


class MemoryAgent:
    def __init__(self, base_url: str, model_name: str):
        self.llm = build_chat_model(
            base_url=base_url,
            model_name=model_name,
            temperature=0.2,
        )

    async def run(
        self,
        request: FitnessRequest,
    ) -> ProfileMemoryUpdate:
        return await invoke_structured_with_retry(
            self.llm,
            ProfileMemoryUpdate,
            [
                SystemMessage(
                    content=MEMORY_UPDATE_SYSTEM_PROMPT
                    + "\n\n"
                    + build_structured_output_instruction(ProfileMemoryUpdate)
                ),
                HumanMessage(
                    content=build_memory_update_user_prompt(
                        request.user_input,
                        request.user_profile.model_dump() if request.user_profile else {},
                    )
                ),
            ],
        )


class PlannerAgent:
    def __init__(self, base_url: str, model_name: str):
        self.llm = build_chat_model(
            base_url=base_url,
            model_name=model_name,
            temperature=0.2,
        )

    async def run(
        self,
        request: FitnessRequest,
        intent: IntentAnalysis,
        completed_steps: List[Dict[str, Any]],
        artifacts: Dict[str, Any],
        available_tools: List[BaseTool],
    ) -> PlanList:
        return await invoke_structured_with_retry(
            self.llm,
            PlanList,
            [
                SystemMessage(
                    content=PLANNER_SYSTEM_PROMPT
                    + "\n\n"
                    + build_structured_output_instruction(PlanList)
                ),
                HumanMessage(
                    content=build_planner_user_prompt(
                        request.user_input,
                        intent.model_dump(),
                        completed_steps,
                        artifacts,
                        available_tools,
                    )
                ),
            ],
        )


class DecisionAgent:
    def __init__(self, base_url: str, model_name: str):
        self.llm = build_chat_model(
            base_url=base_url,
            model_name=model_name,
            temperature=0.1,
        )

    async def run(
        self,
        request: FitnessRequest,
        artifacts: Dict[str, Any],
        latest_observation: str,
        remaining_steps: List[Dict[str, Any]],
        iteration: int,
        max_iterations: int,
    ) -> Decision:
        return await invoke_structured_with_retry(
            self.llm,
            Decision,
            [
                SystemMessage(
                    content=DECISION_SYSTEM_PROMPT
                    + "\n\n"
                    + build_structured_output_instruction(Decision)
                ),
                HumanMessage(
                    content=build_decision_user_prompt(
                        request.user_input,
                        artifacts,
                        latest_observation,
                        remaining_steps,
                        iteration,
                        max_iterations,
                    )
                ),
            ],
        )


__all__ = [
    "build_chat_model",
    "calculate_bmr",
    "calculate_tdee",
    "calculate_macros",
    "enrich_profile",
    "IntentInterpreterAgent",
    "MemoryAgent",
    "PlannerAgent",
    "DecisionAgent",
    "ProfileCollectionAgent",
]
