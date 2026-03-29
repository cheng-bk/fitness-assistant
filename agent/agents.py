from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from .llm import invoke_structured_with_retry, build_chat_model, build_structured_output_instruction
from .models import FitnessRequest, DecisionOutput, IntentAnalysis, PlannerOutput
from .models import ProfileAnswerInterpretation, ProfileQuestionOutput
from .prompts import (
    DECISION_SYSTEM_PROMPT,
    INTENT_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    PROFILE_ANSWER_PARSE_SYSTEM_PROMPT,
    PROFILE_COLLECTION_SYSTEM_PROMPT,
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


def _get_profile_preferences(request: FitnessRequest) -> tuple[Dict[str, Any], Dict[str, Any]]:
    profile = request.user_profile
    if profile is None:
        return {}, {}
    meal_preferences = getattr(profile, "meal_preferences", {}) or {}
    workout_preferences = getattr(profile, "workout_preferences", {}) or {}
    return meal_preferences, workout_preferences


class IntentInterpreterAgent:
    def __init__(self, base_url: str, model_name: str):
        self.llm = build_chat_model(
            base_url=base_url,
            model_name=model_name,
            temperature=0.1,
        )

    async def run(self, request: FitnessRequest) -> IntentAnalysis:
        meal_preferences, workout_preferences = _get_profile_preferences(request)
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
                        meal_preferences,
                        workout_preferences,
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
        available_tools: List[str],
    ) -> PlannerOutput:
        return await invoke_structured_with_retry(
            self.llm,
            PlannerOutput,
            [
                SystemMessage(
                    content=PLANNER_SYSTEM_PROMPT
                    + "\n\n"
                    + build_structured_output_instruction(PlannerOutput)
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
    ) -> DecisionOutput:
        return await invoke_structured_with_retry(
            self.llm,
            DecisionOutput,
            [
                SystemMessage(
                    content=DECISION_SYSTEM_PROMPT
                    + "\n\n"
                    + build_structured_output_instruction(DecisionOutput)
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
    ) -> ProfileQuestionOutput:
        return await invoke_structured_with_retry(
            self.llm,
            ProfileQuestionOutput,
            [
                SystemMessage(
                    content=PROFILE_COLLECTION_SYSTEM_PROMPT
                    + "\n\n"
                    + build_structured_output_instruction(ProfileQuestionOutput)
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


__all__ = [
    "build_chat_model",
    "calculate_bmr",
    "calculate_tdee",
    "calculate_macros",
    "enrich_profile",
    "IntentInterpreterAgent",
    "PlannerAgent",
    "DecisionAgent",
    "ProfileCollectionAgent",
]
