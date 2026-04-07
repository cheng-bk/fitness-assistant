from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from .llm import invoke_structured_with_retry, build_chat_model, build_structured_output_instruction
from .models import Decision, FinalAnswerStructured, FitnessRequest, IntentAnalysis, PlanList
from .models import ProfileAnswerInterpretation, ProfileMemoryUpdate, ProfileQuestion
from .prompts import (
    DECISION_SYSTEM_PROMPT,
    FINAL_ANSWER_SYSTEM_PROMPT,
    INTENT_SYSTEM_PROMPT,
    MEMORY_UPDATE_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    PROFILE_ANSWER_PARSE_SYSTEM_PROMPT,
    PROFILE_COLLECTION_SYSTEM_PROMPT,
    build_memory_update_user_prompt,
    build_decision_user_prompt,
    build_final_answer_user_prompt,
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


def _latest_artifact_value(artifacts: Dict[str, Any], key: str) -> Any:
    value = artifacts.get(key)
    if isinstance(value, list):
        return value[-1] if value else None
    return value


def _format_meal_plan_detail(meal_plan: Dict[str, Any]) -> str:
    if not meal_plan:
        return ""

    lines: List[str] = []
    plan_name = meal_plan.get("plan_name") or "饮食方案"
    target_macros = meal_plan.get("target_macros") or {}
    lines.append(f"Meal plan: {plan_name}")
    if target_macros:
        lines.append(
            "Target macros:"
            f" calories {target_macros.get('calories', '-')},"
            f" protein {target_macros.get('protein_g', '-') }g,"
            f" carbs {target_macros.get('carbs_g', '-') }g,"
            f" fat {target_macros.get('fat_g', '-') }g"
        )

    for day in meal_plan.get("daily_plans") or []:
        day_label = day.get("day_name") or f"Day {day.get('day', '')}".strip()
        lines.append(f"{day_label}:")
        for meal in day.get("meals") or []:
            total_macros = meal.get("total_macros") or {}
            macro_text = (
                f" ({total_macros.get('calories', '-')} kcal, "
                f"P {total_macros.get('protein_g', '-')}g, "
                f"C {total_macros.get('carbs_g', '-')}g, "
                f"F {total_macros.get('fat_g', '-')}g)"
            )
            lines.append(f"- {meal.get('meal_name', 'Meal')}{macro_text}")
            for food in meal.get("foods") or []:
                lines.append(
                    f"  - {food.get('food_name', 'food')}: {food.get('portion', '-')}"
                )
            if meal.get("preparation_notes"):
                lines.append(f"  - notes: {meal.get('preparation_notes')}")
        daily_totals = day.get("daily_totals") or {}
        if daily_totals:
            lines.append(
                "  Daily totals:"
                f" {daily_totals.get('calories', '-')} kcal,"
                f" P {daily_totals.get('protein_g', '-')}g,"
                f" C {daily_totals.get('carbs_g', '-')}g,"
                f" F {daily_totals.get('fat_g', '-')}g"
            )

    if meal_plan.get("key_principles"):
        lines.append("Key principles:")
        lines.extend(f"- {item}" for item in meal_plan.get("key_principles") or [])
    if meal_plan.get("shopping_tips"):
        lines.append("Shopping tips:")
        lines.extend(f"- {item}" for item in meal_plan.get("shopping_tips") or [])

    return "\n".join(lines)


def _format_workout_plan_detail(workout_plan: Dict[str, Any]) -> str:
    if not workout_plan:
        return ""

    lines: List[str] = []
    lines.append(f"Workout plan: {workout_plan.get('plan_name') or '训练方案'}")
    lines.append(
        f"Split: {workout_plan.get('split_type', '-')}, style: {workout_plan.get('training_style', '-')}"
    )
    for day in workout_plan.get("weekly_schedule") or []:
        day_label = day.get("day_name") or f"Day {day.get('day', '')}"
        lines.append(f"{day_label} - {day.get('focus', '-')}:")
        for exercise in day.get("exercises") or []:
            lines.append(
                f"- {exercise.get('exercise_name', 'exercise')}: "
                f"{exercise.get('sets', '-')} sets x {exercise.get('reps', '-')}, "
                f"rest {exercise.get('rest_seconds', '-')}s"
            )
            if exercise.get("notes"):
                lines.append(f"  - notes: {exercise.get('notes')}")
    if workout_plan.get("progression_strategy"):
        lines.append(f"Progression: {workout_plan.get('progression_strategy')}")
    if workout_plan.get("key_principles"):
        lines.append("Key principles:")
        lines.extend(f"- {item}" for item in workout_plan.get("key_principles") or [])
    return "\n".join(lines)


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
                        request.context.model_dump() if request.context else {},
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
            extra_body={"enable_thinking": True}
        )

    async def run(
        self,
        request: FitnessRequest,
        profile: Dict[str, Any],
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
                        profile,
                        intent.model_dump(),
                        completed_steps,
                        artifacts,
                        available_tools,
                        request.context.model_dump() if request.context else {},
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
        profile: Dict[str, Any],
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
                        profile,
                        artifacts,
                        latest_observation,
                        remaining_steps,
                        iteration,
                        max_iterations,
                    )
                ),
            ],
        )


class SummaryAgent:
    def __init__(self, base_url: str, model_name: str):
        self.llm = build_chat_model(
            base_url=base_url,
            model_name=model_name,
            temperature=0.3,
            extra_body={"enable_thinking": True}
        )

    async def run(
        self,
        user_input: str,
        profile: Dict[str, Any],
        artifacts: Dict[str, Any],
        workflow_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        final_answer = await invoke_structured_with_retry(
            self.llm,
            FinalAnswerStructured,
            [
                SystemMessage(
                    content=FINAL_ANSWER_SYSTEM_PROMPT
                    + "\n\n"
                    + build_structured_output_instruction(FinalAnswerStructured)
                ),
                HumanMessage(content=build_final_answer_user_prompt(user_input, profile, artifacts, workflow_context)),
            ],
        )

        lines: List[str] = [final_answer.overview]
        meal_plan = _latest_artifact_value(artifacts, "meal_plan")
        workout_plan = _latest_artifact_value(artifacts, "workout_plan")
        meal_plan_detail = _format_meal_plan_detail(meal_plan) if isinstance(meal_plan, dict) else ""
        workout_plan_detail = _format_workout_plan_detail(workout_plan) if isinstance(workout_plan, dict) else ""
        if meal_plan_detail:
            lines.append(meal_plan_detail)
        if workout_plan_detail:
            lines.append(workout_plan_detail)
        if final_answer.completed_work:
            lines.append("Completed work:")
            lines.extend(f"- {item}" for item in final_answer.completed_work)
        if final_answer.nutrition_guidance:
            lines.append("Nutrition guidance:")
            lines.extend(f"- {item}" for item in final_answer.nutrition_guidance)
        if final_answer.training_guidance:
            lines.append("Training guidance:")
            lines.extend(f"- {item}" for item in final_answer.training_guidance)
        if final_answer.next_steps:
            lines.append("Next steps:")
            lines.extend(f"- {item}" for item in final_answer.next_steps)

        return {
            "final_answer": "\n".join(lines),
            "observation": "Summary agent produced the final answer.",
        }


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
    "SummaryAgent",
    "ProfileCollectionAgent",
]
