from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import invoke_structured_with_retry, build_chat_model, build_structured_output_instruction
from ..models import (
    FinalAnswerStructured,
    MealPlanRequest,
    MealPlanStructured,
    NutritionQuery,
    UserProfile,
    WorkoutPlanRequest,
    WorkoutPlanStructured,
)
from ..prompts import (
    FINAL_ANSWER_SYSTEM_PROMPT,
    MEAL_PLAN_SYSTEM_PROMPT,
    WORKOUT_PLAN_SYSTEM_PROMPT,
    build_final_answer_user_prompt,
    build_meal_plan_user_prompt,
    build_workout_plan_user_prompt,
)
from ..repositories.food_repository import find_foods_by_text
from ..repositories.profile_repository import load_profile, upsert_profile
from ..services.nutrition_service import format_mongo_food_summary, semantic_food_search
from ..services.profile_service import enrich_profile


async def prepare_profile_artifact(
    user_id: str,
    user_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    profile = UserProfile(**user_profile) if user_profile else load_profile(user_id)
    profile.user_id = user_id
    enriched = enrich_profile(profile)
    enriched = upsert_profile(enriched)
    return {
        "user_profile": enriched.model_dump(),
        "observation": (
            f"Profile is ready with targets of {enriched.target_calories} kcal, "
            f"{enriched.target_protein_g}g protein, {enriched.target_carbs_g}g carbs, "
            f"and {enriched.target_fat_g}g fat."
        ),
    }


async def search_food_candidates_artifact(
    user_input: str,
    use_full_database: bool = False,
    user_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    profile = UserProfile(**user_profile) if user_profile else None
    dietary_preferences = profile.dietary_preferences if profile else []
    macro_goals: Dict[str, float] = {}
    if profile and (profile.target_protein_g or 0) >= 120:
        macro_goals["protein_min"] = 15
    if profile and "keto" in [pref.lower() for pref in profile.dietary_preferences]:
        macro_goals["carbs_max"] = 10

    try:
        foods = semantic_food_search(
            NutritionQuery(
                query=user_input,
                use_full_database=use_full_database,
                dietary_restrictions=dietary_preferences,
                macro_goals=macro_goals,
                limit=12,
                similarity_threshold=0.2,
            )
        ).results
    except Exception:
        foods = []

    if not foods:
        try:
            foods = [
                format_mongo_food_summary(item)
                for item in find_foods_by_text(
                    query=user_input,
                    limit=8,
                    use_full_database=use_full_database,
                    include_ingredients=True,
                )
            ]
        except Exception:
            foods = []

    return {
        "food_candidates": foods,
        "observation": f"Nutrition search collected {len(foods)} candidate foods.",
    }


async def generate_meal_plan_artifact(
    user_input: str,
    user_profile: Dict[str, Any],
    meal_preferences: Optional[Dict[str, Any]] = None,
    food_candidates: Optional[List[Dict[str, Any]]] = None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_name: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    profile = UserProfile(**user_profile)
    meal_request = MealPlanRequest(
        meal_count=(meal_preferences or {}).get("meal_count", 5),
        days=(meal_preferences or {}).get("days", 7),
        preferences=meal_preferences or {},
    )

    llm = build_chat_model(
        base_url=base_url,
        model_name=model_name,
        temperature=0.4,
    )
    meal_plan = await invoke_structured_with_retry(
        llm,
        MealPlanStructured,
        [
            SystemMessage(
                content=MEAL_PLAN_SYSTEM_PROMPT
                + "\n\n"
                + build_structured_output_instruction(MealPlanStructured)
            ),
            HumanMessage(
                content=build_meal_plan_user_prompt(
                    user_input,
                    profile.model_dump(),
                    meal_request.model_dump(),
                    food_candidates or [],
                )
            ),
        ],
    )

    return {
        "meal_plan": {
            "user_id": profile.user_id,
            "plan_name": meal_plan.plan_name,
            "days": len(meal_plan.days),
            "target_macros": meal_plan.target_macros.model_dump(),
            "daily_plans": [day.model_dump() for day in meal_plan.days],
            "key_principles": meal_plan.key_principles,
            "shopping_tips": meal_plan.shopping_tips,
            "available_foods_count": len(food_candidates or []),
            "generated_at": datetime.now().isoformat(),
        },
        "observation": f"Meal plan tool generated a {len(meal_plan.days)}-day plan.",
    }


async def generate_workout_plan_artifact(
    user_input: str,
    user_profile: Dict[str, Any],
    workout_preferences: Optional[Dict[str, Any]] = None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_name: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    profile = UserProfile(**user_profile)
    workout_request = WorkoutPlanRequest(
        split_type=(workout_preferences or {}).get("split_type", "full_body"),
        training_style=(workout_preferences or {}).get("training_style", "hypertrophy"),
        days_per_week=(workout_preferences or {}).get("days_per_week", profile.workout_frequency or 3),
        duration_minutes=(workout_preferences or {}).get("duration_minutes", profile.workout_duration or 60),
    )

    llm = build_chat_model(
        base_url=base_url,
        model_name=model_name,
        temperature=0.4,
    )
    workout_plan = await invoke_structured_with_retry(
        llm,
        WorkoutPlanStructured,
        [
            SystemMessage(
                content=WORKOUT_PLAN_SYSTEM_PROMPT
                + "\n\n"
                + build_structured_output_instruction(WorkoutPlanStructured)
            ),
            HumanMessage(
                content=build_workout_plan_user_prompt(
                    user_input,
                    profile.model_dump(),
                    workout_request.model_dump(),
                )
            ),
        ],
    )

    return {
        "workout_plan": {
            "user_id": profile.user_id,
            "plan_name": workout_plan.plan_name,
            "split_type": workout_plan.split_type,
            "training_style": workout_plan.training_style,
            "days_per_week": workout_request.days_per_week,
            "duration_minutes": workout_request.duration_minutes,
            "weekly_schedule": [day.model_dump() for day in workout_plan.weekly_schedule],
            "progression_strategy": workout_plan.progression_strategy,
            "equipment_needed": workout_plan.equipment_needed,
            "key_principles": workout_plan.key_principles,
            "generated_at": datetime.now().isoformat(),
        },
        "observation": f"Workout plan tool generated a {workout_request.days_per_week}-day program.",
    }


async def summarize_final_answer_artifact(
    user_input: str,
    artifacts: Optional[Dict[str, Any]] = None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_name: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    llm = build_chat_model(
        base_url=base_url,
        model_name=model_name,
        temperature=0.3,
    )
    final_answer = await invoke_structured_with_retry(
        llm,
        FinalAnswerStructured,
        [
            SystemMessage(
                content=FINAL_ANSWER_SYSTEM_PROMPT
                + "\n\n"
                + build_structured_output_instruction(FinalAnswerStructured)
            ),
            HumanMessage(content=build_final_answer_user_prompt(user_input, artifacts or {})),
        ],
    )

    lines: List[str] = [final_answer.overview]
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
        "observation": "Final summary tool produced the final answer.",
    }
