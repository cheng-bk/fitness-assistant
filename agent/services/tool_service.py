from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from ..models import (
    MealPlanRequest,
    MealPlanStructured,
    UserProfile,
    WorkoutPlanRequest,
    WorkoutPlanStructured,
)
from ..prompts import (
    MEAL_PLAN_SYSTEM_PROMPT,
    WORKOUT_PLAN_SYSTEM_PROMPT,
    build_meal_plan_user_prompt,
    build_workout_plan_user_prompt,
)

from ..llm import invoke_structured_with_retry, build_chat_model, build_structured_output_instruction
from ..repositories.food_repository import find_foods_by_text
from ..services.nutrition_service import build_meal_candidate_bundle, format_mongo_food_summary


async def search_food_candidates_artifact(
    user_input: str,
    food_types: Optional[List[str]] = None,
    user_profile: Optional[Dict[str, Any]] = None,
    protein_min: Optional[float] = None,
    carbs_min: Optional[float] = None,
    carbs_max: Optional[float] = None,
    calories_max: Optional[float] = None,
    limit_per_slot: int = 6,
) -> Dict[str, Any]:
    profile = UserProfile(**user_profile) if user_profile else None
    dietary_restrictions = [item.name for item in (profile.dietary_notes if profile else []) if not item.enabled]
    macro_goals: Dict[str, float] = {}
    body_weight = profile.weight if profile and profile.weight else None
    fitness_goal = (profile.fitness_goal or "maintenance").lower() if profile else "maintenance"

    if protein_min is not None:
        macro_goals["protein_min"] = protein_min
    elif body_weight:
        if fitness_goal == "cut":
            macro_goals["protein_min"] = round(body_weight * 0.30, 1)
        elif fitness_goal == "bulk":
            macro_goals["protein_min"] = round(body_weight * 0.25, 1)
        else:
            macro_goals["protein_min"] = round(body_weight * 0.22, 1)

    if carbs_min is not None:
        macro_goals["carbs_min"] = carbs_min
    elif body_weight and fitness_goal == "bulk":
        macro_goals["carbs_min"] = round(body_weight * 0.35, 1)
    elif body_weight and fitness_goal == "maintenance":
        macro_goals["carbs_min"] = round(body_weight * 0.22, 1)

    if carbs_max is not None:
        macro_goals["carbs_max"] = carbs_max
    elif profile and any(item.enabled and item.name.lower() == "keto" for item in profile.dietary_notes):
        macro_goals["carbs_max"] = 12
    elif body_weight and fitness_goal == "cut":
        macro_goals["carbs_max"] = round(body_weight * 0.45, 1)

    if calories_max is not None:
        macro_goals["calories_max"] = calories_max
    elif profile and (profile.target_calories or 0) > 0:
        macro_goals["calories_max"] = max(350, round(profile.target_calories / max(profile.workout_frequency or 3, 3)))

    try:
        food_candidates = build_meal_candidate_bundle(
            query=user_input,
            dietary_restrictions=dietary_restrictions,
            macro_goals=macro_goals,
            food_types=food_types or ["foundation"],
            limit_per_slot=limit_per_slot,
        )
    except Exception:
        fallback_foods = [
            format_mongo_food_summary(item)
            for item in find_foods_by_text(
                query=user_input,
                limit=max(limit_per_slot * 2, 12),
                food_types=food_types or ["foundation"],
                include_ingredients=False,
            )
        ]
        food_candidates = {
            "query": user_input,
            "candidate_strategy": {
                "vector_backend": "faiss",
                "retrieval_modes": ["text_fallback"],
                "limit_per_slot": limit_per_slot,
                "macro_goals": macro_goals,
                "food_types": food_types or ["foundation"],
            },
            "top_matches": fallback_foods[:limit_per_slot],
            "slot_candidates": {
                "proteins": [],
                "carbs": [],
                "vegetables": [],
                "fats": [],
                "flexible": fallback_foods[:limit_per_slot],
            },
            "total_candidates": len(fallback_foods[:limit_per_slot]),
        }

    food_candidates.setdefault("candidate_strategy", {})
    food_candidates["candidate_strategy"]["macro_goals"] = macro_goals
    food_candidates["candidate_strategy"]["food_types"] = food_types or ["foundation"]
    top_matches = food_candidates.get("top_matches", [])
    total_candidates = food_candidates.get("total_candidates", len(top_matches))
    return {
        "food_candidates": food_candidates,
        "observation": (
            f"Nutrition search assembled a meal-planning candidate bundle with "
            f"{total_candidates} foods across grouped slots and {len(top_matches)} direct matches."
        ),
    }


async def generate_meal_plan_artifact(
    user_input: str,
    user_profile: Dict[str, Any],
    meal_preferences: Optional[Dict[str, Any]] = None,
    food_candidates: Optional[Dict[str, Any]] = None,
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
                    food_candidates or {},
                )
            ),
        ],
    )

    slot_candidates = (food_candidates or {}).get("slot_candidates", {})
    available_foods_count = sum(len(items) for items in slot_candidates.values())

    return {
        "meal_plan": {
            "user_id": profile.user_id,
            "plan_name": meal_plan.plan_name,
            "days": len(meal_plan.days),
            "target_macros": meal_plan.target_macros.model_dump(),
            "daily_plans": [day.model_dump() for day in meal_plan.days],
            "key_principles": meal_plan.key_principles,
            "shopping_tips": meal_plan.shopping_tips,
            "available_foods_count": available_foods_count,
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
