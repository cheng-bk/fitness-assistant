from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool, StructuredTool

from agent.repositories.vector_index_repository import retrieve_knowledge_hits

from .models import (
    SearchKnowledgeInput,
    SearchFoodEntityInput,
    SearchExerciseEntityInput,
    MealPlanInput,
    WorkoutPlanInput,
)
from .services.exercise_service import generate_workout_plan_artifact, search_exercise_entity_artifact
from .services.food_service import generate_meal_plan_artifact, search_food_entity_artifact


def _compact_text(value: Any, limit: int = 200) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


async def search_knowledge_tool(
    query: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    knowledge_hits = retrieve_knowledge_hits(query, top_k=top_k)
    top_sources = [
        str(item.get("source") or item.get("title") or item.get("name") or "").strip()
        for item in knowledge_hits[:3]
        if str(item.get("source") or item.get("title") or item.get("name") or "").strip()
    ]
    return {
        "knowledge_lookup": {
            "query": query,
            "knowledge_hits": knowledge_hits,
        },
        "observation": (
            f"Knowledge search input: query='{_compact_text(query)}', top_k={top_k}. "
            f"Returned {len(knowledge_hits)} hits"
            + (f"; top sources: {top_sources}." if top_sources else ".")
        ),
    }


async def search_food_entity_tool(
    query: str,
    limit: int = 5,
) -> Dict[str, Any]:
    result = await search_food_entity_artifact(
        query=query,
        food_types=["foundation"],
        limit=limit,
    )
    food_lookup = result.get("food_lookup") or {}
    matched_foods = food_lookup.get("matched_foods") or []
    top_food_names = [str(item.get("name") or "").strip() for item in matched_foods[:3] if item.get("name")]
    result["observation"] = (
        f"Food entity lookup input: query='{_compact_text(query)}', limit={limit}. "
        f"Returned {len(food_lookup.get('name_matches') or [])} name matches and {len(matched_foods)} food records"
        + (f"; top foods: {top_food_names}." if top_food_names else ".")
    )
    return result


async def search_exercise_entity_tool(
    query: str,
    limit: int = 5,
) -> Dict[str, Any]:
    result = await search_exercise_entity_artifact(
        query=query,
        limit=limit,
    )
    exercise_lookup = result.get("exercise_lookup") or {}
    matched_exercises = exercise_lookup.get("matched_exercises") or []
    top_exercise_names = [str(item.get("name") or "").strip() for item in matched_exercises[:3] if item.get("name")]
    result["observation"] = (
        f"Exercise entity lookup input: query='{_compact_text(query)}', limit={limit}. "
        f"Returned {len(exercise_lookup.get('name_matches') or [])} name matches and {len(matched_exercises)} exercise records"
        + (f"; top exercises: {top_exercise_names}." if top_exercise_names else ".")
    )
    return result


async def generate_meal_plan_tool(
    user_input: str,
    user_profile: Dict[str, Any],
    meal_preferences: Optional[Dict[str, Any]] = None,
    prior_food_lookups: Optional[List[Dict[str, Any]]] = None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_name: str = "qwen3.5-plus",
) -> Dict[str, Any]:
    result = await generate_meal_plan_artifact(
        user_input=user_input,
        user_profile=user_profile,
        meal_preferences=meal_preferences,
        prior_food_lookups=prior_food_lookups,
        base_url=base_url,
        model_name=model_name,
    )
    plan = result.get("meal_plan") or {}
    meal_preferences = meal_preferences or {}
    result["observation"] = (
        "Meal plan generation input: "
        f"requested_food_names={meal_preferences.get('requested_food_names') or []}, "
        f"excluded_food_names={meal_preferences.get('excluded_food_names') or []}, "
        f"meal_count={meal_preferences.get('meal_count')}, days={meal_preferences.get('days')}, "
        f"notes={meal_preferences.get('notes') or []}, prior_food_lookups={len(prior_food_lookups or [])}. "
        f"Generated meal plan '{_compact_text(plan.get('plan_name') or '')}' with {plan.get('days', 0)} days."
    )
    return result


async def generate_workout_plan_tool(
    user_input: str,
    user_profile: Dict[str, Any],
    workout_preferences: Optional[Dict[str, Any]] = None,
    prior_exercise_lookups: Optional[List[Dict[str, Any]]] = None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_name: str = "qwen3.5-plus",
) -> Dict[str, Any]:
    result = await generate_workout_plan_artifact(
        user_input=user_input,
        user_profile=user_profile,
        workout_preferences=workout_preferences,
        prior_exercise_lookups=prior_exercise_lookups,
        base_url=base_url,
        model_name=model_name,
    )
    plan = result.get("workout_plan") or {}
    workout_preferences = workout_preferences or {}
    result["observation"] = (
        "Workout plan generation input: "
        f"requested_exercise_names={workout_preferences.get('requested_exercise_names') or []}, "
        f"excluded_exercise_names={workout_preferences.get('excluded_exercise_names') or []}, "
        f"days_per_week={workout_preferences.get('days_per_week')}, duration_minutes={workout_preferences.get('duration_minutes')}, "
        f"split_type={workout_preferences.get('split_type')}, training_style={workout_preferences.get('training_style')}, "
        f"equipment_available={workout_preferences.get('equipment_available') or []}, "
        f"target_muscle_groups={workout_preferences.get('target_muscle_groups') or []}, "
        f"cardio_preference={workout_preferences.get('cardio_preference')}, "
        f"notes={workout_preferences.get('notes') or []}, prior_exercise_lookups={len(prior_exercise_lookups or [])}. "
        f"Generated workout plan '{_compact_text(plan.get('plan_name') or '')}' with {plan.get('days_per_week', 0)} days/week."
    )
    return result


def build_tool_registry(
) -> Dict[str, BaseTool]:
    tools: List[BaseTool] = [
        StructuredTool.from_function(
            coroutine=search_knowledge_tool,
            name="search_knowledge",
            description=(
                "Search the knowledge base for relevant information. "
                "Use this for general factual lookup, explanation, comparison, or verification when the user doesn't explicitly mention a specific food or exercise. "
                "LLM-owned input: query = a short user question or information need in English, not the full user request; "
                "top_k = max knowledge hits to return, default is 5; "
            ),
            args_schema=SearchKnowledgeInput,
        ),
        StructuredTool.from_function(
            coroutine=search_food_entity_tool,
            name="search_food_entity",
            description=(
                "Look up one specific named food in the full food database. "
                "The tool resolves near-name matches from the food name index, then fetches full food records. "
                "Use this for direct factual lookup, explanation, comparison, or verification of a food the user explicitly mentions. "
                "LLM-owned input: query = a short entity-name-like food name string in English, not the full user request; "
                "limit = max matched food records to return, default is 5; "
            ),
            args_schema=SearchFoodEntityInput,
        ),
        StructuredTool.from_function(
            coroutine=search_exercise_entity_tool,
            name="search_exercise_entity",
            description=(
                "Look up one specific named exercise in the full exercise database. "
                "The tool resolves near-name matches from the exercise name index, then fetches full exercise records. "
                "Use this for direct factual lookup, explanation, comparison, or verification of an exercise the user explicitly mentions. "
                "LLM-owned input: query = a short entity-name-like exercise name string in English, not the full user request; "
                "limit = max matched exercise records to return, default is 5; "
            ),
            args_schema=SearchExerciseEntityInput,
        ),
        StructuredTool.from_function(
            coroutine=generate_meal_plan_tool,
            name="generate_meal_plan",
            description=(
                "Generate a structured meal plan. "
                "Use this tool when the user explicitly wants a diet plan, eating schedule, fat loss meal plan, or muscle gain meal plan. "
                "This tool behaves like a sub-agent: it internally retrieves representative candidate foods before generating the final plan. "
                "If the plan depends on specific named foods, especially foods the user prefers, dislikes, wants included, or wants avoided, the planner should usually call search_food_entity first. "
                "LLM-owned input: meal_preferences.requested_food_names = food names that should be included or prioritized; "
                "meal_preferences.excluded_food_names = food names that should be avoided; "
                "meal_preferences.meal_count = preferred meals per day; "
                "meal_preferences.days = preferred plan length in days; "
                "meal_preferences.notes = short temporary planning constraints for this plan only."
            ),
            args_schema=MealPlanInput,
        ),
        StructuredTool.from_function(
            coroutine=generate_workout_plan_tool,
            name="generate_workout_plan",
            description=(
                "Generate a structured workout plan. "
                "Use this tool when the user wants a training schedule, weekly routine, equipment-constrained plan, muscle gain program, or fat loss training plan. "
                "This tool behaves like a sub-agent: it internally retrieves representative candidate exercises before generating the final plan. "
                "If the plan depends on specific named exercises, the planner should usually call search_exercise_entity first. "
                "LLM-owned input: workout_preferences.requested_exercise_names = exercise names that should be included or prioritized; "
                "workout_preferences.excluded_exercise_names = exercise names that should be avoided; "
                "workout_preferences.days_per_week = target training frequency; "
                "workout_preferences.duration_minutes = preferred session duration; "
                "workout_preferences.split_type = preferred split style; "
                "workout_preferences.training_style = preferred training style; "
                "workout_preferences.equipment_available = equipment explicitly available in the current context when it differs from the long-term profile; "
                "workout_preferences.target_muscle_groups = muscle groups the user especially wants to prioritize; "
                "workout_preferences.cardio_preference = preferred cardio emphasis such as none, light, moderate, or high; "
                "workout_preferences.notes = short temporary planning constraints for this plan only."
            ),
            args_schema=WorkoutPlanInput,
        ),
    ]
    return {tool.name: tool for tool in tools}
