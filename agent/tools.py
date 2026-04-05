from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool, StructuredTool

from .models import (
    MealPlanInput,
    SearchFoodEntityInput,
    SearchExerciseEntityInput,
    WorkoutPlanInput,
)
from .services.tool_service import (
    generate_meal_plan_artifact,
    generate_workout_plan_artifact,
    search_exercise_entity_artifact,
    search_food_entity_artifact,
)


async def generate_meal_plan_tool(
    user_input: str,
    user_profile: Dict[str, Any],
    meal_preferences: Optional[Dict[str, Any]] = None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_name: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    return await generate_meal_plan_artifact(
        user_input=user_input,
        user_profile=user_profile,
        meal_preferences=meal_preferences,
        base_url=base_url,
        model_name=model_name,
    )


async def search_food_entity_tool(
    query: str,
    food_types: Optional[List[str]] = None,
    limit: int = 5,
    knowledge_top_k: int = 4,
) -> Dict[str, Any]:
    return await search_food_entity_artifact(
        query=query,
        food_types=food_types,
        limit=limit,
        knowledge_top_k=knowledge_top_k,
    )


async def search_exercise_entity_tool(
    query: str,
    limit: int = 5,
    knowledge_top_k: int = 4,
) -> Dict[str, Any]:
    return await search_exercise_entity_artifact(
        query=query,
        limit=limit,
        knowledge_top_k=knowledge_top_k,
    )


async def generate_workout_plan_tool(
    user_input: str,
    user_profile: Dict[str, Any],
    workout_preferences: Optional[Dict[str, Any]] = None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_name: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    return await generate_workout_plan_artifact(
        user_input=user_input,
        user_profile=user_profile,
        workout_preferences=workout_preferences,
        base_url=base_url,
        model_name=model_name,
    )


def build_tool_registry(
) -> Dict[str, BaseTool]:
    tools: List[BaseTool] = [
        StructuredTool.from_function(
            coroutine=search_food_entity_tool,
            name="search_food_entity",
            description=(
                "Look up a specific named food in the full food database. "
                "Use this tool when the user asks about one particular food, ingredient, or product. "
                "It first resolves near-name matches from the food name index, then fetches full food records and supporting knowledge."
            ),
            args_schema=SearchFoodEntityInput,
        ),
        StructuredTool.from_function(
            coroutine=search_exercise_entity_tool,
            name="search_exercise_entity",
            description=(
                "Look up a specific named exercise in the full exercise database. "
                "Use this tool when the user asks about one particular exercise or movement. "
                "It first resolves near-name matches from the exercise name index, then fetches full exercise records and supporting knowledge."
            ),
            args_schema=SearchExerciseEntityInput,
        ),
        StructuredTool.from_function(
            coroutine=generate_meal_plan_tool,
            name="generate_meal_plan",
            description=(
                "Generate a structured meal plan. "
                "Use this tool when the user explicitly wants a diet plan, eating schedule, fat loss meal plan, or muscle gain meal plan. "
                "A valid user profile should normally exist first. "
                "This tool behaves like a sub-agent: it internally retrieves representative candidate foods and supporting knowledge before generating the final plan."
            ),
            args_schema=MealPlanInput,
        ),
        StructuredTool.from_function(
            coroutine=generate_workout_plan_tool,
            name="generate_workout_plan",
            description=(
                "Generate a structured workout plan. "
                "Use this tool when the user wants a training schedule, weekly routine, equipment-constrained plan, muscle gain program, or fat loss training plan. "
                "A user profile should normally be prepared first so the plan can reflect frequency, duration, goal, and equipment constraints. "
                "This tool behaves like a sub-agent: it internally retrieves representative candidate exercises and supporting knowledge before generating the final plan."
            ),
            args_schema=WorkoutPlanInput,
        ),
    ]
    return {tool.name: tool for tool in tools}
