from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool, StructuredTool

from .models import (
    MealPlanInput,
    WorkoutPlanInput,
    SearchFoodCandidatesInput,
)
from .services.tool_service import (
    generate_meal_plan_artifact,
    generate_workout_plan_artifact,
    search_food_candidates_artifact,
)

async def search_food_candidates_tool(
    user_input: str,
    use_full_database: bool = False,
    user_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return await search_food_candidates_artifact(
        user_input=user_input,
        use_full_database=use_full_database,
        user_profile=user_profile,
    )


async def generate_meal_plan_tool(
    user_input: str,
    user_profile: Dict[str, Any],
    meal_preferences: Optional[Dict[str, Any]] = None,
    food_candidates: Optional[List[Dict[str, Any]]] = None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_name: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    return await generate_meal_plan_artifact(
        user_input=user_input,
        user_profile=user_profile,
        meal_preferences=meal_preferences,
        food_candidates=food_candidates,
        base_url=base_url,
        model_name=model_name,
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
            coroutine=search_food_candidates_tool,
            name="search_food_candidates",
            description=(
                "Search for candidate foods. "
                "Use this tool to collect food candidates from semantic search or text search based on the user request and current profile. "
                "It is useful when the user asks what to eat, wants a meal plan, or needs food options filtered by dietary or macro constraints."
            ),
            args_schema=SearchFoodCandidatesInput,
        ),
        StructuredTool.from_function(
            coroutine=generate_meal_plan_tool,
            name="generate_meal_plan",
            description=(
                "Generate a structured meal plan. "
                "Use this tool when the user explicitly wants a diet plan, eating schedule, fat loss meal plan, or muscle gain meal plan. "
                "A valid user profile should normally exist first, and candidate foods are helpful when you want the plan to reflect realistic food options."
            ),
            args_schema=MealPlanInput,
        ),
        StructuredTool.from_function(
            coroutine=generate_workout_plan_tool,
            name="generate_workout_plan",
            description=(
                "Generate a structured workout plan. "
                "Use this tool when the user wants a training schedule, weekly routine, equipment-constrained plan, muscle gain program, or fat loss training plan. "
                "A user profile should normally be prepared first so the plan can reflect frequency, duration, goal, and equipment constraints."
            ),
            args_schema=WorkoutPlanInput,
        ),
    ]
    return {tool.name: tool for tool in tools}
