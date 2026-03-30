from typing import Any, Awaitable, Callable, Dict, List, Optional

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from .services.tool_service import (
    generate_meal_plan_artifact,
    generate_workout_plan_artifact,
    prepare_profile_artifact,
    search_food_candidates_artifact,
    summarize_final_answer_artifact,
)
from .models import WorkflowEvent


class PrepareProfileInput(BaseModel):
    user_id: str
    user_profile: Optional[Dict[str, Any]] = None


class SearchFoodCandidatesInput(BaseModel):
    user_input: str
    use_full_database: bool = False
    user_profile: Optional[Dict[str, Any]] = None


class GenerateMealPlanInput(BaseModel):
    user_input: str
    user_profile: Dict[str, Any]
    meal_preferences: Dict[str, Any] = Field(default_factory=dict)
    food_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    base_url: str
    model_name: str


class GenerateWorkoutPlanInput(BaseModel):
    user_input: str
    user_profile: Dict[str, Any]
    workout_preferences: Dict[str, Any] = Field(default_factory=dict)
    base_url: str
    model_name: str


class SummarizeFinalAnswerInput(BaseModel):
    user_input: str
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    base_url: str
    model_name: str
    

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


async def summarize_final_answer_tool(
    user_input: str,
    artifacts: Optional[Dict[str, Any]] = None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_name: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    return await summarize_final_answer_artifact(
        user_input=user_input,
        artifacts=artifacts,
        base_url=base_url,
        model_name=model_name,
    )


def build_tool_registry(
    *,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_name: str = "gpt-4o-mini",
    prompt_user: Optional[Callable[[str], Awaitable[str]]] = None,
    notify_user: Optional[Callable[[str], None]] = None,
    event_handler: Optional[Callable[[WorkflowEvent], None]] = None,
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
            args_schema=GenerateMealPlanInput,
        ),
        StructuredTool.from_function(
            coroutine=generate_workout_plan_tool,
            name="generate_workout_plan",
            description=(
                "Generate a structured workout plan. "
                "Use this tool when the user wants a training schedule, weekly routine, equipment-constrained plan, muscle gain program, or fat loss training plan. "
                "A user profile should normally be prepared first so the plan can reflect frequency, duration, goal, and equipment constraints."
            ),
            args_schema=GenerateWorkoutPlanInput,
        ),
        StructuredTool.from_function(
            coroutine=summarize_final_answer_tool,
            name="summarize_final_answer",
            description=(
                "Summarize the final answer for the user. "
                "Use this tool to turn the current artifacts into the final user-facing response when the system already has enough information. "
                "It is suitable both at the end of a completed workflow and when limited information is still sufficient for a concise answer and next-step guidance."
            ),
            args_schema=SummarizeFinalAnswerInput,
        ),
    ]
    return {tool.name: tool for tool in tools}
