import os
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """功能简介：描述用户基础信息、营养目标与训练偏好的核心数据模型。"""

    user_id: str
    age: Optional[int] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    activity_level: Optional[str] = "moderate"
    fitness_goal: Optional[str] = "maintenance"
    workout_frequency: Optional[int] = None
    workout_duration: Optional[int] = None
    target_calories: Optional[int] = None
    target_protein_g: Optional[float] = None
    target_carbs_g: Optional[float] = None
    target_fat_g: Optional[float] = None
    allergies: List[str] = Field(default_factory=list)
    dietary_preferences: List[str] = Field(default_factory=list)
    equipment_available: List[str] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class MealPlanRequest(BaseModel):
    """功能简介：描述 meal plan 生成请求的简化参数。"""

    meal_count: int = 5
    days: int = 7
    preferences: Dict[str, Any] = Field(default_factory=dict)


class WorkoutPlanRequest(BaseModel):
    """功能简介：描述 workout plan 生成请求的简化参数。"""

    split_type: str = "full_body"
    training_style: str = "hypertrophy"
    days_per_week: int = 3
    duration_minutes: int = 60


class FitnessRequest(BaseModel):
    """功能简介：agentic 主流程入口请求模型。"""

    user_input: str
    user_profile: Optional[UserProfile] = None
    use_full_database: bool = False
    max_iterations: int = Field(default=6, ge=2, le=12)


ProfileFieldName = Literal[
    "age",
    "weight",
    "height",
    "activity_level",
    "fitness_goal",
    "workout_frequency",
    "workout_duration",
]


class IntentAnalysis(BaseModel):
    """功能简介：IntentInterpreterAgent 输出的结构化意图。"""

    primary_goal: str
    needs_profile_update: bool = True
    needs_nutrition_search: bool = True
    generate_meal_plan: bool = True
    generate_workout_plan: bool = True
    answer_directly: bool = False
    success_criteria: List[str] = Field(default_factory=list)


class PlanStep(BaseModel):
    """功能简介：PlannerAgent 生成的单个工具执行步骤。"""

    id: str
    tool_name: Literal[
        "prepare_profile",
        "search_food_candidates",
        "generate_meal_plan",
        "generate_workout_plan",
        "summarize_final_answer",
    ]
    objective: str
    tool_input: Dict[str, Any] = Field(default_factory=dict)
    status: Literal["pending", "completed", "failed"] = "pending"


class PlannerOutput(BaseModel):
    """功能简介：PlannerAgent 的结构化输出，描述下一步及剩余计划。"""

    reasoning: str
    next_step: Optional[PlanStep] = None
    remaining_steps: List[PlanStep] = Field(default_factory=list)


class DecisionOutput(BaseModel):
    """功能简介：DecisionAgent 的结构化输出，描述 loop 下一步动作。"""

    decision: Literal["continue", "replan", "finish"]
    reasoning: str
    should_finish: bool = False


class ActionRecord(BaseModel):
    """功能简介：记录每轮 tool 执行结果，方便调试与学习流程。"""

    iteration: int
    step_id: str
    tool_name: str
    objective: str
    status: Literal["completed", "failed"]
    observation: str
    artifact_keys: List[str] = Field(default_factory=list)


class AgenticFitnessResponse(BaseModel):
    """功能简介：对外返回的 agentic 主流程响应模型。"""

    user_id: str
    workflow_status: str
    final_answer: str
    iterations: int
    intent: Optional[IntentAnalysis] = None
    executed_steps: List[ActionRecord] = Field(default_factory=list)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    generated_at: datetime


class MealMacros(BaseModel):
    """功能简介：描述一个 meal 或一天总计的宏量营养信息。"""

    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float


class MealFood(BaseModel):
    """功能简介：描述 meal 中单个 food item 及其营养值。"""

    food_name: str
    portion: str
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float


class Meal(BaseModel):
    """功能简介：描述单餐结构。"""

    meal_name: str
    foods: List[MealFood]
    total_macros: MealMacros
    preparation_notes: Optional[str] = None


class DailyMealPlan(BaseModel):
    """功能简介：描述某一天的 meal plan。"""

    day: int
    day_name: str
    meals: List[Meal]
    daily_totals: MealMacros


class MealPlanStructured(BaseModel):
    """功能简介：meal planning tool 期望 LLM 返回的结构化 meal plan。"""

    plan_name: str
    days: List[DailyMealPlan]
    target_macros: MealMacros
    key_principles: List[str] = Field(default_factory=list)
    shopping_tips: List[str] = Field(default_factory=list)


class Exercise(BaseModel):
    """功能简介：描述单个训练动作。"""

    exercise_name: str
    sets: int
    reps: str
    rest_seconds: int
    notes: Optional[str] = None


class WorkoutDay(BaseModel):
    """功能简介：描述某一天的训练安排。"""

    day: int
    day_name: str
    focus: str
    exercises: List[Exercise]
    estimated_duration: int
    warm_up: List[str] = Field(default_factory=list)
    cool_down: List[str] = Field(default_factory=list)


class WorkoutPlanStructured(BaseModel):
    """功能简介：workout planning tool 期望 LLM 返回的结构化 workout plan。"""

    plan_name: str
    split_type: str
    training_style: str
    weekly_schedule: List[WorkoutDay]
    progression_strategy: str
    equipment_needed: List[str] = Field(default_factory=list)
    key_principles: List[str] = Field(default_factory=list)


class FinalAnswerStructured(BaseModel):
    """功能简介：final summary tool 期望 LLM 返回的结构化最终回答。"""

    overview: str
    completed_work: List[str] = Field(default_factory=list)
    nutrition_guidance: List[str] = Field(default_factory=list)
    training_guidance: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)


class NutritionQuery(BaseModel):
    """功能简介：nutrition 检索请求模型。"""

    query: str
    dietary_restrictions: List[str] = Field(default_factory=list)
    macro_goals: Dict[str, float] = Field(default_factory=dict)
    limit: int = 10
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    use_full_database: bool = False


class VectorSearchResponse(BaseModel):
    """功能简介：semantic search 返回结果模型。"""

    query: str
    results_found: int
    results: List[Dict[str, Any]] = Field(default_factory=list)
    search_time_ms: int


class HybridSearchResponse(BaseModel):
    """功能简介：hybrid search 返回结果模型。"""

    query: str
    search_type: str = "hybrid"
    semantic_weight: float
    traditional_weight: float
    results_found: int
    results: List[Dict[str, Any]] = Field(default_factory=list)


class IndexStatus(BaseModel):
    """功能简介：单个向量索引的状态信息。"""

    exists: bool
    loaded: bool
    loading: bool
    index_size: Optional[int] = None
    embedding_dimension: Optional[int] = None
    file_size_mb: Optional[float] = None
    last_modified: Optional[str] = None
    error: Optional[str] = None
    memory_usage_mb: Optional[float] = None


class VectorIndexStatusResponse(BaseModel):
    """功能简介：所有向量索引与系统资源的综合状态响应。"""

    full_database: IndexStatus
    sample_database: IndexStatus
    legacy_index: IndexStatus
    system_info: Dict[str, Any]


class DatabaseAvailabilityResponse(BaseModel):
    """功能简介：描述 full/sample 数据库集合是否可用。"""

    full_database: Dict[str, Any]
    sample_database: Dict[str, Any]
    recommendation: str


class ProfileQuestionOutput(BaseModel):
    field_name: ProfileFieldName
    question: str


class ProfileAnswerInterpretation(BaseModel):
    field_name: ProfileFieldName
    is_valid: bool
    age: Optional[int] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    activity_level: Optional[Literal["sedentary", "light", "moderate", "active", "very_active"]] = None
    fitness_goal: Optional[Literal["cut", "bulk", "maintenance"]] = None
    workout_frequency: Optional[int] = None
    workout_duration: Optional[int] = None
    acknowledgement: Optional[str] = None
    follow_up_question: Optional[str] = None


class WorkflowEvent(BaseModel):
    event_type: str
    phase: Literal["onboarding", "main_workflow"]
    node: str
    summary: str
    data: Dict[str, Any] = Field(default_factory=dict)
