from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# Workflow inputs / outputs

class WorkflowEvent(BaseModel):
    event_type: str
    phase: Literal["onboarding", "main_workflow"]
    node: str
    summary: str
    data: Dict[str, Any] = Field(default_factory=dict)


class PreferenceFlag(BaseModel):
    name: str
    enabled: bool


class UserProfile(BaseModel):
    user_id: str
    age: Optional[int] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    gender: Optional[Literal["male", "female"]] = None
    activity_level: Optional[str] = "moderate"
    fitness_goal: Optional[str] = "maintenance"
    workout_frequency: Optional[int] = None
    workout_duration: Optional[int] = None
    target_calories: Optional[int] = None
    target_protein_g: Optional[float] = None
    target_carbs_g: Optional[float] = None
    target_fat_g: Optional[float] = None
    dietary_notes: List[PreferenceFlag] = Field(default_factory=list)
    equipment_notes: List[PreferenceFlag] = Field(default_factory=list)
    other_notes: List[str] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class WorkflowContext(BaseModel):
    summary: Optional[str] = None
    carry_over_notes: List[str] = Field(default_factory=list)
    prior_artifacts: Dict[str, Any] = Field(default_factory=dict)


class FitnessRequest(BaseModel):
    user_input: str
    user_profile: Optional[UserProfile] = None
    context: Optional[WorkflowContext] = None
    food_types: List[str] = Field(default_factory=lambda: ["foundation"])
    max_iterations: int = Field(default=6, ge=2, le=12)


# Agent inputs / outputs

PROFILE_FIELDS = Literal[
    "age",
    "weight",
    "height",
    "gender",
    "activity_level",
    "fitness_goal",
    "workout_frequency",
    "workout_duration",
]


class ProfileQuestion(BaseModel):
    field_name: PROFILE_FIELDS
    question: str


class ProfileAnswerInterpretation(BaseModel):
    field_name: PROFILE_FIELDS
    is_valid: bool
    age: Optional[int] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    gender: Optional[Literal["male", "female"]] = None
    activity_level: Optional[Literal["sedentary", "light", "moderate", "active", "very_active"]] = None
    fitness_goal: Optional[Literal["cut", "bulk", "maintenance"]] = None
    workout_frequency: Optional[int] = None
    workout_duration: Optional[int] = None
    acknowledgement: Optional[str] = None
    follow_up_question: Optional[str] = None


class IntentAnalysis(BaseModel):
    primary_goal: str
    profile_update: bool = False
    food_entity_lookup: bool = False
    exercise_entity_lookup: bool = False
    generate_meal_plan: bool = False
    generate_workout_plan: bool = False
    success_criteria: List[str] = Field(default_factory=list)


class ProfileMemoryUpdate(BaseModel):
    should_update_profile: bool = False
    reasoning: str = ""
    dietary_notes: List[PreferenceFlag] = Field(default_factory=list)
    equipment_notes: List[PreferenceFlag] = Field(default_factory=list)
    other_notes: List[str] = Field(default_factory=list)
    dietary_notes_to_remove: List[str] = Field(default_factory=list)
    equipment_notes_to_remove: List[str] = Field(default_factory=list)
    other_notes_to_remove: List[str] = Field(default_factory=list)
    acknowledgement: Optional[str] = None


class PlanStep(BaseModel):
    id: str
    tool_name: Literal[
        "search_knowledge",
        "search_food_entity",
        "search_exercise_entity",
        "generate_meal_plan",
        "generate_workout_plan",
    ]
    objective: str
    tool_input: Dict[str, Any] = Field(default_factory=dict)
    status: Literal["pending", "completed", "failed"] = "pending"


class PlanList(BaseModel):
    reasoning: str
    next_step: Optional[PlanStep] = None
    remaining_steps: List[PlanStep] = Field(default_factory=list)


class Decision(BaseModel):
    decision: Literal["continue", "replan", "finish"]
    reasoning: str
    should_finish: bool = False


class ActionRecord(BaseModel):
    iteration: int
    step_id: str
    tool_name: str
    objective: str
    status: Literal["completed", "failed"]
    observation: str
    artifact_keys: List[str] = Field(default_factory=list)


# Tool inputs / outputs

class SearchFoodEntityInput(BaseModel):
    query: str = Field(description="A short food name string to look up, not the full user request.")
    limit: int = Field(default=5, description="Maximum number of matched food records to return.")


class SearchExerciseEntityInput(BaseModel):
    query: str = Field(description="A short exercise name string to look up, not the full user request.")
    limit: int = Field(default=5, description="Maximum number of matched exercise records to return.")
    
    
class SearchKnowledgeInput(BaseModel):
    query: str = Field(description="A short user question or information need in English, not the full user request.")
    top_k: int = Field(default=5, description="Maximum number of knowledge hits to return.")


class MealPreferencesInput(BaseModel):
    requested_food_names: List[str] = Field(default_factory=list, description="Food names the plan should include or prioritize.")
    excluded_food_names: List[str] = Field(default_factory=list, description="Food names the plan should avoid.")
    meal_count: Optional[int] = Field(default=None, description="Preferred meals per day.")
    days: Optional[int] = Field(default=None, description="Preferred plan length in days.")
    notes: List[str] = Field(default_factory=list, description="Short temporary constraints for this plan only.")


class MealPlanInput(BaseModel):
    meal_preferences: MealPreferencesInput = Field(default_factory=MealPreferencesInput, description="Structured short-term meal-planning preferences.")


class WorkoutPreferencesInput(BaseModel):
    requested_exercise_names: List[str] = Field(default_factory=list, description="Exercise names the plan should include or prioritize.")
    excluded_exercise_names: List[str] = Field(default_factory=list, description="Exercise names the plan should avoid.")
    days_per_week: Optional[int] = Field(default=None, description="Preferred training frequency per week.")
    duration_minutes: Optional[int] = Field(default=None, description="Preferred session length in minutes.")
    split_type: Optional[str] = Field(default=None, description="Preferred training split, such as full_body or upper_lower.")
    training_style: Optional[str] = Field(default=None, description="Preferred training style, such as hypertrophy or strength.")
    equipment_available: List[str] = Field(default_factory=list, description="Equipment available for this plan, when the current context overrides the default profile.")
    target_muscle_groups: List[str] = Field(default_factory=list, description="Muscle groups the user especially wants to prioritize.")
    cardio_preference: Optional[str] = Field(default=None, description="Preferred cardio emphasis, such as none, light, moderate, or high.")
    notes: List[str] = Field(default_factory=list, description="Short temporary constraints for this plan only.")


class WorkoutPlanInput(BaseModel):
    workout_preferences: WorkoutPreferencesInput = Field(default_factory=WorkoutPreferencesInput, description="Structured short-term workout-planning preferences.")
    

class WorkoutPlanRequest(BaseModel):
    split_type: str = "full_body"
    training_style: str = "hypertrophy"
    days_per_week: int = 3
    duration_minutes: int = 60


# Other shared / structured models

class MealMacros(BaseModel):
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float


class MealFood(BaseModel):
    food_name: str
    portion: str
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float


class Meal(BaseModel):
    meal_name: str
    foods: List[MealFood]
    total_macros: MealMacros
    preparation_notes: Optional[str] = None


class DailyMealPlan(BaseModel):
    day: int
    day_name: str
    meals: List[Meal]
    daily_totals: MealMacros


class MealPlanStructured(BaseModel):
    plan_name: str
    days: List[DailyMealPlan]
    target_macros: MealMacros
    key_principles: List[str] = Field(default_factory=list)
    shopping_tips: List[str] = Field(default_factory=list)


class Exercise(BaseModel):
    exercise_name: str
    sets: int
    reps: str
    rest_seconds: int
    notes: Optional[str] = None


class WorkoutDay(BaseModel):
    day: int
    day_name: str
    focus: str
    exercises: List[Exercise]
    estimated_duration: int
    warm_up: List[str] = Field(default_factory=list)
    cool_down: List[str] = Field(default_factory=list)


class WorkoutPlanStructured(BaseModel):
    plan_name: str
    split_type: str
    training_style: str
    weekly_schedule: List[WorkoutDay]
    progression_strategy: str
    equipment_needed: List[str] = Field(default_factory=list)
    key_principles: List[str] = Field(default_factory=list)


class FinalAnswerStructured(BaseModel):
    overview: str
    completed_work: List[str] = Field(default_factory=list)
    nutrition_guidance: List[str] = Field(default_factory=list)
    training_guidance: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)


class NutritionQuery(BaseModel):
    query: str
    dietary_restrictions: List[str] = Field(default_factory=list)
    macro_goals: Dict[str, float] = Field(default_factory=dict)
    limit: int = 10
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    food_types: List[str] = Field(default_factory=lambda: ["foundation"])


class VectorSearchResponse(BaseModel):
    query: str
    results_found: int
    results: List[Dict[str, Any]] = Field(default_factory=list)
    search_time_ms: int


class HybridSearchResponse(BaseModel):
    query: str
    search_type: str = "hybrid"
    semantic_weight: float
    traditional_weight: float
    results_found: int
    results: List[Dict[str, Any]] = Field(default_factory=list)


class IndexStatus(BaseModel):
    exists: bool
    loaded: bool
    loading: bool
    file_size_mb: Optional[float] = None
    last_modified: Optional[str] = None
    error: Optional[str] = None
    index_size: Optional[int] = None
    embedding_dimension: Optional[int] = None
    memory_usage_mb: Optional[float] = None


class VectorIndexStatusResponse(BaseModel):
    system_info: Dict[str, Any]
    full_database: IndexStatus
    sample_database: IndexStatus
    legacy_index: IndexStatus
