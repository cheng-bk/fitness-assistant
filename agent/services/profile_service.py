from datetime import datetime
from typing import Dict, List, Optional

from ..models import ProfileAnswerInterpretation, UserProfile

PROFILE_CORE_FIELDS = [
    "age",
    "weight",
    "height",
    "gender",
    "activity_level",
    "fitness_goal",
    "workout_frequency",
    "workout_duration",
]
PROFILE_FIELD_LABELS = {
    "age": "年龄",
    "weight": "体重",
    "height": "身高",
    "gender": "性别",
    "activity_level": "活动水平",
    "fitness_goal": "健身目标",
    "workout_frequency": "每周训练次数",
    "workout_duration": "单次训练时长",
}
PROFILE_VALUE_LABELS = {
    "male": "男性",
    "female": "女性",
    "sedentary": "久坐少动",
    "light": "轻度活动",
    "moderate": "中等活动",
    "active": "高活动量",
    "very_active": "非常高活动量",
    "cut": "减脂",
    "bulk": "增肌",
    "maintenance": "维持",
}


def calculate_bmr(profile: UserProfile) -> float:
    if not all([profile.age, profile.weight, profile.height, profile.gender]):
        return 2000

    base = (10 * profile.weight) + (6.25 * profile.height) - (5 * profile.age)
    if profile.gender == "female":
        return base - 161
    return base + 5


def calculate_tdee(profile: UserProfile) -> float:
    multipliers = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9,
    }
    return calculate_bmr(profile) * multipliers.get(profile.activity_level, 1.55)


def calculate_macros(profile: UserProfile) -> Dict[str, float]:
    tdee = calculate_tdee(profile)
    if profile.fitness_goal == "cut":
        target_calories = tdee * 0.8
    elif profile.fitness_goal == "bulk":
        target_calories = tdee * 1.1
    else:
        target_calories = tdee

    return {
        "calories": round(target_calories),
        "protein_g": round((target_calories * 0.30) / 4),
        "carbs_g": round((target_calories * 0.40) / 4),
        "fat_g": round((target_calories * 0.30) / 9),
    }


def enrich_profile(profile: UserProfile) -> UserProfile:
    macros = calculate_macros(profile)
    profile.target_calories = int(macros["calories"])
    profile.target_protein_g = macros["protein_g"]
    profile.target_carbs_g = macros["carbs_g"]
    profile.target_fat_g = macros["fat_g"]
    if profile.created_at is None:
        profile.created_at = datetime.now()
    profile.updated_at = datetime.now()
    return profile


def dedupe_profile_fields(field_names: List[str]) -> List[str]:
    normalized: List[str] = []
    for field_name in field_names:
        if field_name in PROFILE_CORE_FIELDS and field_name not in normalized:
            normalized.append(field_name)
    return normalized


def get_missing_profile_fields(profile: UserProfile, field_names: Optional[List[str]] = None) -> List[str]:
    required_fields = dedupe_profile_fields(field_names or PROFILE_CORE_FIELDS)
    missing_fields: List[str] = []
    for field_name in required_fields:
        if getattr(profile, field_name, None) in (None, ""):
            missing_fields.append(field_name)
    return missing_fields


def apply_profile_interpretation(
    profile: UserProfile,
    interpretation: ProfileAnswerInterpretation,
) -> UserProfile:
    if not interpretation.is_valid:
        return profile

    value = getattr(interpretation, interpretation.field_name, None)
    if value is not None:
        setattr(profile, interpretation.field_name, value)
    return profile


def clear_profile_fields(profile: UserProfile, field_names: List[str]) -> UserProfile:
    for field_name in dedupe_profile_fields(field_names):
        setattr(profile, field_name, None)
    return profile


def format_profile_summary(profile: UserProfile, field_names: Optional[List[str]] = None) -> str:
    fields = dedupe_profile_fields(field_names or PROFILE_CORE_FIELDS)
    value_map = {
        "age": str(profile.age) if profile.age is not None else "未填写",
        "weight": f"{profile.weight} kg" if profile.weight is not None else "未填写",
        "height": f"{profile.height} cm" if profile.height is not None else "未填写",
        "gender": PROFILE_VALUE_LABELS.get(profile.gender or "", profile.gender or "未填写"),
        "activity_level": PROFILE_VALUE_LABELS.get(profile.activity_level or "", profile.activity_level or "未填写"),
        "fitness_goal": PROFILE_VALUE_LABELS.get(profile.fitness_goal or "", profile.fitness_goal or "未填写"),
        "workout_frequency": (
            f"每周 {profile.workout_frequency} 次" if profile.workout_frequency is not None else "未填写"
        ),
        "workout_duration": (
            f"每次 {profile.workout_duration} 分钟" if profile.workout_duration is not None else "未填写"
        ),
    }
    lines = ["当前会使用到的资料："]
    for field_name in fields:
        lines.append(f"- {PROFILE_FIELD_LABELS[field_name]}：{value_map[field_name]}")
    return "\n".join(lines)


def parse_profile_modification_request(
    user_input: str,
    allowed_fields: Optional[List[str]] = None,
) -> List[str]:
    text = user_input.lower()
    allowed = set(dedupe_profile_fields(allowed_fields or PROFILE_CORE_FIELDS))
    selected: List[str] = []
    field_keywords = {
        "age": ["年龄", "age"],
        "weight": ["体重", "weight", "公斤", "kg"],
        "height": ["身高", "height", "厘米", "cm"],
        "gender": ["性别", "男", "女", "gender", "male", "female"],
        "activity_level": ["活动", "活动水平", "运动量", "activity"],
        "fitness_goal": ["目标", "健身目标", "减脂", "增肌", "维持", "goal"],
        "workout_frequency": ["训练次数", "每周训练", "每周几次", "频率", "次数", "frequency"],
        "workout_duration": ["训练时长", "单次时长", "每次多久", "时长", "duration", "分钟"],
    }

    for field_name, keywords in field_keywords.items():
        if field_name in allowed and any(keyword in text for keyword in keywords):
            selected.append(field_name)

    if any(token in text for token in ["全部", "所有", "all"]):
        return list(allowed)
    return dedupe_profile_fields(selected)
