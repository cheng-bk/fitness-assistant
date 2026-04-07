from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import build_chat_model, build_structured_output_instruction, invoke_structured_with_retry
from ..models import UserProfile, WorkoutPlanRequest, WorkoutPlanStructured
from ..prompts import WORKOUT_PLAN_SYSTEM_PROMPT, build_workout_plan_user_prompt
from ..repositories.exercise_repository import (
    COMPOUND_MUSCLES,
    ACCESSORY_MUSCLES,
    CORE_MUSCLES,
    find_exercises_for_workout_slot,
    get_exercise_collection,
)
from ..repositories.vector_index_repository import retrieve_name_matches


WORKOUT_SLOT_MUSCLE_HINTS = {
    "strength_compounds": set(COMPOUND_MUSCLES),
    "strength_accessories": set(ACCESSORY_MUSCLES),
    "core_pool": set(CORE_MUSCLES),
}


def _sanitize_exercise_document(document: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(document)
    cleaned.pop("_id", None)
    cleaned.pop("candidate_flags", None)
    return cleaned


def _profile_unavailable_equipment(profile: Optional[UserProfile]) -> Set[str]:
    if profile is None:
        return set()
    return {
        item.name.strip().lower()
        for item in profile.equipment_notes
        if not item.enabled and item.name.strip()
    }


def _normalize_name_list(names: Optional[List[str]]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for raw_name in names or []:
        name = str(raw_name or "").strip()
        if not name:
            continue
        lowered = name.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(name)
    return normalized


def _normalize_string_list(values: Optional[List[str]]) -> List[str]:
    return _normalize_name_list(values)


def _fetch_exercise_documents_by_ids(document_ids: List[str]) -> List[Dict[str, Any]]:
    if not document_ids:
        return []
    documents = list(get_exercise_collection().find({"_id": {"$in": document_ids}}))
    by_id = {str(item.get("_id")): item for item in documents}
    return [_sanitize_exercise_document(by_id[document_id]) for document_id in document_ids if document_id in by_id]


async def search_exercise_entity_artifact(
    query: str,
    limit: int = 5,
) -> Dict[str, Any]:
    name_matches = retrieve_name_matches(entity_type="exercise", query=query, top_k=limit)
    document_ids = [str(item.get("document_id")) for item in name_matches if item.get("document_id")]
    matched_exercises = _fetch_exercise_documents_by_ids(document_ids=document_ids)

    if not matched_exercises:
        regex = {"$regex": query, "$options": "i"}
        fallback_documents = list(
            get_exercise_collection().find(
                {
                    "category": {"$in": ["strength", "cardio"]},
                    "$or": [
                        {"name": regex},
                        {"category": regex},
                        {"equipment": regex},
                        {"primaryMuscles": regex},
                    ],
                }
            ).limit(limit)
        )
        matched_exercises = [_sanitize_exercise_document(item) for item in fallback_documents]

    exercise_lookup = {
        "query": query,
        "name_matches": name_matches,
        "matched_exercises": matched_exercises,
    }
    return {
        "exercise_lookup": exercise_lookup,
        "observation": (
            f"Exercise entity lookup found {len(matched_exercises)} exercise records, "
            f"{len(name_matches)} name matches."
        ),
    }


def _allowed_equipment_for_workout(
    profile: Optional[UserProfile],
    workout_preferences: Dict[str, Any],
) -> Optional[List[str]]:
    requested_equipment = _normalize_string_list(workout_preferences.get("equipment_available"))
    if requested_equipment:
        return requested_equipment

    available = [
        item.name.strip().lower()
        for item in (profile.equipment_notes if profile else [])
        if item.enabled and item.name.strip()
    ]
    return available or None


def _slot_priority_boost(slot_name: str, target_muscles: Set[str]) -> int:
    if not target_muscles:
        return 0
    return len(WORKOUT_SLOT_MUSCLE_HINTS.get(slot_name, set()) & target_muscles)


def _build_exercise_candidate_bundle(
    profile: Optional[UserProfile],
    workout_preferences: Dict[str, Any],
    limit_per_slot: int = 6,
) -> Dict[str, Any]:
    blocked_equipment = _profile_unavailable_equipment(profile)
    allowed_equipment = _allowed_equipment_for_workout(profile, workout_preferences)
    target_muscles = {item.lower() for item in _normalize_string_list(workout_preferences.get("target_muscle_groups"))}
    cardio_preference = str(workout_preferences.get("cardio_preference") or "").strip().lower()

    slot_candidates: Dict[str, List[Dict[str, Any]]] = {}
    slot_configs = [
        ("strength_compounds", limit_per_slot),
        ("strength_accessories", limit_per_slot),
        ("core_pool", max(3, limit_per_slot // 2)),
    ]

    cardio_limit = max(2, limit_per_slot // 2)
    if cardio_preference in {"moderate", "high"} or (profile and profile.fitness_goal == "cut"):
        cardio_limit = max(3, limit_per_slot // 2 + 1)
    slot_configs.append(("cardio_modes", cardio_limit))

    slot_configs.sort(key=lambda item: (_slot_priority_boost(item[0], target_muscles), item[0]), reverse=True)

    for slot_name, slot_limit in slot_configs:
        slot_from_db = find_exercises_for_workout_slot(
            slot_name=slot_name,
            limit=slot_limit,
            candidate_only=True,
            allowed_equipment=allowed_equipment,
            blocked_equipment=blocked_equipment,
        )
        slot_candidates[slot_name] = [_sanitize_exercise_document(item) for item in slot_from_db]

    total_candidates = sum(len(items) for items in slot_candidates.values())
    return {
        "candidate_strategy": {
            "limit_per_slot": limit_per_slot,
            "allowed_equipment": allowed_equipment or [],
            "target_muscle_groups": sorted(target_muscles),
            "cardio_preference": cardio_preference or None,
        },
        "slot_candidates": slot_candidates,
        "total_candidates": total_candidates,
    }


def _resolved_days_per_week(profile: UserProfile, workout_preferences: Dict[str, Any]) -> int:
    return int(
        workout_preferences.get("days_per_week")
        or profile.workout_frequency
        or 3
    )


def _resolved_duration_minutes(profile: UserProfile, workout_preferences: Dict[str, Any]) -> int:
    return int(
        workout_preferences.get("duration_minutes")
        or profile.workout_duration
        or 60
    )


async def generate_workout_plan_artifact(
    user_input: str,
    user_profile: Dict[str, Any],
    workout_preferences: Optional[Dict[str, Any]] = None,
    prior_exercise_lookups: Optional[List[Dict[str, Any]]] = None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_name: str = "qwen3.5-plus",
) -> Dict[str, Any]:
    profile = UserProfile(**user_profile)
    resolved_workout_preferences = dict(workout_preferences or {})

    requested_exercise_names = _normalize_name_list(resolved_workout_preferences.get("requested_exercise_names"))
    excluded_exercise_names = _normalize_name_list(resolved_workout_preferences.get("excluded_exercise_names"))
    resolved_prior_exercise_lookups = [item for item in (prior_exercise_lookups or []) if item]

    workout_request = WorkoutPlanRequest(
        split_type=resolved_workout_preferences.get("split_type") or "full_body",
        training_style=resolved_workout_preferences.get("training_style") or "hypertrophy",
        days_per_week=_resolved_days_per_week(profile, resolved_workout_preferences),
        duration_minutes=_resolved_duration_minutes(profile, resolved_workout_preferences),
    )

    exercise_candidates = _build_exercise_candidate_bundle(
        profile=profile,
        workout_preferences=resolved_workout_preferences,
        limit_per_slot=6,
    )
    resolved_workout_context = {
        "requested_exercise_names": requested_exercise_names,
        "excluded_exercise_names": excluded_exercise_names,
        "days_per_week": workout_request.days_per_week,
        "duration_minutes": workout_request.duration_minutes,
        "split_type": workout_request.split_type,
        "training_style": workout_request.training_style,
        "equipment_available": _normalize_string_list(resolved_workout_preferences.get("equipment_available")),
        "target_muscle_groups": _normalize_string_list(resolved_workout_preferences.get("target_muscle_groups")),
        "cardio_preference": resolved_workout_preferences.get("cardio_preference"),
        "notes": resolved_workout_preferences.get("notes", []) or [],
    }

    llm = build_chat_model(base_url=base_url, model_name=model_name, temperature=0.5, extra_body={"enable_thinking": True})
    workout_plan = await invoke_structured_with_retry(
        llm,
        WorkoutPlanStructured,
        [
            SystemMessage(
                content=WORKOUT_PLAN_SYSTEM_PROMPT + "\n\n" + build_structured_output_instruction(WorkoutPlanStructured)
            ),
            HumanMessage(
                content=build_workout_plan_user_prompt(
                    user_input,
                    profile.model_dump(),
                    exercise_candidates=exercise_candidates,
                    workout_preferences=resolved_workout_context,
                    prior_exercise_lookups=resolved_prior_exercise_lookups,
                )
            ),
        ],
    )

    slot_candidates = (exercise_candidates or {}).get("slot_candidates", {})
    available_exercises_count = sum(len(items) for items in slot_candidates.values())
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
            "available_exercises_count": available_exercises_count,
            "prior_exercise_lookup_count": len(resolved_prior_exercise_lookups),
            "generated_at": datetime.now().isoformat(),
        },
        "observation": f"Workout plan tool generated a {workout_request.days_per_week}-day program.",
    }
