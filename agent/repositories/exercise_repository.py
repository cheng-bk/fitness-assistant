from typing import Any, Dict, List, Optional, Set

import pymongo

from ..infrastructure.database import get_database


EXERCISE_COLLECTION = "exercises"
DEFAULT_EXERCISE_CATEGORIES = ["strength", "cardio"]
COMMON_EQUIPMENT = [
    "body only",
    "dumbbell",
    "barbell",
    "machine",
    "cable",
    "kettlebells",
]
CARDIO_EQUIPMENT = COMMON_EQUIPMENT + ["other", "bands", "exercise ball", "medicine ball", None]
COMPOUND_MUSCLES = [
    "chest",
    "shoulders",
    "quadriceps",
    "hamstrings",
    "glutes",
    "lats",
    "middle back",
    "lower back",
]
ACCESSORY_MUSCLES = [
    "biceps",
    "triceps",
    "calves",
    "forearms",
    "traps",
    "abductors",
    "adductors",
    "shoulders",
]
CORE_MUSCLES = ["abdominals"]


EXERCISE_SLOT_QUERIES: Dict[str, Dict[str, Any]] = {
    "strength_compounds": {
        "category": "strength",
        "primaryMuscles": {"$in": COMPOUND_MUSCLES},
        "equipment": {"$in": COMMON_EQUIPMENT},
    },
    "strength_accessories": {
        "category": "strength",
        "primaryMuscles": {"$in": ACCESSORY_MUSCLES},
        "equipment": {"$in": COMMON_EQUIPMENT},
    },
    "core_pool": {
        "category": "strength",
        "primaryMuscles": {"$in": CORE_MUSCLES},
        "equipment": {"$in": COMMON_EQUIPMENT + ["bands", "exercise ball"]},
    },
    "cardio_modes": {
        "category": "cardio",
        "equipment": {"$in": CARDIO_EQUIPMENT},
    },
}

EXERCISE_SLOT_SORTS: Dict[str, List[tuple[str, int]]] = {
    "strength_compounds": [
        ("candidate_flags.rank_in_bucket", pymongo.ASCENDING),
        ("name", pymongo.ASCENDING),
    ],
    "strength_accessories": [
        ("candidate_flags.rank_in_bucket", pymongo.ASCENDING),
        ("name", pymongo.ASCENDING),
    ],
    "core_pool": [
        ("candidate_flags.rank_in_bucket", pymongo.ASCENDING),
        ("name", pymongo.ASCENDING),
    ],
    "cardio_modes": [
        ("candidate_flags.rank_in_bucket", pymongo.ASCENDING),
        ("name", pymongo.ASCENDING),
    ],
}


def get_exercise_collection():
    return get_database()[EXERCISE_COLLECTION]


def _merge_filters(*filters: Dict[str, Any]) -> Dict[str, Any]:
    normalized_filters = [item for item in filters if item]
    if not normalized_filters:
        return {}
    if len(normalized_filters) == 1:
        return normalized_filters[0]
    return {"$and": normalized_filters}


def _equipment_filter(
    allowed_equipment: Optional[List[str]] = None,
    blocked_equipment: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    clauses: List[Dict[str, Any]] = []
    normalized_allowed = [str(item).strip().lower() for item in (allowed_equipment or []) if str(item).strip()]
    normalized_blocked = {str(item).strip().lower() for item in (blocked_equipment or set()) if str(item).strip()}

    if normalized_allowed:
        passthrough = {"body only", "other"}
        clauses.append({"equipment": {"$in": sorted(set(normalized_allowed) | passthrough)}})

    if normalized_blocked:
        clauses.append(
            {
                "$or": [
                    {"equipment": None},
                    {"equipment": {"$nin": sorted(normalized_blocked)}},
                ]
            }
        )

    return _merge_filters(*clauses)


def find_exercises_for_workout_slot(
    slot_name: str,
    limit: int,
    candidate_only: bool = True,
    allowed_equipment: Optional[List[str]] = None,
    blocked_equipment: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    slot_query = EXERCISE_SLOT_QUERIES.get(slot_name, {})
    base_filter: Dict[str, Any] = {"category": {"$in": DEFAULT_EXERCISE_CATEGORIES}}
    if candidate_only:
        base_filter = _merge_filters(base_filter, {"candidate_flags.is_candidate": True})
    equipment_query = _equipment_filter(allowed_equipment=allowed_equipment, blocked_equipment=blocked_equipment)
    mongo_query = _merge_filters(base_filter, slot_query, equipment_query)
    sort_spec = EXERCISE_SLOT_SORTS.get(slot_name, [("name", pymongo.ASCENDING)])
    cursor = get_exercise_collection().find(mongo_query).sort(sort_spec)
    return list(cursor.limit(limit))

