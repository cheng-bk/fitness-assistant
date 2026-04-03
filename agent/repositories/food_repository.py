from typing import Any, Dict, Iterable, List, Optional

import pymongo

from ..infrastructure.database import get_database


FOOD_COLLECTION = "foods"
DEFAULT_FOOD_TYPES = ["foundation"]

MEAL_SLOT_QUERIES: Dict[str, Dict[str, Any]] = {
    "proteins": {
        "$or": [
            {"category": {"$regex": "poultry|beef|pork|lamb|veal|fish|shellfish|egg|legume|soy|bean|lentil|meat", "$options": "i"}},
            {"per_100g.protein_g": {"$gte": 10}},
        ],
        "per_100g.calories_kcal": {"$lte": 360},
    },
    "carbs": {
        "$or": [
            {"category": {"$regex": "grain|cereal|rice|pasta|bread|fruit|potato|starch|baked", "$options": "i"}},
            {"per_100g.carbs_g": {"$gte": 12}},
        ],
        "per_100g.fat_g": {"$lte": 15},
    },
    "vegetables": {
        "$or": [
            {"category": {"$regex": "vegetable|fungi|mushroom|herb", "$options": "i"}},
            {"per_100g.calories_kcal": {"$lte": 80}},
        ],
        "per_100g.fat_g": {"$lte": 5},
        "per_100g.carbs_g": {"$lte": 15},
    },
    "fats": {
        "$or": [
            {"category": {"$regex": "fat|oil|nut|seed|olive|avocado", "$options": "i"}},
            {"per_100g.fat_g": {"$gte": 8}},
        ],
    },
}

MEAL_SLOT_SORTS: Dict[str, List[tuple[str, int]]] = {
    "proteins": [("per_100g.protein_g", pymongo.DESCENDING), ("per_100g.calories_kcal", pymongo.ASCENDING)],
    "carbs": [("per_100g.carbs_g", pymongo.DESCENDING), ("per_100g.fat_g", pymongo.ASCENDING)],
    "vegetables": [("per_100g.calories_kcal", pymongo.ASCENDING), ("per_100g.carbs_g", pymongo.ASCENDING)],
    "fats": [("per_100g.fat_g", pymongo.DESCENDING), ("per_100g.calories_kcal", pymongo.ASCENDING)],
}


def get_food_collection_name(food_types: Optional[List[str]] = None) -> str:
    return FOOD_COLLECTION


def get_food_collection():
    return get_database()[FOOD_COLLECTION]


def count_food_documents(food_types: Optional[List[str]] = None) -> int:
    return int(get_food_collection().count_documents(_base_collection_filter(food_types)))


def iterate_food_documents(
    food_types: Optional[List[str]] = None,
    max_documents: Optional[int] = None,
) -> Iterable[Dict[str, Any]]:
    cursor = get_food_collection().find(_base_collection_filter(food_types))
    if max_documents is not None:
        cursor = cursor.limit(max_documents)
    return cursor


def _base_collection_filter(food_types: Optional[List[str]]) -> Dict[str, Any]:
    return {"type": {"$in": food_types or DEFAULT_FOOD_TYPES}}


def _merge_filters(*filters: Dict[str, Any]) -> Dict[str, Any]:
    normalized_filters = [item for item in filters if item]
    if not normalized_filters:
        return {}
    if len(normalized_filters) == 1:
        return normalized_filters[0]
    return {"$and": normalized_filters}


def _build_text_match_filters(normalized_query: str) -> List[Dict[str, Any]]:
    base_fields = [
        "name",
        "category",
        "brand_name",
        "search_terms",
        "measurements.modifier",
        "measurements.unit_name",
        "measurements.unit_abbreviation",
    ]
    filters = [{field: {"$regex": normalized_query, "$options": "i"}} for field in base_fields]

    tokens = [token for token in normalized_query.split() if len(token) >= 3]
    for token in tokens:
        filters.extend(
            {field: {"$regex": token, "$options": "i"}}
            for field in ["name", "category", "search_terms", "measurements.modifier"]
        )
    return filters


def find_foods_by_text(
    query: str,
    limit: int,
    food_types: Optional[List[str]] = None,
    meal_candidates_only: bool = False,
) -> List[Dict[str, Any]]:
    base_filter = _base_collection_filter(food_types)
    if meal_candidates_only:
        base_filter = _merge_filters(base_filter, {"candidate_flags.is_meal_candidate": True})
    normalized_query = query.strip()
    if not normalized_query:
        cursor = get_food_collection().find(base_filter).sort([("name", pymongo.ASCENDING)])
        return list(cursor.limit(limit))

    mongo_query = _merge_filters(base_filter, {"$or": _build_text_match_filters(normalized_query)})
    cursor = get_food_collection().find(mongo_query).sort([("name", pymongo.ASCENDING)])
    return list(cursor.limit(limit))


def find_foods_with_macro_filters(
    query: str,
    limit: int,
    protein_min: float,
    carbs_min: float,
    carbs_max: float,
    calories_max: float,
    food_types: Optional[List[str]] = None,
    meal_candidates_only: bool = False,
) -> List[Dict[str, Any]]:
    text_match = find_foods_by_text(
        query=query,
        limit=max(limit * 4, 50),
        food_types=food_types,
        meal_candidates_only=meal_candidates_only,
    )
    candidate_ids = [item["_id"] for item in text_match]
    if not candidate_ids:
        return []

    filters: List[Dict[str, Any]] = [
        _base_collection_filter(food_types),
        {"_id": {"$in": candidate_ids}},
    ]
    if meal_candidates_only:
        filters.append({"candidate_flags.is_meal_candidate": True})
    if protein_min > 0:
        filters.append({"per_100g.protein_g": {"$gte": protein_min}})
    if carbs_min > 0:
        filters.append({"per_100g.carbs_g": {"$gte": carbs_min}})
    if carbs_max < 999:
        filters.append({"per_100g.carbs_g": {"$lte": carbs_max}})
    if calories_max < 999:
        filters.append({"per_100g.calories_kcal": {"$lte": calories_max}})

    cursor = get_food_collection().find(_merge_filters(*filters)).sort(
        [("per_100g.protein_g", pymongo.DESCENDING), ("per_100g.calories_kcal", pymongo.ASCENDING)]
    )
    return list(cursor.limit(limit))


def find_foods_for_meal_slot(
    slot_name: str,
    limit: int,
    food_types: Optional[List[str]] = None,
    meal_candidates_only: bool = True,
) -> List[Dict[str, Any]]:
    slot_query = MEAL_SLOT_QUERIES.get(slot_name, {})
    base_filter = _base_collection_filter(food_types)
    if meal_candidates_only:
        base_filter = _merge_filters(base_filter, {"candidate_flags.is_meal_candidate": True})
    mongo_query = _merge_filters(base_filter, slot_query)
    sort_spec = MEAL_SLOT_SORTS.get(slot_name, [("name", pymongo.ASCENDING)])
    cursor = get_food_collection().find(mongo_query).sort(sort_spec)
    return list(cursor.limit(limit))
