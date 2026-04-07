from typing import Any, Dict, Iterable, List, Optional

import pymongo

from ..infrastructure.database import get_database


FOOD_COLLECTION = "foods"
DEFAULT_FOOD_TYPES = ["foundation"]
PROTEIN_CATEGORIES = [
    "Beef Products",
    "Finfish and Shellfish Products",
    "Lamb, Veal, and Game Products",
    "Pork Products",
    "Poultry Products",
    "Dairy and Egg Products",
]
CARB_CATEGORIES = [
    "Cereal Grains and Pasta",
]
VEGETABLE_CATEGORIES = [
    "Vegetables and Vegetable Products",
]
FAT_CATEGORIES = [
    "Nut and Seed Products",
]
OIL_CATEGORIES = [
    "Fats and Oils",
]
OIL_NAME_WHITELIST = [
    "Oil, peanut",
    "Oil, soybean",
    "Oil, olive, extra virgin",
    "Oil, olive, extra light",
]
FLEXIBLE_CATEGORIES = [
    "Fruits and Fruit Juices",
]


MEAL_SLOT_QUERIES: Dict[str, Dict[str, Any]] = {
    "proteins": {
        "category": {"$in": PROTEIN_CATEGORIES},
        "per_100g.calories_kcal": {"$lte": 360},
    },
    "carbs": {
        "category": {"$in": CARB_CATEGORIES},
        "per_100g.fat_g": {"$lte": 15},
    },
    "vegetables": {
        "category": {"$in": VEGETABLE_CATEGORIES},
        "per_100g.fat_g": {"$lte": 5},
        "per_100g.carbs_g": {"$lte": 15},
    },
    "fats": {
        "category": {"$in": FAT_CATEGORIES},
    },
    "oil": {
        "category": {"$in": OIL_CATEGORIES},
        "name": {"$in": OIL_NAME_WHITELIST},
    },
    "flexible": {
        "category": {"$in": FLEXIBLE_CATEGORIES},
    },
}

MEAL_SLOT_SORTS: Dict[str, List[tuple[str, int]]] = {
    "proteins": [("per_100g.protein_g", pymongo.DESCENDING), ("per_100g.calories_kcal", pymongo.ASCENDING)],
    "carbs": [("per_100g.carbs_g", pymongo.DESCENDING), ("per_100g.fat_g", pymongo.ASCENDING)],
    "vegetables": [("per_100g.calories_kcal", pymongo.ASCENDING), ("per_100g.carbs_g", pymongo.ASCENDING)],
    "fats": [("per_100g.fat_g", pymongo.DESCENDING), ("per_100g.calories_kcal", pymongo.ASCENDING)],
    "oil": [("name", pymongo.ASCENDING)],
    "flexible": [("per_100g.carbs_g", pymongo.DESCENDING), ("name", pymongo.ASCENDING)],
}


def get_food_collection():
    return get_database()[FOOD_COLLECTION]


def _base_collection_filter(food_types: Optional[List[str]]) -> Dict[str, Any]:
    return {"type": {"$in": food_types or DEFAULT_FOOD_TYPES}}


def _merge_filters(*filters: Dict[str, Any]) -> Dict[str, Any]:
    normalized_filters = [item for item in filters if item]
    if not normalized_filters:
        return {}
    if len(normalized_filters) == 1:
        return normalized_filters[0]
    return {"$and": normalized_filters}


def find_foods_for_meal_slot(
    slot_name: str,
    limit: int,
    food_types: Optional[List[str]] = None,
    meal_candidates_only: bool = True,
) -> List[Dict[str, Any]]:
    slot_query = MEAL_SLOT_QUERIES.get(slot_name, {})
    base_filter = _base_collection_filter(food_types)
    if meal_candidates_only and slot_name != "oil":
        base_filter = _merge_filters(base_filter, {"candidate_flags.is_meal_candidate": True})
    mongo_query = _merge_filters(base_filter, slot_query)
    sort_spec = MEAL_SLOT_SORTS.get(slot_name, [("name", pymongo.ASCENDING)])
    cursor = get_food_collection().find(mongo_query).sort(sort_spec)
    return list(cursor.limit(limit))
