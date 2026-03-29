from typing import Any, Dict, Iterable, List, Optional

import pymongo

from ..infrastructure.database import get_database


def get_food_collection_name(use_full_database: bool) -> str:
    return "branded_foods" if use_full_database else "branded_foods_sample"


def get_food_collection(use_full_database: bool):
    return get_database()[get_food_collection_name(use_full_database)]


def find_foods_by_text(
    query: str,
    limit: int,
    use_full_database: bool,
    include_ingredients: bool = True,
) -> List[Dict[str, Any]]:
    filters = [
        {"description": {"$regex": query, "$options": "i"}},
        {"brandOwner": {"$regex": query, "$options": "i"}},
    ]
    if include_ingredients:
        filters.append({"ingredients": {"$regex": query, "$options": "i"}})
    return list(get_food_collection(use_full_database).find({"$or": filters}).limit(limit))


def find_foods_with_macro_filters(
    query: str,
    limit: int,
    use_full_database: bool,
    protein_min: float,
    carbs_max: float,
    calories_max: float,
) -> List[Dict[str, Any]]:
    mongo_query = {
        "$and": [
            {
                "$or": [
                    {"description": {"$regex": query, "$options": "i"}},
                    {"brandOwner": {"$regex": query, "$options": "i"}},
                    {"ingredients": {"$regex": query, "$options": "i"}},
                ]
            },
            {"nutrition_enhanced.per_100g.protein_g": {"$gte": protein_min}},
            {"nutrition_enhanced.per_100g.carbs_g": {"$lte": carbs_max}},
            {"nutrition_enhanced.per_100g.energy_kcal": {"$lte": calories_max}},
        ]
    }
    return list(get_food_collection(use_full_database).find(mongo_query).limit(limit))


def count_food_documents(use_full_database: bool) -> int:
    return get_food_collection(use_full_database).count_documents({})


def iterate_food_documents(use_full_database: bool, max_documents: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    cursor = get_food_collection(use_full_database).find({})
    if max_documents:
        cursor = cursor.limit(max_documents)
    return cursor


def create_sample_food_indexes() -> None:
    branded_foods = get_food_collection(False)
    branded_foods.create_index([("foodClass", pymongo.ASCENDING)])
    branded_foods.create_index([("brandOwner", pymongo.ASCENDING)])
    branded_foods.create_index([("foodCategory", pymongo.ASCENDING)])
    branded_foods.create_index([("gtinUpc", pymongo.ASCENDING)])
    branded_foods.create_index(
        [("description", pymongo.TEXT), ("ingredients", pymongo.TEXT)],
        name="search_text_index",
    )
    branded_foods.create_index(
        [("nutrition_enhanced.macro_breakdown.primary_macro_category", pymongo.ASCENDING)]
    )
    branded_foods.create_index(
        [("nutrition_enhanced.macro_breakdown.is_high_protein", pymongo.ASCENDING)]
    )
    branded_foods.create_index([("nutrition_enhanced.macro_breakdown.is_high_fat", pymongo.ASCENDING)])
    branded_foods.create_index([("nutrition_enhanced.macro_breakdown.is_high_carb", pymongo.ASCENDING)])
    branded_foods.create_index([("nutrition_enhanced.macro_breakdown.is_balanced", pymongo.ASCENDING)])
    branded_foods.create_index([("nutrition_enhanced.per_100g.protein_g", pymongo.DESCENDING)])
    branded_foods.create_index([("nutrition_enhanced.per_100g.energy_kcal", pymongo.ASCENDING)])
    branded_foods.create_index([("nutrition_enhanced.nutrition_density_score", pymongo.DESCENDING)])
    branded_foods.create_index([("nutrition_enhanced.macro_breakdown.protein_percent", pymongo.DESCENDING)])


def insert_sample_food_batch(batch: List[Dict[str, Any]]) -> int:
    get_food_collection(False).insert_many(batch, ordered=False)
    return len(batch)


def count_enhanced_sample_foods() -> int:
    return get_food_collection(False).count_documents({"nutrition_enhanced": {"$exists": True}})


def list_collection_names() -> List[str]:
    return get_database().list_collection_names()


def get_collection_stats(collection_name: str) -> Dict[str, Any]:
    collection = get_database()[collection_name]
    sample = collection.find_one()
    return {
        "document_count": collection.count_documents({}),
        "sample_fields": list(sample.keys())[:10] if sample else [],
        "total_fields": len(sample.keys()) if sample else 0,
    }
