"""
Compact meal-planner schema stored in MongoDB:

{
  "_id": "fdc_id:<fdc_id>",
  "type": "foundation",
  "name": "Hummus, commercial",
  "per_100g": {
    "calories_kcal_mean": 229.0,
    "protein_g_mean": 7.35,
    "fat_g_mean": 17.1,
    "carbs_g_mean": 14.9
  }
}

"""

import json
import os
from pymongo import UpdateOne
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Dict, Iterable, List, Optional

from agent.infrastructure.database import get_mongo_client


FOUNDATION_DATA_PATH = Path("data/json/food/FoodData_Central_foundation_food_json_2025-12-18.json")
BATCH_SIZE = 128
COLLECTION = "foods"

# USDA nutrient ids / numbers used for the compact meal planner schema.
NUTRIENT_KEYS = {
    "protein_g": {"id": 1003, "number": "203"},
    "fat_g": {"id": 1004, "number": "204"},
    "carbs_g_difference": {"id": 1005, "number": "205"},
    "carbs_g_summation": {"id": 1050, "number": "205.2"},
    "energy_kcal": {"id": 1008, "number": "208"},
    "energy_kj": {"id": 1062, "number": "268"},
}


def load_foundation_foods(input_path: Path) -> List[Dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    foods = payload.get("FoundationFoods")
    return foods


def extract_nutrient_amount(food_nutrients: List[Dict[str, Any]], nutrient_key: str) -> Optional[float]:
    keys = NUTRIENT_KEYS[nutrient_key]

    for nutrient_entry in food_nutrients:
        nutrient = nutrient_entry.get("nutrient", {})
        nutrient_id = nutrient.get("id")
        nutrient_number = str(nutrient.get("number")) if nutrient.get("number") is not None else None

        if nutrient_id == keys["id"] or nutrient_number == keys["number"]:
            value = nutrient_entry.get("amount")
            if value is None:
                value = nutrient_entry.get("median")
            if value is None:
                return None
            return round(float(value), 2)

    return None


def extract_energy_kcal(food_nutrients: List[Dict[str, Any]]) -> Optional[float]:
    energy_kcal = extract_nutrient_amount(food_nutrients, "energy_kcal")
    if energy_kcal is not None:
        return energy_kcal

    energy_kj = extract_nutrient_amount(food_nutrients, "energy_kj")
    if energy_kj is None:
        return None

    return round(energy_kj * 0.239005736, 2)


def extract_carbs_g(food_nutrients: List[Dict[str, Any]]) -> Optional[float]:
    carbs_by_difference = extract_nutrient_amount(food_nutrients, "carbs_g_difference")
    if carbs_by_difference is not None:
        return carbs_by_difference

    return extract_nutrient_amount(food_nutrients, "carbs_g_summation")


def extract_measurements(food: Dict[str, Any]) -> List[Dict[str, Any]]:
    measurements = food.get("foodPortions", [])
    res = []
    for measurement in measurements:
        res.append({
            "value": measurement.get("value"),
            "unit_name": measurement.get("measureUnit").get("name"),
            "unit_abbreviation": measurement.get("measureUnit").get("abbreviation"),
            "modifier": measurement.get("modifier"),
            "gram_weight": measurement.get("gramWeight")
        })
    return res


def build_meal_planner_document(food: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    fdc_id = food.get("fdcId")
    name = food.get("description")
    category = food.get("foodCategory").get("description") if food.get("foodCategory") else None
    nutrients = food.get("foodNutrients", [])

    if not fdc_id or not name or not isinstance(nutrients, list):
        return None

    document = {
        "_id": f"fcid:{fdc_id}",
        "type": "foundation",
        "name": name,
        "category": category,
        "per_100g": {
            "calories_kcal": extract_energy_kcal(nutrients),
            "protein_g": extract_nutrient_amount(nutrients, "protein_g"),
            "fat_g": extract_nutrient_amount(nutrients, "fat_g"),
            "carbs_g": extract_carbs_g(nutrients),
        },
        "measurements": extract_measurements(food)
    }

    return document


def iter_compact_documents(foods: Iterable[Dict[str, Any]], limit: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    processed = 0
    for food in foods:
        if limit is not None and processed >= limit:
            break

        compact_document = build_meal_planner_document(food)
        if compact_document is None:
            continue

        processed += 1
        yield compact_document


def create_indexes(collection) -> None:
    collection.create_index("type")
    collection.create_index([("type", 1), ("name", 1)])
    collection.create_index([("type", 1), ("category", 1)])
    collection.create_index([("type", 1), ("per_100g.calories_kcal", 1)])
    collection.create_index([("type", 1), ("per_100g.carbs_g", 1)])
    collection.create_index([("type", 1), ("per_100g.protein_g", -1)])
    collection.create_index([("type", 1), ("per_100g.fat_g", -1)])


def write_documents(collection, documents: Iterable[Dict[str, Any]], batch_size: int) -> int:
    total_written = 0
    batch = []

    for document in documents:
        batch.append(
            UpdateOne(
                {"_id": document["_id"]},
                {"$set": document},
                upsert=True,
            )
        )

        if len(batch) >= batch_size:
            collection.bulk_write(batch, ordered=False)
            total_written += len(batch)
            batch = []

    if batch:
        collection.bulk_write(batch, ordered=False)
        total_written += len(batch)

    return total_written


def main() -> None:
    load_dotenv()
    
    database_name = os.getenv("MONGO_DB_NAME", "fitness_assistant")

    foods = load_foundation_foods(FOUNDATION_DATA_PATH)
    compact_documents = list(iter_compact_documents(foods))

    if not compact_documents:
        raise RuntimeError("No compact documents were produced from the input file.")

    database = get_mongo_client()[database_name]
    collection = database[COLLECTION]
    create_indexes(collection)
    written = write_documents(collection, compact_documents, batch_size=BATCH_SIZE)
    print(
        f"Upserted {written} foundation food documents into "
        f"{database.name}.{COLLECTION} from {FOUNDATION_DATA_PATH}."
    )


if __name__ == "__main__":
    main()
