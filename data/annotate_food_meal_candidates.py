import os
from pprint import pprint
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from pymongo import UpdateOne

from agent.infrastructure.database import get_mongo_client


COLLECTION = "foods"
FOOD_TYPES = ["foundation"]
BATCH_SIZE = 256

STRONG_TAGS = {
    "raw",
    "dry",
    "dried",
    "canned",
    "frozen",
    "cooked",
    "refrigerated",
    "shelf stable",
    "shelf-stable",
    "plain",
    "unsweetened",
    "from concentrate",
    "whole grain",
    "nonfat",
    "lowfat",
    "full fat",
    "whole milk",
    "not fortified",
    "enriched",
    "unenriched",
    "unbleached",
    "pasteurized",
    "american",
    "greek",
    "all-purpose",
    "wild caught",
    "farm raised",
}

WEAK_TAGS = {
    "green",
    "white",
    "red",
    "yellow",
    "black",
    "whole",
    "sliced",
    "solid",
    "ground",
    "boneless",
    "skinless",
    "with skin",
    "without skin",
    "peeled",
    "seeded",
    "seedless",
    "drained and rinsed",
    "sodium added",
    "with salt added",
    "sugar added",
    "sweet",
    "large",
    "baby",
    "bell",
    "bulb",
    "leaf",
    "round",
    "root removed",
    "kidney",
    "choice",
    "select",
    "grade a",
    "trimmed to 1/8\" fat",
    "trimmed to 0\" fat",
    "separable lean only",
    "meat and skin",
    "breast",
    "loin",
    "pork",
    "salmon",
    "rice",
    "grain",
    "wheat",
}

ALIASES = {
    "dried": "dry",
    "shelf-stable": "shelf stable",
}


@dataclass
class SimilarityConfig:
    relative_threshold: float = 0.3
    calories_min_active: float = 50.0
    protein_min_active: float = 5.0
    fat_min_active: float = 5.0
    carbs_min_active: float = 10.0


similarity_config = SimilarityConfig()


def get_food_collection(database_name: str):
    return get_mongo_client()[database_name][COLLECTION]


def normalize_tag(tag: str) -> str:
    normalized = tag.strip().lower()
    return ALIASES.get(normalized, normalized)


def extract_base_tag(name: str) -> str:
    name_parts = [normalize_tag(part) for part in name.split(", ") if part.strip()]
    if not name_parts:
        return "unknown"
    return name_parts[0] or "unknown"


def extract_tags(name: str) -> Tuple[List[str], List[str]]:
    strong_tags: List[str] = []
    weak_tags: List[str] = []
    for part in [normalize_tag(part) for part in name.split(", ") if part.strip()][1:]:
        if not part:
            continue
        if part in STRONG_TAGS:
            strong_tags.append(part)
            continue
        if part in WEAK_TAGS:
            weak_tags.append(part)
            continue
        weak_tags.append(part)

    return list(dict.fromkeys(strong_tags)), list(dict.fromkeys(weak_tags))


def get_per_100g(food_item: Dict[str, Any]) -> Dict[str, Optional[float]]:
    per_100g = food_item.get("per_100g", {})
    return {
        "calories_kcal": safe_float(per_100g.get("calories_kcal")),
        "protein_g": safe_float(per_100g.get("protein_g")),
        "fat_g": safe_float(per_100g.get("fat_g")),
        "carbs_g": safe_float(per_100g.get("carbs_g")),
    }


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_thresholds(config: SimilarityConfig) -> Dict[str, float]:
    return {
        "calories_kcal": config.calories_min_active,
        "protein_g": config.protein_min_active,
        "fat_g": config.fat_min_active,
        "carbs_g": config.carbs_min_active,
    }


def foods_are_similar(
    food_a: Dict[str, Any],
    food_b: Dict[str, Any],
    config: SimilarityConfig,
) -> bool:
    values_a = get_per_100g(food_a)
    values_b = get_per_100g(food_b)
    active_comparisons = 0

    for metric_name, min_active in metric_thresholds(config).items():
        value_a = values_a.get(metric_name)
        value_b = values_b.get(metric_name)

        if value_a is None or value_b is None:
            continue

        scale = max(abs(value_a), abs(value_b))
        if scale < min_active:
            continue

        relative_diff = abs(value_a - value_b) / scale
        active_comparisons += 1
        if relative_diff > config.relative_threshold:
            return False

    return True


def representative_score(food_item: Dict[str, Any]) -> Tuple[int, int, int, int]:
    per_100g = get_per_100g(food_item)
    nutrient_completeness = sum(value is not None for value in per_100g.values())
    measurement_count = len(food_item.get("measurements", []) or [])
    name = food_item.get("name", "")
    weak_tag_count = len(extract_tags(name)[1])
    return (
        nutrient_completeness,
        measurement_count,
        -weak_tag_count,
        -len(name),
    )


def sort_group_items(group_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        group_items,
        key=lambda item: representative_score(item),
        reverse=True,
    )


def build_candidate_annotation(
    food_item: Dict[str, Any],
    representative: Dict[str, Any],
    config: SimilarityConfig,
) -> Dict[str, Any]:
    base_tag = extract_base_tag(food_item.get("name", ""))
    strong_tags, weak_tags = extract_tags(food_item.get("name", ""))
    representative_id = representative["_id"]
    is_candidate = food_item["_id"] == representative_id
    return {
        "base_tag": base_tag,
        "strong_tags": strong_tags,
        "weak_tags": weak_tags,
        "representative_id": representative_id,
        "representative_name": representative.get("name"),
        "is_meal_candidate": is_candidate,
        "similarity_policy": {
            "relative_threshold": config.relative_threshold,
            "calories_min_active": config.calories_min_active,
            "protein_min_active": config.protein_min_active,
            "fat_min_active": config.fat_min_active,
            "carbs_min_active": config.carbs_min_active,
        },
    }


def build_group_clusters(
    group_items: List[Dict[str, Any]],
    config: SimilarityConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    representatives: List[Dict[str, Any]] = []
    assignments: Dict[str, Dict[str, Any]] = {}

    for food_item in sort_group_items(group_items):
        matched_representative: Optional[Dict[str, Any]] = None
        for representative in representatives:
            if foods_are_similar(food_item, representative, config):
                matched_representative = representative
                break

        if matched_representative is None:
            representatives.append(food_item)
            matched_representative = food_item

        assignments[food_item["_id"]] = build_candidate_annotation(
            food_item=food_item,
            representative=matched_representative,
            config=config,
        )

    return representatives, assignments


def iter_food_documents(
    collection,
    food_types: List[str],
) -> Iterable[Dict[str, Any]]:
    return collection.find({"type": {"$in": food_types}})


def create_indexes(collection) -> None:
    collection.create_index([("type", 1), ("candidate_flags.is_meal_candidate", 1)])
    collection.create_index([("type", 1), ("candidate_flags.base_tag", 1)])
    collection.create_index([("type", 1), ("candidate_flags.group_key", 1)])
    collection.create_index([("type", 1), ("candidate_flags.representative_id", 1)])


def clear_candidate_flags(collection, food_types: List[str]) -> int:
    result = collection.update_many(
        {"type": {"$in": food_types}},
        {"$unset": {"candidate_flags": ""}},
    )
    return int(result.modified_count)


def build_operations(
    collection,
    food_types: List[str],
    config: Dict[str, float],
) -> Tuple[List[UpdateOne], Dict[str, Any]]:
    grouped_items: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    type_filter = {"type": {"$in": food_types}}
    for food_item in collection.find(type_filter):
        base_tag = extract_base_tag(food_item.get("name", ""))
        canonical_bucket = f"{food_item.get('type', 'foundation')}::{base_tag}"
        grouped_items[canonical_bucket].append(food_item)

    operations: List[UpdateOne] = []
    total_groups = len(grouped_items)
    representative_count = 0
    compressed_count = 0

    for group_items in grouped_items.values():
        representatives, assignments = build_group_clusters(group_items, config)
        representative_count += len(representatives)
        compressed_count += max(0, len(group_items) - len(representatives))
        for food_id, annotation in assignments.items():
            operations.append(
                UpdateOne(
                    {"_id": food_id},
                    {"$set": {"candidate_flags": annotation}},
                    upsert=False,
                )
            )

    stats = {
        "groups": total_groups,
        "documents": len(operations),
        "representatives": representative_count,
        "compressed_documents": compressed_count,
        "compression_ratio": round(compressed_count / len(operations), 4) if operations else 0.0,
        "food_types": food_types,
    }
    return operations, stats


def write_operations(collection, operations: List[UpdateOne], batch_size: int) -> int:
    written = 0
    batch: List[UpdateOne] = []
    for operation in operations:
        batch.append(operation)
        if len(batch) >= batch_size:
            collection.bulk_write(batch, ordered=False)
            written += len(batch)
            batch = []

    if batch:
        collection.bulk_write(batch, ordered=False)
        written += len(batch)

    return written


def main() -> None:
    load_dotenv()

    database_name = os.getenv("MONGO_DB_NAME", "fitness_assistant")
    collection = get_food_collection(database_name)
    create_indexes(collection)
    cleared = clear_candidate_flags(collection, FOOD_TYPES)

    operations, stats = build_operations(
        collection=collection,
        food_types=FOOD_TYPES,
        config=similarity_config,
    )

    written = write_operations(collection, operations, batch_size=BATCH_SIZE)
    
    print(
        f"Cleared candidate_flags on {cleared} foods, then annotated {written} foods "
        f"in {database_name}.{COLLECTION}."
    )
    pprint(stats)


if __name__ == "__main__":
    main()
