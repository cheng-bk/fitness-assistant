from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import psutil
from fastapi import HTTPException
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from ..models import (
    HybridSearchResponse,
    IndexStatus,
    NutritionQuery,
    VectorIndexStatusResponse,
    VectorSearchResponse,
)
from ..repositories.food_repository import (
    count_food_documents,
    find_foods_by_text,
    find_foods_for_meal_slot,
    find_foods_with_macro_filters,
    get_food_collection_name,
    iterate_food_documents,
)
from ..repositories.vector_index_repository import (
    cache_vector_store,
    check_file_info,
    drop_vector_store,
    get_embeddings,
    get_index_directory,
    index_exists,
    load_vector_store,
    save_vector_store,
)


VEGETABLE_KEYWORDS = {
    "broccoli",
    "spinach",
    "lettuce",
    "cabbage",
    "cauliflower",
    "zucchini",
    "eggplant",
    "asparagus",
    "pepper",
    "tomato",
    "carrot",
    "cucumber",
    "onion",
    "mushroom",
    "kale",
    "celery",
}

CATEGORY_ROLE_HINTS = {
    "vegetables": [
        "vegetable",
        "vegetables",
        "mushroom",
        "fungi",
        "herb",
        "seaweed",
    ],
    "proteins": [
        "poultry",
        "beef",
        "pork",
        "lamb",
        "veal",
        "game",
        "fish",
        "shellfish",
        "egg",
        "sausage",
        "meat",
        "legume",
        "bean",
        "lentil",
        "soy",
    ],
    "carbs": [
        "grain",
        "cereal",
        "rice",
        "pasta",
        "bread",
        "baked",
        "fruit",
        "potato",
        "starch",
    ],
    "fats": [
        "fat",
        "oil",
        "nut",
        "seed",
        "olive",
        "avocado",
    ],
}


def _get_per_100g(food_item: Dict[str, Any]) -> Dict[str, float]:
    per_100g = food_item.get("per_100g", {})
    return {
        "calories_kcal": float(per_100g.get("calories_kcal", 0) or 0),
        "protein_g": float(per_100g.get("protein_g", 0) or 0),
        "fat_g": float(per_100g.get("fat_g", 0) or 0),
        "carbs_g": float(per_100g.get("carbs_g", 0) or 0),
    }


def _get_category(food_item: Dict[str, Any]) -> str:
    return str(food_item.get("category") or "").strip()


def _format_measurements(food_item: Dict[str, Any], limit: int = 3) -> List[str]:
    formatted: List[str] = []
    seen = set()
    for measurement in food_item.get("measurements", []) or []:
        if not isinstance(measurement, dict):
            continue
        value = measurement.get("value")
        unit_name = measurement.get("unit_name")
        modifier = measurement.get("modifier")
        gram_weight = measurement.get("gram_weight")
        parts: List[str] = []
        if value:
            parts.append(str(value))
        if unit_name:
            parts.append(str(unit_name))
        if modifier:
            parts.append(str(modifier))
        label = " ".join(parts).strip()
        if gram_weight:
            label = f"{label} ({round(float(gram_weight), 1)}g)".strip()
        if not label:
            continue
        normalized_label = label.lower()
        if normalized_label in seen:
            continue
        seen.add(normalized_label)
        formatted.append(label)
        if len(formatted) >= limit:
            break
    return formatted


def _category_matches(category: str, role: str) -> bool:
    category_lower = category.lower()
    return any(keyword in category_lower for keyword in CATEGORY_ROLE_HINTS.get(role, []))


def infer_meal_role(food_item: Dict[str, Any]) -> str:
    name = str(food_item.get("name", "")).lower()
    category = _get_category(food_item)
    per_100g = _get_per_100g(food_item)
    protein = per_100g["protein_g"]
    fat = per_100g["fat_g"]
    carbs = per_100g["carbs_g"]
    calories = per_100g["calories_kcal"]

    if category:
        if _category_matches(category, "vegetables") and calories <= 90:
            return "vegetables"
        if _category_matches(category, "fats") and fat >= 8:
            return "fats"
        if _category_matches(category, "proteins") and protein >= 8:
            return "proteins"
        if _category_matches(category, "carbs") and carbs >= 10:
            return "carbs"
    if any(keyword in name for keyword in VEGETABLE_KEYWORDS) and calories <= 90:
        return "vegetables"
    if protein >= 10 and protein >= carbs and protein * 0.9 >= fat:
        return "proteins"
    if fat >= 8 and fat >= protein and fat >= carbs:
        return "fats"
    if carbs >= 12 and carbs >= protein and carbs >= fat:
        return "carbs"
    if calories <= 80 and fat <= 5:
        return "vegetables"
    return "flexible"


def normalize_food_document(food_item: Dict[str, Any], similarity_score: float = 0.0) -> Dict[str, Any]:
    per_100g = _get_per_100g(food_item)
    role = infer_meal_role(food_item)
    category = _get_category(food_item)
    measurements = _format_measurements(food_item)
    return {
        "id": food_item.get("_id"),
        "fdc_id": food_item.get("fdc_id"),
        "name": food_item.get("name"),
        "type": food_item.get("type", "foundation"),
        "category": category or None,
        "brand_name": food_item.get("brand_name"),
        "meal_role": role,
        "measurements": measurements,
        "similarity_score": round(float(similarity_score), 4),
        "per_100g": {
            "calories_kcal": round(per_100g["calories_kcal"], 2),
            "protein_g": round(per_100g["protein_g"], 2),
            "fat_g": round(per_100g["fat_g"], 2),
            "carbs_g": round(per_100g["carbs_g"], 2),
        },
    }


def create_food_text_representation(food_item: Dict[str, Any]) -> str:
    normalized = normalize_food_document(food_item)
    per_100g = normalized["per_100g"]
    parts = [
        f"Food: {normalized.get('name', '')}",
        f"Type: {normalized.get('type', 'foundation')}",
        f"Meal role: {normalized.get('meal_role', 'flexible')}",
        (
            f"Per 100g: {per_100g.get('calories_kcal', 0)} kcal, "
            f"{per_100g.get('protein_g', 0)}g protein, "
            f"{per_100g.get('fat_g', 0)}g fat, "
            f"{per_100g.get('carbs_g', 0)}g carbs"
        ),
    ]
    if normalized.get("category"):
        parts.append(f"Category: {normalized['category']}")
    if normalized.get("measurements"):
        parts.append(f"Common measures: {', '.join(normalized['measurements'])}")
    if normalized.get("brand_name"):
        parts.append(f"Brand: {normalized['brand_name']}")
    return " | ".join(parts)


def _matches_dietary_restrictions(content_lower: str, restrictions: List[str]) -> bool:
    for restriction in restrictions:
        restriction_lower = restriction.lower()
        if restriction_lower == "vegan" and any(
            token in content_lower
            for token in ["milk", "cheese", "butter", "egg", "meat", "chicken", "beef", "pork", "fish"]
        ):
            return False
        if restriction_lower == "gluten-free" and any(
            token in content_lower for token in ["wheat", "barley", "rye", "gluten"]
        ):
            return False
    return True


def _matches_macro_goals(metadata: Dict[str, Any], macro_goals: Dict[str, float]) -> bool:
    protein = metadata.get("protein_per_100g", 0)
    carbs = metadata.get("carbs_per_100g", 0)
    calories = metadata.get("calories_per_100g", 0)
    for goal, value in macro_goals.items():
        if goal == "protein_min" and protein < value:
            return False
        if goal == "carbs_min" and carbs < value:
            return False
        if goal == "carbs_max" and carbs > value:
            return False
        if goal == "calories_max" and calories > value:
            return False
    return True


def _format_semantic_result(doc: Document, normalized_score: float) -> Dict[str, Any]:
    metadata = doc.metadata
    return {
        "id": metadata.get("id"),
        "fdc_id": metadata.get("fdc_id"),
        "name": metadata.get("name"),
        "type": metadata.get("type", "foundation"),
        "category": metadata.get("category"),
        "meal_role": metadata.get("meal_role", "flexible"),
        "measurements": metadata.get("measurements", []),
        "similarity_score": normalized_score,
        "per_100g": {
            "calories_kcal": metadata.get("calories_per_100g", 0),
            "protein_g": metadata.get("protein_per_100g", 0),
            "fat_g": metadata.get("fat_per_100g", 0),
            "carbs_g": metadata.get("carbs_per_100g", 0),
        },
        "matched_content": doc.page_content[:220] + ("..." if len(doc.page_content) > 220 else ""),
    }


def format_mongo_food_summary(food_item: Dict[str, Any]) -> Dict[str, Any]:
    return normalize_food_document(food_item, similarity_score=0.0)


def format_mongo_food_detail(food_item: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_food_document(food_item, similarity_score=0.0)
    normalized["source_document"] = food_item
    return normalized


def _safe_similarity_search(query_data: NutritionQuery) -> List[Dict[str, Any]]:
    try:
        return semantic_food_search(query_data).results
    except Exception:
        return []


def semantic_food_search(query_data: NutritionQuery) -> VectorSearchResponse:
    start_time = datetime.now()
    vector_store = load_vector_store(food_types=query_data.food_types)
    if vector_store is None:
        db_type = ",".join(query_data.food_types or ["foundation"])
        raise HTTPException(
            status_code=404,
            detail=f"FAISS vector index for {db_type} foods not found. Please create it first.",
        )

    docs_and_scores = vector_store.similarity_search_with_score(query_data.query, k=query_data.limit * 3)
    results: List[Dict[str, Any]] = []
    for doc, raw_score in docs_and_scores:
        normalized_score = max(0.0, 1.0 / (1.0 + float(raw_score)))
        if normalized_score < query_data.similarity_threshold:
            continue
        content_lower = doc.page_content.lower()
        if query_data.dietary_restrictions and not _matches_dietary_restrictions(
            content_lower, query_data.dietary_restrictions
        ):
            continue
        if query_data.macro_goals and not _matches_macro_goals(doc.metadata, query_data.macro_goals):
            continue
        results.append(_format_semantic_result(doc, normalized_score))
        if len(results) >= query_data.limit:
            break

    return VectorSearchResponse(
        query=query_data.query,
        results_found=len(results),
        results=results,
        search_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
    )


def _dedupe_candidates(candidates: Iterable[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen_ids = set()
    for candidate in candidates:
        candidate_id = candidate.get("id") or candidate.get("fdc_id") or candidate.get("name")
        if candidate_id in seen_ids:
            continue
        seen_ids.add(candidate_id)
        deduped.append(candidate)
        if len(deduped) >= limit:
            break
    return deduped


def build_meal_candidate_bundle(
    query: str,
    dietary_restrictions: List[str],
    macro_goals: Dict[str, float],
    food_types: Optional[List[str]] = None,
    limit_per_slot: int = 6,
) -> Dict[str, Any]:
    normalized_food_types = food_types or ["foundation"]
    semantic_hits = _safe_similarity_search(
        NutritionQuery(
            query=query,
            dietary_restrictions=dietary_restrictions,
            macro_goals=macro_goals,
            limit=limit_per_slot * 3,
            similarity_threshold=0.15,
            food_types=normalized_food_types,
        )
    )
    text_hits = [
        format_mongo_food_summary(item)
        for item in find_foods_by_text(
            query=query,
            limit=limit_per_slot * 3,
            food_types=normalized_food_types,
            meal_candidates_only=True,
        )
    ]

    slot_candidates: Dict[str, List[Dict[str, Any]]] = {}
    for slot_name in ["proteins", "carbs", "vegetables", "fats"]:
        if slot_name == "proteins":
            slot_macro_goals = {
                key: value for key, value in macro_goals.items() if key in {"protein_min", "calories_max"}
            }
        elif slot_name == "carbs":
            slot_macro_goals = {
                key: value for key, value in macro_goals.items() if key in {"carbs_min", "carbs_max", "calories_max"}
            }
        elif slot_name == "vegetables":
            slot_macro_goals = {
                key: value for key, value in macro_goals.items() if key in {"calories_max"}
            }
        else:
            slot_macro_goals = {}

        slot_from_query = [
            item
            for item in [*semantic_hits, *text_hits]
            if item.get("meal_role") == slot_name and _matches_macro_goals(
                {
                    "protein_per_100g": item.get("per_100g", {}).get("protein_g", 0),
                    "carbs_per_100g": item.get("per_100g", {}).get("carbs_g", 0),
                    "calories_per_100g": item.get("per_100g", {}).get("calories_kcal", 0),
                },
                slot_macro_goals,
            )
        ]
        slot_from_db = [
            format_mongo_food_summary(item)
            for item in find_foods_for_meal_slot(
                slot_name=slot_name,
                limit=limit_per_slot * 3,
                food_types=normalized_food_types,
                meal_candidates_only=True,
            )
        ]
        slot_candidates[slot_name] = _dedupe_candidates([*slot_from_query, *slot_from_db], limit_per_slot)

    flexible_candidates = _dedupe_candidates(
        [
            item
            for item in [*semantic_hits, *text_hits]
            if item.get("meal_role") not in {"proteins", "carbs", "vegetables", "fats"}
        ],
        limit_per_slot,
    )
    if not flexible_candidates:
        flexible_candidates = _dedupe_candidates(
            [
                format_mongo_food_summary(item)
                for item in find_foods_by_text(
                    "",
                    limit_per_slot * 2,
                    normalized_food_types,
                    meal_candidates_only=True,
                )
            ],
            limit_per_slot,
        )

    slot_candidates["flexible"] = flexible_candidates
    top_matches = _dedupe_candidates([*semantic_hits, *text_hits], limit_per_slot)
    total_candidates = sum(len(items) for items in slot_candidates.values())
    return {
        "query": query,
        "candidate_strategy": {
            "vector_backend": "faiss",
            "retrieval_modes": ["semantic", "text", "slot_pool"],
            "limit_per_slot": limit_per_slot,
            "macro_goals": macro_goals,
            "food_types": normalized_food_types,
        },
        "top_matches": top_matches,
        "slot_candidates": slot_candidates,
        "total_candidates": total_candidates,
    }


def hybrid_food_search(
    query: str,
    dietary_restrictions: str = "",
    protein_min: float = 0,
    carbs_min: float = 0,
    carbs_max: float = 999,
    calories_max: float = 999,
    limit: int = 10,
    semantic_weight: float = 0.7,
    food_types: Optional[List[str]] = None,
) -> HybridSearchResponse:
    normalized_food_types = food_types or ["foundation"]
    restrictions = [item.strip() for item in dietary_restrictions.split(",") if item.strip()]
    semantic_results = _safe_similarity_search(
        NutritionQuery(
            query=query,
            dietary_restrictions=restrictions,
            macro_goals={
                "protein_min": protein_min,
                "carbs_min": carbs_min,
                "carbs_max": carbs_max,
                "calories_max": calories_max,
            },
            limit=limit * 2,
            similarity_threshold=0.2,
            food_types=normalized_food_types,
        )
    )
    mongo_results = [
        format_mongo_food_summary(item)
        for item in find_foods_with_macro_filters(
            query=query,
            limit=limit * 2,
            protein_min=protein_min,
            carbs_min=carbs_min,
            carbs_max=carbs_max,
            calories_max=calories_max,
            food_types=normalized_food_types,
            meal_candidates_only=True,
        )
    ]

    hybrid_results: Dict[Any, Dict[str, Any]] = {}
    for result in semantic_results:
        hybrid_results[result["id"]] = {
            **result,
            "hybrid_score": result["similarity_score"] * semantic_weight,
            "semantic_score": result["similarity_score"],
            "traditional_score": 0.0,
        }

    traditional_weight = 1 - semantic_weight
    query_lower = query.lower()
    for food_item in mongo_results:
        name = str(food_item.get("name", "")).lower()
        score = 0.0
        if query_lower and query_lower in name:
            score += 0.8
        for word in query_lower.split():
            if word in name:
                score += 0.1
        traditional_score = min(score or 0.2, 1.0)

        if food_item["id"] in hybrid_results:
            hybrid_results[food_item["id"]]["hybrid_score"] += traditional_score * traditional_weight
            hybrid_results[food_item["id"]]["traditional_score"] = traditional_score
        else:
            hybrid_results[food_item["id"]] = {
                **food_item,
                "hybrid_score": traditional_score * traditional_weight,
                "semantic_score": 0.0,
                "traditional_score": traditional_score,
            }

    sorted_results = sorted(hybrid_results.values(), key=lambda item: item["hybrid_score"], reverse=True)[:limit]
    return HybridSearchResponse(
        query=query,
        semantic_weight=semantic_weight,
        traditional_weight=traditional_weight,
        results_found=len(sorted_results),
        results=sorted_results,
    )


def mongo_food_search(
    query: str = "chicken",
    limit: int = 10,
    food_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    normalized_food_types = food_types or ["foundation"]
    results = find_foods_by_text(
        query=query,
        limit=limit,
        food_types=normalized_food_types,
    )
    return {
        "query": query,
        "collection_name": get_food_collection_name(normalized_food_types),
        "results_found": len(results),
        "results": [format_mongo_food_detail(item) for item in results],
    }


def create_vector_index(
    food_types: Optional[List[str]] = None,
    batch_size: int = 1000,
    max_documents: Optional[int] = None,
    recreate: bool = False,
) -> Dict[str, Any]:
    normalized_food_types = food_types or ["foundation"]
    start_time = datetime.now()
    collection_name = get_food_collection_name(normalized_food_types)
    index_path = get_index_directory(normalized_food_types)

    if recreate:
        drop_vector_store(normalized_food_types)

    if index_exists(index_path) and not recreate:
        vector_store = load_vector_store(food_types=normalized_food_types)
        return {
            "status": "success",
            "message": f"Loaded existing FAISS index for {collection_name}",
            "collection_name": collection_name,
            "index_size": vector_store.index.ntotal if vector_store else 0,
            "index_path": index_path,
        }

    total_docs = count_food_documents(normalized_food_types)
    if total_docs == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' not found or empty. Please import data first.",
        )

    embeddings = get_embeddings()
    vector_store = None
    batch: List[Document] = []
    processed_count = 0
    for food_item in iterate_food_documents(normalized_food_types, max_documents=max_documents):
        try:
            normalized = normalize_food_document(food_item)
            per_100g = normalized["per_100g"]
            batch.append(
                Document(
                    page_content=create_food_text_representation(food_item),
                    metadata={
                        "id": normalized.get("id"),
                        "fdc_id": normalized.get("fdc_id"),
                        "name": normalized.get("name", ""),
                        "type": normalized.get("type", "foundation"),
                        "category": normalized.get("category"),
                        "meal_role": normalized.get("meal_role", "flexible"),
                        "measurements": normalized.get("measurements", []),
                        "calories_per_100g": per_100g.get("calories_kcal", 0),
                        "protein_per_100g": per_100g.get("protein_g", 0),
                        "fat_per_100g": per_100g.get("fat_g", 0),
                        "carbs_per_100g": per_100g.get("carbs_g", 0),
                    },
                )
            )
            processed_count += 1
            if len(batch) >= batch_size:
                partial_store = FAISS.from_documents(batch, embeddings)
                if vector_store is None:
                    vector_store = partial_store
                else:
                    vector_store.merge_from(partial_store)
                batch = []
        except Exception:
            continue

    if batch:
        partial_store = FAISS.from_documents(batch, embeddings)
        if vector_store is None:
            vector_store = partial_store
        else:
            vector_store.merge_from(partial_store)

    if vector_store is None:
        raise HTTPException(status_code=500, detail="No documents were indexed.")

    save_vector_store(vector_store, index_path)
    cache_vector_store(food_types=normalized_food_types, vector_store=vector_store)
    return {
        "status": "success",
        "message": f"FAISS vector index created successfully for {collection_name}",
        "collection_name": collection_name,
        "total_documents_processed": processed_count,
        "index_size": vector_store.index.ntotal,
        "processing_time_seconds": round((datetime.now() - start_time).total_seconds(), 2),
        "index_path": index_path,
        "batch_size_used": batch_size,
    }


def get_vector_index_status() -> VectorIndexStatusResponse:
    system_info = {
        "memory_usage_percent": psutil.virtual_memory().percent,
        "memory_available_gb": psutil.virtual_memory().available / (1024**3),
        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "timestamp": datetime.now().isoformat(),
    }

    response: Dict[str, Any] = {"system_info": system_info}
    for name, food_types in {
        "full_database": ["foundation", "branded"],
        "sample_database": ["foundation"],
    }.items():
        path = get_index_directory(food_types)
        file_info = check_file_info(path)
        status = IndexStatus(
            exists=file_info["exists"],
            loaded=False,
            loading=False,
            file_size_mb=file_info.get("file_size_mb"),
            last_modified=file_info.get("last_modified"),
            error=file_info.get("error"),
        )
        if status.exists:
            vector_store = load_vector_store(food_types=food_types)
            if vector_store is not None:
                status.loaded = True
                status.index_size = vector_store.index.ntotal
                status.embedding_dimension = vector_store.index.d
                if status.index_size and status.embedding_dimension:
                    status.memory_usage_mb = (status.index_size * status.embedding_dimension * 4) / (1024**2)
        response[name] = status

    legacy_info = check_file_info("./nutrition_faiss_index")
    response["legacy_index"] = IndexStatus(
        exists=legacy_info.get("exists", False),
        loaded=False,
        loading=False,
        file_size_mb=legacy_info.get("file_size_mb"),
        last_modified=legacy_info.get("last_modified"),
        error=legacy_info.get("error"),
    )
    return VectorIndexStatusResponse(**response)
