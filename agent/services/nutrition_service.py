from datetime import datetime
from typing import Any, Dict, List, Optional

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
    find_foods_with_macro_filters,
    get_food_collection_name,
    iterate_food_documents,
)
from ..repositories.vector_index_repository import (
    cache_vector_store,
    check_file_info,
    get_embeddings,
    get_index_directory,
    index_exists,
    load_vector_store,
    save_vector_store,
)


def create_food_text_representation(food_item: Dict[str, Any]) -> str:
    parts: List[str] = []
    if food_item.get("description"):
        parts.append(f"Food: {food_item['description']}")
    if food_item.get("brandOwner"):
        parts.append(f"Brand: {food_item['brandOwner']}")
    if food_item.get("brandName"):
        parts.append(f"Product: {food_item['brandName']}")
    if food_item.get("foodCategory"):
        parts.append(f"Category: {food_item['foodCategory']}")
    if food_item.get("ingredients"):
        parts.append(f"Ingredients: {food_item['ingredients'][:500]}")

    nutrition = food_item.get("nutrition_enhanced", {})
    per_100g = nutrition.get("per_100g", {})
    if per_100g:
        parts.append(
            f"Per 100g: {per_100g.get('energy_kcal', 0)} calories, "
            f"{per_100g.get('protein_g', 0)}g protein, "
            f"{per_100g.get('total_fat_g', 0)}g fat, "
            f"{per_100g.get('carbs_g', 0)}g carbs"
        )

    primary_macro = nutrition.get("macro_breakdown", {}).get("primary_macro_category")
    if primary_macro and primary_macro != "unknown":
        parts.append(f"Primary macro: {primary_macro}")

    if food_item.get("servingSize") and food_item.get("servingSizeUnit"):
        parts.append(f"Serving: {food_item['servingSize']}{food_item['servingSizeUnit']}")
    if food_item.get("householdServingFullText"):
        parts.append(f"Serving description: {food_item['householdServingFullText']}")
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
    fat = metadata.get("fat_per_100g", 0)
    carbs = metadata.get("carbs_per_100g", 0)
    calories = metadata.get("calories_per_100g", 0)
    for goal, value in macro_goals.items():
        if goal == "protein_min" and protein < value:
            return False
        if goal == "protein_max" and protein > value:
            return False
        if goal == "fat_min" and fat < value:
            return False
        if goal == "fat_max" and fat > value:
            return False
        if goal == "carbs_min" and carbs < value:
            return False
        if goal == "carbs_max" and carbs > value:
            return False
        if goal == "calories_min" and calories < value:
            return False
        if goal == "calories_max" and calories > value:
            return False
    return True


def _format_semantic_result(doc: Document, normalized_score: float) -> Dict[str, Any]:
    metadata = doc.metadata
    return {
        "fdc_id": metadata.get("fdc_id"),
        "description": metadata.get("description"),
        "brand_owner": metadata.get("brand_owner"),
        "brand_name": metadata.get("brand_name"),
        "food_category": metadata.get("food_category"),
        "similarity_score": normalized_score,
        "nutrition_per_100g": {
            "calories": metadata.get("calories_per_100g", 0),
            "protein_g": metadata.get("protein_per_100g", 0),
            "fat_g": metadata.get("fat_per_100g", 0),
            "carbs_g": metadata.get("carbs_per_100g", 0),
        },
        "primary_macro_category": metadata.get("primary_macro", "unknown"),
        "is_high_protein": metadata.get("is_high_protein", False),
        "nutrition_density_score": metadata.get("nutrition_density_score", 0),
        "serving_size": metadata.get("serving_size", 0),
        "matched_content": doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""),
    }


def format_mongo_food_summary(food_item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "fdc_id": food_item.get("fdcId"),
        "description": food_item.get("description"),
        "brand_owner": food_item.get("brandOwner"),
        "brand_name": food_item.get("brandName"),
        "food_category": food_item.get("foodCategory"),
        "similarity_score": 0.0,
        "nutrition_per_100g": food_item.get("nutrition_enhanced", {}).get("per_100g", {}),
    }


def format_mongo_food_detail(food_item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "fdc_id": food_item.get("fdcId"),
        "description": food_item.get("description"),
        "brand_owner": food_item.get("brandOwner"),
        "brand_name": food_item.get("brandName"),
        "food_class": food_item.get("foodClass"),
        "food_category": food_item.get("foodCategory"),
        "gtin_upc": food_item.get("gtinUpc"),
        "ingredients": food_item.get("ingredients"),
        "serving_size": food_item.get("servingSize"),
        "serving_size_unit": food_item.get("servingSizeUnit"),
        "household_serving_fulltext": food_item.get("householdServingFullText"),
        "nutrition_enhanced": food_item.get("nutrition_enhanced", {}),
    }


def semantic_food_search(query_data: NutritionQuery) -> VectorSearchResponse:
    start_time = datetime.now()
    vector_store = load_vector_store(use_full_database=query_data.use_full_database)
    if vector_store is None:
        db_type = "full" if query_data.use_full_database else "sample"
        raise HTTPException(
            status_code=404,
            detail=f"Vector index for {db_type} database not found. Please create it first.",
        )

    docs_and_scores = vector_store.similarity_search_with_score(query_data.query, k=query_data.limit * 2)
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


def hybrid_food_search(
    query: str,
    dietary_restrictions: str = "",
    protein_min: float = 0,
    carbs_max: float = 999,
    calories_max: float = 999,
    limit: int = 10,
    semantic_weight: float = 0.7,
    use_full_database: bool = False,
) -> HybridSearchResponse:
    restrictions = [item.strip() for item in dietary_restrictions.split(",") if item.strip()]
    semantic_results = semantic_food_search(
        NutritionQuery(
            query=query,
            dietary_restrictions=restrictions,
            macro_goals={
                "protein_min": protein_min,
                "carbs_max": carbs_max,
                "calories_max": calories_max,
            },
            limit=limit * 2,
            similarity_threshold=0.2,
            use_full_database=use_full_database,
        )
    )
    mongo_results = find_foods_with_macro_filters(
        query=query,
        limit=limit * 2,
        use_full_database=use_full_database,
        protein_min=protein_min,
        carbs_max=carbs_max,
        calories_max=calories_max,
    )

    hybrid_results: Dict[Any, Dict[str, Any]] = {}
    for result in semantic_results.results:
        fdc_id = result["fdc_id"]
        hybrid_results[fdc_id] = {
            **result,
            "hybrid_score": result["similarity_score"] * semantic_weight,
            "semantic_score": result["similarity_score"],
            "traditional_score": 0,
        }

    traditional_weight = 1 - semantic_weight
    for food_item in mongo_results:
        fdc_id = food_item.get("fdcId")
        description = food_item.get("description", "").lower()
        brand = food_item.get("brandOwner", "").lower()
        ingredients = food_item.get("ingredients", "").lower()
        query_lower = query.lower()

        score = 0.0
        if query_lower in description:
            score += 0.4
        if query_lower in brand:
            score += 0.3
        if query_lower in ingredients:
            score += 0.3
        for word in query_lower.split():
            if word in description:
                score += 0.1
            if word in brand:
                score += 0.05
        traditional_score = min(score, 1.0)

        if fdc_id in hybrid_results:
            hybrid_results[fdc_id]["hybrid_score"] += traditional_score * traditional_weight
            hybrid_results[fdc_id]["traditional_score"] = traditional_score
        else:
            nutrition = food_item.get("nutrition_enhanced", {})
            per_100g = nutrition.get("per_100g", {})
            hybrid_results[fdc_id] = {
                "fdc_id": fdc_id,
                "description": food_item.get("description"),
                "brand_owner": food_item.get("brandOwner"),
                "brand_name": food_item.get("brandName"),
                "food_category": food_item.get("foodCategory"),
                "nutrition_per_100g": {
                    "calories": per_100g.get("energy_kcal", 0),
                    "protein_g": per_100g.get("protein_g", 0),
                    "fat_g": per_100g.get("total_fat_g", 0),
                    "carbs_g": per_100g.get("carbs_g", 0),
                },
                "primary_macro_category": nutrition.get("macro_breakdown", {}).get(
                    "primary_macro_category", "unknown"
                ),
                "is_high_protein": nutrition.get("macro_breakdown", {}).get("is_high_protein", False),
                "nutrition_density_score": nutrition.get("nutrition_density_score", 0),
                "serving_size": food_item.get("servingSize", 0),
                "hybrid_score": traditional_score * traditional_weight,
                "semantic_score": 0,
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
    query: str = "coca cola",
    limit: int = 10,
    use_full_database: bool = True,
) -> Dict[str, Any]:
    results = find_foods_by_text(
        query=query,
        limit=limit,
        use_full_database=use_full_database,
        include_ingredients=False,
    )
    return {
        "query": query,
        "collection_name": get_food_collection_name(use_full_database),
        "results_found": len(results),
        "results": [format_mongo_food_detail(item) for item in results],
    }


def create_vector_index(
    use_full_database: bool = False,
    batch_size: int = 1000,
    max_documents: Optional[int] = None,
    recreate: bool = False,
) -> Dict[str, Any]:
    start_time = datetime.now()
    collection_name = get_food_collection_name(use_full_database)
    index_path = get_index_directory(use_full_database)

    if index_exists(index_path) and not recreate:
        vector_store = load_vector_store(use_full_database=use_full_database)
        return {
            "status": "success",
            "message": f"Loaded existing FAISS index for {collection_name}",
            "collection_name": collection_name,
            "index_size": vector_store.index.ntotal if vector_store else 0,
            "index_path": index_path,
        }

    total_docs = count_food_documents(use_full_database)
    if total_docs == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' not found or empty. Please import data first.",
        )

    embeddings = get_embeddings()
    vector_store = None
    batch: List[Document] = []
    processed_count = 0
    for food_item in iterate_food_documents(use_full_database, max_documents=max_documents):
        try:
            batch.append(
                Document(
                    page_content=create_food_text_representation(food_item),
                    metadata={
                        "fdc_id": food_item.get("fdcId"),
                        "description": food_item.get("description", ""),
                        "brand_owner": food_item.get("brandOwner", ""),
                        "brand_name": food_item.get("brandName", ""),
                        "food_category": food_item.get("foodCategory", ""),
                        "gtin_upc": food_item.get("gtinUpc", ""),
                        "serving_size": food_item.get("servingSize", 0),
                        "calories_per_100g": food_item.get("nutrition_enhanced", {}).get("per_100g", {}).get("energy_kcal", 0),
                        "protein_per_100g": food_item.get("nutrition_enhanced", {}).get("per_100g", {}).get("protein_g", 0),
                        "fat_per_100g": food_item.get("nutrition_enhanced", {}).get("per_100g", {}).get("total_fat_g", 0),
                        "carbs_per_100g": food_item.get("nutrition_enhanced", {}).get("per_100g", {}).get("carbs_g", 0),
                        "primary_macro": food_item.get("nutrition_enhanced", {}).get("macro_breakdown", {}).get("primary_macro_category", "unknown"),
                        "is_high_protein": food_item.get("nutrition_enhanced", {}).get("macro_breakdown", {}).get("is_high_protein", False),
                        "nutrition_density_score": food_item.get("nutrition_enhanced", {}).get("nutrition_density_score", 0),
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
    cache_vector_store(use_full_database=use_full_database, vector_store=vector_store)
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
    for name, path in {
        "full_database": get_index_directory(True),
        "sample_database": get_index_directory(False),
        "legacy_index": "./nutrition_faiss_index",
    }.items():
        file_info = check_file_info(path)
        status = IndexStatus(
            exists=file_info["exists"],
            loaded=False,
            loading=False,
            file_size_mb=file_info.get("file_size_mb"),
            last_modified=file_info.get("last_modified"),
            error=file_info.get("error"),
        )
        if name in {"full_database", "sample_database"} and status.exists:
            vector_store = load_vector_store(use_full_database=(name == "full_database"))
            if vector_store is not None:
                status.loaded = True
                status.index_size = vector_store.index.ntotal
                status.embedding_dimension = vector_store.index.d
                if status.index_size and status.embedding_dimension:
                    status.memory_usage_mb = (status.index_size * status.embedding_dimension * 4) / (1024**2)
        response[name] = status

    return VectorIndexStatusResponse(**response)
