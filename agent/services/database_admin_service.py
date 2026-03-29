import json
import os
from typing import Any, Dict

from fastapi import HTTPException
from pymongo import MongoClient, errors

from ..infrastructure.database import get_database, get_mongo_client
from ..models import DatabaseAvailabilityResponse
from ..repositories.food_repository import (
    count_enhanced_sample_foods,
    count_food_documents,
    create_sample_food_indexes,
    get_collection_stats,
    insert_sample_food_batch,
    list_collection_names,
)


def test_mongo_connection() -> Dict[str, Any]:
    mongo_user = os.getenv("MONGO_USER")
    mongo_password = os.getenv("MONGO_PASSWORD")
    mongo_db_name = os.getenv("MONGO_DB_NAME")
    attempts = [("mongodb_ai_fitness_planner", 27017), ("localhost", 27019)]

    for host, port in attempts:
        try:
            client = get_mongo_client() if host == "mongodb_ai_fitness_planner" else None
            if client is None:
                mongo_uri = f"mongodb://{mongo_user}:{mongo_password}@{host}:{port}/admin"
                client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
                client.admin.command("ping")
            db = client[mongo_db_name] if mongo_db_name else client["usda_nutrition"]
            collection_names = db.list_collection_names()
            collections_sample = []
            for collection_name in collection_names[:5]:
                try:
                    collections_sample.append(
                        {
                            "name": collection_name,
                            "document_count": db[collection_name].count_documents({}),
                        }
                    )
                except Exception as exc:
                    collections_sample.append({"name": collection_name, "error": str(exc)})
            return {
                "status": "success",
                "connected_host": host,
                "port": port,
                "database_name": mongo_db_name or "usda_nutrition",
                "total_collections": len(collection_names),
                "collections_sample": collections_sample,
                "environment_vars": {
                    "MONGO_USER": mongo_user,
                    "MONGO_PASSWORD": "***" if mongo_password else None,
                    "MONGO_DB_NAME": mongo_db_name,
                },
            }
        except (errors.ServerSelectionTimeoutError, errors.OperationFailure, RuntimeError):
            continue
        except Exception:
            continue
    raise HTTPException(status_code=500, detail="Failed to connect to MongoDB.")


def get_database_stats() -> Dict[str, Any]:
    collections = list_collection_names()
    collection_details: Dict[str, Any] = {}
    for collection_name in collections:
        try:
            collection_details[collection_name] = get_collection_stats(collection_name)
        except Exception as exc:
            collection_details[collection_name] = {"error": str(exc)}

    db_stats = get_database().command("dbstats")
    return {
        "database_name": os.getenv("MONGO_DB_NAME", "usda_nutrition"),
        "total_collections": len(collections),
        "collection_details": collection_details,
        "database_size_mb": round(db_stats.get("dataSize", 0) / (1024 * 1024), 2),
        "storage_size_mb": round(db_stats.get("storageSize", 0) / (1024 * 1024), 2),
        "indexes": db_stats.get("indexes", 0),
    }


def import_sampled_data(
    sample_file: str = "./fast_api/app/api/nutrition_data/samples/usda_sampled_5000_foods.json",
) -> Dict[str, Any]:
    if not os.path.exists(sample_file):
        raise HTTPException(status_code=404, detail=f"Sample file not found: {sample_file}")

    try:
        create_sample_food_indexes()
    except Exception:
        pass

    with open(sample_file, "r", encoding="utf-8") as file:
        sample_data = json.load(file)

    foods = sample_data.get("BrandedFoods", [])
    metadata = sample_data.get("metadata", {})
    total_processed = 0
    batch_size = 1000
    for start in range(0, len(foods), batch_size):
        batch = foods[start : start + batch_size]
        try:
            total_processed += insert_sample_food_batch(batch)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Import failed: {exc}") from exc

    final_count = count_food_documents(False)
    enhanced_final_count = count_enhanced_sample_foods()
    return {
        "status": "success",
        "message": "Sampled USDA data imported successfully",
        "sample_metadata": metadata,
        "total_documents_imported": total_processed,
        "enhanced_documents": enhanced_final_count,
        "final_document_count": final_count,
        "source_file": sample_file,
    }


def check_database_availability() -> DatabaseAvailabilityResponse:
    full_count = count_food_documents(True)
    sample_count = count_food_documents(False)
    return DatabaseAvailabilityResponse(
        full_database={
            "available": full_count > 0,
            "document_count": full_count,
            "collection_name": "branded_foods",
        },
        sample_database={
            "available": sample_count > 0,
            "document_count": sample_count,
            "collection_name": "branded_foods_sample",
        },
        recommendation="full" if full_count > 0 else "sample" if sample_count > 0 else "none",
    )
