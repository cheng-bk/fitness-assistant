import os
from typing import Any, Dict, Optional

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient


_mongo_client: Optional[MongoClient] = None
_embeddings_model: Optional[OpenAIEmbeddings] = None
_vector_stores: Dict[str, Any] = {}


def get_mongo_client() -> MongoClient:
    global _mongo_client
    if _mongo_client is not None:
        return _mongo_client

    mongo_user = os.getenv("MONGO_USER")
    mongo_password = os.getenv("MONGO_PASSWORD")
    mongo_host = os.getenv("MONGO_HOST", "localhost")
    mongo_port = os.getenv("MONGO_PORT", "27017")
    mongo_auth_source = os.getenv("MONGO_AUTH_SOURCE", "admin")

    uri = f"mongodb://{mongo_user}:{mongo_password}@{mongo_host}:{mongo_port}/?authSource={mongo_auth_source}"

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        _mongo_client = client
        return client
    except Exception as exc:
        raise RuntimeError(f"Failed to connect to MongoDB: {exc}") from exc


def get_database():
    client = get_mongo_client()
    return client[os.getenv("MONGO_DB_NAME", "fitness_assistant")]


def get_embeddings_model() -> OpenAIEmbeddings:
    global _embeddings_model
    if _embeddings_model is None:
        _embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    return _embeddings_model


def _vector_store_key(food_types: Optional[list[str]] = None) -> str:
    normalized = sorted({item.strip() for item in (food_types or ["foundation"]) if item and item.strip()})
    return "|".join(normalized) if normalized else "foundation"


def get_vector_store(food_types: Optional[list[str]] = None):
    key = _vector_store_key(food_types)
    if key in _vector_stores:
        return _vector_stores[key]

    index_path = get_index_path(food_types=food_types)
    if not os.path.exists(index_path) and key == "foundation":
        legacy_path = "./nutrition_faiss_index"
        if os.path.exists(legacy_path):
            index_path = legacy_path

    if not os.path.exists(index_path):
        return None

    vector_store = FAISS.load_local(
        index_path,
        get_embeddings_model(),
        allow_dangerous_deserialization=True,
    )
    _vector_stores[key] = vector_store
    return vector_store


def set_vector_store(food_types: Optional[list[str]], vector_store: Any) -> None:
    key = _vector_store_key(food_types)
    if vector_store is None and key in _vector_stores:
        del _vector_stores[key]
        return
    _vector_stores[key] = vector_store


def get_index_path(food_types: Optional[list[str]] = None) -> str:
    key = _vector_store_key(food_types).replace("|", "_")
    return f"./nutrition_faiss_index_{key}"
