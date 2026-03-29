import os
from typing import Any, Dict, List, Optional

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
    uri = f"mongodb://{mongo_user}:{mongo_password}@localhost:27019/?authSource=admin"

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


def get_vector_store(use_full_database: bool = False):
    key = "full" if use_full_database else "sample"
    if key in _vector_stores:
        return _vector_stores[key]

    index_path = "./nutrition_faiss_index_full" if use_full_database else "./nutrition_faiss_index_sample"
    if not os.path.exists(index_path) and not use_full_database:
        index_path = "./nutrition_faiss_index"

    if not os.path.exists(index_path):
        return None

    vector_store = FAISS.load_local(
        index_path,
        get_embeddings_model(),
        allow_dangerous_deserialization=True,
    )
    _vector_stores[key] = vector_store
    return vector_store


def set_vector_store(use_full_database: bool, vector_store: Any) -> None:
    key = "full" if use_full_database else "sample"
    _vector_stores[key] = vector_store


def get_index_path(use_full_database: bool = False) -> str:
    return "./nutrition_faiss_index_full" if use_full_database else "./nutrition_faiss_index_sample"
