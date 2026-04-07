import os

from pymongo import MongoClient
from typing import Optional



_mongo_client: Optional[MongoClient] = None



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
    return get_mongo_client()[os.getenv("MONGO_DB_NAME", "fitness_assistant")]


