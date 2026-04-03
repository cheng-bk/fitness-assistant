import os
import shutil
from datetime import datetime
from typing import Any, Dict, Optional

from ..infrastructure.database import (
    get_embeddings_model,
    get_index_path,
    get_vector_store,
    set_vector_store,
)


def get_index_directory(food_types: Optional[list[str]] = None) -> str:
    return get_index_path(food_types=food_types)


def index_exists(path: str) -> bool:
    return os.path.exists(path)


def load_vector_store(food_types: Optional[list[str]] = None):
    return get_vector_store(food_types=food_types)


def get_embeddings():
    return get_embeddings_model()


def cache_vector_store(food_types: Optional[list[str]], vector_store: Any) -> None:
    set_vector_store(food_types=food_types, vector_store=vector_store)


def save_vector_store(vector_store: Any, path: str) -> None:
    os.makedirs(path, exist_ok=True)
    vector_store.save_local(path)


def drop_vector_store(food_types: Optional[list[str]] = None) -> None:
    path = get_index_directory(food_types)
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)
    set_vector_store(food_types, None)


def check_file_info(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"exists": False}
    try:
        total_size = 0
        latest_modified = 0.0
        if os.path.isdir(path):
            for file_name in os.listdir(path):
                file_path = os.path.join(path, file_name)
                if os.path.isfile(file_path):
                    stat = os.stat(file_path)
                    total_size += stat.st_size
                    latest_modified = max(latest_modified, stat.st_mtime)
        else:
            stat = os.stat(path)
            total_size = stat.st_size
            latest_modified = stat.st_mtime
        return {
            "exists": True,
            "file_size_mb": total_size / (1024 * 1024),
            "last_modified": datetime.fromtimestamp(latest_modified).isoformat() if latest_modified else None,
        }
    except Exception as exc:
        return {"exists": True, "error": f"Failed to get file info: {exc}"}
