import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import faiss
from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from marshmallow import pprint

from agent.infrastructure.database import get_mongo_client


OUTPUT_DIR = Path("data/processed")
FAISS_DIR = OUTPUT_DIR / "faiss_store" / "names"
INDEX_CONFIG = {
    "food": {
        "collection": "foods",
        "persist_dir": FAISS_DIR / "food",
        "index_path": FAISS_DIR / "food" / "faiss.index",
        "query_test": "chicken breast",
    },
    "exercise": {
        "collection": "exercises",
        "persist_dir": FAISS_DIR / "exercise",
        "index_path": FAISS_DIR / "exercise" / "faiss.index",
        "query_test": "bench press",
    },
}

for config in INDEX_CONFIG.values():
    config["persist_dir"].mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DEVICE = "cuda"
EMBED_DIM = 1024


embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL_NAME,
    device=EMBEDDING_DEVICE,
)

Settings.embed_model = embed_model


def get_collection(database_name: str, collection_name: str):
    return get_mongo_client()[database_name][collection_name]


def iter_named_documents(collection_name: str, database_name: str) -> Iterable[Dict[str, Any]]:
    collection = get_collection(database_name, collection_name)
    cursor = collection.find({})
    return cursor


def build_food_name_metadata(document: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "entity_type": "food",
        "collection": "foods",
        "document_id": str(document.get("_id")),
        "name": str(document.get("name", "")),
    }


def build_exercise_name_metadata(document: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "entity_type": "exercise",
        "collection": "exercises",
        "document_id": str(document.get("_id")),
        "name": str(document.get("name", "")),
    }


def build_name_nodes(entity_type: str, database_name: str) -> List[TextNode]:
    config = INDEX_CONFIG[entity_type]
    documents = iter_named_documents(
        collection_name=config["collection"],
        database_name=database_name,
    )

    nodes: List[TextNode] = []
    for idx, document in enumerate(documents):
        name = str(document.get("name", "")).strip()
        if not name:
            continue

        if entity_type == "food":
            text = name
            metadata = build_food_name_metadata(document)
        else:
            text = name
            metadata = build_exercise_name_metadata(document)

        metadata["chunk_id"] = idx
        metadata["chunk_len"] = len(text)
        nodes.append(TextNode(text=text, metadata=metadata))

    return nodes


def build_storage_context() -> tuple[faiss.IndexFlatL2, StorageContext]:
    faiss_index = faiss.IndexFlatL2(EMBED_DIM)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return faiss_index, storage_context


def build_and_persist_single_index(entity_type: str, nodes: List[TextNode]) -> None:
    config = INDEX_CONFIG[entity_type]
    faiss_index, storage_context = build_storage_context()

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True,
    )

    index.storage_context.persist(persist_dir=str(config["persist_dir"]))
    faiss.write_index(faiss_index, str(config["index_path"]))

    print(f"[INFO] Saved {entity_type} llama-index storage to: {config['persist_dir']}")
    print(f"[INFO] Saved {entity_type} faiss index to: {config['index_path']}")


def build_and_persist_indexes(database_name: str) -> None:
    for entity_type in ["food", "exercise"]:
        nodes = build_name_nodes(entity_type=entity_type, database_name=database_name)
        print(f"[INFO] Indexed {entity_type} name nodes: {len(nodes)}")
        if not nodes:
            print(f"[WARN] No {entity_type} name nodes found. Skipping index build.")
            continue
        build_and_persist_single_index(entity_type=entity_type, nodes=nodes)

    print(f"[INFO] Saved name indexes under: {FAISS_DIR}")


def load_index_from_disk(entity_type: str) -> VectorStoreIndex:
    config = INDEX_CONFIG[entity_type]
    loaded_faiss_index = faiss.read_index(str(config["index_path"]))
    loaded_vector_store = FaissVectorStore(faiss_index=loaded_faiss_index)

    loaded_storage_context = StorageContext.from_defaults(
        persist_dir=str(config["persist_dir"]),
        vector_store=loaded_vector_store,
    )

    loaded_index = load_index_from_storage(loaded_storage_context)
    return loaded_index


def test_similarity_from_disk(entity_type: str, query: Optional[str] = None, top_k: int = 3) -> None:
    config = INDEX_CONFIG[entity_type]
    loaded_index = load_index_from_disk(entity_type=entity_type)
    retriever = loaded_index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query or config["query_test"])

    print(f"\n[TEST RETRIEVE RESULT - LOADED FROM DISK - {entity_type.upper()}]")
    for i, node in enumerate(nodes, 1):
        print(f"\n{'-' * 50} Result {i} {'-' * 50}")
        print(node.text)
        pprint(node.metadata)


def main() -> None:
    force_rebuild = True

    load_dotenv()

    database_name = os.getenv("MONGO_DB_NAME", "fitness_assistant")

    food_index_path = INDEX_CONFIG["food"]["index_path"]
    exercise_index_path = INDEX_CONFIG["exercise"]["index_path"]

    if food_index_path.exists() and exercise_index_path.exists() and not force_rebuild:
        print(f"[INFO] Reusing existing name indexes: {FAISS_DIR}")
    else:
        print("[INFO] Building / rebuilding name indexes from MongoDB collections.")
        build_and_persist_indexes(database_name=database_name)

    test_similarity_from_disk(entity_type="food", top_k=3)
    test_similarity_from_disk(entity_type="exercise", top_k=3)


if __name__ == "__main__":
    main()
