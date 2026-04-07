import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import faiss
import torch
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from llama_index.core import Settings, StorageContext, load_index_from_storage

_embedding_model: Optional[HuggingFaceEmbedding] = None
_reranker_tokenizer: Optional[Any] = None
_reranker_model: Optional[Any] = None
_llama_index_cache: Dict[str, Any] = {}


def get_model_cache_dir() -> str:
    cache_dir = os.getenv("HF_MODEL_CACHE_DIR")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_embedding_model() -> HuggingFaceEmbedding:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbedding(
            model_name=os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3"),
            device="cuda" if torch.cuda.is_available() else "cpu",
            cache_folder=get_model_cache_dir(),
        )
        Settings.embed_model = _embedding_model
    return _embedding_model


def get_reranker_components() -> Tuple[Any, Any]:
    global _reranker_tokenizer, _reranker_model
    if _reranker_tokenizer is not None and _reranker_model is not None:
        return _reranker_tokenizer, _reranker_model

    model_name = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = get_model_cache_dir()

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
    model.eval()
    model.to(device)

    _reranker_tokenizer = tokenizer
    _reranker_model = model

    return tokenizer, model


def load_llama_index(persist_dir: str | Path, cache_key: str):
    if cache_key in _llama_index_cache:
        return _llama_index_cache[cache_key]

    persist_path = Path(persist_dir)
    index_path = persist_path / "faiss.index"
    if not persist_path.exists() or not index_path.exists():
        return None

    get_embedding_model()
    loaded_faiss_index = faiss.read_index(str(index_path))
    loaded_vector_store = FaissVectorStore(faiss_index=loaded_faiss_index)
    loaded_storage_context = StorageContext.from_defaults(
        persist_dir=str(persist_path),
        vector_store=loaded_vector_store,
    )
    loaded_index = load_index_from_storage(loaded_storage_context)
    _llama_index_cache[cache_key] = loaded_index
    return loaded_index


def clear_llama_index_cache(cache_key: Optional[str] = None) -> None:
    if cache_key is None:
        _llama_index_cache.clear()
        return
    _llama_index_cache.pop(cache_key, None)
