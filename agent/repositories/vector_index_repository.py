import math
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..infrastructure.faiss_llama_index import load_llama_index, get_reranker_components


NAME_INDEX_DIR = Path("data/processed/faiss_store/names")
KNOWLEDGE_INDEX_DIR = Path("data/processed/faiss_store/knowledge")


def _retrieve_index_nodes(
    persist_dir: Path,
    cache_key: str,
    query: str,
    top_k: int,
) -> List[Any]:
    if not query.strip():
        return []

    index = load_llama_index(persist_dir=persist_dir, cache_key=cache_key)
    if index is None:
        return []

    retriever = index.as_retriever(similarity_top_k=top_k)
    return list(retriever.retrieve(query))


def _score_from_node(node: Any) -> float:
    score = getattr(node, "score", None)
    if score is None:
        return 0.0
    return round(float(score), 4)


def retrieve_name_matches(entity_type: str, query: str, top_k: int) -> List[Dict[str, Any]]:
    raw_nodes = _retrieve_index_nodes(
        persist_dir=NAME_INDEX_DIR / entity_type,
        cache_key=f"name:{entity_type}",
        query=query,
        top_k=top_k,
    )
    matches: List[Dict[str, Any]] = []
    for node in raw_nodes:
        metadata = dict(getattr(node, "metadata", {}) or {})
        matches.append(
            {
                "score": _score_from_node(node),
                "entity_type": metadata.get("entity_type", entity_type),
                "document_id": metadata.get("document_id"),
                "name": metadata.get("name") or getattr(node, "text", ""),
            }
        )
    return matches


def _serialize_knowledge_node(node: Any) -> Dict[str, Any]:
    metadata = dict(getattr(node, "metadata", {}) or {})
    text = getattr(node, "text", "") or ""
    return {
        "score": _score_from_node(node),
        "vector_score": _score_from_node(node),
        "text": text,
        "chunk_type": metadata.get("chunk_type"),
        "header_path": metadata.get("header_path"),
        "file_name": metadata.get("file_name"),
        "domain": metadata.get("domain"),
        "year": metadata.get("year"),
    }


def _rerank_query_documents(query: str, documents: List[str], batch_size: int = 16) -> Optional[List[float]]:

    tokenizer, model = get_reranker_components()
    device = model.device

    scores: List[float] = []
    with torch.inference_mode():
        for start in range(0, len(documents), batch_size):
            batch_docs = documents[start:start + batch_size]
            pairs = [[query, document] for document in batch_docs]
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            batch_scores = model(**inputs, return_dict=True).logits.view(-1, ).float().cpu().tolist()
            scores.extend(float(score) for score in batch_scores)
    return scores


def retrieve_knowledge_hits(
    query: str,
    top_k: int = 3,
    candidate_multiplier: float = 2.0,
) -> List[Dict[str, Any]]:
    candidate_total = math.ceil(top_k * max(candidate_multiplier, 1.0))
    text_k = max(1, candidate_total // 2)
    table_k = max(1, candidate_total - text_k)
    raw_nodes = [
        *_retrieve_index_nodes(
            persist_dir=KNOWLEDGE_INDEX_DIR / "text",
            cache_key="knowledge:text",
            query=query,
            top_k=text_k,
        ),
        *_retrieve_index_nodes(
            persist_dir=KNOWLEDGE_INDEX_DIR / "table",
            cache_key="knowledge:table",
            query=query,
            top_k=table_k,
        ),
    ]
    hits = [_serialize_knowledge_node(node) for node in raw_nodes]
    rerank_scores = _rerank_query_documents(
        query=query,
        documents=[str(item.get("text") or "") for item in hits],
    )
    if rerank_scores is not None and len(rerank_scores) == len(hits):
        for hit, rerank_score in zip(hits, rerank_scores):
            hit["rerank_score"] = round(float(rerank_score), 4)
        hits.sort(
            key=lambda item: (
                float(item.get("rerank_score", 0.0)),
                float(item.get("vector_score", 0.0)),
            ),
            reverse=True,
        )
    else:
        hits.sort(key=lambda item: float(item.get("vector_score", 0.0)), reverse=True)

    final_hits: List[Dict[str, Any]] = []
    for hit in hits[:top_k]:
        cleaned = dict(hit)
        cleaned["score"] = cleaned.get("rerank_score", cleaned.get("vector_score", 0.0))
        final_hits.append(cleaned)
    return final_hits
