import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import faiss
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..infrastructure.database import get_mongo_client
from ..llm import build_chat_model, build_structured_output_instruction, invoke_structured_with_retry
from ..models import (
    MealPlanRequest,
    MealPlanStructured,
    UserProfile,
    WorkoutPlanRequest,
    WorkoutPlanStructured,
)
from ..prompts import (
    MEAL_PLAN_SYSTEM_PROMPT,
    WORKOUT_PLAN_SYSTEM_PROMPT,
    build_meal_plan_user_prompt,
    build_workout_plan_user_prompt,
)
from ..repositories.food_repository import find_foods_by_text
from ..services.nutrition_service import (
    build_meal_candidate_bundle,
    format_mongo_food_detail,
    format_mongo_food_summary,
)


DATABASE_NAME = os.getenv("MONGO_DB_NAME", "fitness_assistant")
FOOD_COLLECTION = "foods"
EXERCISE_COLLECTION = "exercises"
NAME_INDEX_DIR = Path("data/processed/faiss_store/names")
KNOWLEDGE_INDEX_DIR = Path("data/processed/faiss_store")

_index_cache: Dict[str, Any] = {}
_embed_model: Optional[HuggingFaceEmbedding] = None


class RequestedEntityNames(BaseModel):
    entity_names: List[str] = Field(default_factory=list)


def _get_database_name() -> str:
    return os.getenv("MONGO_DB_NAME", DATABASE_NAME)


def _ensure_embed_model() -> None:
    global _embed_model
    if _embed_model is not None:
        return

    _embed_model = HuggingFaceEmbedding(
        model_name=os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-m3"),
        device=os.getenv("RAG_EMBEDDING_DEVICE", "cpu"),
    )
    Settings.embed_model = _embed_model


def _load_llama_index(persist_dir: Path, index_path: Path, cache_key: str):
    if cache_key in _index_cache:
        return _index_cache[cache_key]

    if not persist_dir.exists() or not index_path.exists():
        return None

    _ensure_embed_model()

    loaded_faiss_index = faiss.read_index(str(index_path))
    loaded_vector_store = FaissVectorStore(faiss_index=loaded_faiss_index)
    loaded_storage_context = StorageContext.from_defaults(
        persist_dir=str(persist_dir),
        vector_store=loaded_vector_store,
    )
    loaded_index = load_index_from_storage(loaded_storage_context)
    _index_cache[cache_key] = loaded_index
    return loaded_index


def _retrieve_index_nodes(
    persist_dir: Path,
    index_path: Path,
    cache_key: str,
    query: str,
    top_k: int,
) -> List[Any]:
    if not query.strip():
        return []

    index = _load_llama_index(persist_dir=persist_dir, index_path=index_path, cache_key=cache_key)
    if index is None:
        return []

    retriever = index.as_retriever(similarity_top_k=top_k)
    return list(retriever.retrieve(query))


def _score_from_node(node: Any) -> float:
    score = getattr(node, "score", None)
    if score is None:
        return 0.0
    return round(float(score), 4)


def _serialize_knowledge_node(node: Any) -> Dict[str, Any]:
    metadata = dict(getattr(node, "metadata", {}) or {})
    text = getattr(node, "text", "") or ""
    return {
        "score": _score_from_node(node),
        "text": text[:600] + ("..." if len(text) > 600 else ""),
        "chunk_type": metadata.get("chunk_type"),
        "header_path": metadata.get("header_path"),
        "file_name": metadata.get("file_name"),
        "domain": metadata.get("domain"),
        "year": metadata.get("year"),
    }


def _retrieve_knowledge_hits(query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    text_k = max(1, top_k // 2)
    table_k = max(1, top_k - text_k)

    raw_nodes = [
        *_retrieve_index_nodes(
            persist_dir=KNOWLEDGE_INDEX_DIR / "text",
            index_path=KNOWLEDGE_INDEX_DIR / "text" / "faiss.index",
            cache_key="knowledge:text",
            query=query,
            top_k=text_k,
        ),
        *_retrieve_index_nodes(
            persist_dir=KNOWLEDGE_INDEX_DIR / "table",
            index_path=KNOWLEDGE_INDEX_DIR / "table" / "faiss.index",
            cache_key="knowledge:table",
            query=query,
            top_k=table_k,
        ),
    ]
    serialized = [_serialize_knowledge_node(node) for node in raw_nodes]
    serialized.sort(key=lambda item: item["score"], reverse=True)
    return serialized[:top_k]


def _retrieve_name_matches(entity_type: str, query: str, top_k: int) -> List[Dict[str, Any]]:
    raw_nodes = _retrieve_index_nodes(
        persist_dir=NAME_INDEX_DIR / entity_type,
        index_path=NAME_INDEX_DIR / entity_type / "faiss.index",
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


async def _extract_requested_entity_names(
    user_input: str,
    entity_type: str,
    base_url: str,
    model_name: str,
) -> List[str]:
    llm = build_chat_model(
        base_url=base_url,
        model_name=model_name,
        temperature=0.1,
    )
    entity_label = "foods" if entity_type == "food" else "exercises"
    response = await invoke_structured_with_retry(
        llm,
        RequestedEntityNames,
        [
            SystemMessage(
                content=(
                    f"You extract explicitly requested {entity_label} from a user's planning request. "
                    f"Return only distinct concrete {entity_label} that the user clearly names and wants included, prioritized, avoided, checked, or considered in the plan. "
                    "If none are clearly named, return an empty list. "
                    + build_structured_output_instruction(RequestedEntityNames)
                )
            ),
            HumanMessage(content=f"User request: {user_input}"),
        ],
    )
    return list(dict.fromkeys(name.strip() for name in response.entity_names if name and name.strip()))


async def _build_requested_food_lookups(
    user_input: str,
    base_url: str,
    model_name: str,
    food_types: Optional[List[str]] = None,
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    entity_names = await _extract_requested_entity_names(
        user_input=user_input,
        entity_type="food",
        base_url=base_url,
        model_name=model_name,
    )
    lookups: List[Dict[str, Any]] = []
    for entity_name in entity_names[:top_n]:
        result = await search_food_entity_artifact(
            query=entity_name,
            food_types=food_types,
            limit=3,
            knowledge_top_k=2,
        )
        lookups.append(result.get("food_lookup", {}))
    return lookups


async def _build_requested_exercise_lookups(
    user_input: str,
    base_url: str,
    model_name: str,
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    entity_names = await _extract_requested_entity_names(
        user_input=user_input,
        entity_type="exercise",
        base_url=base_url,
        model_name=model_name,
    )
    lookups: List[Dict[str, Any]] = []
    for entity_name in entity_names[:top_n]:
        result = await search_exercise_entity_artifact(
            query=entity_name,
            limit=3,
            knowledge_top_k=2,
        )
        lookups.append(result.get("exercise_lookup", {}))
    return lookups


def _get_food_collection():
    return get_mongo_client()[_get_database_name()][FOOD_COLLECTION]


def _get_exercise_collection():
    return get_mongo_client()[_get_database_name()][EXERCISE_COLLECTION]


def _fetch_food_documents_by_ids(
    document_ids: List[str],
    food_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if not document_ids:
        return []

    mongo_filter: Dict[str, Any] = {"_id": {"$in": document_ids}}
    if food_types:
        mongo_filter["type"] = {"$in": food_types}

    documents = list(_get_food_collection().find(mongo_filter))
    by_id = {str(item.get("_id")): item for item in documents}
    return [by_id[document_id] for document_id in document_ids if document_id in by_id]


def _fetch_exercise_documents_by_ids(document_ids: List[str]) -> List[Dict[str, Any]]:
    if not document_ids:
        return []
    documents = list(_get_exercise_collection().find({"_id": {"$in": document_ids}}))
    by_id = {str(item.get("_id")): item for item in documents}
    return [by_id[document_id] for document_id in document_ids if document_id in by_id]


def _format_exercise_summary(document: Dict[str, Any]) -> Dict[str, Any]:
    candidate_flags = document.get("candidate_flags", {}) or {}
    return {
        "id": document.get("_id"),
        "name": document.get("name"),
        "category": document.get("category"),
        "equipment": document.get("equipment"),
        "level": document.get("level"),
        "mechanic": document.get("mechanic"),
        "force": document.get("force"),
        "primary_muscles": document.get("primaryMuscles", []) or [],
        "secondary_muscles": document.get("secondaryMuscles", []) or [],
        "is_candidate": bool(candidate_flags.get("is_candidate")),
        "bucket_key": candidate_flags.get("bucket_key"),
        "rank_in_bucket": candidate_flags.get("rank_in_bucket"),
    }


def _format_exercise_detail(document: Dict[str, Any]) -> Dict[str, Any]:
    detail = _format_exercise_summary(document)
    detail["instructions"] = document.get("instructions", []) or []
    detail["source_document"] = document
    return detail


def _profile_unavailable_equipment(profile: Optional[UserProfile]) -> set[str]:
    if profile is None:
        return set()
    return {item.name.strip().lower() for item in profile.equipment_notes if not item.enabled and item.name.strip()}


def _profile_available_equipment(profile: Optional[UserProfile]) -> set[str]:
    if profile is None:
        return set()
    return {item.name.strip().lower() for item in profile.equipment_notes if item.enabled and item.name.strip()}


def _exercise_search_text(document: Dict[str, Any]) -> str:
    fields = [
        str(document.get("name", "")),
        str(document.get("category", "")),
        str(document.get("equipment", "")),
        " ".join(str(item) for item in document.get("primaryMuscles", []) or []),
        " ".join(str(item) for item in document.get("secondaryMuscles", []) or []),
    ]
    return " ".join(part for part in fields if part).lower()


def _score_exercise_document(document: Dict[str, Any], query_tokens: List[str]) -> float:
    if not query_tokens:
        return 0.0

    searchable = _exercise_search_text(document)
    name = str(document.get("name", "")).lower()
    score = 0.0
    for token in query_tokens:
        if token in name:
            score += 2.0
        elif token in searchable:
            score += 0.75
    return score


def _filter_exercise_documents(
    documents: Iterable[Dict[str, Any]],
    profile: Optional[UserProfile],
) -> List[Dict[str, Any]]:
    unavailable = _profile_unavailable_equipment(profile)
    available = _profile_available_equipment(profile)
    filtered: List[Dict[str, Any]] = []

    for document in documents:
        equipment = str(document.get("equipment") or "").strip().lower()
        if equipment and equipment in unavailable:
            continue
        if available and equipment and equipment not in available and equipment not in {"body only", "none", "other"}:
            continue
        filtered.append(document)

    return filtered


def _build_exercise_candidate_bundle(
    query: str,
    profile: Optional[UserProfile],
    limit_per_bucket: int,
    limit_total: int,
) -> Dict[str, Any]:
    collection = _get_exercise_collection()
    candidate_documents = list(collection.find({"candidate_flags.is_candidate": True}))
    candidate_documents = _filter_exercise_documents(candidate_documents, profile)

    query_tokens = [token for token in query.lower().split() if len(token) >= 2]
    scored_documents = sorted(
        candidate_documents,
        key=lambda item: (
            _score_exercise_document(item, query_tokens),
            -int((item.get("candidate_flags", {}) or {}).get("rank_in_bucket") or 9999),
            str(item.get("name", "")),
        ),
        reverse=True,
    )

    if not any(_score_exercise_document(item, query_tokens) > 0 for item in scored_documents):
        scored_documents = sorted(
            candidate_documents,
            key=lambda item: (
                int((item.get("candidate_flags", {}) or {}).get("rank_in_bucket") or 9999),
                str(item.get("name", "")),
            ),
        )

    top_matches = [_format_exercise_summary(item) for item in scored_documents[:limit_total]]

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for document in scored_documents:
        bucket_key = str((document.get("candidate_flags", {}) or {}).get("bucket_key") or document.get("category") or "general")
        if len(grouped[bucket_key]) >= limit_per_bucket:
            continue
        grouped[bucket_key].append(_format_exercise_summary(document))

    bucket_candidates = [
        {"bucket_key": bucket_key, "items": items}
        for bucket_key, items in grouped.items()
        if items
    ]

    return {
        "query": query,
        "candidate_strategy": {
            "retrieval_modes": ["candidate_text", "bucket_pool", "knowledge"],
            "limit_per_bucket": limit_per_bucket,
            "limit_total": limit_total,
        },
        "top_matches": top_matches,
        "bucket_candidates": bucket_candidates,
        "knowledge_hits": _retrieve_knowledge_hits(query, top_k=4),
        "total_candidates": sum(len(item["items"]) for item in bucket_candidates),
    }


async def search_food_candidates_artifact(
    user_input: str,
    food_types: Optional[List[str]] = None,
    user_profile: Optional[Dict[str, Any]] = None,
    protein_min: Optional[float] = None,
    carbs_min: Optional[float] = None,
    carbs_max: Optional[float] = None,
    calories_max: Optional[float] = None,
    limit_per_slot: int = 6,
) -> Dict[str, Any]:
    profile = UserProfile(**user_profile) if user_profile else None
    dietary_restrictions = [item.name for item in (profile.dietary_notes if profile else []) if not item.enabled]
    macro_goals: Dict[str, float] = {}
    body_weight = profile.weight if profile and profile.weight else None
    fitness_goal = (profile.fitness_goal or "maintenance").lower() if profile else "maintenance"

    if protein_min is not None:
        macro_goals["protein_min"] = protein_min
    elif body_weight:
        if fitness_goal == "cut":
            macro_goals["protein_min"] = round(body_weight * 0.30, 1)
        elif fitness_goal == "bulk":
            macro_goals["protein_min"] = round(body_weight * 0.25, 1)
        else:
            macro_goals["protein_min"] = round(body_weight * 0.22, 1)

    if carbs_min is not None:
        macro_goals["carbs_min"] = carbs_min
    elif body_weight and fitness_goal == "bulk":
        macro_goals["carbs_min"] = round(body_weight * 0.35, 1)
    elif body_weight and fitness_goal == "maintenance":
        macro_goals["carbs_min"] = round(body_weight * 0.22, 1)

    if carbs_max is not None:
        macro_goals["carbs_max"] = carbs_max
    elif profile and any(item.enabled and item.name.lower() == "keto" for item in profile.dietary_notes):
        macro_goals["carbs_max"] = 12
    elif body_weight and fitness_goal == "cut":
        macro_goals["carbs_max"] = round(body_weight * 0.45, 1)

    if calories_max is not None:
        macro_goals["calories_max"] = calories_max
    elif profile and (profile.target_calories or 0) > 0:
        macro_goals["calories_max"] = max(350, round(profile.target_calories / max(profile.workout_frequency or 3, 3)))

    try:
        food_candidates = build_meal_candidate_bundle(
            query=user_input,
            dietary_restrictions=dietary_restrictions,
            macro_goals=macro_goals,
            food_types=food_types or ["foundation"],
            limit_per_slot=limit_per_slot,
        )
    except Exception:
        fallback_foods = [
            format_mongo_food_summary(item)
            for item in find_foods_by_text(
                query=user_input,
                limit=max(limit_per_slot * 2, 12),
                food_types=food_types or ["foundation"],
                meal_candidates_only=True,
            )
        ]
        food_candidates = {
            "query": user_input,
            "candidate_strategy": {
                "vector_backend": "faiss",
                "retrieval_modes": ["text_fallback"],
                "limit_per_slot": limit_per_slot,
                "macro_goals": macro_goals,
                "food_types": food_types or ["foundation"],
            },
            "top_matches": fallback_foods[:limit_per_slot],
            "slot_candidates": {
                "proteins": [],
                "carbs": [],
                "vegetables": [],
                "fats": [],
                "flexible": fallback_foods[:limit_per_slot],
            },
            "total_candidates": len(fallback_foods[:limit_per_slot]),
        }

    food_candidates.setdefault("candidate_strategy", {})
    food_candidates["candidate_strategy"]["macro_goals"] = macro_goals
    food_candidates["candidate_strategy"]["food_types"] = food_types or ["foundation"]
    food_candidates["knowledge_hits"] = _retrieve_knowledge_hits(user_input, top_k=4)
    top_matches = food_candidates.get("top_matches", [])
    total_candidates = food_candidates.get("total_candidates", len(top_matches))
    return {
        "food_candidates": food_candidates,
        "observation": (
            f"Nutrition search assembled a meal-planning candidate bundle with "
            f"{total_candidates} foods across grouped slots, {len(top_matches)} direct matches, "
            f"and {len(food_candidates.get('knowledge_hits', []))} knowledge hits."
        ),
    }


async def search_exercise_candidates_artifact(
    user_input: str,
    user_profile: Optional[Dict[str, Any]] = None,
    query: Optional[str] = None,
    limit_per_bucket: int = 4,
    limit_total: int = 16,
) -> Dict[str, Any]:
    profile = UserProfile(**user_profile) if user_profile else None
    candidate_query = (query or user_input).strip() or user_input
    exercise_candidates = _build_exercise_candidate_bundle(
        query=candidate_query,
        profile=profile,
        limit_per_bucket=limit_per_bucket,
        limit_total=limit_total,
    )
    return {
        "exercise_candidates": exercise_candidates,
        "observation": (
            f"Exercise search assembled {exercise_candidates.get('total_candidates', 0)} candidate exercises, "
            f"{len(exercise_candidates.get('top_matches', []))} top matches, "
            f"and {len(exercise_candidates.get('knowledge_hits', []))} knowledge hits."
        ),
    }


async def search_food_entity_artifact(
    query: str,
    food_types: Optional[List[str]] = None,
    limit: int = 5,
    knowledge_top_k: int = 4,
) -> Dict[str, Any]:
    name_matches = _retrieve_name_matches(entity_type="food", query=query, top_k=limit)
    document_ids = [str(item.get("document_id")) for item in name_matches if item.get("document_id")]
    matched_foods = [
        format_mongo_food_detail(item)
        for item in _fetch_food_documents_by_ids(document_ids=document_ids, food_types=food_types)
    ]

    if not matched_foods:
        matched_foods = [
            format_mongo_food_detail(item)
            for item in find_foods_by_text(
                query=query,
                limit=limit,
                food_types=food_types or ["foundation"],
                meal_candidates_only=False,
            )
        ]

    food_lookup = {
        "query": query,
        "name_matches": name_matches,
        "matched_foods": matched_foods,
        "knowledge_hits": _retrieve_knowledge_hits(query, top_k=knowledge_top_k),
    }
    return {
        "food_lookup": food_lookup,
        "observation": (
            f"Food entity lookup found {len(matched_foods)} food records, "
            f"{len(name_matches)} name matches, and {len(food_lookup['knowledge_hits'])} knowledge hits."
        ),
    }


async def search_exercise_entity_artifact(
    query: str,
    limit: int = 5,
    knowledge_top_k: int = 4,
) -> Dict[str, Any]:
    name_matches = _retrieve_name_matches(entity_type="exercise", query=query, top_k=limit)
    document_ids = [str(item.get("document_id")) for item in name_matches if item.get("document_id")]
    matched_exercises = [
        _format_exercise_detail(item)
        for item in _fetch_exercise_documents_by_ids(document_ids=document_ids)
    ]

    if not matched_exercises:
        regex = {"$regex": query, "$options": "i"}
        fallback_documents = list(
            _get_exercise_collection().find(
                {
                    "$or": [
                        {"name": regex},
                        {"category": regex},
                        {"equipment": regex},
                        {"primaryMuscles": regex},
                    ]
                }
            ).limit(limit)
        )
        matched_exercises = [_format_exercise_detail(item) for item in fallback_documents]

    exercise_lookup = {
        "query": query,
        "name_matches": name_matches,
        "matched_exercises": matched_exercises,
        "knowledge_hits": _retrieve_knowledge_hits(query, top_k=knowledge_top_k),
    }
    return {
        "exercise_lookup": exercise_lookup,
        "observation": (
            f"Exercise entity lookup found {len(matched_exercises)} exercise records, "
            f"{len(name_matches)} name matches, and {len(exercise_lookup['knowledge_hits'])} knowledge hits."
        ),
    }


async def generate_meal_plan_artifact(
    user_input: str,
    user_profile: Dict[str, Any],
    meal_preferences: Optional[Dict[str, Any]] = None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_name: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    profile = UserProfile(**user_profile)
    resolved_meal_preferences = dict(meal_preferences or {})
    meal_request = MealPlanRequest(
        meal_count=resolved_meal_preferences.get("meal_count", 5),
        days=resolved_meal_preferences.get("days", 7),
        preferences=resolved_meal_preferences,
    )
    food_candidates = (await search_food_candidates_artifact(
        user_input=user_input,
        food_types=["foundation"],
        user_profile=profile.model_dump(),
        protein_min=resolved_meal_preferences.get("protein_min"),
        carbs_min=resolved_meal_preferences.get("carbs_min"),
        carbs_max=resolved_meal_preferences.get("carbs_max"),
        calories_max=resolved_meal_preferences.get("calories_max"),
        limit_per_slot=resolved_meal_preferences.get("limit_per_slot", 6),
    )).get("food_candidates", {})
    if "food_lookup" in resolved_meal_preferences:
        food_candidates["requested_food_lookups"] = [resolved_meal_preferences["food_lookup"]]
    else:
        food_candidates["requested_food_lookups"] = await _build_requested_food_lookups(
            user_input=user_input,
            base_url=base_url,
            model_name=model_name,
            food_types=["foundation"],
        )

    llm = build_chat_model(
        base_url=base_url,
        model_name=model_name,
        temperature=0.4,
    )
    meal_plan = await invoke_structured_with_retry(
        llm,
        MealPlanStructured,
        [
            SystemMessage(
                content=MEAL_PLAN_SYSTEM_PROMPT
                + "\n\n"
                + build_structured_output_instruction(MealPlanStructured)
            ),
            HumanMessage(
                content=build_meal_plan_user_prompt(
                    user_input,
                    profile.model_dump(),
                    meal_request.model_dump(),
                    food_candidates or {},
                )
            ),
        ],
    )

    slot_candidates = (food_candidates or {}).get("slot_candidates", {})
    available_foods_count = sum(len(items) for items in slot_candidates.values())

    return {
        "meal_plan": {
            "user_id": profile.user_id,
            "plan_name": meal_plan.plan_name,
            "days": len(meal_plan.days),
            "target_macros": meal_plan.target_macros.model_dump(),
            "daily_plans": [day.model_dump() for day in meal_plan.days],
            "key_principles": meal_plan.key_principles,
            "shopping_tips": meal_plan.shopping_tips,
            "available_foods_count": available_foods_count,
            "knowledge_hits_used": len((food_candidates or {}).get("knowledge_hits", [])),
            "generated_at": datetime.now().isoformat(),
        },
        "observation": f"Meal plan tool generated a {len(meal_plan.days)}-day plan.",
    }


async def generate_workout_plan_artifact(
    user_input: str,
    user_profile: Dict[str, Any],
    workout_preferences: Optional[Dict[str, Any]] = None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_name: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    profile = UserProfile(**user_profile)
    resolved_workout_preferences = dict(workout_preferences or {})
    workout_request = WorkoutPlanRequest(
        split_type=resolved_workout_preferences.get("split_type", "full_body"),
        training_style=resolved_workout_preferences.get("training_style", "hypertrophy"),
        days_per_week=resolved_workout_preferences.get("days_per_week", profile.workout_frequency or 3),
        duration_minutes=resolved_workout_preferences.get("duration_minutes", profile.workout_duration or 60),
    )
    exercise_candidates = (await search_exercise_candidates_artifact(
        user_input=user_input,
        user_profile=profile.model_dump(),
        query=resolved_workout_preferences.get("query"),
        limit_per_bucket=resolved_workout_preferences.get("limit_per_bucket", 4),
        limit_total=resolved_workout_preferences.get("limit_total", 16),
    )).get("exercise_candidates", {})
    if "exercise_lookup" in resolved_workout_preferences:
        exercise_candidates["requested_exercise_lookups"] = [resolved_workout_preferences["exercise_lookup"]]
    else:
        exercise_candidates["requested_exercise_lookups"] = await _build_requested_exercise_lookups(
            user_input=user_input,
            base_url=base_url,
            model_name=model_name,
        )

    llm = build_chat_model(
        base_url=base_url,
        model_name=model_name,
        temperature=0.4,
    )
    workout_plan = await invoke_structured_with_retry(
        llm,
        WorkoutPlanStructured,
        [
            SystemMessage(
                content=WORKOUT_PLAN_SYSTEM_PROMPT
                + "\n\n"
                + build_structured_output_instruction(WorkoutPlanStructured)
            ),
            HumanMessage(
                content=build_workout_plan_user_prompt(
                    user_input,
                    profile.model_dump(),
                    workout_request.model_dump(),
                    exercise_candidates or {},
                )
            ),
        ],
    )

    return {
        "workout_plan": {
            "user_id": profile.user_id,
            "plan_name": workout_plan.plan_name,
            "split_type": workout_plan.split_type,
            "training_style": workout_plan.training_style,
            "days_per_week": workout_request.days_per_week,
            "duration_minutes": workout_request.duration_minutes,
            "weekly_schedule": [day.model_dump() for day in workout_plan.weekly_schedule],
            "progression_strategy": workout_plan.progression_strategy,
            "equipment_needed": workout_plan.equipment_needed,
            "key_principles": workout_plan.key_principles,
            "available_exercises_count": len((exercise_candidates or {}).get("top_matches", [])),
            "knowledge_hits_used": len((exercise_candidates or {}).get("knowledge_hits", [])),
            "generated_at": datetime.now().isoformat(),
        },
        "observation": f"Workout plan tool generated a {workout_request.days_per_week}-day program.",
    }
