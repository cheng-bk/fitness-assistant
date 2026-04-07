from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from ..infrastructure.database import get_database
from ..llm import build_chat_model, build_structured_output_instruction, invoke_structured_with_retry
from ..models import UserProfile, WorkoutPlanRequest, WorkoutPlanStructured
from ..prompts import WORKOUT_PLAN_SYSTEM_PROMPT, build_workout_plan_user_prompt
from ..repositories.vector_index_repository import retrieve_knowledge_hits, retrieve_name_matches


EXERCISE_COLLECTION = "exercises"


def _get_exercise_collection():
    return get_database()[EXERCISE_COLLECTION]


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


def _filter_exercise_documents(documents: Iterable[Dict[str, Any]], profile: Optional[UserProfile]) -> List[Dict[str, Any]]:
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


def _normalize_name_list(names: Optional[List[str]]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for raw_name in names or []:
        name = str(raw_name or "").strip()
        if not name:
            continue
        lowered = name.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(name)
    return normalized


def _fetch_exercise_documents_by_ids(document_ids: List[str]) -> List[Dict[str, Any]]:
    if not document_ids:
        return []
    documents = list(_get_exercise_collection().find({"_id": {"$in": document_ids}}))
    by_id = {str(item.get("_id")): item for item in documents}
    return [by_id[document_id] for document_id in document_ids if document_id in by_id]


def build_exercise_candidate_bundle(
    query: str,
    profile: Optional[UserProfile],
    limit_per_bucket: int,
    limit_total: int,
) -> Dict[str, Any]:
    candidate_documents = list(_get_exercise_collection().find({"candidate_flags.is_candidate": True}))
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

    return {
        "query": query,
        "candidate_strategy": {
            "retrieval_modes": ["candidate_text", "bucket_pool", "knowledge"],
            "limit_per_bucket": limit_per_bucket,
            "limit_total": limit_total,
        },
        "top_matches": top_matches,
        "bucket_candidates": [{"bucket_key": bucket_key, "items": items} for bucket_key, items in grouped.items() if items],
        "knowledge_hits": retrieve_knowledge_hits(query, top_k=4),
        "total_candidates": sum(len(items) for items in grouped.values() if items),
    }


async def search_exercise_entity_artifact(
    query: str,
    limit: int = 5,
    knowledge_top_k: int = 3,
) -> Dict[str, Any]:
    name_matches = retrieve_name_matches(entity_type="exercise", query=query, top_k=limit)
    document_ids = [str(item.get("document_id")) for item in name_matches if item.get("document_id")]
    matched_exercises = [_format_exercise_detail(item) for item in _fetch_exercise_documents_by_ids(document_ids=document_ids)]

    if not matched_exercises:
        regex = {"$regex": query, "$options": "i"}
        fallback_documents = list(
            _get_exercise_collection().find(
                {"$or": [{"name": regex}, {"category": regex}, {"equipment": regex}, {"primaryMuscles": regex}]}
            ).limit(limit)
        )
        matched_exercises = [_format_exercise_detail(item) for item in fallback_documents]

    exercise_lookup = {
        "query": query,
        "name_matches": name_matches,
        "matched_exercises": matched_exercises,
        "knowledge_hits": retrieve_knowledge_hits(query, top_k=knowledge_top_k),
    }
    return {
        "exercise_lookup": exercise_lookup,
        "observation": (
            f"Exercise entity lookup found {len(matched_exercises)} exercise records, "
            f"{len(name_matches)} name matches, and {len(exercise_lookup['knowledge_hits'])} knowledge hits."
        ),
    }


async def _build_exercise_lookups_from_names(entity_names: Optional[List[str]], top_n: int = 3) -> List[Dict[str, Any]]:
    lookups: List[Dict[str, Any]] = []
    for entity_name in _normalize_name_list(entity_names)[:top_n]:
        result = await search_exercise_entity_artifact(query=entity_name, limit=3, knowledge_top_k=2)
        lookups.append(result.get("exercise_lookup", {}))
    return lookups


async def build_exercise_candidates_artifact(
    user_input: str,
    user_profile: Optional[Dict[str, Any]] = None,
    query: Optional[str] = None,
    limit_per_bucket: int = 4,
    limit_total: int = 16,
) -> Dict[str, Any]:
    profile = UserProfile(**user_profile) if user_profile else None
    candidate_query = (query or user_input).strip() or user_input
    exercise_candidates = build_exercise_candidate_bundle(
        query=candidate_query,
        profile=profile,
        limit_per_bucket=limit_per_bucket,
        limit_total=limit_total,
    )
    return {
        "exercise_candidates": exercise_candidates,
        "observation": (
            f"Exercise candidate search assembled {exercise_candidates.get('total_candidates', 0)} candidate exercises, "
            f"{len(exercise_candidates.get('top_matches', []))} top matches, "
            f"and {len(exercise_candidates.get('knowledge_hits', []))} knowledge hits."
        ),
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
    requested_exercise_names = _normalize_name_list(resolved_workout_preferences.get("requested_exercise_names"))
    excluded_exercise_names = _normalize_name_list(resolved_workout_preferences.get("excluded_exercise_names"))
    prior_exercise_lookups = [item for item in (resolved_workout_preferences.get("exercise_lookups") or []) if item]

    workout_request = WorkoutPlanRequest(
        split_type=resolved_workout_preferences.get("split_type", "full_body"),
        training_style=resolved_workout_preferences.get("training_style", "hypertrophy"),
        days_per_week=resolved_workout_preferences.get("days_per_week", profile.workout_frequency or 3),
        duration_minutes=resolved_workout_preferences.get("duration_minutes", profile.workout_duration or 60),
    )
    exercise_candidates = (
        await build_exercise_candidates_artifact(
            user_input=user_input,
            user_profile=profile.model_dump(),
            query=resolved_workout_preferences.get("query"),
        )
    ).get("exercise_candidates", {})
    exercise_candidates["requested_exercise_names"] = requested_exercise_names
    exercise_candidates["excluded_exercise_names"] = excluded_exercise_names
    exercise_candidates["requested_exercise_lookups"] = (
        prior_exercise_lookups if prior_exercise_lookups else await _build_exercise_lookups_from_names(requested_exercise_names)
    )
    exercise_candidates["excluded_exercise_lookups"] = await _build_exercise_lookups_from_names(excluded_exercise_names)

    llm = build_chat_model(base_url=base_url, model_name=model_name, temperature=0.4)
    workout_plan = await invoke_structured_with_retry(
        llm,
        WorkoutPlanStructured,
        [
            SystemMessage(content=WORKOUT_PLAN_SYSTEM_PROMPT + "\n\n" + build_structured_output_instruction(WorkoutPlanStructured)),
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
