from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import build_chat_model, build_structured_output_instruction, invoke_structured_with_retry
from ..models import MealPlanStructured, UserProfile
from ..prompts import MEAL_PLAN_SYSTEM_PROMPT, build_meal_plan_user_prompt
from ..repositories.food_repository import (
    CARB_CATEGORIES,
    FAT_CATEGORIES,
    FLEXIBLE_CATEGORIES,
    OIL_CATEGORIES,
    PROTEIN_CATEGORIES,
    VEGETABLE_CATEGORIES,
    find_foods_for_meal_slot,
)
from ..repositories.vector_index_repository import retrieve_name_matches
from ..repositories.food_repository import get_food_collection

MEAL_ROLE_CATEGORY_MAP = {
    "proteins": set(PROTEIN_CATEGORIES),
    "carbs": set(CARB_CATEGORIES),
    "vegetables": set(VEGETABLE_CATEGORIES),
    "fats": set(FAT_CATEGORIES),
    "oil": set(OIL_CATEGORIES),
    "flexible": set(FLEXIBLE_CATEGORIES),
}


def _fetch_food_documents_by_ids(document_ids: List[str], food_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    if not document_ids:
        return []
    mongo_filter: Dict[str, Any] = {"_id": {"$in": document_ids}}
    if food_types:
        mongo_filter["type"] = {"$in": food_types}
    documents = list(get_food_collection().find(mongo_filter))
    by_id = {str(item.get("_id")): item for item in documents}
    return [by_id[document_id] for document_id in document_ids]


async def search_food_entity_artifact(
    query: str,
    food_types: Optional[List[str]] = None,
    limit: int = 5,
) -> Dict[str, Any]:
    name_matches = retrieve_name_matches(entity_type="food", query=query, top_k=limit)
    document_ids = [str(item.get("document_id")) for item in name_matches if item.get("document_id")]
    matched_foods = [item for item in _fetch_food_documents_by_ids(document_ids=document_ids, food_types=food_types)]
    for item in matched_foods:
        item.pop("_id", None)
        item.pop("candidate_flags", None)

    food_lookup = {
        "query": query,
        "name_matches": name_matches,
        "matched_foods": matched_foods,
    }
    return {
        "food_lookup": food_lookup,
        "observation": (
            f"Food entity lookup found {len(matched_foods)} food records, "
            f"{len(name_matches)} name matches."
        ),
    }


def _build_food_candidate_bundle(
    food_types: Optional[List[str]] = None,
    limit_per_slot: int = 6,
) -> Dict[str, Any]:
    normalized_food_types = food_types or ["foundation"]
    slot_candidates: Dict[str, List[Dict[str, Any]]] = {}
    for slot_name in ["proteins", "carbs", "vegetables", "fats", "oil", "flexible"]:
        slot_from_db = find_foods_for_meal_slot(
            slot_name=slot_name,
            limit=limit_per_slot,
            food_types=normalized_food_types,
            meal_candidates_only=True,
        )
        for item in slot_from_db:
            item.pop("_id", None)
            item.pop("candidate_flags", None)
        slot_candidates[slot_name] = slot_from_db
    
    total_candidates = sum(len(items) for items in slot_candidates.values())
    return {
        "candidate_strategy": {
            "limit_per_slot": limit_per_slot,
            "food_types": normalized_food_types,
        },
        "slot_candidates": slot_candidates,
        "total_candidates": total_candidates,
    }
    

def _build_food_candidates(limit_per_slot: int = 6) -> Dict[str, Any]:
    return _build_food_candidate_bundle(
        food_types=["foundation"],
        limit_per_slot=limit_per_slot,
    )


async def generate_meal_plan_artifact(
    user_input: str,
    user_profile: Dict[str, Any],
    meal_preferences: Optional[Dict[str, Any]] = None,
    prior_food_lookups: Optional[List[Dict[str, Any]]] = None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_name: str = "qwen3.5-plus",
) -> Dict[str, Any]:
    profile = UserProfile(**user_profile)
    resolved_meal_preferences = dict(meal_preferences or {})
    
    requested_food_names = [
        str(item).strip() for item in (resolved_meal_preferences.get("requested_food_names") or []) if str(item).strip()
    ]
    excluded_food_names = [
        str(item).strip() for item in (resolved_meal_preferences.get("excluded_food_names") or []) if str(item).strip()
    ]
    resolved_prior_food_lookups = [item for item in (prior_food_lookups or []) if item]

    food_candidates = _build_food_candidates(limit_per_slot=6)
    resolved_meal_context = {
        "requested_food_names": requested_food_names,
        "excluded_food_names": excluded_food_names,
        "meal_count": resolved_meal_preferences.get("meal_count", 3),
        "days": resolved_meal_preferences.get("days", 7),
        "notes": resolved_meal_preferences.get("notes", []) or [],
    }

    llm = build_chat_model(base_url=base_url, model_name=model_name, temperature=0.7, extra_body={"enable_thinking": True})
    meal_plan = await invoke_structured_with_retry(
        llm,
        MealPlanStructured,
        [
            SystemMessage(content=MEAL_PLAN_SYSTEM_PROMPT + "\n\n" + build_structured_output_instruction(MealPlanStructured)),
            HumanMessage(
                content=build_meal_plan_user_prompt(
                    user_input,
                    profile.model_dump(),
                    food_candidates=food_candidates,
                    meal_preferences=resolved_meal_context,
                    prior_food_lookups=resolved_prior_food_lookups,
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
            "prior_food_lookup_count": len(resolved_prior_food_lookups),
            "generated_at": datetime.now().isoformat(),
        },
        "observation": f"Meal plan tool generated a {len(meal_plan.days)}-day plan.",
    }
