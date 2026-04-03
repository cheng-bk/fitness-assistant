import argparse
import asyncio
import os
from collections import defaultdict
from pprint import pprint
from typing import Any, Dict, Iterable, List, Tuple

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from pymongo import UpdateOne

from agent.llm import (
    build_chat_model,
    build_structured_output_instruction,
    invoke_structured_with_retry,
)

from agent.infrastructure.database import get_mongo_client


COLLECTION = "exercises"
DEFAULT_BATCH_SIZE = 256
DEFAULT_MIN_SELECTIONS = 3
DEFAULT_MAX_SELECTIONS = 5
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL_NAME = "gpt-4o-mini"


class ExerciseSelectionResponse(BaseModel):
    reasoning: str
    selected_names: List[str] = Field(default_factory=list)

def get_exercise_collection(database_name: str):
    return get_mongo_client()[database_name][COLLECTION]


def normalize_text(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip().lower()
    return text or fallback


def get_primary_muscles(exercise: Dict[str, Any]) -> List[str]:
    muscles = exercise.get("primaryMuscles") or []
    normalized = [normalize_text(item) for item in muscles if str(item).strip()]
    return normalized or ["unknown"]


def build_bucket_keys(exercise: Dict[str, Any]) -> List[str]:
    category = normalize_text(exercise.get("category"))
    equipment = normalize_text(exercise.get("equipment"))
    return [f"{category}::{equipment}::{muscle}" for muscle in get_primary_muscles(exercise)]


def clear_candidate_flags(collection) -> int:
    result = collection.update_many({}, {"$unset": {"candidate_flags": ""}})
    return int(result.modified_count)


def create_indexes(collection) -> None:
    collection.create_index("candidate_flags.is_common_candidate")
    collection.create_index("candidate_flags.common_bucket_keys")
    collection.create_index("candidate_flags.selection_method")


def list_bucket_candidates(collection) -> Dict[str, List[Dict[str, Any]]]:
    bucket_to_exercises: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for exercise in collection.find({}):
        for bucket_key in build_bucket_keys(exercise):
            bucket_to_exercises[bucket_key].append(exercise)
    return bucket_to_exercises


def build_bucket_prompt(bucket_key: str, exercises: List[Dict[str, Any]], min_select: int, max_select: int) -> str:
    category, equipment, primary_muscle = bucket_key.split("::", 2)
    exercise_lines: List[str] = []
    for exercise in exercises:
        exercise_lines.append(
            (
                f"- name: {exercise.get('name', '')}\n"
                f"  level: {exercise.get('level', '')}\n"
                f"  mechanic: {exercise.get('mechanic', '')}\n"
                f"  force: {exercise.get('force', '')}\n"
                f"  primary_muscles: {exercise.get('primaryMuscles', [])}\n"
                f"  secondary_muscles: {exercise.get('secondaryMuscles', [])}\n"
                f"  instruction_count: {len(exercise.get('instructions') or [])}"
            )
        )

    return (
        f"Bucket category: {category}\n"
        f"Bucket equipment: {equipment}\n"
        f"Bucket primary muscle: {primary_muscle}\n"
        f"Candidate exercises:\n" + "\n".join(exercise_lines) + "\n\n"
        f"Select {min_select} to {max_select} exercises that are the most common, standard, and practical choices "
        "for general population workout planning. Prefer exercises that are broadly recognizable, easy to program, "
        "and representative of this bucket. Avoid quirky, niche, redundant, or overly specialized variants when a "
        "more standard movement exists.\n"
        "Return only exercise names that exist exactly in the candidate list."
    )


async def select_bucket_candidates_with_llm(
    llm,
    bucket_key: str,
    exercises: List[Dict[str, Any]],
    min_select: int,
    max_select: int,
) -> ExerciseSelectionResponse:
    if len(exercises) <= max_select:
        selected_names = [str(exercise.get("name", "")) for exercise in exercises if exercise.get("name")]
        return ExerciseSelectionResponse(
            reasoning="Bucket size is already within the target range, so all exercises were kept.",
            selected_names=selected_names,
        )

    response = await invoke_structured_with_retry(
        llm,
        ExerciseSelectionResponse,
        [
            SystemMessage(
                content=(
                    "You are labeling common workout exercises for a fitness assistant. "
                    "Your job is to select the most standard and broadly useful exercises within a bucket. "
                    "Prefer common foundational movements over unusual, gimmicky, or overly specific variants. "
                    "Do not invent exercise names. "
                    + build_structured_output_instruction(ExerciseSelectionResponse)
                )
            ),
            HumanMessage(content=build_bucket_prompt(bucket_key, exercises, min_select, max_select)),
        ],
    )

    valid_names = {str(exercise.get("name", "")) for exercise in exercises}
    selected_names = [name for name in response.selected_names if name in valid_names]
    selected_names = list(dict.fromkeys(selected_names))

    if len(selected_names) < min_select:
        fallback_names = [str(exercise.get("name", "")) for exercise in exercises if exercise.get("name")]
        for name in fallback_names:
            if name not in selected_names:
                selected_names.append(name)
            if len(selected_names) >= min(min_select, len(valid_names)):
                break

    return ExerciseSelectionResponse(
        reasoning=response.reasoning,
        selected_names=selected_names[: min(max_select, len(valid_names))],
    )


async def select_all_buckets(
    bucket_to_exercises: Dict[str, List[Dict[str, Any]]],
    base_url: str,
    model_name: str,
    min_select: int,
    max_select: int,
) -> Dict[str, ExerciseSelectionResponse]:
    llm = build_chat_model(
        base_url=base_url,
        model_name=model_name,
        temperature=0.1,
    )

    selections: Dict[str, ExerciseSelectionResponse] = {}
    for bucket_key, exercises in bucket_to_exercises.items():
        selections[bucket_key] = await select_bucket_candidates_with_llm(
            llm=llm,
            bucket_key=bucket_key,
            exercises=exercises,
            min_select=min_select,
            max_select=max_select,
        )
    return selections


def build_operations(
    collection,
    selections_by_bucket: Dict[str, ExerciseSelectionResponse],
    min_select: int,
    max_select: int,
    model_name: str,
) -> Tuple[List[UpdateOne], Dict[str, Any]]:
    exercises = list(collection.find({}))
    selected_by_exercise: Dict[str, Dict[str, Any]] = {}
    selected_total = 0

    for bucket_key, selection in selections_by_bucket.items():
        for rank, exercise_name in enumerate(selection.selected_names, start=1):
            selected_total += 1
            record = selected_by_exercise.setdefault(
                exercise_name,
                {
                    "common_bucket_keys": [],
                    "bucket_rankings": {},
                    "selection_reasons": {},
                },
            )
            record["common_bucket_keys"].append(bucket_key)
            record["bucket_rankings"][bucket_key] = rank
            record["selection_reasons"][bucket_key] = selection.reasoning

    operations: List[UpdateOne] = []
    selected_candidates = 0
    for exercise in exercises:
        exercise_id = str(exercise.get("_id"))
        selected_record = selected_by_exercise.get(exercise_id)
        is_common_candidate = selected_record is not None
        if is_common_candidate:
            selected_candidates += 1

        annotation = {
            "is_common_candidate": is_common_candidate,
            "common_bucket_keys": selected_record["common_bucket_keys"] if selected_record else [],
            "bucket_rankings": selected_record["bucket_rankings"] if selected_record else {},
            "selection_reasons": selected_record["selection_reasons"] if selected_record else {},
            "selection_method": "llm",
            "selection_model": model_name,
            "selection_policy": {
                "min_select": min_select,
                "max_select": max_select,
                "bucket_definition": ["category", "equipment", "primaryMuscles"],
            },
        }
        operations.append(
            UpdateOne(
                {"_id": exercise_id},
                {"$set": {"candidate_flags": annotation}},
                upsert=False,
            )
        )

    stats = {
        "documents": len(exercises),
        "buckets": len(selections_by_bucket),
        "selected_slot_count": selected_total,
        "selected_candidates": selected_candidates,
        "compressed_documents": len(exercises) - selected_candidates,
        "compression_ratio": round((len(exercises) - selected_candidates) / len(exercises), 4) if exercises else 0.0,
    }
    return operations, stats


def write_operations(collection, operations: Iterable[UpdateOne], batch_size: int) -> int:
    written = 0
    batch: List[UpdateOne] = []
    for operation in operations:
        batch.append(operation)
        if len(batch) >= batch_size:
            collection.bulk_write(batch, ordered=False)
            written += len(batch)
            batch = []

    if batch:
        collection.bulk_write(batch, ordered=False)
        written += len(batch)

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate common exercise candidates with an LLM.")
    parser.add_argument("--min-select", type=int, default=DEFAULT_MIN_SELECTIONS)
    parser.add_argument("--max-select", type=int, default=DEFAULT_MAX_SELECTIONS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    if args.min_select < 1 or args.max_select < args.min_select:
        raise ValueError("Invalid selection range.")

    load_dotenv()
    database_name = os.getenv("MONGO_DB_NAME", "fitness_assistant")
    collection = get_exercise_collection(database_name)
    create_indexes(collection)
    bucket_to_exercises = list_bucket_candidates(collection)
    selections_by_bucket = await select_all_buckets(
        bucket_to_exercises=bucket_to_exercises,
        base_url=args.base_url,
        model_name=args.model_name,
        min_select=args.min_select,
        max_select=args.max_select,
    )

    cleared = clear_candidate_flags(collection)
    operations, stats = build_operations(
        collection=collection,
        selections_by_bucket=selections_by_bucket,
        min_select=args.min_select,
        max_select=args.max_select,
        model_name=args.model_name,
    )
    written = write_operations(collection, operations, batch_size=args.batch_size)

    print(
        f"Cleared candidate_flags on {cleared} exercises, then annotated {written} exercises "
        f"in {database_name}.{COLLECTION}."
    )
    pprint(stats)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
