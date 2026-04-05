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
BATCH_SIZE = 256
MIN_SELECTIONS = 2
MAX_SELECTIONS = 4
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL_NAME = "glm-5"


BUCKET_SYSTEM_PROMPT = (
    "You are labeling common workout exercises for a fitness assistant. "
    "Your job is to select the most standard and broadly useful exercises within a bucket. "
    "Prefer common foundational movements over unusual, gimmicky, or overly specific variants. "
    "Do not invent exercise names. "
)


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


class ExerciseSelectionResponse(BaseModel):
    reasoning: str
    selected_names: List[str] = Field(default_factory=list)


def get_exercise_collection(database_name: str):
    return get_mongo_client()[database_name][COLLECTION]


def normalize_text(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip().lower()
    return text or fallback


def build_bucket_key(exercise: Dict[str, Any]) -> str:
    category = normalize_text(exercise.get("category"))
    equipment = normalize_text(exercise.get("equipment"))
    primary_muscle = normalize_text(exercise.get("primaryMuscles")[0])
    return f"{category}::{equipment}::{primary_muscle}"


def clear_candidate_flags(collection) -> int:
    result = collection.update_many({}, {"$unset": {"candidate_flags": ""}})
    return int(result.modified_count)


def create_indexes(collection) -> None:
    collection.create_index("candidate_flags.is_candidate")
    collection.create_index("candidate_flags.bucket_key")


def list_bucket_candidates(collection) -> Dict[str, List[Dict[str, Any]]]:
    bucket_to_exercises: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for exercise in collection.find({}):
        bucket_key = build_bucket_key(exercise)
        bucket_to_exercises[bucket_key].append(exercise)
    return bucket_to_exercises


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
                    BUCKET_SYSTEM_PROMPT
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
        appended_names: List[str] = []
        for name in fallback_names:
            if name not in selected_names:
                selected_names.append(name)
                appended_names.append(name)
            if len(selected_names) >= min(min_select, len(valid_names)):
                break
        if appended_names:
            response.reasoning = (
                f"{response.reasoning} "
                f"Note: The LLM selected only {len(selected_names) - len(appended_names)} valid exercises, which is below the minimum of {min_select}. "
                f"The following exercises were appended in original order because the LLM selected too few valid items: {', '.join(appended_names)}. "
            )

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
        temperature=0.2,
        extra_body={
            "enable_thinking": True,
        }
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

    for bucket_key, selection in selections_by_bucket.items():
        for rank, exercise_name in enumerate(selection.selected_names, start=1):
            selected_by_exercise[exercise_name] = {
                "bucket_key": bucket_key,
                "rank": rank,
                "selection_reason": selection.reasoning,
            }

    operations: List[UpdateOne] = []
    for exercise in exercises:
        exercise_name = str(exercise.get("name"))
        selected_record = selected_by_exercise.get(exercise_name)
        is_candidate = selected_record is not None

        annotation = {
            "is_candidate": is_candidate,
            "bucket_key": selected_record["bucket_key"] if selected_record else None,
            "rank_in_bucket": selected_record["rank"] if selected_record else None,
            "selection_reason": selected_record["selection_reason"] if selected_record else None,
            "selection_method": "llm",
            "selection_model": model_name,
            "selection_policy": {
                "min_select": min_select,
                "max_select": max_select,
                "bucket_definition": ["category", "equipment", "primaryMuscle"],
            },
        }
        operations.append(
            UpdateOne(
                {"_id": exercise_name},
                {"$set": {"candidate_flags": annotation}},
                upsert=False,
            )
        )

    stats = {
        "documents": len(exercises),
        "buckets": len(selections_by_bucket),
        "selected_candidates": len(selected_by_exercise),
        "compressed_documents": len(exercises) - len(selected_by_exercise),
        "compression_ratio": round((len(exercises) - len(selected_by_exercise)) / len(exercises), 4) if exercises else 0.0,
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


async def main_async() -> None:
    load_dotenv()
    database_name = os.getenv("MONGO_DB_NAME", "fitness_assistant")
    collection = get_exercise_collection(database_name)
    create_indexes(collection)
    bucket_to_exercises = list_bucket_candidates(collection)
    selections_by_bucket = await select_all_buckets(
        bucket_to_exercises=bucket_to_exercises,
        base_url=BASE_URL,
        model_name=MODEL_NAME,
        min_select=MIN_SELECTIONS,
        max_select=MAX_SELECTIONS,
    )

    cleared = clear_candidate_flags(collection)
    operations, stats = build_operations(
        collection=collection,
        selections_by_bucket=selections_by_bucket,
        min_select=MIN_SELECTIONS,
        max_select=MAX_SELECTIONS,
        model_name=MODEL_NAME,
    )
    written = write_operations(collection, operations, batch_size=BATCH_SIZE)

    print(
        f"Cleared candidate_flags on {cleared} exercises, then annotated {written} exercises "
        f"in {database_name}.{COLLECTION}."
    )
    pprint(stats)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
