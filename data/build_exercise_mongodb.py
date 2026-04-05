import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from dotenv import load_dotenv
from pymongo import UpdateOne

from agent.infrastructure.database import get_mongo_client


EXERCISES_DATA_PATH = Path("data/json/workout/exercises.json")
BATCH_SIZE = 128
COLLECTION = "exercises"


def load_exercises(input_path: Path) -> List[Dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    exercises = payload.get("exercises")
    return exercises


def build_exercise_document(exercise: Dict[str, Any]) -> Dict[str, Any]:
    name = exercise.get("name")
    return {
        "_id": name,
        **exercise,
    }


def create_indexes(collection) -> None:
    collection.create_index("level")
    collection.create_index("primaryMuscles")
    collection.create_index("category")
    collection.create_index("equipment")


def write_documents(collection, documents: Iterable[Dict[str, Any]], batch_size: int) -> int:
    total_written = 0
    batch = []

    for document in documents:
        batch.append(
            UpdateOne(
                {"_id": document["_id"]},
                {"$set": document},
                upsert=True,
            )
        )

        if len(batch) >= batch_size:
            collection.bulk_write(batch, ordered=False)
            total_written += len(batch)
            batch = []

    if batch:
        collection.bulk_write(batch, ordered=False)
        total_written += len(batch)

    return total_written


def main() -> None:
    load_dotenv()

    database_name = os.getenv("MONGO_DB_NAME", "fitness_assistant")

    exercises = load_exercises(EXERCISES_DATA_PATH)
    
    if not exercises:
        raise RuntimeError("No exercise documents were produced from the input file.")
    documents = [build_exercise_document(exercise) for exercise in exercises]

    database = get_mongo_client()[database_name]
    collection = database[COLLECTION]
    create_indexes(collection)
    written = write_documents(collection, documents, batch_size=BATCH_SIZE)
    print(
        f"Upserted {written} exercise documents into "
        f"{database.name}.{COLLECTION} from {EXERCISES_DATA_PATH}."
    )


if __name__ == "__main__":
    main()
