from typing import Optional

from ..infrastructure.database import get_database
from ..models import UserProfile

PROFILE_COLLECTION_NAME = "user_profiles"


def load_profile(user_id: str) -> Optional[UserProfile]:
    try:
        db = get_database()
        existing = db[PROFILE_COLLECTION_NAME].find_one({"user_id": user_id})
        if existing:
            existing.pop("_id", None)
            return UserProfile(**existing)
    except Exception:
        pass
    return None


def upsert_profile(profile: UserProfile) -> UserProfile:
    try:
        db = get_database()
        db[PROFILE_COLLECTION_NAME].update_one(
            {"user_id": profile.user_id},
            {"$set": profile.model_dump()},
            upsert=True,
        )
    except Exception:
        pass
    return profile
