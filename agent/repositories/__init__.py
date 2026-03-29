"""Repository layer for persistence and external data access."""

from .profile_repository import load_profile, upsert_profile

__all__ = ["load_profile", "upsert_profile"]
