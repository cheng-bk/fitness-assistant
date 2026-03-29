from typing import Any, Awaitable, Callable, Dict, Optional

from ..workflows import FitnessGraph, ProfileOnboardingGraph
from ..models import FitnessRequest, UserProfile, WorkflowEvent


async def run_fitness_workflow(
    fitness_request: FitnessRequest,
    base_url: str,
    model_name: str,
    planner_model_name: Optional[str],
    prompt_user: Callable[[str], Awaitable[str]],
    notify_user: Callable[[str], None],
    event_handler: Optional[Callable[[WorkflowEvent], None]] = None,
) -> Dict[str, Any]:
    graph = FitnessGraph(
        base_url=base_url,
        model_name=model_name,
        planner_model_name=planner_model_name,
        prompt_user=prompt_user,
        notify_user=notify_user,
        event_handler=event_handler,
    )
    return await graph.run(fitness_request)


async def run_profile_onboarding_workflow(
    profile: UserProfile,
    base_url: str,
    model_name: str,
    prompt_user: Callable[[str], Awaitable[str]],
    notify_user: Callable[[str], None],
    event_handler: Optional[Callable[[WorkflowEvent], None]] = None,
) -> UserProfile:
    graph = ProfileOnboardingGraph(
        base_url=base_url,
        model_name=model_name,
        prompt_user=prompt_user,
        notify_user=notify_user,
        event_handler=event_handler,
    )
    return await graph.run(profile)
