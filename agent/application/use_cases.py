from typing import Any, Awaitable, Callable, Dict, Optional

from ..workflows import FitnessGraph, ProfileOnboardingGraph
from ..models import FitnessRequest, UserProfile, WorkflowEvent


async def run_fitness_workflow(
    base_url: str,
    model_name: str,
    strong_model_name: Optional[str],
    prompt_user: Callable[[str], Awaitable[str]],
    notify_user: Callable[[str], None],
    event_handler: Optional[Callable[[WorkflowEvent], None]] = None,
    fitness_request: Optional[FitnessRequest] = None,
    profile: Optional[UserProfile] = None,
    max_iterations: int = 6,
) -> Dict[str, Any]:
    request = fitness_request
    if request is None:
        user_input = await prompt_user("")
        request = FitnessRequest(
            user_input=user_input,
            user_profile=profile,
            max_iterations=max_iterations,
        )

    graph = FitnessGraph(
        base_url=base_url,
        model_name=model_name,
        strong_model_name=strong_model_name,
        prompt_user=prompt_user,
        notify_user=notify_user,
        event_handler=event_handler,
    )
    return await graph.run(request)


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
