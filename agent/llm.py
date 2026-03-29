import asyncio
from typing import Type

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


def build_chat_model(base_url: str, model_name: str, temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(base_url=base_url, model=model_name, temperature=temperature, extra_body={"enable_thinking": False})


def build_structured_output_instruction(schema_model: Type[BaseModel]) -> str:
    keys = list(schema_model.model_fields.keys())
    lines = ["Return valid JSON only.", "", "Use exactly these keys:"]
    lines.extend(f"- {key}" for key in keys)
    return "\n".join(lines)


async def invoke_structured_with_retry(
    llm: ChatOpenAI,
    schema_model: Type[BaseModel],
    messages: list[BaseMessage],
    max_attempts: int = 3,
) -> BaseModel:
    structured_llm = llm.with_structured_output(schema_model)
    
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        
        retry_message = HumanMessage(
        content=(
            "Your previous response did not match the required structured output. "
            "The exception was: " + (str(last_error) if last_error else "Unknown error") + "\n\n"
            "Try again and follow the JSON constraints strictly.\n\n"
            + build_structured_output_instruction(schema_model)
        )
    )
        
        current_messages = messages if attempt == 0 else [*messages, retry_message]
        try:
            return await structured_llm.ainvoke(current_messages)
        except Exception as exc:
            last_error = exc
            if attempt == max_attempts - 1:
                raise
            await asyncio.sleep(min(5.0, 0.5 * 2**attempt))

    if last_error is not None:
        raise last_error
    raise RuntimeError("Structured output retry failed without an explicit error.")
