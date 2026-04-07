import asyncio
from typing import Any, Type, get_args, get_origin

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


def build_chat_model(base_url: str, model_name: str, temperature: float = 0.2, extra_body: dict = {"enable_thinking": False}) -> ChatOpenAI:
    return ChatOpenAI(base_url=base_url, model=model_name, temperature=temperature, extra_body=extra_body)


def _is_base_model_type(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    origin = get_origin(annotation)
    if origin is None:
        return annotation, False

    args = get_args(annotation)
    non_none_args = [arg for arg in args if arg is not type(None)]
    if len(non_none_args) != len(args):
        if len(non_none_args) == 1:
            return non_none_args[0], True
        return non_none_args[0] if non_none_args else Any, True
    return annotation, False


def _annotation_label(annotation: Any) -> str:
    annotation, is_optional = _unwrap_optional(annotation)
    origin = get_origin(annotation)

    if origin is None:
        if annotation is Any:
            label = "any"
        elif _is_base_model_type(annotation):
            label = annotation.__name__
        elif hasattr(annotation, "__name__"):
            label = annotation.__name__
        else:
            label = str(annotation)
        return f"{label} | null" if is_optional else label

    args = get_args(annotation)
    if origin in {list, tuple, set}:
        inner = _annotation_label(args[0] if args else Any)
        label = f"array[{inner}]"
    elif origin is dict:
        key_type = _annotation_label(args[0] if len(args) > 0 else Any)
        value_type = _annotation_label(args[1] if len(args) > 1 else Any)
        label = f"object[{key_type}, {value_type}]"
    else:
        label = str(annotation).replace("typing.", "")

    return f"{label} | null" if is_optional else label


def _schema_lines_for_model(schema_model: Type[BaseModel], indent: int = 0) -> list[str]:
    prefix = "  " * indent
    lines: list[str] = []

    for field_name, field_info in schema_model.model_fields.items():
        annotation, _ = _unwrap_optional(field_info.annotation)
        required_label = "required" if field_info.is_required() else "optional"
        type_label = _annotation_label(field_info.annotation)
        description = field_info.description or ""

        line = f"{prefix}- {field_name} ({type_label}, {required_label})"
        if description:
            line += f": {description}"
        lines.append(line)

        origin = get_origin(annotation)
        nested_model: Any = None
        if _is_base_model_type(annotation):
            nested_model = annotation
        elif origin in {list, tuple, set}:
            item_annotation = get_args(annotation)[0] if get_args(annotation) else Any
            item_annotation, _ = _unwrap_optional(item_annotation)
            if _is_base_model_type(item_annotation):
                nested_model = item_annotation

        if nested_model is not None:
            child_prefix = "  " * (indent + 1)
            if origin in {list, tuple, set}:
                lines.append(f"{child_prefix}Each array item must contain:")
            else:
                lines.append(f"{child_prefix}Nested object fields:")
            lines.extend(_schema_lines_for_model(nested_model, indent + 2))

    return lines


def build_structured_output_instruction(schema_model: Type[BaseModel]) -> str:
    lines = [
        "Return valid JSON only.",
        "Do not omit required fields.",
        "Follow this field structure exactly, including nested fields:",
        "",
        f"Root object: {schema_model.__name__}",
    ]
    lines.extend(_schema_lines_for_model(schema_model, indent=0))
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
