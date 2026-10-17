"""Bridge client-executed Responses tool discovery without changing its protocol."""

from collections.abc import Mapping, Sequence
from typing import Final

from openai.types.responses.response_tool_search_call import ResponseToolSearchCall
from pydantic import TypeAdapter

TOOL_SEARCH_NAME: Final = "tool_search"
_ARGUMENTS_ADAPTER: Final = TypeAdapter(dict[str, object])
_TOOLS_ADAPTER: Final = TypeAdapter(list[dict[str, object]])


def has_client_tool_search(tools: Sequence[Mapping[str, object]] | None) -> bool:
    return any(tool.get("type") == "tool_search" and tool.get("execution") == "client" for tool in tools or ())


def merge_discovered_tools(tools: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    """Combine discovery history and current definitions, keeping the latest schema.

    A namespace may be discovered in separate searches with different children.
    Merge those children too, instead of losing an earlier search's tools.
    """
    merged: Final[dict[tuple[object, object], Mapping[str, object]]] = {}  # mutable-ok: ordered merge
    for tool in tools:
        key = (tool.get("type"), tool.get("name"))
        previous = merged.get(key)
        children = tool.get("tools")
        prior_children = previous.get("tools") if previous is not None else None
        if tool.get("type") == "namespace" and isinstance(children, list) and isinstance(prior_children, list):
            merged[key] = {  # mutable-ok: these definitions are serialized by json.dumps downstream
                **tool,
                "tools": list(  # mutable-ok: downstream namespace descriptors require JSON arrays
                    merge_discovered_tools(
                        (*_TOOLS_ADAPTER.validate_python(prior_children), *_TOOLS_ADAPTER.validate_python(children))
                    )
                ),
            }
        else:
            merged[key] = tool
    return tuple(merged.values())


def build_tool_search_call(call_id: str, arguments: str, status: str) -> ResponseToolSearchCall:
    # Arguments are an object in Responses, unlike a function_call's JSON string.
    parsed: Final = _ARGUMENTS_ADAPTER.validate_json(arguments or "{}", strict=True)
    return ResponseToolSearchCall(
        type="tool_search_call",
        id=call_id if call_id.startswith("tsc_") else f"tsc_{call_id}",
        call_id=call_id,
        execution="client",
        arguments=parsed,
        status="completed" if status == "completed" else "in_progress",
    )
