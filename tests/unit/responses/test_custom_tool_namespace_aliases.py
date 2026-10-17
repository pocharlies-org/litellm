import json

import pytest

from litellm.responses.litellm_completion_transformation.custom_tools import (
    build_tool_call_item_kwargs,
    extract_custom_tool_names,
)
from litellm.responses.litellm_completion_transformation.transformation import LiteLLMCompletionResponsesConfig
from litellm.types.utils import ChatCompletionMessageToolCall, Choices, Function, Message, ModelResponse

PATCH = "*** Begin Patch\n*** Add File: demo.txt\n+hello\n*** End Patch"
NAMESPACE = {"type": "namespace", "name": "functions", "tools": [{"type": "custom", "name": "apply_patch"}]}


@pytest.mark.parametrize("name", ["apply_patch", "functions__apply_patch"])
def test_custom_namespace_alias_restores_custom_response(name):
    response = ModelResponse(
        choices=[
            Choices(
                finish_reason="tool_calls",
                message=Message(
                    role="assistant",
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_patch",
                            type="function",
                            function=Function(name=name, arguments=json.dumps({"content": PATCH})),
                        )
                    ],
                ),
            )
        ]
    )
    items = LiteLLMCompletionResponsesConfig.transform_chat_completion_tools_to_responses_tools(
        response, responses_api_request={"tools": [NAMESPACE]}
    )
    assert len(items) == 1
    assert items[0].type == "custom_tool_call"
    assert items[0].name == "apply_patch"
    assert items[0].namespace == "functions"
    assert items[0].input == PATCH


@pytest.mark.parametrize("status", ["in_progress", "completed"])
def test_streaming_item_builder_classifies_bare_custom_alias(status):
    item = build_tool_call_item_kwargs(
        "call_patch", "apply_patch", json.dumps({"content": PATCH}), status, extract_custom_tool_names([NAMESPACE])
    )
    assert item["type"] == "custom_tool_call"
    assert item["input"] == (PATCH if status == "completed" else "")
    assert "arguments" not in item


@pytest.mark.parametrize(
    "other",
    [
        {"type": "function", "name": "apply_patch"},
        {"type": "namespace", "name": "other", "tools": [{"type": "function", "name": "apply_patch"}]},
        {"type": "namespace", "name": "other", "tools": [{"type": "custom", "name": "apply_patch"}]},
    ],
)
def test_ambiguous_custom_alias_is_not_guessed(other):
    names = extract_custom_tool_names([NAMESPACE, other])
    assert "apply_patch" not in names
    assert "functions__apply_patch" in names


def test_top_level_custom_keeps_its_identity_when_namespace_collides():
    names = extract_custom_tool_names([NAMESPACE, {"type": "custom", "name": "apply_patch"}])
    assert "apply_patch" in names
    mapping = LiteLLMCompletionResponsesConfig.namespace_tool_name_map(
        [NAMESPACE, {"type": "custom", "name": "apply_patch"}]
    )
    assert "apply_patch" not in mapping
