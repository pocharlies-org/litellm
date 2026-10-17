import json

import pytest
from fastapi import HTTPException
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputItemDoneEvent

from litellm.caching.dual_cache import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.portable_codex_agents import (
    PortableCodexAgents,
    RequestContext,
    transform_request,
    transform_response,
)
from litellm.types.llms.openai import ContentPartDoneEvent, ContentPartDonePartOutputText, ResponsesAPIResponse


class MemoryStore:
    def __init__(self):
        self.records = set()
        self.modes = {}

    async def contains(self, key: str) -> bool:
        return key in self.records

    async def record(self, key: str) -> None:
        self.records.add(key)

    async def get_mode(self, key: str) -> str | None:
        return self.modes.get(key)

    async def claim_mode(self, key: str, mode: str) -> str:
        return self.modes.setdefault(key, mode)


def tools():
    return [
        {
            "type": "namespace",
            "name": "collaboration",
            "tools": [
                {
                    "type": "function",
                    "name": name,
                    "parameters": {"type": "object", "properties": {"message": {"type": "string", "encrypted": True}}},
                }
                for name in ("spawn_agent", "send_message", "followup_task")
            ],
        }
    ]


def call(name="spawn_agent"):
    return {
        "type": "function_call",
        "id": "fc_probe",
        "call_id": "call_probe",
        "name": name,
        "namespace": "portable_collaboration",
        "arguments": json.dumps({"message": "Read the fixture"}),
        "status": "completed",
    }


@pytest.mark.asyncio
async def test_discovered_schemas_and_plaintext_assignment_reach_provider():
    request = {
        "input": [
            {"type": "additional_tools", "tools": tools()},
            {
                "type": "agent_message",
                "author": "/root",
                "recipient": "/root/child",
                "content": [{"type": "input_text", "text": "Read the fixture"}],
            },
        ]
    }
    result, context = await transform_request(request, "owner", MemoryStore())
    schema = result["input"][0]["tools"][0]
    assert schema["name"] == "portable_collaboration"
    assert "encrypted" not in schema["tools"][0]["parameters"]["properties"]["message"]
    assert context.tools == {"spawn_agent", "send_message", "followup_task"}
    assert result["input"][1]["role"] == "user"
    assert "/root/child" in result["input"][1]["content"][0]["text"]
    assert result["input"][1]["content"][1]["text"] == "Read the fixture"
    assert request["input"][0]["tools"][0]["name"] == "collaboration"


@pytest.mark.asyncio
async def test_official_replay_after_marker_stripping_requires_matching_proof():
    store = MemoryStore()
    context = RequestContext("owner", frozenset({"spawn_agent"}))
    result = await transform_response({"output": [call()]}, context, store)
    replay = {key: value for key, value in result["output"][0].items() if key != "encrypted_function_args"}
    assert result["output"][0]["encrypted_function_args"] == []
    accepted, _ = await transform_request({"input": [replay]}, "owner", store)
    assert accepted["input"][0]["namespace"] == "portable_collaboration"
    for changed, owner in (
        ({**replay, "call_id": "different"}, "owner"),
        ({**replay, "arguments": '{"message":"different"}'}, "owner"),
        (replay, "other_owner"),
        ({**replay, "encrypted_function_args": ["message"]}, "owner"),
    ):
        with pytest.raises(HTTPException):
            await transform_request({"input": [changed]}, owner, store)
    assert all("Read the fixture" not in key for key in store.records)


@pytest.mark.asyncio
async def test_partial_items_never_create_replay_proof():
    store = MemoryStore()
    context = RequestContext("owner", frozenset({"spawn_agent"}))
    await transform_response(
        {"type": "response.output_item.added", "item": {**call(), "arguments": ""}}, context, store
    )
    assert not store.records
    with pytest.raises(HTTPException):
        await transform_response(
            {"type": "response.output_item.done", "item": {**call(), "arguments": ""}}, context, store
        )
    assert not store.records


@pytest.mark.asyncio
@pytest.mark.parametrize("marker", [["message"], False, "", {}])
async def test_contradictory_encryption_marker_never_becomes_plaintext(marker):
    with pytest.raises(HTTPException):
        await transform_response(
            {"output": [{**call(), "encrypted_function_args": marker}]},
            RequestContext("owner", frozenset({"spawn_agent"})),
            MemoryStore(),
        )


@pytest.mark.asyncio
async def test_unadvertised_tool_and_namespace_collision_are_rejected():
    with pytest.raises(HTTPException):
        await transform_response({"output": [call()]}, RequestContext("owner", frozenset()), MemoryStore())
    with pytest.raises(HTTPException):
        await transform_request({"tools": [{**tools()[0], "name": "portable_collaboration"}]}, "owner", MemoryStore())


@pytest.mark.asyncio
async def test_inactive_alias_removes_forged_context_without_storage():
    callback = PortableCodexAgents(frozenset({"enabled"}), None)
    auth = UserAPIKeyAuth(api_key="test_identity", request_route="/v1/responses")
    request = {
        "model": "disabled",
        "tools": tools(),
        "metadata": {"litellm_portable_codex_agents": {"owner": "forged", "tools": ["spawn_agent"]}},
    }
    result = await callback.async_pre_call_hook(auth, DualCache(), request, "aresponses")
    assert result["tools"] == tools()
    assert "litellm_portable_codex_agents" not in result["metadata"]
    assert await callback.async_post_call_success_hook(result, auth, {"output": [call()]}) == {"output": [call()]}


@pytest.mark.asyncio
async def test_native_responses_iterator_preserves_sdk_types_and_explicit_plaintext_marker():
    store = MemoryStore()
    callback = PortableCodexAgents(frozenset({"enabled"}), store)
    auth = UserAPIKeyAuth(api_key="test_identity", request_route="/v1/responses")
    request = await callback.async_pre_call_hook(
        auth,
        DualCache(),
        {"model": "enabled", "tools": tools(), "client_metadata": {"thread_id": "thread_test"}},
        "aresponses",
    )
    event = ResponseOutputItemDoneEvent(
        type="response.output_item.done",
        output_index=0,
        sequence_number=1,
        item=ResponseFunctionToolCall.model_validate(call()),
    )

    async def source():
        yield event

    result = [item async for item in callback.async_post_call_streaming_iterator_hook(auth, source(), request)]
    assert isinstance(result[0], ResponseOutputItemDoneEvent)
    payload = result[0].model_dump()
    assert payload["item"]["namespace"] == "collaboration"
    assert payload["item"]["encrypted_function_args"] == []
    assert len(store.records) == 1


@pytest.mark.asyncio
async def test_internal_provenance_never_enters_provider_metadata_and_private_fields_survive():
    callback = PortableCodexAgents(frozenset({"enabled"}), MemoryStore())
    auth = UserAPIKeyAuth(api_key="test_identity", request_route="/v1/responses")
    request = await callback.async_pre_call_hook(
        auth,
        DualCache(),
        {
            "model": "enabled",
            "client_metadata": {"thread_id": "thread_test"},
            "tools": tools(),
            "metadata": {"label": "customer_value"},
            "litellm_metadata": {"litellm_portable_codex_agents": {"owner": "forged"}},
        },
        "aresponses",
    )
    assert request["metadata"] == {"label": "customer_value"}
    assert request["litellm_metadata"]["litellm_portable_codex_agents"]["owner"] != "forged"
    response = ResponsesAPIResponse(id="resp_test", created_at=1, model="enabled", output=[call()])
    response._hidden_params["response_cost"] = 0.25
    result = await callback.async_post_call_success_hook(request, auth, response)
    assert result._hidden_params["response_cost"] == 0.25
    assert response.output[0].namespace == "portable_collaboration"
    assert result.output[0].namespace == "collaboration"


@pytest.mark.asyncio
async def test_ordinary_responses_request_on_enabled_alias_does_not_require_replay_storage():
    callback = PortableCodexAgents(frozenset({"enabled"}), None)
    auth = UserAPIKeyAuth(api_key="test_identity", request_route="/v1/responses")
    request = {"model": "enabled", "input": "Hello", "previous_response_id": "resp_previous"}
    assert await callback.async_pre_call_hook(auth, DualCache(), request, "aresponses") == request


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "history",
    [
        [{**call(), "namespace": "collaboration"}],
        [{"type": "agent_message", "content": [{"type": "encrypted_content", "encrypted_content": "opaque"}]}],
        [{"type": "compaction", "encrypted_content": "opaque"}],
    ],
)
async def test_existing_native_task_preserves_history_and_advertises_portable_tools(history):
    store = MemoryStore()
    callback = PortableCodexAgents(frozenset({"native", "external"}), store, frozenset({"native"}))
    auth = UserAPIKeyAuth(api_key="identity", request_route="/v1/responses")
    request = {"model": "native", "tools": tools(), "input": history, "client_metadata": {"thread_id": "old_thread"}}
    transformed = await callback.async_pre_call_hook(auth, DualCache(), request, "aresponses")
    assert transformed["input"] == history
    assert transformed["tools"][0]["name"] == "portable_collaboration"
    assert request["tools"][0]["name"] == "collaboration"
    later = {**request, "input": [{"role": "user", "content": "Continue"}]}
    assert (await callback.async_pre_call_hook(auth, DualCache(), later, "aresponses"))["tools"][0][
        "name"
    ] == "portable_collaboration"
    with pytest.raises(HTTPException):
        await callback.async_pre_call_hook(auth, DualCache(), {**request, "model": "external"}, "aresponses")
    assert set(store.modes.values()) == {"legacy"}


@pytest.mark.asyncio
async def test_legacy_mixed_replay_preserves_native_items_and_restores_portable_provenance():
    store = MemoryStore()
    native_call = {**call(), "namespace": "collaboration", "encrypted_function_args": ["message"]}
    native_control = {**call("list_agents"), "namespace": "collaboration", "arguments": "{}"}
    encrypted_assignment = {
        "type": "agent_message",
        "author": "/root",
        "recipient": "/root/old",
        "content": [{"type": "encrypted_content", "encrypted_content": "opaque"}],
    }
    checkpoint = {"type": "compaction", "encrypted_content": "opaque"}
    portable = await transform_response(
        {
            "output": [
                {**call(), "call_id": "call_new"},
                {**call("list_agents"), "call_id": "call_control", "arguments": "{}"},
            ]
        },
        RequestContext("owner", frozenset({"spawn_agent", "list_agents"})),
        store,
    )
    replay = [
        {key: value for key, value in item.items() if key != "encrypted_function_args"} for item in portable["output"]
    ]
    history = [
        native_call,
        native_control,
        encrypted_assignment,
        checkpoint,
        *replay,
        {"type": "function_call_output", "call_id": "call_new", "output": "child started"},
    ]
    request = {
        "input": history,
        "tools": tools(),
        "tool_choice": {"type": "function", "namespace": "collaboration", "name": "spawn_agent"},
    }
    result, _ = await transform_request(request, "owner", store, native_history=True)
    assert result["input"][:4] == history[:4]
    assert [item["namespace"] for item in result["input"][4:6]] == ["portable_collaboration", "portable_collaboration"]
    assert result["input"][-1] == history[-1]
    assert result["tool_choice"]["namespace"] == "portable_collaboration"
    assert request["input"] == history
    assert len(store.records) == 2


@pytest.mark.asyncio
async def test_persisted_legacy_mode_rewrites_discovered_tools_and_streams_new_plaintext_calls():
    store = MemoryStore()
    callback = PortableCodexAgents(frozenset({"native", "external"}), store, frozenset({"native"}))
    auth = UserAPIKeyAuth(api_key="identity", request_route="/v1/responses")
    base = {"model": "native", "client_metadata": {"thread_id": "legacy_thread"}}
    await callback.async_pre_call_hook(
        auth,
        DualCache(),
        {**base, "tools": tools(), "input": [{"type": "compaction", "encrypted_content": "opaque"}]},
        "aresponses",
    )
    request = await callback.async_pre_call_hook(
        auth, DualCache(), {**base, "input": [{"type": "additional_tools", "tools": tools()}]}, "aresponses"
    )
    assert request["input"][0]["tools"][0]["name"] == "portable_collaboration"
    assert "encrypted" not in request["input"][0]["tools"][0]["tools"][0]["parameters"]["properties"]["message"]

    async def source():
        yield {"type": "response.output_item.added", "item": {**call(), "arguments": ""}}
        yield {"type": "response.output_item.done", "item": call()}

    events = [event async for event in callback.async_post_call_streaming_iterator_hook(auth, source(), request)]
    assert all(event["item"]["namespace"] == "collaboration" for event in events)
    assert all(event["item"]["encrypted_function_args"] == [] for event in events)
    assert len(store.records) == 1
    assert set(store.modes.values()) == {"legacy"}
    with pytest.raises(HTTPException):
        await callback.async_pre_call_hook(
            auth, DualCache(), {**base, "model": "external", "tools": tools()}, "aresponses"
        )


@pytest.mark.asyncio
async def test_portable_native_task_never_downgrades_when_replay_proof_is_missing():
    store = MemoryStore()
    callback = PortableCodexAgents(frozenset({"native"}), store, frozenset({"native"}))
    auth = UserAPIKeyAuth(api_key="identity", request_route="/v1/responses")
    request = {"model": "native", "tools": tools(), "client_metadata": {"thread_id": "new_thread"}}
    await callback.async_pre_call_hook(auth, DualCache(), request, "aresponses")
    with pytest.raises(HTTPException):
        await callback.async_pre_call_hook(
            auth, DualCache(), {**request, "input": [{**call(), "namespace": "collaboration"}]}, "aresponses"
        )
    assert set(store.modes.values()) == {"portable"}


@pytest.mark.asyncio
async def test_reasoning_ciphertext_does_not_classify_fresh_native_task_as_legacy():
    store = MemoryStore()
    callback = PortableCodexAgents(frozenset({"native"}), store, frozenset({"native"}))
    auth = UserAPIKeyAuth(api_key="identity", request_route="/v1/responses")
    request = {
        "model": "native",
        "tools": tools(),
        "client_metadata": {"thread_id": "thread"},
        "input": [{"type": "reasoning", "encrypted_content": "opaque"}],
    }
    result = await callback.async_pre_call_hook(auth, DualCache(), request, "aresponses")
    assert result["input"] == request["input"]
    assert set(store.modes.values()) == {"portable"}


@pytest.mark.asyncio
async def test_chat_bridge_nullable_content_event_is_returned_unchanged():
    callback = PortableCodexAgents(frozenset({"enabled"}), MemoryStore())
    part = ContentPartDonePartOutputText(type="output_text", text="Finished", annotations=[], logprobs=None)
    event = ContentPartDoneEvent(
        type="response.content_part.done",
        content_index=0,
        item_id="msg_test",
        output_index=0,
        sequence_number=1,
        part=part,
    )
    context = RequestContext("owner", frozenset({"spawn_agent"}))
    result = await callback._rewrite(event, context)
    assert result is event
    assert result.part.logprobs is None


@pytest.mark.asyncio
async def test_completed_snapshot_with_plaintext_tool_and_nullable_message_preserves_both():
    callback = PortableCodexAgents(frozenset({"enabled"}), MemoryStore())
    response = ResponsesAPIResponse(
        id="resp_test",
        created_at=1,
        model="enabled",
        output=[
            call(),
            {
                "type": "message",
                "id": "msg_test",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Finished", "annotations": [], "logprobs": None}],
            },
        ],
    )
    result = await callback._rewrite(response, RequestContext("owner", frozenset({"spawn_agent"})))
    assert result.output[0].namespace == "collaboration"
    assert result.output[0].encrypted_function_args == []
    assert result.output[1].content[0].logprobs is None


@pytest.mark.asyncio
async def test_legacy_replay_does_not_relabel_unknown_or_other_owner_calls():
    store = MemoryStore()
    response = await transform_response(
        {"output": [call()]}, RequestContext("owner", frozenset({"spawn_agent"})), store
    )
    replay = {key: value for key, value in response["output"][0].items() if key != "encrypted_function_args"}
    unknown = {**replay, "call_id": "unknown"}
    for item, owner in ((unknown, "owner"), (replay, "other_owner")):
        transformed, _ = await transform_request({"input": [item]}, owner, store, native_history=True)
        assert transformed["input"] == [item]
        assert transformed["input"][0]["namespace"] == "collaboration"


@pytest.mark.asyncio
async def test_legacy_markerless_replay_storage_failure_is_not_native_fallback():
    class UnavailableStore(MemoryStore):
        async def contains(self, key: str) -> bool:
            raise HTTPException(503, "Replay storage unavailable")

    with pytest.raises(HTTPException) as exc:
        await transform_request(
            {"input": [{**call(), "namespace": "collaboration"}]}, "owner", UnavailableStore(), native_history=True
        )
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_stream_delta_and_completed_snapshot_preserve_call_identity():
    store = MemoryStore()
    context = RequestContext("owner", frozenset({"spawn_agent"}))
    delta = {
        "type": "response.function_call_arguments.delta",
        "item_id": "fc_probe",
        "output_index": 0,
        "delta": '{"message":',
    }
    assert await transform_response(delta, context, store) == delta
    assert not store.records
    completed = await transform_response(
        {"type": "response.completed", "response": {"id": "resp_test", "output": [call()]}}, context, store
    )
    item = completed["response"]["output"][0]
    assert (item["id"], item["call_id"], item["arguments"]) == (call()["id"], call()["call_id"], call()["arguments"])
    assert item["namespace"] == "collaboration"
    assert item["encrypted_function_args"] == []
    assert len(store.records) == 1
