"""Opt-in plaintext collaboration transport for official Codex clients."""

import hashlib
import json
import os
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Final, Protocol

from fastapi import HTTPException
from pydantic import BaseModel, TypeAdapter
from redis.asyncio import Redis
from redis.exceptions import RedisError

from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import CallTypesLiteral

_NAMESPACE: Final = "portable_collaboration"
_MESSAGE_TOOLS: Final = frozenset({"spawn_agent", "send_message", "followup_task"})
_CONTEXT: Final = "litellm_portable_codex_agents"
_MAPPING: Final = TypeAdapter(dict[str, object])
_ROUTES: Final = frozenset({"/responses", "/v1/responses", "/openai/responses", "/openai/v1/responses"})


class ReplayStore(Protocol):
    async def contains(self, key: str) -> bool: ...

    async def record(self, key: str) -> None: ...

    async def get_mode(self, key: str) -> str | None: ...

    async def claim_mode(self, key: str, mode: str) -> str: ...


class RedisReplayStore:
    def __init__(self, client: Redis) -> None:
        self.client = client

    async def contains(self, key: str) -> bool:
        try:
            return bool(await self.client.exists(key))
        except RedisError:
            raise HTTPException(503, "Portable agent replay storage is unavailable.") from None

    async def record(self, key: str) -> None:
        try:
            await self.client.set(key, "1")
        except RedisError:
            raise HTTPException(503, "Portable agent replay storage is unavailable.") from None

    async def get_mode(self, key: str) -> str | None:
        try:
            value: Final = await self.client.get(key)
        except RedisError:
            raise HTTPException(503, "Portable agent replay storage is unavailable.") from None
        if value is not None and value not in ("legacy", "portable"):
            raise HTTPException(503, "Portable agent thread mode is invalid.")
        return value

    async def claim_mode(self, key: str, mode: str) -> str:
        try:
            await self.client.set(key, mode, nx=True)
        except RedisError:
            raise HTTPException(503, "Portable agent replay storage is unavailable.") from None
        selected: Final = await self.get_mode(key)
        if selected is None:
            raise HTTPException(503, "Portable agent thread mode could not be recorded.")
        return selected


@dataclass(frozen=True, slots=True)
class RequestContext:
    owner: str
    tools: frozenset[str]


def _object(value: object) -> dict[str, object]:
    return _MAPPING.validate_python(value) if isinstance(value, Mapping) else {}


def _items(value: object) -> tuple[object, ...]:
    return tuple(value) if isinstance(value, list) else ()


def _has_collaboration_tool(value: object) -> bool:
    tool: Final = _object(value)
    return (tool.get("type") == "namespace" and tool.get("name") in ("collaboration", _NAMESPACE)) or any(
        _has_collaboration_tool(item) for item in _items(tool.get("tools"))
    )


def _needs_transport(data: Mapping[str, object]) -> bool:
    return any(_has_collaboration_tool(tool) for tool in _items(data.get("tools"))) or any(
        _object(item).get("type") == "agent_message"
        or _object(item).get("namespace") == "collaboration"
        or (_object(item).get("type") == "additional_tools" and _has_collaboration_tool(item))
        for item in _items(data.get("input"))
    )


def _owner(auth: UserAPIKeyAuth) -> str:
    identity: Final = auth.api_key or auth.user_id
    if not identity:
        raise HTTPException(400, "Portable agent transport requires an authenticated identity.")
    return hashlib.sha256(json.dumps((auth.team_id, identity)).encode()).hexdigest()


def _mode_key(data: Mapping[str, object], owner: str) -> str:
    metadata: Final = _object(data.get("client_metadata"))
    headers: Final = _object(_object(data.get("proxy_server_request")).get("headers"))
    thread: Final = metadata.get("thread_id") or headers.get("thread_id")
    if not isinstance(thread, str) or not thread:
        raise HTTPException(400, "Portable collaboration requires the stable Codex thread_id.")
    return "litellm:portable_codex_agents:mode:v1:" + hashlib.sha256(json.dumps((owner, thread)).encode()).hexdigest()


async def _has_legacy_history(data: Mapping[str, object], owner: str, store: ReplayStore) -> bool:
    for value in _items(data.get("input")):
        item = _object(value)
        if item.get("type") in ("compaction", "context_compaction"):
            return True
        if item.get("type") == "agent_message" and any(
            _object(block).get("type") == "encrypted_content" for block in _items(item.get("content"))
        ):
            return True
        if (
            item.get("type") == "function_call"
            and item.get("namespace") == "collaboration"
            and item.get("name") in _MESSAGE_TOOLS
        ):
            marker = item.get("encrypted_function_args")
            if marker not in (None, []):
                return True
            if marker is None and not await store.contains(_replay_key(owner, item)):
                return True
    return False


def _replay_key(owner: str, item: Mapping[str, object]) -> str:
    call_id: Final = item.get("call_id")
    arguments: Final = item.get("arguments")
    if not isinstance(call_id, str) or not call_id or not isinstance(arguments, str):
        raise HTTPException(400, "Portable agent call is missing its identity or arguments.")
    digest: Final = hashlib.sha256(json.dumps((owner, item.get("name"), call_id, arguments)).encode()).hexdigest()
    return "litellm:portable_codex_agents:v1:" + digest


def _rewrite_tool(value: object) -> tuple[object, frozenset[str]]:
    tool: Final = _object(value)
    if tool.get("type") == "namespace" and tool.get("name") == _NAMESPACE:
        raise HTTPException(400, "Portable collaboration namespace is reserved by the adapter.")
    if tool.get("type") != "namespace":
        return value, frozenset()
    if tool.get("name") != "collaboration":
        nested: Final = tuple(_rewrite_tool(item) for item in _items(tool.get("tools")))
        return {**tool, "tools": [item for item, _ in nested]}, frozenset(name for _, names in nested for name in names)
    functions: Final = tuple(_object(item) for item in _items(tool.get("tools")))
    return {
        **tool,
        "name": _NAMESPACE,
        "tools": [_plaintext_schema(function) for function in functions],
    }, frozenset(str(function["name"]) for function in functions if isinstance(function.get("name"), str))


def _plaintext_schema(function: dict[str, object]) -> dict[str, object]:
    if function.get("name") not in _MESSAGE_TOOLS:
        return function
    parameters: Final = _object(function.get("parameters"))
    properties: Final = _object(parameters.get("properties"))
    message: Final = _object(properties.get("message"))
    if message.get("type") != "string":
        raise HTTPException(400, "Portable collaboration requires a string message schema.")
    return {
        **function,
        "parameters": {
            **parameters,
            "properties": {
                **properties,
                "message": {key: value for key, value in message.items() if key != "encrypted"},
            },
        },
    }


async def transform_request(
    original: Mapping[str, object], owner: str, store: ReplayStore, *, native_history: bool = False
) -> tuple[dict[str, object], RequestContext]:
    if original.get("previous_response_id"):
        raise HTTPException(400, "Portable agent transport requires full history; previous_response_id is unsupported.")
    tools: Final = tuple(_rewrite_tool(tool) for tool in _items(original.get("tools")))
    discovered: Final = tuple(
        names
        for item in _items(original.get("input"))
        if _object(item).get("type") == "additional_tools"
        for _, names in (_rewrite_tool(tool) for tool in _items(_object(item).get("tools")))
    )
    input_items: Final = [
        await _request_item(item, owner, store, native_history=native_history) for item in _items(original.get("input"))
    ]
    choice: Final = _object(original.get("tool_choice"))
    rewritten_choice: Final = (
        {**choice, "namespace": _NAMESPACE} if choice.get("namespace") == "collaboration" else choice
    )
    return {
        **original,
        **({"tools": [tool for tool, _ in tools]} if "tools" in original else {}),
        **({"input": input_items} if isinstance(original.get("input"), list) else {}),
        **({"tool_choice": rewritten_choice} if choice else {}),
    }, RequestContext(
        owner, frozenset(name for names in (*discovered, *(names for _, names in tools)) for name in names)
    )


async def _request_item(value: object, owner: str, store: ReplayStore, *, native_history: bool = False) -> object:
    item: Final = _object(value)
    if item.get("type") == "additional_tools":
        return {**item, "tools": [_rewrite_tool(tool)[0] for tool in _items(item.get("tools"))]}
    if item.get("type") == "function_call" and item.get("namespace") == "collaboration":
        if native_history:
            native_marker: Final = item.get("encrypted_function_args")
            if native_marker not in (None, []):
                return value
            if native_marker is None and not await store.contains(_replay_key(owner, item)):
                return value
        if item.get("name") in _MESSAGE_TOOLS:
            marker: Final = item.get("encrypted_function_args")
            if marker not in (None, []):
                raise HTTPException(400, "Encrypted collaboration history cannot use portable transport.")
            if marker is None and not await store.contains(_replay_key(owner, item)):
                raise HTTPException(400, "Portable replay proof is unavailable. Start a fresh task on this endpoint.")
        return {**item, "namespace": _NAMESPACE}
    if item.get("type") != "agent_message":
        return value
    content: Final = item.get("content")
    blocks: Final = ({"type": "input_text", "text": content},) if isinstance(content, str) else _items(content)
    if not blocks or any(_object(block).get("type") != "input_text" for block in blocks):
        if native_history:
            return value
        raise HTTPException(400, "Encrypted agent assignments require a fresh task using portable transport.")
    return {
        "type": "message",
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": json.dumps({"author": item.get("author"), "recipient": item.get("recipient")}),
            },
            *blocks,
        ],
    }


async def _response_item(value: object, context: RequestContext, store: ReplayStore, complete: bool) -> object:
    item: Final = _object(value)
    if item.get("type") != "function_call" or item.get("namespace") != _NAMESPACE:
        return value
    if item.get("name") not in context.tools:
        raise HTTPException(502, "Upstream returned an unadvertised portable collaboration tool.")
    if item.get("name") not in _MESSAGE_TOOLS:
        if complete:
            await store.record(_replay_key(context.owner, item))
        return {**item, "namespace": "collaboration"}
    if item.get("encrypted_function_args") not in (None, []):
        raise HTTPException(502, "Upstream returned encrypted portable collaboration arguments.")
    if complete:
        try:
            arguments: Final = json.loads(str(item.get("arguments")))
        except ValueError:
            raise HTTPException(502, "Upstream returned invalid portable collaboration arguments.") from None
        if not isinstance(_object(arguments).get("message"), str):
            raise HTTPException(502, "Upstream returned a portable call without a plaintext message string.")
        await store.record(_replay_key(context.owner, item))
    return {**item, "namespace": "collaboration", "encrypted_function_args": []}


async def transform_response(
    value: Mapping[str, object], context: RequestContext, store: ReplayStore
) -> dict[str, object]:
    direct: Final = _object(await _response_item(value, context, store, True))
    nested: Final = _object(direct.get("response"))
    return {
        **direct,
        **(
            {
                "item": await _response_item(
                    direct["item"], context, store, direct.get("type") == "response.output_item.done"
                )
            }
            if "item" in direct
            else {}
        ),
        **(
            {"output": [await _response_item(item, context, store, True) for item in _items(direct["output"])]}
            if "output" in direct
            else {}
        ),
        **({"response": await transform_response(nested, context, store)} if nested else {}),
    }


class PortableCodexAgents(CustomLogger):
    def __init__(
        self, models: frozenset[str], store: ReplayStore | None, native_models: frozenset[str] = frozenset()
    ) -> None:
        super().__init__()
        self.models = models
        self.store = store
        self.native_models = native_models

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, object],
        call_type: CallTypesLiteral,
    ) -> dict[str, object]:
        metadata: Final = _object(data.get("metadata"))
        clean_metadata: Final = {key: value for key, value in metadata.items() if key != _CONTEXT}
        internal: Final = _object(data.get("litellm_metadata"))
        clean_internal: Final = {key: value for key, value in internal.items() if key != _CONTEXT}
        if call_type != "aresponses" or data.get("model") not in self.models or not _needs_transport(data):
            return {
                **data,
                **({"metadata": clean_metadata} if "metadata" in data else {}),
                **({"litellm_metadata": clean_internal} if "litellm_metadata" in data else {}),
            }
        if self.store is None:
            raise HTTPException(503, "Portable Codex transport requires PORTABLE_CODEX_REDIS_URL.")
        owner: Final = _owner(user_api_key_dict)
        key: Final = _mode_key(data, owner)
        existing: Final = await self.store.get_mode(key)
        legacy: Final = existing is None and await _has_legacy_history(data, owner, self.store)
        if legacy and data.get("model") not in self.native_models:
            raise HTTPException(
                400, "Legacy encrypted collaboration requires a native provider or a fresh portable task."
            )
        mode: Final = existing or await self.store.claim_mode(key, "legacy" if legacy else "portable")
        if mode == "legacy":
            if data.get("model") not in self.native_models:
                raise HTTPException(
                    400, "This legacy task requires a native Responses provider. Start a fresh portable task."
                )
        transformed, context = await transform_request(data, owner, self.store, native_history=mode == "legacy")
        return {
            **transformed,
            **({"metadata": clean_metadata} if "metadata" in data else {}),
            "litellm_metadata": {**clean_internal, _CONTEXT: {"owner": context.owner, "tools": sorted(context.tools)}},
        }

    def _context(self, data: Mapping[str, object], auth: UserAPIKeyAuth) -> RequestContext | None:
        fields: Final = _object(_object(data.get("litellm_metadata")).get(_CONTEXT))
        if not fields or auth.request_route not in _ROUTES:
            return None
        if fields.get("owner") != _owner(auth):
            raise HTTPException(400, "Portable transport request identity changed.")
        return RequestContext(str(fields["owner"]), frozenset(str(name) for name in _items(fields.get("tools"))))

    async def _rewrite(self, response: object, context: RequestContext) -> object:
        if self.store is None:
            raise HTTPException(503, "Portable Codex replay storage is not configured.")
        if isinstance(response, BaseModel):
            original: Final = response.model_dump(mode="json")
            rewritten: Final = await transform_response(original, context, self.store)
            if rewritten == original:
                return response
            validated: Final = type(response).model_validate(rewritten)
            return response.model_copy(deep=True, update=validated.__dict__)
        if isinstance(response, Mapping):
            return await transform_response(_object(response), context, self.store)
        return response

    async def async_post_call_success_hook(
        self,
        data: dict[str, object],
        user_api_key_dict: UserAPIKeyAuth,
        response: object,
    ) -> object:
        context: Final = self._context(data, user_api_key_dict)
        return await self._rewrite(response, context) if context else response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: AsyncIterator[object],
        request_data: dict[str, object],
    ) -> AsyncIterator[object]:
        context: Final = self._context(request_data, user_api_key_dict)
        async for chunk in response:
            yield await self._rewrite(chunk, context) if context else chunk


def _configured_callback() -> PortableCodexAgents:
    models: Final = frozenset(
        name.strip() for name in os.environ.get("PORTABLE_CODEX_MODELS", "").split(",") if name.strip()
    )
    url: Final = os.environ.get("PORTABLE_CODEX_REDIS_URL", "")
    host: Final = os.environ.get("REDIS_HOST")
    client: Final = (
        Redis.from_url(url, decode_responses=True, socket_timeout=5, socket_connect_timeout=5)
        if url
        else Redis(
            host=host,
            port=int(os.environ.get("REDIS_PORT", "6379")),
            username=os.environ.get("REDIS_USERNAME") or None,
            password=os.environ.get("REDIS_PASSWORD") or None,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
        )
        if host
        else None
    )
    native_models: Final = frozenset(
        name.strip() for name in os.environ.get("PORTABLE_CODEX_NATIVE_MODELS", "").split(",") if name.strip()
    )
    return PortableCodexAgents(models, RedisReplayStore(client) if client else None, native_models)


portable_codex_agents: Final = _configured_callback()
