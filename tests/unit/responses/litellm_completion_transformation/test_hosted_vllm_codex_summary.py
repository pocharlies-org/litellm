import asyncio
from unittest.mock import AsyncMock

import pytest

import litellm
from litellm.responses.litellm_completion_transformation import hosted_vllm_codex_summary as summary
from litellm.responses.litellm_completion_transformation.streaming_iterator import LiteLLMCompletionStreamingIterator
from litellm.responses.litellm_completion_transformation.transformation import LiteLLMCompletionResponsesConfig
from litellm.types.utils import Delta, ModelResponse, ModelResponseStream, StreamingChoices, Usage


class Stream:
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.logging_obj = None
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.chunks)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self):
        self.closed = True


def chunk(reasoning=None, content=None, finish=None, tools=None):
    return ModelResponseStream(
        id="chatcmpl-test",
        model="test",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(reasoning_content=reasoning, content=content, tool_calls=tools),
                finish_reason=finish,
            )
        ],
    )


def bridge(chunks):
    return LiteLLMCompletionStreamingIterator(
        model="test",
        litellm_custom_stream_wrapper=Stream(chunks),
        request_input="secret question",
        responses_api_request={"reasoning": {"summary": "auto"}},
        custom_llm_provider="hosted_vllm_codex",
    )


def test_summary_request_is_strictly_opt_in():
    assert summary.summary_requested("hosted_vllm_codex", {"reasoning": {"summary": "auto"}})
    for provider, request in [
        ("hosted_vllm", {"reasoning": {"summary": "auto"}}),
        ("hosted_vllm_codex", {}),
        ("hosted_vllm_codex", {"reasoning": {"effort": "high"}}),
    ]:
        assert not summary.summary_requested(provider, request)


def test_request_contains_only_reasoning_and_connection_settings():
    args = summary.summary_args(
        {
            "model": "hosted_vllm_codex/qwen",
            "api_base": "http://localhost/v1",
            "api_key": "test",
            "messages": [{"role": "user", "content": "private"}],
            "tools": [1],
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
            "reasoning_effort": "high",
            "max_tokens": 9000,
        },
        "original reasoning",
    )
    assert args["model"] == "qwen" and args["custom_llm_provider"] == "hosted_vllm"
    assert args["api_key"] == "test" and args["max_tokens"] == 500
    assert args["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert args["messages"][1]["content"] == "original reasoning"
    assert "tools" not in args and "reasoning_effort" not in args and "thinking" not in args


@pytest.mark.asyncio
async def test_concurrent_summary_preserves_raw_reasoning_tools_ids_and_usage(monkeypatch):
    release = asyncio.Event()
    started = asyncio.Event()

    async def request(**kwargs):
        assert kwargs["messages"][1]["content"] == "one two"
        started.set()

        async def stream():
            await release.wait()
            yield chunk(content="Short summary.")
            yield ModelResponseStream(
                id="summary", choices=[], usage=Usage(prompt_tokens=7, completion_tokens=3, total_tokens=10)
            )

        return stream()

    monkeypatch.setattr(litellm, "acompletion", request)
    source = bridge(
        [
            chunk(reasoning="one "),
            chunk(),
            chunk(reasoning="two"),
            chunk(
                tools=[
                    {
                        "index": 0,
                        "id": "call_codex_summary_concurrency",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ]
            ),
            chunk(finish="tool_calls"),
        ]
    )
    wrapped = summary.HostedVLLMCodexSummaryStream(source, {"model": "test"})

    class SignalPrimary(Stream):
        async def __anext__(self):
            event = await super().__anext__()
            if event.choices[0].finish_reason:
                await started.wait()
                release.set()
            return event

    source.litellm_custom_stream_wrapper = SignalPrimary(source.litellm_custom_stream_wrapper.chunks)
    events = [event async for event in wrapped]
    assert release.is_set(), "Primary must be consumed while summary delivery is pending"
    summary_end = next(i for i, event in enumerate(events) if event.type == "response.reasoning_summary_part.done")
    tool_start = next(i for i, event in enumerate(events) if event.type == "response.function_call_arguments.delta")
    assert summary_end < tool_start
    assert [e.sequence_number for e in events] == list(range(len(events)))
    completed = events[-1].response
    reasoning = next(item for item in completed.output if item.type == "reasoning")
    assert reasoning.content[0].text == "one two"
    assert reasoning.summary[0].text == "Short summary."
    summary_events = [e for e in events if e.type.startswith("response.reasoning_summary_")]
    assert all(e.item_id == reasoning.id and e.output_index == 0 and e.summary_index == 0 for e in summary_events)
    assert next(i for i, e in enumerate(events) if e.type == "response.reasoning_summary_part.done") < next(
        i for i, e in enumerate(events) if e.type == "response.output_item.done" and e.item.type == "reasoning"
    )
    assert completed._hidden_params["reasoning_summary_usage"]["total_tokens"] == 10
    assert source.completed_response is wrapped.completed_response is events[-1]
    assert source.litellm_custom_stream_wrapper.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "exception"])
async def test_auxiliary_failure_keeps_original_output(monkeypatch, failure):
    async def request(**kwargs):
        if failure == "exception":
            raise RuntimeError("backend failed")
        await asyncio.Event().wait()

    monkeypatch.setattr(litellm, "acompletion", request)
    monkeypatch.setattr(summary, "SUMMARY_TIMEOUT_SECONDS", 0.01)
    events = [
        e
        async for e in summary.HostedVLLMCodexSummaryStream(
            bridge([chunk(reasoning="raw"), chunk(content="answer"), chunk(finish="stop")]), {"model": "test"}
        )
    ]
    assert events[-1].type == "response.completed"
    assert events[-1].response.output[0].content[0].text == "raw"
    assert events[-1].response.output[0].summary == []
    assert not any(e.type.startswith("response.reasoning_summary_") for e in events)


@pytest.mark.asyncio
async def test_disconnect_cancels_auxiliary_and_closes_primary(monkeypatch):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def request(**kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(litellm, "acompletion", request)
    source = bridge([chunk(reasoning="raw"), chunk(content="answer"), chunk(finish="stop")])
    wrapped = summary.HostedVLLMCodexSummaryStream(source, {"model": "test"})
    async for event in wrapped:
        if event.type == "response.reasoning_text.done":
            await started.wait()
            await wrapped.aclose()
            break
    assert cancelled.is_set() and source.litellm_custom_stream_wrapper.closed


@pytest.mark.asyncio
async def test_nonstream_summary_keeps_reasoning_and_accounts_usage(monkeypatch):
    monkeypatch.setattr(
        litellm,
        "acompletion",
        AsyncMock(
            return_value=Stream(
                [
                    chunk(content="Brief"),
                    ModelResponseStream(choices=[], usage=Usage(prompt_tokens=2, completion_tokens=1, total_tokens=3)),
                ]
            )
        ),
    )
    response = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
        chat_completion_response=ModelResponse(
            model="test",
            choices=[{"message": {"role": "assistant", "content": "Answer", "reasoning_content": "raw"}}],
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        ),
        request_input="question",
        responses_api_request={},
    )
    await summary.summarize_response(response, {"model": "test"})
    assert response.output[0].summary[0].text == "Brief"
    assert response.output[0].content[0].text == "raw"
    assert response.usage.total_tokens == 30
    assert response._hidden_params["reasoning_summary_usage"]["total_tokens"] == 3


def test_sync_stream_summary(monkeypatch):
    monkeypatch.setattr(litellm, "acompletion", AsyncMock(return_value=Stream([chunk(content="Brief")])))
    wrapped = summary.HostedVLLMCodexSummaryStream(
        bridge([chunk(reasoning="raw"), chunk(content="answer"), chunk(finish="stop")]), {"model": "test"}, sync=True
    )
    events = list(wrapped)
    assert events[-1].response.output[0].summary[0].text == "Brief"


@pytest.mark.asyncio
async def test_capacity_exhaustion_skips_auxiliary_without_waiting(monkeypatch):
    import threading

    semaphore = threading.BoundedSemaphore(1)
    semaphore.acquire()
    monkeypatch.setattr(summary, "_SUMMARY_CAPACITY", semaphore)
    mock = AsyncMock()
    monkeypatch.setattr(litellm, "acompletion", mock)
    events = [
        e
        async for e in summary.HostedVLLMCodexSummaryStream(
            bridge([chunk(reasoning="raw"), chunk(content="answer"), chunk(finish="stop")]), {"model": "test"}
        )
    ]
    assert events[-1].type == "response.completed"
    mock.assert_not_called()


@pytest.mark.asyncio
async def test_summary_deltas_arrive_before_primary_finishes(monkeypatch):
    release = asyncio.Event()
    source = bridge([chunk(reasoning="raw"), chunk(content="answer"), chunk(finish="stop")])

    class PausedStream(Stream):
        async def __anext__(self):
            result = await super().__anext__()
            if result.choices[0].finish_reason:
                await release.wait()
            return result

    source.litellm_custom_stream_wrapper = PausedStream(
        [chunk(reasoning="raw"), chunk(content="answer"), chunk(finish="stop")]
    )
    monkeypatch.setattr(litellm, "acompletion", AsyncMock(return_value=Stream([chunk(content="Brief")])))
    events = []
    async with asyncio.timeout(2):
        async for event in summary.HostedVLLMCodexSummaryStream(source, {"model": "test"}):
            events.append(event)
            if event.type == "response.reasoning_summary_text.delta":
                release.set()
    assert release.is_set() and events[-1].type == "response.completed"


@pytest.mark.asyncio
async def test_close_before_iteration_closes_primary():
    source = bridge([])
    wrapper = summary.HostedVLLMCodexSummaryStream(source, {"model": "test"})
    await wrapper.aclose()
    await wrapper.aclose()
    assert source.litellm_custom_stream_wrapper.closed


def test_sync_close_before_iteration_closes_primary():
    source = bridge([])
    wrapper = summary.HostedVLLMCodexSummaryStream(source, {"model": "test"}, sync=True)
    wrapper.close()
    assert source.litellm_custom_stream_wrapper.closed


def test_auxiliary_attribution_excludes_parent_reservation_and_logging():
    args = summary.summary_args(
        {
            "model": "test",
            "user": "end-user",
            "litellm_call_id": "parent-call",
            "metadata": {
                "user_api_key": "hashed-key",
                "user_api_key_user_id": "owner",
                "user_api_key_team_id": "team",
                "user_api_key_budget_reservation": "parent-reservation",
                "user_api_key_auth": {"secret": "parent"},
                "other": "private",
            },
        },
        "reasoning",
    )
    assert args["metadata"] == {
        "user_api_key": "hashed-key",
        "user_api_key_user_id": "owner",
        "user_api_key_team_id": "team",
        "hosted_vllm_codex_auxiliary": "reasoning_summary",
    }
    assert args["user"] == "end-user" and "litellm_call_id" not in args


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,requested,calls",
    [("hosted_vllm_codex", True, 2), ("hosted_vllm_codex", False, 1), ("hosted_vllm", True, 1)],
)
async def test_handler_nonstream_enables_only_opted_in_provider(monkeypatch, provider, requested, calls):
    from litellm.responses.litellm_completion_transformation.handler import LiteLLMCompletionTransformationHandler

    main = ModelResponse(
        model="test", choices=[{"message": {"role": "assistant", "content": "Answer", "reasoning_content": "raw"}}]
    )
    mock = AsyncMock(side_effect=[main, Stream([chunk(content="Brief")])])
    monkeypatch.setattr(litellm, "acompletion", mock)
    result = await LiteLLMCompletionTransformationHandler().async_response_api_handler(
        litellm_completion_request={"model": "test", "custom_llm_provider": provider},
        request_input="question",
        responses_api_request={"reasoning": {"summary": "auto"}} if requested else {},
    )
    assert mock.call_count == calls
    assert bool(result.output[0].summary) == (calls == 2)


def test_sync_auxiliary_real_stream_keeps_delayed_logging_alive(monkeypatch):
    import threading
    from unittest.mock import MagicMock

    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

    logged = threading.Event()
    logging = MagicMock()
    logging.model_call_details = {"custom_llm_provider": "hosted_vllm", "litellm_params": {}}
    logging.call_type = "acompletion"
    logging.stream_options = {"include_usage": True}
    logging.messages = [{"role": "user", "content": "reasoning"}]
    logging.completion_start_time = None
    logging._llm_caching_handler = None
    logging._on_deferred_stream_complete = None

    async def delayed_log(*args, **kwargs):
        await asyncio.sleep(0.05)
        logged.set()

    logging.dispatch_success_handlers = delayed_log
    actual = CustomStreamWrapper(
        completion_stream=Stream([chunk(content="Brief"), chunk(finish="stop")]),
        model="test",
        logging_obj=logging,
        custom_llm_provider="hosted_vllm",
    )
    monkeypatch.setattr(litellm, "acompletion", AsyncMock(return_value=actual))
    wrapped = summary.HostedVLLMCodexSummaryStream(
        bridge([chunk(reasoning="raw"), chunk(content="answer"), chunk(finish="stop")]), {"model": "test"}, sync=True
    )
    events = list(wrapped)
    assert events[-1].response.output[0].summary[0].text == "Brief"
    assert logged.wait(2), "Auxiliary spend callback must survive the sync response's return"
    with pytest.raises(StopIteration):
        next(wrapped)
    with pytest.raises(StopIteration):
        next(wrapped)


@pytest.mark.asyncio
async def test_completed_snapshot_preserves_primary_hidden_metadata_and_identity(monkeypatch):
    monkeypatch.setattr(litellm, "acompletion", AsyncMock(return_value=Stream([chunk(content="Brief")])))
    source = bridge([chunk(reasoning="raw"), chunk(content="answer"), chunk(finish="stop")])
    original_emit = source._emit_response_completed_event
    captured = []
    hidden = {
        "custom_llm_provider": "hosted_vllm_codex",
        "response_cost": 0.123,
        "additional_headers": {"x-source-marker": "primary"},
    }

    def capture_completed(response):
        event = original_emit(response)
        assert event is not None
        event.response._hidden_params.update(hidden)
        captured.append(event)
        return event

    monkeypatch.setattr(source, "_emit_response_completed_event", capture_completed)
    wrapped = summary.HostedVLLMCodexSummaryStream(source, {"model": "test"})
    events = [event async for event in wrapped]
    assert len(captured) == 1
    assert events[-1] is captured[0] is source.completed_response is wrapped.completed_response
    assert {key: events[-1].response._hidden_params[key] for key in hidden} == hidden
    reasoning = next(item for item in events[-1].response.output if item.type == "reasoning")
    assert reasoning.content[0].text == "raw"
    assert reasoning.summary[0].text == "Brief"


@pytest.mark.asyncio
async def test_disconnect_while_primary_delivery_is_buffered_closes_both_streams(monkeypatch):
    primary_waiting = asyncio.Event()
    auxiliary_started = asyncio.Event()
    auxiliary_cancelled = asyncio.Event()

    class PausedPrimary(Stream):
        async def __anext__(self):
            event = await super().__anext__()
            if event.choices[0].finish_reason:
                primary_waiting.set()
                await asyncio.Event().wait()
            return event

    async def request(**kwargs):
        async def response():
            auxiliary_started.set()
            try:
                await asyncio.Event().wait()
                yield chunk(content="Never delivered")
            finally:
                auxiliary_cancelled.set()

        return response()

    monkeypatch.setattr(litellm, "acompletion", request)
    source = bridge([])
    primary = PausedPrimary([chunk(reasoning="raw"), chunk(content="buffered"), chunk(finish="stop")])
    source.litellm_custom_stream_wrapper = primary
    delivered = []
    wrapper = summary.HostedVLLMCodexSummaryStream(source, {"model": "test"})

    async def consume():
        async for event in wrapper:
            delivered.append(event)

    consumer = asyncio.create_task(consume())
    async with asyncio.timeout(3):
        await primary_waiting.wait()
        await auxiliary_started.wait()
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
    assert auxiliary_cancelled.is_set() and primary.closed
    assert not any(event.type == "response.output_text.delta" for event in delivered)


@pytest.mark.asyncio
@pytest.mark.parametrize("partial_summary", [False, True])
async def test_primary_failure_drains_received_events_then_preserves_original_error(monkeypatch, partial_summary):
    injected_error = RuntimeError("synthetic primary failure")
    allow_error = asyncio.Event()
    auxiliary_started = asyncio.Event()
    auxiliary_closed = asyncio.Event()
    received = []

    class BrokenPrimary(Stream):
        async def __anext__(self):
            try:
                return next(self.chunks)
            except StopIteration:
                await allow_error.wait()
                raise injected_error

    class RecordingBridge(LiteLLMCompletionStreamingIterator):
        async def __anext__(self):
            event = await super().__anext__()
            received.append(event)
            return event

    async def request(**kwargs):
        async def response():
            auxiliary_started.set()
            try:
                if partial_summary:
                    yield chunk(content="Partial before failure")
                await asyncio.Event().wait()
            finally:
                auxiliary_closed.set()

        return response()

    monkeypatch.setattr(litellm, "acompletion", request)
    primary = BrokenPrimary([chunk(reasoning="raw"), chunk(content="visible-before-failure")])
    source = RecordingBridge(
        model="test",
        litellm_custom_stream_wrapper=primary,
        request_input="Synthetic failure check",
        responses_api_request={},
        custom_llm_provider="hosted_vllm_codex",
    )
    wrapper = summary.HostedVLLMCodexSummaryStream(source, {"model": "test"})
    delivered = []

    async def drain():
        async for event in wrapper:
            delivered.append(event)
            if event.type == (
                "response.reasoning_summary_text.delta" if partial_summary else "response.reasoning_text.done"
            ):
                await auxiliary_started.wait()
                allow_error.set()

    async with asyncio.timeout(3):
        with pytest.raises(RuntimeError) as failure:
            await drain()
    assert failure.value is injected_error
    assert [id(event) for event in delivered if not event.type.startswith("response.reasoning_summary_")] == [
        id(event) for event in received
    ], "Already received primary events must survive the upstream error in original FIFO order"
    assert (
        "".join(event.delta for event in delivered if event.type == "response.output_text.delta")
        == "visible-before-failure"
    )
    assert not any(event.type == "response.completed" for event in delivered)
    assert wrapper.completed_response is None
    assert auxiliary_closed.is_set() and primary.closed
    assert [event.sequence_number for event in delivered] == list(range(len(delivered)))
    terminals = [event for event in delivered if event.type == "response.reasoning_summary_text.done"]
    assert len(terminals) == int(partial_summary)
    reasoning_done = next(
        event for event in delivered if event.type == "response.output_item.done" and event.item.type == "reasoning"
    )
    if partial_summary:
        assert terminals[0].text == "Partial before failure"
        assert reasoning_done.model_dump()["item"]["summary"][0]["text"] == "Partial before failure"
        assert delivered.index(terminals[0]) < delivered.index(reasoning_done)
    else:
        assert reasoning_done.item.summary == []


@pytest.mark.asyncio
async def test_consumer_cancellation_during_overflow_auxiliary_cleanup_propagates(monkeypatch):
    closing = asyncio.Event()
    started = asyncio.Event()
    primary = Stream([chunk(reasoning="raw"), chunk(content="buffered"), chunk(finish="stop")])

    class SlowClosingSummary:
        def __aiter__(self):
            return self

        async def __anext__(self):
            started.set()
            await asyncio.Event().wait()

        async def aclose(self):
            closing.set()
            await asyncio.Event().wait()

    async def request(**kwargs):
        return SlowClosingSummary()

    monkeypatch.setattr(litellm, "acompletion", request)
    monkeypatch.setattr(summary, "SUMMARY_BUFFER_MAX_EVENTS", 1)
    source = bridge([])
    source.litellm_custom_stream_wrapper = primary
    wrapper = summary.HostedVLLMCodexSummaryStream(source, {"model": "test"})
    delivered = []

    async def consume():
        async for event in wrapper:
            delivered.append(event)
            if event.type == "response.reasoning_text.done":
                await started.wait()

    consumer = asyncio.create_task(consume())
    async with asyncio.timeout(3):
        await closing.wait()
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
    assert primary.closed
    assert not any(event.type in ("response.output_text.delta", "response.completed") for event in delivered)


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_output", [False, True])
async def test_proxy_json_serialization_preserves_sequence_and_reasoning_item(monkeypatch, tool_output):
    import json

    monkeypatch.setattr(litellm, "acompletion", AsyncMock(return_value=Stream([chunk(content="Wire summary")])))
    output = (
        chunk(
            tools=[
                {
                    "index": 0,
                    "id": "call_summary_wire",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ]
        )
        if tool_output
        else chunk(content="Wire answer")
    )
    source = bridge([chunk(reasoning="Wire raw"), output, chunk(finish="tool_calls" if tool_output else "stop")])
    wrapper = summary.HostedVLLMCodexSummaryStream(source, {"model": "test"})
    events = [event async for event in wrapper]
    wire = [json.loads(event.model_dump_json(exclude_none=True)) for event in events]
    assert [event.get("sequence_number") for event in wire] == list(range(len(wire)))
    completed = wire[-1]["response"]
    reasoning = next(item for item in completed["output"] if item["type"] == "reasoning")
    done_items = [event["item"] for event in wire if event["type"] == "response.output_item.done"]
    reasoning_done = next(item for item in done_items if item.get("type") == "reasoning")
    assert reasoning_done["id"] == reasoning["id"]
    assert reasoning_done["content"] == [{"type": "reasoning_text", "text": "Wire raw"}]
    assert reasoning_done["summary"] == [{"type": "summary_text", "text": "Wire summary"}]
    summary_events = [event for event in wire if event["type"].startswith("response.reasoning_summary_")]
    assert summary_events and all(event["item_id"] == reasoning["id"] for event in summary_events)
    assert events[-1] is wrapper.completed_response is source.completed_response
