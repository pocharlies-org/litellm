import asyncio
from collections import defaultdict

import pytest

import litellm
from litellm.responses.litellm_completion_transformation.hosted_vllm_codex_summary import HostedVLLMCodexSummaryStream
from litellm.responses.litellm_completion_transformation.streaming_iterator import LiteLLMCompletionStreamingIterator
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices


class PrimaryChunks:
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.logging_obj = None

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
        pass


def primary_chunk(*, reasoning=None, content=None, tools=None, finish=None):
    return ModelResponseStream(
        id="chatcmpl-client-order",
        model="test",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(reasoning_content=reasoning, content=content, tool_calls=tools),
                finish_reason=finish,
            )
        ],
    )


class PrimaryWithCompletionSignal(LiteLLMCompletionStreamingIterator):
    def __init__(self, chunks, completed):
        super().__init__(
            model="test",
            litellm_custom_stream_wrapper=PrimaryChunks(chunks),
            request_input="Synthetic ordering check",
            responses_api_request={"reasoning": {"summary": "auto"}},
            custom_llm_provider="hosted_vllm_codex",
        )
        self.primary_completed = completed

    async def __anext__(self):
        event = await super().__anext__()
        if event.type == "response.completed":
            self.primary_completed.set()
        return event


class InstalledCodexConsumer:
    """Codex turn.rs routes summary deltas by active_item; its SSE parser discards item_id."""

    def __init__(self):
        self.active_item = None
        self.summary_by_item = defaultdict(str)
        self.unattached_summary = ""

    def consume(self, event):
        if event.type == "response.output_item.added" and event.item.type in ("reasoning", "message"):
            self.active_item = event.item.id
        elif event.type == "response.output_item.done":
            self.active_item = None
        elif event.type == "response.reasoning_summary_text.delta":
            if self.active_item is None:
                self.unattached_summary += event.delta
            else:
                self.summary_by_item[self.active_item] += event.delta


@pytest.mark.asyncio
@pytest.mark.parametrize("output_kind", ["message", "function_call"])
async def test_delayed_summary_is_visible_on_original_reasoning_before_next_item(monkeypatch, output_kind):
    primary_completed = asyncio.Event()

    async def request(**kwargs):
        assert kwargs["messages"][1]["content"] == "First inspect. Then answer."

        async def response():
            await primary_completed.wait()
            yield primary_chunk(content="Inspect, then answer.")

        return response()

    monkeypatch.setattr(litellm, "acompletion", request)
    chunks = [primary_chunk(reasoning="First inspect. "), primary_chunk(reasoning="Then answer.")]
    if output_kind == "message":
        chunks.extend([primary_chunk(content="ORDER_"), primary_chunk(content="OK"), primary_chunk(finish="stop")])
    else:
        chunks.extend(
            [
                primary_chunk(
                    tools=[
                        {
                            "index": 0,
                            "id": "call_order",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"key":'},
                        }
                    ]
                ),
                primary_chunk(tools=[{"index": 0, "function": {"arguments": '"ORDER_OK"}'}}]),
                primary_chunk(finish="tool_calls"),
            ]
        )
    source = PrimaryWithCompletionSignal(chunks, primary_completed)
    consumer = InstalledCodexConsumer()
    events = []
    async with asyncio.timeout(3):
        async for event in HostedVLLMCodexSummaryStream(source, {"model": "test"}):
            events.append(event)
            consumer.consume(event)
    completed = events[-1].response
    reasoning = next(item for item in completed.output if item.type == "reasoning")
    assert reasoning.content[0].text == "First inspect. Then answer."
    assert reasoning.summary[0].text == "Inspect, then answer."
    if output_kind == "message":
        assert "".join(e.delta for e in events if e.type == "response.output_text.delta") == "ORDER_OK"
    else:
        tool = next(item for item in completed.output if item.type == "function_call")
        assert (tool.call_id, tool.name, tool.arguments) == ("call_order", "lookup", '{"key":"ORDER_OK"}')
        assert "".join(e.delta for e in events if e.type == "response.function_call_arguments.delta") == tool.arguments
    assert consumer.summary_by_item[reasoning.id] == "Inspect, then answer.", (
        "Installed Codex lost or attached summary to a later item: "
        f"routed={dict(consumer.summary_by_item)!r}, unattached={consumer.unattached_summary!r}"
    )
    assert consumer.unattached_summary == ""
    summary_positions = [i for i, e in enumerate(events) if e.type == "response.reasoning_summary_text.delta"]
    next_item_position = next(
        i for i, e in enumerate(events) if e.type == "response.output_item.added" and e.item.type == output_kind
    )
    assert max(summary_positions) < next_item_position


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", ["events", "bytes"])
async def test_buffer_overflow_finishes_partial_summary_and_preserves_every_primary_event(monkeypatch, limit):
    from litellm.responses.litellm_completion_transformation import hosted_vllm_codex_summary as summary

    partial_delivered = asyncio.Event()
    auxiliary_cancelled = asyncio.Event()
    primary_events = []

    async def request(**kwargs):
        async def response():
            try:
                yield primary_chunk(content="Partial summary.")
                await asyncio.Event().wait()
            finally:
                auxiliary_cancelled.set()

        return response()

    class PrimaryAfterPartial(LiteLLMCompletionStreamingIterator):
        async def __anext__(self):
            event = await super().__anext__()
            if event.type == "response.output_item.done" and event.item.type == "reasoning":
                await partial_delivered.wait()
            primary_events.append(event)
            return event

    monkeypatch.setattr(litellm, "acompletion", request)
    monkeypatch.setattr(summary, "SUMMARY_BUFFER_MAX_EVENTS", 2 if limit == "events" else 10000)
    monkeypatch.setattr(summary, "SUMMARY_BUFFER_MAX_BYTES", 1 if limit == "bytes" else 10000000)
    source = PrimaryAfterPartial(
        model="test",
        litellm_custom_stream_wrapper=PrimaryChunks(
            [
                primary_chunk(reasoning="Original reasoning"),
                primary_chunk(
                    tools=[
                        {
                            "index": 0,
                            "id": "call_overflow",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"key":'},
                        }
                    ]
                ),
                primary_chunk(tools=[{"index": 0, "function": {"arguments": '"still here"}'}}]),
                primary_chunk(finish="tool_calls"),
            ]
        ),
        request_input="Synthetic overflow check",
        responses_api_request={},
        custom_llm_provider="hosted_vllm_codex",
    )
    events = []
    async with asyncio.timeout(3):
        async for event in HostedVLLMCodexSummaryStream(source, {"model": "test"}):
            events.append(event)
            if event.type == "response.reasoning_summary_text.delta":
                partial_delivered.set()
    assert auxiliary_cancelled.is_set()
    assert [id(event) for event in events if not event.type.startswith("response.reasoning_summary_")] == [
        id(event) for event in primary_events
    ], "Overflow must preserve FIFO identity, including the event that hits the limit"
    assert [event.sequence_number for event in events] == list(range(len(events)))
    completed = events[-1].response
    reasoning = next(item for item in completed.output if item.type == "reasoning")
    assert reasoning.content[0].text == "Original reasoning"
    assert reasoning.summary[0].text == "Partial summary."
    tool = next(item for item in completed.output if item.type == "function_call")
    assert (tool.call_id, tool.name, tool.arguments) == ("call_overflow", "lookup", '{"key":"still here"}')
    assert "".join(e.delta for e in events if e.type == "response.function_call_arguments.delta") == tool.arguments
    assert sum(e.type == "response.reasoning_summary_text.done" for e in events) == 1
    assert sum(e.type == "response.reasoning_summary_part.done" for e in events) == 1
    consumer = InstalledCodexConsumer()
    for event in events:
        consumer.consume(event)
    assert consumer.summary_by_item[reasoning.id] == "Partial summary."
    assert consumer.unattached_summary == ""


@pytest.mark.asyncio
async def test_early_summary_waits_for_original_reasoning_done_before_releasing_later_items(monkeypatch):
    summary_terminal_delivered = asyncio.Event()
    original_events = []

    async def request(**kwargs):
        async def response():
            yield primary_chunk(content="Fast summary.")

        return response()

    class DelayedReasoningDone(LiteLLMCompletionStreamingIterator):
        async def __anext__(self):
            event = await super().__anext__()
            if event.type == "response.output_item.done" and event.item.type == "reasoning":
                await summary_terminal_delivered.wait()
            original_events.append(event)
            return event

    monkeypatch.setattr(litellm, "acompletion", request)
    source = DelayedReasoningDone(
        model="test",
        litellm_custom_stream_wrapper=PrimaryChunks(
            [primary_chunk(reasoning="raw"), primary_chunk(content="Answer"), primary_chunk(finish="stop")]
        ),
        request_input="Synthetic early completion check",
        responses_api_request={},
        custom_llm_provider="hosted_vllm_codex",
    )
    events = []
    async with asyncio.timeout(3):
        async for event in HostedVLLMCodexSummaryStream(source, {"model": "test"}):
            events.append(event)
            if event.type == "response.reasoning_summary_part.done":
                summary_terminal_delivered.set()
    assert [id(event) for event in events if not event.type.startswith("response.reasoning_summary_")] == [
        id(event) for event in original_events
    ]
    reasoning_done = next(
        i
        for i, event in enumerate(events)
        if event.type == "response.output_item.done" and event.item.type == "reasoning"
    )
    message_start = next(
        i
        for i, event in enumerate(events)
        if event.type == "response.output_item.added" and event.item.type == "message"
    )
    terminal = next(i for i, event in enumerate(events) if event.type == "response.reasoning_summary_part.done")
    assert terminal < reasoning_done < message_start
    assert events[-1].type == "response.completed"
