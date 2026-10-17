import json

import pytest
import respx

import litellm
from litellm.utils import ProviderConfigManager


def test_provider_resolution_and_responses_bridge_selection(monkeypatch):
    monkeypatch.setenv("HOSTED_VLLM_API_BASE", "https://codex-vllm.invalid/v1")
    monkeypatch.setenv("HOSTED_VLLM_API_KEY", "test-key")
    assert litellm.get_llm_provider("hosted_vllm_codex/Qwen/test") == (
        "Qwen/test",
        "hosted_vllm_codex",
        "test-key",
        "https://codex-vllm.invalid/v1",
    )
    assert ProviderConfigManager.get_provider_responses_api_config("hosted_vllm_codex", "Qwen/test") is None
    assert isinstance(
        ProviderConfigManager.get_provider_responses_api_config("hosted_vllm", "Qwen/test"),
        litellm.HostedVLLMResponsesAPIConfig,
    )
    assert "thinking" in litellm.get_supported_openai_params("hosted_vllm_codex/Qwen/test")


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("via_router", [False, True])
@pytest.mark.asyncio
async def test_completion_uses_hosted_vllm_transport_and_transformation(is_async, via_router, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    params = {
        "model": "hosted_vllm_codex/Qwen/test",
        "api_base": "https://codex-vllm.invalid/v1",
        "api_key": "test-key",
    }
    router = litellm.Router(
        model_list=[{"model_name": "qwen3.8-flash-next-codex", "litellm_params": params}], num_retries=0
    )
    with respx.mock(assert_all_called=True) as mock:
        route = mock.post("https://codex-vllm.invalid/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-codex",
                "object": "chat.completion",
                "created": 1,
                "model": "Qwen/test",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Done"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            },
        )
        kwargs = {
            **({"model": "qwen3.8-flash-next-codex"} if via_router else params),
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": {"type": "disabled"},
        }
        client = router if via_router else litellm
        response = await client.acompletion(**kwargs) if is_async else client.completion(**kwargs)
        assert response.choices[0].message.content == "Done"
        payload = json.loads(route.calls[-1].request.content)
        assert payload["model"] == "Qwen/test"
        assert payload["reasoning_effort"] == "off"
        assert "thinking" not in payload
        assert route.calls[-1].request.headers["authorization"] == "Bearer test-key"
