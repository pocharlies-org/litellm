# Codex through the Responses bridge

Use a separate model alias to route Codex Responses requests through Chat Completions

```yaml
model_list:
  - model_name: qwen3.8-flash-next-codex
    litellm_params:
      model: hosted_vllm/qwen3.8-flash-next
      api_base: http://localhost:8000/v1
      use_chat_completions_api: true
      extra_body:
        merge_system_messages: true
```

`merge_system_messages` is an opt-in LiteLLM setting for chat templates that accept only one initial system message. It combines system and developer text, in their original relative order, into the initial system message. Other messages retain their order. LiteLLM consumes the setting before sending the request to vLLM. System content must be text or text blocks

The Responses bridge lifts `additional_tools`, qualifies namespace tool names for Chat Completions, and restores names, namespaces, call IDs and custom tool inputs in Responses output and streaming events. Custom tool grammars are included in the function description; this does not enforce constrained decoding on the backend

`agent_message` items are represented as assistant messages containing their author, recipient and content. Opaque encrypted content remains opaque. Local Codex compaction can summarize the visible conversation; the bridge does not decrypt remote compaction payloads

Keep the ordinary Qwen alias without these opt-in settings for clients that already use Chat Completions
