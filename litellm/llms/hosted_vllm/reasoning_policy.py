from collections.abc import Mapping
from typing import Final

from litellm.exceptions import BadRequestError

DEEPSEEK_V4_MODEL_GROUP: Final = "deepseek-v4-flash-fp8-mtp"
DEEPSEEK_V4_UPSTREAM_MODEL: Final = "deepseek-ai/DeepSeek-V4-Flash"
DEEPSEEK_V4_REASONING_EFFORTS: Final = frozenset({"off", "low", "high", "max"})


def get_model_group(litellm_params: object) -> object:
    if isinstance(litellm_params, Mapping):
        metadata: Final = litellm_params.get("metadata") or litellm_params.get("litellm_metadata")
        return metadata.get("model_group") if isinstance(metadata, Mapping) else None
    metadata_attribute: Final = getattr(litellm_params, "metadata", None) or getattr(
        litellm_params, "litellm_metadata", None
    )
    return metadata_attribute.get("model_group") if isinstance(metadata_attribute, Mapping) else None


def is_deepseek_v4_target(model: str, model_group: object) -> bool:
    upstream_model: Final = model.removeprefix("hosted_vllm/")
    return model_group == DEEPSEEK_V4_MODEL_GROUP and upstream_model == DEEPSEEK_V4_UPSTREAM_MODEL


def normalize_deepseek_v4_reasoning_effort(
    *,
    model: str,
    model_group: object,
    reasoning_effort: object,
    supplied: bool,
) -> object:
    if not supplied or not is_deepseek_v4_target(model=model, model_group=model_group):
        return reasoning_effort
    if not isinstance(reasoning_effort, str) or reasoning_effort not in DEEPSEEK_V4_REASONING_EFFORTS:
        invalid_value: Final = repr(reasoning_effort)[:100]
        raise BadRequestError(
            message=(
                f"Invalid reasoning_effort {invalid_value} for model group {DEEPSEEK_V4_MODEL_GROUP}. "
                "Allowed values are: off, low, high, max"
            ),
            model=DEEPSEEK_V4_MODEL_GROUP,
            llm_provider="hosted_vllm",
        )
    if reasoning_effort == "off":
        return "none"
    return reasoning_effort


def normalize_deepseek_v4_responses_reasoning(
    *,
    model: str,
    model_group: object,
    request_data: Mapping[str, object],
) -> dict[str, object]:  # mutable-ok: framework contract requires mutable request or response containers
    if not is_deepseek_v4_target(model=model, model_group=model_group):
        return dict(request_data)  # mutable-ok: framework contract requires mutable request or response containers
    reasoning: Final = request_data.get("reasoning")
    nested_supplied: Final = isinstance(reasoning, Mapping) and "effort" in reasoning
    compatibility_supplied: Final = "reasoning_effort" in request_data
    typed_reasoning: Final = reasoning if isinstance(reasoning, Mapping) else None
    nested_effort: Final[object] = typed_reasoning.get("effort") if typed_reasoning is not None else None
    compatibility_effort: Final = request_data.get("reasoning_effort")
    if nested_supplied and compatibility_supplied and nested_effort != compatibility_effort:
        nested_value: Final = repr(nested_effort)[:100]
        compatibility_value: Final = repr(compatibility_effort)[:100]
        raise BadRequestError(
            message=(
                f"Conflicting reasoning values for model group {DEEPSEEK_V4_MODEL_GROUP}: "
                f"reasoning.effort={nested_value} and reasoning_effort={compatibility_value}. "
                "Supply only one representation or use the same allowed value: off, low, high, max"
            ),
            model=DEEPSEEK_V4_MODEL_GROUP,
            llm_provider="hosted_vllm",
        )
    supplied: Final = nested_supplied or compatibility_supplied
    authoritative_effort: Final[object] = nested_effort if nested_supplied else compatibility_effort
    normalized_effort: Final = normalize_deepseek_v4_reasoning_effort(
        model=model,
        model_group=model_group,
        reasoning_effort=authoritative_effort,
        supplied=supplied,
    )
    without_compatibility: Final = {  # mutable-ok: framework contract requires mutable request or response containers
        key: value for key, value in request_data.items() if key != "reasoning_effort"
    }  # mutable-ok: framework contract requires mutable request or response containers
    if not supplied:
        return without_compatibility
    normalized_reasoning: Final[  # mutable-ok: framework contract requires mutable request or response containers
        dict[str, object]
    ] = {  # mutable-ok: framework contract requires mutable request or response containers
        **(
            dict(typed_reasoning)  # mutable-ok: framework contract requires mutable request or response containers
            if typed_reasoning is not None
            else {}  # mutable-ok: framework contract requires mutable request or response containers
        ),  # mutable-ok: framework contract requires mutable request or response containers
        "effort": normalized_effort,
    }
    return {  # mutable-ok: framework contract requires mutable request or response containers
        **without_compatibility,
        "reasoning": normalized_reasoning,
    }  # mutable-ok: framework contract requires mutable request or response containers
