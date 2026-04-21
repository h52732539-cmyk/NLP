from __future__ import annotations

import importlib.util
import warnings
from typing import Any

from latent_semantic_eval.config import ModelConfig


def load_tokenizer_and_model(model_config: ModelConfig) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {
        "torch_dtype": resolve_torch_dtype(model_config.torch_dtype),
        "device_map": model_config.device_map,
        "trust_remote_code": model_config.trust_remote_code,
    }
    resolved_attn_implementation = resolve_attention_implementation(
        model_config.attn_implementation,
        torch,
    )
    if resolved_attn_implementation:
        kwargs["attn_implementation"] = resolved_attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **kwargs)
    model.eval()
    return tokenizer, model


def resolve_attention_implementation(attn_implementation: str | None, torch_module) -> str | None:
    if not attn_implementation:
        return None

    if attn_implementation != "flash_attention_2":
        return attn_implementation

    if not torch_module.cuda.is_available():
        warnings.warn(
            "flash_attention_2 was requested but CUDA is unavailable; falling back to the default attention implementation.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    if importlib.util.find_spec("flash_attn") is None:
        warnings.warn(
            "flash_attention_2 was requested but flash-attn is not installed; falling back to the default attention implementation.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    return attn_implementation


def resolve_torch_dtype(dtype_name: str):
    import torch

    normalized = dtype_name.lower()
    if normalized == "float32":
        return torch.float32
    if normalized == "float16":
        return torch.float16
    if normalized == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")