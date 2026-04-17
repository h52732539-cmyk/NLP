from __future__ import annotations

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
    if model_config.attn_implementation:
        kwargs["attn_implementation"] = model_config.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **kwargs)
    model.eval()
    return tokenizer, model


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