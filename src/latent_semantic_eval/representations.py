from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from latent_semantic_eval.config import ModelConfig
from latent_semantic_eval.modeling import load_tokenizer_and_model


@dataclass(slots=True)
class TextEncodingBundle:
    hidden_states: list[np.ndarray]
    prompt_hidden_states: list[np.ndarray] | None
    token_count: int
    prompt_token_count: int
    answer_start: int


class QwenRepresentationExtractor:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self._tokenizer = None
        self._model = None

    def extract_bundle(self, prompt: str, text: str) -> TextEncodingBundle:
        tokenizer, model = self._ensure_loaded()
        combined_text = format_prompt_and_answer(prompt, text)
        prompt_only_text = format_prompt_and_answer(prompt, "") if prompt.strip() else ""

        combined_hidden_states, token_count = self._forward(tokenizer, model, combined_text)
        prompt_hidden_states = None
        prompt_token_count = 0
        if prompt_only_text:
            prompt_hidden_states, prompt_token_count = self._forward(tokenizer, model, prompt_only_text)

        answer_start = min(prompt_token_count, token_count - 1) if token_count > 0 else 0
        return TextEncodingBundle(
            hidden_states=combined_hidden_states,
            prompt_hidden_states=prompt_hidden_states,
            token_count=token_count,
            prompt_token_count=prompt_token_count,
            answer_start=answer_start,
        )

    def resolve_layers(self, requested_layers: Sequence[int]) -> list[int]:
        tokenizer, model = self._ensure_loaded()
        del tokenizer, model
        total_hidden_states = self._hidden_state_count()
        resolved: list[int] = []
        for layer in requested_layers:
            actual = layer if layer >= 0 else total_hidden_states + layer
            if actual < 0 or actual >= total_hidden_states:
                raise ValueError(f"Layer index out of range: {layer}")
            if actual not in resolved:
                resolved.append(actual)
        return resolved

    def pool_vector(
        self,
        bundle: TextEncodingBundle,
        layer_index: int,
        pooling: str,
        use_prompt_residual: bool,
    ) -> np.ndarray:
        if use_prompt_residual and bundle.prompt_hidden_states is not None and bundle.prompt_token_count > 0:
            combined_tokens = bundle.hidden_states[layer_index][: bundle.token_count]
            prompt_tokens = bundle.prompt_hidden_states[layer_index][: bundle.prompt_token_count]
            return pool_tokens(combined_tokens, pooling) - pool_tokens(prompt_tokens, pooling)

        answer_tokens = bundle.hidden_states[layer_index][bundle.answer_start : bundle.token_count]
        if answer_tokens.size == 0:
            answer_tokens = bundle.hidden_states[layer_index][: bundle.token_count]
        return pool_tokens(answer_tokens, pooling)

    def _ensure_loaded(self):
        if self._tokenizer is None or self._model is None:
            self._tokenizer, self._model = load_tokenizer_and_model(self.model_config)
        return self._tokenizer, self._model

    def _forward(self, tokenizer, model, text: str) -> tuple[list[np.ndarray], int]:
        import torch

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_config.max_input_length,
        )
        inputs = inputs.to(model.device)
        token_count = int(inputs["attention_mask"][0].sum().item())

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        hidden_states = [
            state[0, :token_count, :].detach().float().cpu().numpy()
            for state in outputs.hidden_states
        ]
        return hidden_states, token_count

    def _hidden_state_count(self) -> int:
        tokenizer, model = self._ensure_loaded()
        del tokenizer
        return int(model.config.num_hidden_layers) + 1


def format_prompt_and_answer(prompt: str, text: str) -> str:
    prompt = prompt.strip()
    text = text.strip()
    if not prompt:
        return text
    return f"Prompt:\n{prompt}\n\nAnswer:\n{text}"


def pool_tokens(token_matrix: np.ndarray, pooling: str) -> np.ndarray:
    if token_matrix.ndim != 2:
        raise ValueError("Expected a 2D token matrix.")
    if token_matrix.shape[0] == 0:
        raise ValueError("Cannot pool an empty token matrix.")

    if pooling == "mean":
        return token_matrix.mean(axis=0)
    if pooling == "last_token":
        return token_matrix[-1]
    if pooling == "max":
        return token_matrix.max(axis=0)
    if pooling == "first_token":
        return token_matrix[0]
    raise ValueError(f"Unsupported pooling strategy: {pooling}")