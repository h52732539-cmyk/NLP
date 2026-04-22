from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from latent_semantic_eval.config import ModelConfig
from latent_semantic_eval.modeling import load_tokenizer_and_model


@dataclass(slots=True)
class GenerationOutput:
    text: str
    reasoning: str | None = None


class QwenGenerator:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self._tokenizer = None
        self._model = None

    def generate(self, prompts: Sequence[str]) -> list[GenerationOutput]:
        import torch

        tokenizer, model = self._ensure_loaded()
        chat_prompts = [self._build_chat_prompt(tokenizer, prompt) for prompt in prompts]
        inputs = tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_config.max_input_length,
        )
        inputs = inputs.to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=self.model_config.max_new_tokens,
                temperature=self.model_config.temperature,
                top_p=self.model_config.top_p,
                do_sample=self.model_config.temperature > 0.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        outputs: list[GenerationOutput] = []
        for row_index, token_ids in enumerate(generated):
            prompt_length = int(inputs["attention_mask"][row_index].sum().item())
            completion_ids = token_ids[prompt_length:]
            decoded = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            reasoning, answer = split_reasoning(decoded)
            outputs.append(GenerationOutput(text=answer, reasoning=reasoning))
        return outputs

    def _ensure_loaded(self):
        if self._tokenizer is None or self._model is None:
            self._tokenizer, self._model = load_tokenizer_and_model(self.model_config)
        return self._tokenizer, self._model

    def _build_chat_prompt(self, tokenizer, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt.strip()}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.model_config.enable_thinking,
            )
        except TypeError:
            content = prompt.strip()
            if not self.model_config.enable_thinking:
                content = content + "\n/no_think"
            fallback_messages = [{"role": "user", "content": content}]
            return tokenizer.apply_chat_template(
                fallback_messages,
                tokenize=False,
                add_generation_prompt=True,
            )


def split_reasoning(decoded_text: str) -> tuple[str | None, str]:
    marker = "</think>"
    if marker not in decoded_text:
        return None, decoded_text.strip()
    reasoning, answer = decoded_text.split(marker, maxsplit=1)
    reasoning = reasoning.replace("<think>", "").strip()
    return reasoning or None, answer.strip()