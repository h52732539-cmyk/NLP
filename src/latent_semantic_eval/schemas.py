from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EvaluationRecord:
    record_id: str
    prompt: str
    candidate: str
    reference: str
    human_score: float | None = None
    binary_label: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "prompt": self.prompt,
            "candidate": self.candidate,
            "reference": self.reference,
            "human_score": self.human_score,
            "binary_label": self.binary_label,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluationRecord":
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        binary_label = payload.get("binary_label")
        return cls(
            record_id=str(payload.get("record_id", "")),
            prompt=str(payload.get("prompt", "")),
            candidate=str(payload.get("candidate", "")),
            reference=str(payload.get("reference", "")),
            human_score=_safe_float(payload.get("human_score")),
            binary_label=_safe_int(binary_label),
            metadata=metadata,
        )


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)