from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class NERSpan:
    text: str
    label: str
    score: float
    start: int
    end: int


# Maps d4data/biomedical-ner-all entity groups to clinical categories
_CONDITION_LABELS = {"Disease_disorder", "Sign_symptom"}
_MEDICATION_LABELS = {"Medication"}

_MODEL_ID = "d4data/biomedical-ner-all"


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


class BiomedicalNER:
    """Lazy-loading biomedical NER using a BERT-based HuggingFace model.

    Runs on GPU automatically when CUDA is available (NVIDIA NIM-compatible stack).
    """

    def __init__(self) -> None:
        self._pipe: object | None = None

    def _load(self) -> None:
        if self._pipe is not None:
            return
        from transformers import pipeline  # type: ignore[import-untyped]

        device = 0 if _cuda_available() else -1
        self._pipe = pipeline(
            "ner",
            model=_MODEL_ID,
            aggregation_strategy="simple",
            device=device,
        )

    def extract(self, text: str) -> list[NERSpan]:
        self._load()
        assert self._pipe is not None
        results = self._pipe(text)  # type: ignore[operator]
        return [
            NERSpan(
                text=r["word"],
                label=r["entity_group"],
                score=float(r["score"]),
                start=r["start"],
                end=r["end"],
            )
            for r in results
        ]

    def conditions(self, text: str) -> list[tuple[str, float]]:
        seen: set[str] = set()
        out: list[tuple[str, float]] = []
        for span in self.extract(text):
            if span.label in _CONDITION_LABELS:
                key = span.text.lower()
                if key not in seen:
                    out.append((key, span.score))
                    seen.add(key)
        return out

    def medications(self, text: str) -> list[tuple[str, float]]:
        seen: set[str] = set()
        out: list[tuple[str, float]] = []
        for span in self.extract(text):
            if span.label in _MEDICATION_LABELS:
                key = span.text.lower()
                if key not in seen:
                    out.append((key, span.score))
                    seen.add(key)
        return out


def build_ner() -> BiomedicalNER | None:
    if os.getenv("SANARE_ENABLE_NER") != "1":
        return None
    try:
        import transformers  # noqa: F401  # type: ignore[import-untyped]

        return BiomedicalNER()
    except ImportError:
        return None
