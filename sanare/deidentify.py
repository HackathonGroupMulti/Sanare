from __future__ import annotations

import re
import os
from dataclasses import dataclass

from sanare.schemas import DeidentifiedNote


@dataclass(frozen=True)
class RegexRecognizer:
    label: str
    pattern: re.Pattern[str]


FALLBACK_RECOGNIZERS = [
    RegexRecognizer("EMAIL_ADDRESS", re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b")),
    RegexRecognizer("PHONE_NUMBER", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    RegexRecognizer("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    RegexRecognizer("MRN", re.compile(r"\b(?:MRN|mrn)[:#\s-]*[A-Za-z0-9-]{4,}\b")),
    RegexRecognizer("DATE", re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")),
    # Catches names introduced by clinical context keywords (e.g. "patient John Smith")
    RegexRecognizer(
        "PERSON",
        re.compile(
            r"\b(?:patient|pt|name|provider|physician|dr\.?|doctor)\s+([A-Z][a-z]+(?: [A-Z][a-z]+){1,2})\b"
        ),
    ),
    # Catches standalone Title Case pairs not preceded by a digit (avoids matching "Type 2")
    RegexRecognizer(
        "PERSON",
        re.compile(r"(?<!\d )(?<!\d)\b([A-Z][a-z]{1,20} [A-Z][a-z]{1,20})\b(?! (mg|ml|mcg|units?|mmhg|bpm))"),
    ),
]


class ClinicalDeidentifier:
    def __init__(self) -> None:
        self._presidio = self._load_presidio()

    def redact(self, text: str) -> DeidentifiedNote:
        if self._presidio:
            return self._presidio_redact(text)
        return self._fallback_redact(text)

    def _load_presidio(self) -> tuple[object, object] | None:
        if os.getenv("SANARE_ENABLE_PRESIDIO") != "1":
            return None
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
        except ImportError:
            return None
        return AnalyzerEngine(), AnonymizerEngine()

    def _presidio_redact(self, text: str) -> DeidentifiedNote:
        analyzer, anonymizer = self._presidio
        results = analyzer.analyze(text=text, language="en")
        anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
        entities = sorted({result.entity_type for result in results})
        return DeidentifiedNote(text=anonymized.text, phi_entities=entities)

    def _fallback_redact(self, text: str) -> DeidentifiedNote:
        redacted = text
        entities: set[str] = set()
        for recognizer in FALLBACK_RECOGNIZERS:
            if recognizer.pattern.search(redacted):
                entities.add(recognizer.label)
                redacted = recognizer.pattern.sub(f"<{recognizer.label}>", redacted)
        return DeidentifiedNote(text=redacted, phi_entities=sorted(entities))

