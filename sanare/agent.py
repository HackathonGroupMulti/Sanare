from __future__ import annotations

import json
import re
from typing import Any, Generator

from pydantic import ValidationError

from sanare.llm_client import LLMClient, LLMUnavailableError
from sanare.risk import infer_next_step, infer_risk
from sanare.schemas import ClinicalAnalysis

SYSTEM_PROMPT = """You are a clinical extraction engine.

Convert input into STRICT JSON.

Fields:
- patient_summary
- conditions
- medications
- risk_level
- next_step

Rules:
- no hallucinated entities
- infer risk from condition severity and symptoms
- risk_level must be one of: low, moderate, high
- no text outside JSON
"""

REPAIR_PROMPT = """Repair this response into STRICT JSON matching exactly:
{
  "patient_summary": "string",
  "conditions": ["string"],
  "medications": ["string"],
  "risk_level": "low|moderate|high",
  "next_step": "string"
}

No text outside JSON.
"""

CONDITION_ALIASES = {
    "htn": "hypertension",
    "hypertension": "hypertension",
    "chest pain": "chest pain",
    "sob": "shortness of breath",
    "shortness of breath": "shortness of breath",
    "dyspnea": "shortness of breath",
    "diabetes": "diabetes",
    "dm": "diabetes",
    "asthma": "asthma",
    "copd": "copd",
}

COMMON_MEDICATIONS = {
    "lisinopril",
    "metformin",
    "atorvastatin",
    "amlodipine",
    "albuterol",
    "insulin",
    "warfarin",
    "apixaban",
    "aspirin",
    "losartan",
}


class ClinicalExtractionAgent:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def analyze(self, note: str) -> ClinicalAnalysis:
        try:
            raw = self.llm_client.complete_json(SYSTEM_PROMPT, self._analysis_prompt(note))
            return self._validated_or_repaired(note, raw)
        except LLMUnavailableError:
            return self._heuristic_analysis(note)

    def analyze_streaming(self, note: str) -> Generator[str | ClinicalAnalysis, None, None]:
        """Yield str tokens as the LLM generates, then yield the final ClinicalAnalysis."""
        try:
            raw_parts: list[str] = []
            for token in self.llm_client.stream_tokens(SYSTEM_PROMPT, self._analysis_prompt(note)):
                raw_parts.append(token)
                yield token
            yield self._validated_or_repaired(note, "".join(raw_parts))
        except (LLMUnavailableError, Exception):
            yield self._heuristic_analysis(note)

    def _analysis_prompt(self, note: str) -> str:
        return f'Clinical text:\n"""{note.strip()}"""'

    def _validated_or_repaired(self, note: str, raw: str) -> ClinicalAnalysis:
        try:
            return self._validate(note, raw)
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as first_error:
            repair_input = f"{REPAIR_PROMPT}\n\nOriginal note:\n{note}\n\nBad response:\n{raw}\n\nError:\n{first_error}"
            try:
                repaired = self.llm_client.complete_json(REPAIR_PROMPT, repair_input)
                return self._validate(note, repaired)
            except (json.JSONDecodeError, ValidationError, TypeError, ValueError, LLMUnavailableError):
                return self._heuristic_analysis(note)

    def _validate(self, note: str, raw: str) -> ClinicalAnalysis:
        data = self._extract_json(raw)
        analysis = ClinicalAnalysis.model_validate(data)
        return self._apply_risk_layer(note, analysis)

    def _extract_json(self, raw: str) -> dict[str, Any]:
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                text = text[start : end + 1]
        return json.loads(text)

    def _apply_risk_layer(self, note: str, analysis: ClinicalAnalysis) -> ClinicalAnalysis:
        conditions: list[str] = []
        for item in analysis.conditions:
            if self._condition_is_grounded(item, note):
                normalized = CONDITION_ALIASES.get(item.lower(), item)
                if normalized not in conditions:
                    conditions.append(normalized)
        medications = []
        for item in analysis.medications:
            if self._text_is_grounded(item, note) and item not in medications:
                medications.append(item)
        for condition in self._extract_conditions(note):
            if condition not in conditions:
                conditions.append(condition)
        for medication in self._extract_medications(note):
            if medication not in medications:
                medications.append(medication)
        risk_level = infer_risk(note, conditions)
        next_step = infer_next_step(risk_level, conditions, note)
        return analysis.model_copy(
            update={
                "patient_summary": self._summary(note, conditions, medications),
                "conditions": conditions,
                "medications": medications,
                "risk_level": risk_level,
                "next_step": next_step,
            }
        )

    def _condition_is_grounded(self, condition: str, note: str) -> bool:
        if self._text_is_grounded(condition, note):
            return True
        normalized = condition.lower()
        note_text = note.lower()
        return any(
            normalized == alias_value and re.search(rf"\b{re.escape(alias_key)}\b", note_text)
            for alias_key, alias_value in CONDITION_ALIASES.items()
        )

    def _text_is_grounded(self, value: str, note: str) -> bool:
        return re.search(rf"\b{re.escape(value.lower())}\b", note.lower()) is not None

    def _heuristic_analysis(self, note: str) -> ClinicalAnalysis:
        conditions = self._extract_conditions(note)
        medications = self._extract_medications(note)
        risk_level = infer_risk(note, conditions)
        return ClinicalAnalysis(
            patient_summary=self._summary(note, conditions, medications),
            conditions=conditions,
            medications=medications,
            risk_level=risk_level,
            next_step=infer_next_step(risk_level, conditions, note),
        )

    def _extract_conditions(self, note: str) -> list[str]:
        text = note.lower()
        conditions: list[str] = []
        for source, normalized in CONDITION_ALIASES.items():
            if re.search(rf"\b{re.escape(source)}\b", text) and normalized not in conditions:
                conditions.append(normalized)
        return conditions

    def _extract_medications(self, note: str) -> list[str]:
        text = note.lower()
        return [med for med in sorted(COMMON_MEDICATIONS) if re.search(rf"\b{re.escape(med)}\b", text)]

    def _summary(self, note: str, conditions: list[str], medications: list[str]) -> str:
        age_sex = re.search(r"\b(\d{1,3})\s*([mf])\b", note.lower())
        patient = "patient"
        if age_sex:
            sex = "male" if age_sex.group(2) == "m" else "female"
            patient = f"{age_sex.group(1)}-year-old {sex}"

        details: list[str] = []
        if conditions:
            details.append(f"with {', '.join(conditions)}")
        if medications:
            details.append(f"on {', '.join(medications)}")
        if not details:
            return f"{patient} with clinical note provided"
        return f"{patient} {' and '.join(details)}"

