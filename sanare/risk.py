from __future__ import annotations

import re

HIGH_RISK_TERMS = [
    "stroke",
    "sepsis",
    "myocardial infarction",
    "heart attack",
    "pulmonary embolism",
    "respiratory distress",
    "syncope",
]

MODERATE_RISK_TERMS = [
    "chest pain",
    "shortness of breath",
    "sob",
    "dyspnea",
    "exertional dyspnea",
    "abdominal pain",
    "uncontrolled diabetes",
    "hypertension",
    "htn",
]

LOW_RISK_TERMS = [
    "rash",
    "cough",
    "cold",
    "allergic rhinitis",
    "medication refill",
]

_HIGH_RE = re.compile("|".join(rf"\b{re.escape(t)}\b" for t in HIGH_RISK_TERMS))
_MODERATE_RE = re.compile("|".join(rf"\b{re.escape(t)}\b" for t in MODERATE_RISK_TERMS))
_LOW_RE = re.compile("|".join(rf"\b{re.escape(t)}\b" for t in LOW_RISK_TERMS))


def infer_risk(note: str, conditions: list[str]) -> str:
    text = " ".join([note.lower(), *conditions])
    if _HIGH_RE.search(text):
        return "high"
    if _MODERATE_RE.search(text):
        return "moderate"
    if _LOW_RE.search(text):
        return "low"
    return "low"


def infer_next_step(risk_level: str, conditions: list[str], note: str) -> str:
    text = " ".join([note.lower(), *[c.lower() for c in conditions]])

    # Emergency-specific routing first — most specific wins
    if re.search(r"\b(myocardial infarction|heart attack)\b", text):
        return "emergency cardiology intervention"
    if re.search(r"\b(stroke|cva)\b", text):
        return "neurology referral"
    if re.search(r"\bsepsis\b", text):
        return "icu admission evaluation"
    if re.search(r"\bpulmonary embolism\b", text):
        return "emergency pulmonology consult"
    if re.search(r"\brespiratory distress\b", text):
        return "urgent respiratory evaluation"

    # Symptom-combination routing
    if re.search(r"\bchest pain\b", text) and re.search(r"\b(shortness of breath|sob|dyspnea)\b", text):
        return "cardiology referral"
    if re.search(r"\b(copd|asthma)\b", text) and re.search(r"\b(shortness of breath|sob|dyspnea)\b", text):
        return "pulmonology referral"

    # Condition-specific specialty routing
    if re.search(r"\bdiabetes\b", text) and risk_level in ("moderate", "high"):
        return "endocrinology follow-up"
    if re.search(r"\bmedication refill\b", text):
        return "medication management"

    # Generic risk-level fallback
    if risk_level == "high":
        return "urgent clinical evaluation"
    if risk_level == "moderate":
        return "clinician follow-up"
    return "routine follow-up"
