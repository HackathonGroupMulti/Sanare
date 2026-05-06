HIGH_RISK_TERMS = {
    "stroke",
    "sepsis",
    "myocardial infarction",
    "heart attack",
    "pulmonary embolism",
    "respiratory distress",
    "syncope",
}

MODERATE_RISK_TERMS = {
    "chest pain",
    "shortness of breath",
    "sob",
    "dyspnea",
    "exertional dyspnea",
    "abdominal pain",
    "uncontrolled diabetes",
    "hypertension",
    "htn",
}

LOW_RISK_TERMS = {
    "rash",
    "cough",
    "cold",
    "allergic rhinitis",
    "medication refill",
}


def infer_risk(note: str, conditions: list[str]) -> str:
    text = " ".join([note.lower(), *conditions])
    if any(term in text for term in HIGH_RISK_TERMS):
        return "high"
    if any(term in text for term in MODERATE_RISK_TERMS):
        return "moderate"
    if any(term in text for term in LOW_RISK_TERMS):
        return "low"
    return "low"


def infer_next_step(risk_level: str, conditions: list[str], note: str) -> str:
    text = " ".join([note.lower(), *conditions])
    if "chest pain" in text and ("shortness of breath" in text or "sob" in text):
        return "cardiology referral"
    if risk_level == "high":
        return "urgent clinical evaluation"
    if risk_level == "moderate":
        return "clinician follow-up"
    return "routine follow-up"

