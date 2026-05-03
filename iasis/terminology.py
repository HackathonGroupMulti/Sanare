from __future__ import annotations

from iasis.schemas import CodedConcept

ICD10 = "http://hl7.org/fhir/sid/icd-10-cm"
RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"
SNOMED = "http://snomed.info/sct"

CONDITION_CODES = {
    "hypertension": ("I10", "Essential hypertension"),
    "chest pain": ("R07.9", "Chest pain, unspecified"),
    "shortness of breath": ("R06.02", "Shortness of breath"),
    "diabetes": ("E11.9", "Type 2 diabetes mellitus without complications"),
    "asthma": ("J45.909", "Unspecified asthma, uncomplicated"),
    "copd": ("J44.9", "Chronic obstructive pulmonary disease, unspecified"),
}

MEDICATION_CODES = {
    "lisinopril": ("29046", "lisinopril"),
    "metformin": ("6809", "metformin"),
    "atorvastatin": ("83367", "atorvastatin"),
    "amlodipine": ("17767", "amlodipine"),
    "albuterol": ("435", "albuterol"),
    "insulin": ("5856", "insulin"),
    "warfarin": ("11289", "warfarin"),
    "apixaban": ("1364430", "apixaban"),
    "aspirin": ("1191", "aspirin"),
    "losartan": ("52175", "losartan"),
}

SYMPTOM_OBSERVATION_TERMS = {"chest pain", "shortness of breath"}


class TerminologyMapper:
    def condition(self, text: str) -> CodedConcept | None:
        code = CONDITION_CODES.get(text.lower())
        if not code:
            return None
        return CodedConcept(text=text, system=ICD10, code=code[0], display=code[1])

    def medication(self, text: str) -> CodedConcept | None:
        code = MEDICATION_CODES.get(text.lower())
        if not code:
            return None
        return CodedConcept(text=text, system=RXNORM, code=code[0], display=code[1])

    def is_observation(self, condition: str) -> bool:
        return condition.lower() in SYMPTOM_OBSERVATION_TERMS
