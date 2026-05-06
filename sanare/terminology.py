from __future__ import annotations

import urllib.parse
from functools import lru_cache

import httpx

from sanare.schemas import CodedConcept

ICD10 = "http://hl7.org/fhir/sid/icd-10-cm"
RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"
SNOMED = "http://snomed.info/sct"

SYMPTOM_OBSERVATION_TERMS = {"chest pain", "shortness of breath"}

_CONDITION_FALLBACK: dict[str, tuple[str, str]] = {
    "hypertension": ("I10", "Essential hypertension"),
    "chest pain": ("R07.9", "Chest pain, unspecified"),
    "shortness of breath": ("R06.02", "Shortness of breath"),
    "diabetes": ("E11.9", "Type 2 diabetes mellitus without complications"),
    "asthma": ("J45.909", "Unspecified asthma, uncomplicated"),
    "copd": ("J44.9", "Chronic obstructive pulmonary disease, unspecified"),
}

_MEDICATION_FALLBACK: dict[str, tuple[str, str]] = {
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

_RXNORM_URL = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
_ICD10_URL = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
_TIMEOUT = 3.0


@lru_cache(maxsize=512)
def _lookup_rxcui(name: str) -> tuple[str, str] | None:
    try:
        resp = httpx.get(
            _RXNORM_URL,
            params={"name": name, "search": "0"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        ids = resp.json().get("idGroup", {}).get("rxnormId")
        if ids:
            return (ids[0], name)
    except Exception:
        pass
    return None


@lru_cache(maxsize=512)
def _lookup_icd10(term: str) -> tuple[str, str] | None:
    try:
        resp = httpx.get(
            _ICD10_URL,
            params={"terms": term, "maxList": "1"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        # Response shape: [total, [codes], extra, [[code, display], ...]]
        if data[0] > 0 and data[3]:
            code, display = data[3][0]
            return (code, display)
    except Exception:
        pass
    return None


class TerminologyMapper:
    def condition(self, text: str) -> CodedConcept | None:
        key = text.lower()
        result = _lookup_icd10(key) or _CONDITION_FALLBACK.get(key)
        if not result:
            return None
        return CodedConcept(text=text, system=ICD10, code=result[0], display=result[1])

    def medication(self, text: str) -> CodedConcept | None:
        key = text.lower()
        result = _lookup_rxcui(key) or _MEDICATION_FALLBACK.get(key)
        if not result:
            return None
        return CodedConcept(text=text, system=RXNORM, code=result[0], display=result[1])

    def is_observation(self, condition: str) -> bool:
        return condition.lower() in SYMPTOM_OBSERVATION_TERMS
