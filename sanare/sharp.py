from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel


class SharpContext(BaseModel):
    patient_id: str | None = None
    encounter_id: str | None = None
    fhir_base_url: str | None = None
    fhir_access_token: str | None = None
    scope: str | None = None
    user: str | None = None

    @property
    def has_fhir(self) -> bool:
        return bool(self.fhir_base_url and self.fhir_access_token and self.patient_id)

    @property
    def patient_resource_id(self) -> str | None:
        if not self.patient_id:
            return None
        # Accept "Patient/123" or bare "123"
        return self.patient_id.split("/")[-1]


def fetch_patient_context(ctx: SharpContext, timeout: float = 5.0) -> dict[str, Any]:
    """Fetch existing conditions and medications from the FHIR server for SHARP context."""
    if not ctx.has_fhir:
        return {"existing_conditions": [], "existing_medications": []}

    base = ctx.fhir_base_url.rstrip("/")  # type: ignore[union-attr]
    pid = ctx.patient_resource_id
    headers = {
        "Authorization": f"Bearer {ctx.fhir_access_token}",
        "Accept": "application/fhir+json",
    }

    existing_conditions: list[str] = []
    existing_medications: list[str] = []

    try:
        resp = httpx.get(f"{base}/Condition", params={"patient": pid}, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            for entry in resp.json().get("entry", []):
                text = entry.get("resource", {}).get("code", {}).get("text", "")
                if text:
                    existing_conditions.append(text)
    except Exception:
        pass

    try:
        resp = httpx.get(f"{base}/MedicationStatement", params={"patient": pid}, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            for entry in resp.json().get("entry", []):
                text = entry.get("resource", {}).get("medicationCodeableConcept", {}).get("text", "")
                if text:
                    existing_medications.append(text)
    except Exception:
        pass

    return {
        "existing_conditions": existing_conditions,
        "existing_medications": existing_medications,
    }


def enrich_note_with_context(note: str, patient_context: dict[str, Any]) -> str:
    """Prepend known patient context to the note so the LLM has full picture."""
    parts: list[str] = []
    if patient_context.get("existing_conditions"):
        parts.append("Known conditions: " + ", ".join(patient_context["existing_conditions"]))
    if patient_context.get("existing_medications"):
        parts.append("Current medications: " + ", ".join(patient_context["existing_medications"]))
    if parts:
        return "\n".join(parts) + "\n\nNew note:\n" + note
    return note
