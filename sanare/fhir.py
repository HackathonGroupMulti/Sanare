from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sanare.schemas import ClinicalAnalysis
from sanare.terminology import SNOMED, TerminologyMapper


class FhirMapper:
    def __init__(self, terminology: TerminologyMapper | None = None) -> None:
        self.terminology = terminology or TerminologyMapper()

    def bundle(self, run_id: str, analysis: ClinicalAnalysis) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        entries: list[dict[str, Any]] = [self._entry(self._patient(run_id))]

        for index, condition in enumerate(analysis.conditions, start=1):
            if self.terminology.is_observation(condition):
                entries.append(self._entry(self._observation(run_id, index, condition, now)))
            else:
                entries.append(self._entry(self._condition(run_id, index, condition, now)))

        for index, medication in enumerate(analysis.medications, start=1):
            entries.append(self._entry(self._medication_statement(run_id, index, medication, now)))

        return {
            "resourceType": "Bundle",
            "type": "collection",
            "id": f"bundle-{run_id}",
            "timestamp": now,
            "entry": entries,
        }

    def _entry(self, resource: dict[str, Any]) -> dict[str, Any]:
        return {"fullUrl": f"urn:uuid:{resource['id']}", "resource": resource}

    def _patient(self, run_id: str) -> dict[str, Any]:
        return {
            "resourceType": "Patient",
            "id": f"patient-{run_id}",
            "identifier": [{"system": "urn:sanare:deidentified", "value": run_id}],
        }

    def _condition(self, run_id: str, index: int, condition: str, recorded_date: str) -> dict[str, Any]:
        concept = self.terminology.condition(condition)
        coding = []
        if concept:
            coding.append({"system": concept.system, "code": concept.code, "display": concept.display})
        return {
            "resourceType": "Condition",
            "id": f"condition-{run_id}-{index}",
            "clinicalStatus": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                        "code": "active",
                        "display": "Active",
                    }
                ]
            },
            "code": {"coding": coding, "text": condition},
            "subject": {"reference": f"Patient/patient-{run_id}"},
            "recordedDate": recorded_date,
        }

    def _observation(self, run_id: str, index: int, symptom: str, effective_date: str) -> dict[str, Any]:
        concept = self.terminology.condition(symptom)
        coding = []
        if concept:
            coding.append({"system": concept.system, "code": concept.code, "display": concept.display})
        return {
            "resourceType": "Observation",
            "id": f"observation-{run_id}-{index}",
            "status": "final",
            "code": {"coding": coding, "text": symptom},
            "subject": {"reference": f"Patient/patient-{run_id}"},
            "effectiveDateTime": effective_date,
            "valueCodeableConcept": {
                "coding": [{"system": SNOMED, "code": "52101004", "display": "Present"}],
                "text": "present",
            },
        }

    def _medication_statement(self, run_id: str, index: int, medication: str, effective_date: str) -> dict[str, Any]:
        concept = self.terminology.medication(medication)
        coding = []
        if concept:
            coding.append({"system": concept.system, "code": concept.code, "display": concept.display})
        return {
            "resourceType": "MedicationStatement",
            "id": f"medication-{run_id}-{index}",
            "status": "active",
            "medicationCodeableConcept": {"coding": coding, "text": medication},
            "subject": {"reference": f"Patient/patient-{run_id}"},
            "effectiveDateTime": effective_date,
        }

