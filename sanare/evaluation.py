from __future__ import annotations

from sanare.agent import CONDITION_ALIASES, ClinicalExtractionAgent
from sanare.schemas import EvaluationRequest, EvaluationResult


class ClinicalEvaluator:
    def __init__(self, agent: ClinicalExtractionAgent | None = None) -> None:
        self.agent = agent or ClinicalExtractionAgent()

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        total_fields = 0
        exact_matches = 0
        risk_matches = 0
        hallucination_violations = 0
        failures: list[dict[str, object]] = []

        for index, case in enumerate(request.cases):
            actual = self.agent.analyze(case.text)
            expected = case.expected

            field_results = {
                "patient_summary": actual.patient_summary == expected.patient_summary,
                "conditions": actual.conditions == expected.conditions,
                "medications": actual.medications == expected.medications,
                "risk_level": actual.risk_level == expected.risk_level,
                "next_step": actual.next_step == expected.next_step,
            }
            total_fields += len(field_results)
            exact_matches += sum(1 for matched in field_results.values() if matched)
            risk_matches += int(field_results["risk_level"])

            violations = self._hallucination_violations(case.text, actual.conditions + actual.medications)
            hallucination_violations += violations

            if not all(field_results.values()) or violations:
                failures.append(
                    {
                        "case_index": index,
                        "field_results": field_results,
                        "hallucination_violations": violations,
                        "actual": actual.model_dump(mode="json"),
                        "expected": expected.model_dump(mode="json"),
                    }
                )

        return EvaluationResult(
            total_cases=len(request.cases),
            exact_field_accuracy=exact_matches / total_fields if total_fields else 0.0,
            risk_accuracy=risk_matches / len(request.cases),
            hallucination_violations=hallucination_violations,
            failures=failures,
        )

    def _hallucination_violations(self, text: str, entities: list[str]) -> int:
        source = text.lower()
        violations = 0
        for entity in entities:
            normalized = entity.lower()
            if normalized in source:
                continue
            if any(normalized == value and alias in source for alias, value in CONDITION_ALIASES.items()):
                continue
            violations += 1
        return violations

