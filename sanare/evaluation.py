from __future__ import annotations

from sanare.agent import CONDITION_ALIASES, ClinicalExtractionAgent
from sanare.schemas import EvaluationRequest, EvaluationResult, FieldF1


def _list_f1(predicted: list[str], expected: list[str]) -> tuple[float, float, float]:
    pred_set = set(predicted)
    exp_set = set(expected)
    tp = len(pred_set & exp_set)
    precision = tp / len(pred_set) if pred_set else (1.0 if not exp_set else 0.0)
    recall = tp / len(exp_set) if exp_set else (1.0 if not pred_set else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


class ClinicalEvaluator:
    def __init__(self, agent: ClinicalExtractionAgent | None = None) -> None:
        self.agent = agent or ClinicalExtractionAgent()

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        total_fields = 0
        exact_matches = 0
        risk_matches = 0
        hallucination_violations = 0
        failures: list[dict[str, object]] = []

        cond_precisions: list[float] = []
        cond_recalls: list[float] = []
        cond_f1s: list[float] = []
        med_precisions: list[float] = []
        med_recalls: list[float] = []
        med_f1s: list[float] = []

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

            cp, cr, cf = _list_f1(actual.conditions, expected.conditions)
            cond_precisions.append(cp)
            cond_recalls.append(cr)
            cond_f1s.append(cf)

            mp, mr, mf = _list_f1(actual.medications, expected.medications)
            med_precisions.append(mp)
            med_recalls.append(mr)
            med_f1s.append(mf)

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

        n = len(request.cases)

        def _avg(vals: list[float]) -> float:
            return round(sum(vals) / n, 4) if n else 0.0

        return EvaluationResult(
            total_cases=n,
            exact_field_accuracy=exact_matches / total_fields if total_fields else 0.0,
            risk_accuracy=risk_matches / n if n else 0.0,
            conditions_f1=FieldF1(
                precision=_avg(cond_precisions),
                recall=_avg(cond_recalls),
                f1=_avg(cond_f1s),
            ),
            medications_f1=FieldF1(
                precision=_avg(med_precisions),
                recall=_avg(med_recalls),
                f1=_avg(med_f1s),
            ),
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
