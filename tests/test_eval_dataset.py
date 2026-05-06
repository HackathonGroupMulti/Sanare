import json
from pathlib import Path

from sanare.agent import ClinicalExtractionAgent
from sanare.evaluation import ClinicalEvaluator
from sanare.llm_client import LLMUnavailableError
from sanare.schemas import EvaluationRequest


class OfflineLLMClient:
    def complete_json(self, _system_prompt: str, _user_prompt: str) -> str:
        raise LLMUnavailableError("offline eval mode")


def test_eval_dataset_has_100_cases_and_scores_cleanly() -> None:
    payload = json.loads(Path("examples/eval_cases.json").read_text(encoding="utf-8"))
    assert len(payload["cases"]) == 100

    request = EvaluationRequest.model_validate(payload)
    evaluator = ClinicalEvaluator(agent=ClinicalExtractionAgent(llm_client=OfflineLLMClient()))
    result = evaluator.evaluate(request)

    assert result.total_cases == 100
    assert result.exact_field_accuracy == 1.0
    assert result.risk_accuracy == 1.0
    assert result.hallucination_violations == 0
