from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sanare.agent import ClinicalExtractionAgent
from sanare.config import load_environment
from sanare.evaluation import ClinicalEvaluator
from sanare.llm_client import LLMUnavailableError
from sanare.schemas import EvaluationRequest


class OfflineLLMClient:
    def complete_json(self, _system_prompt: str, _user_prompt: str) -> str:
        raise LLMUnavailableError("offline eval mode")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Sanare golden-case evaluation")
    parser.add_argument("dataset", type=Path, help="Path to an EvaluationRequest JSON file")
    return parser.parse_args()


def main() -> None:
    load_environment()
    args = parse_args()
    request = EvaluationRequest.model_validate_json(args.dataset.read_text(encoding="utf-8"))
    agent = ClinicalExtractionAgent(llm_client=OfflineLLMClient())
    result = ClinicalEvaluator(agent=agent).evaluate(request)
    print(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
