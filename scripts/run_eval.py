"""Run Sanare golden-case and clinical evaluation datasets."""
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
    parser = argparse.ArgumentParser(description="Run Sanare evaluation")
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=ROOT / "examples" / "eval_cases.json",
        help="Path to an EvaluationRequest JSON file (default: examples/eval_cases.json)",
    )
    parser.add_argument(
        "--clinical",
        action="store_true",
        help="Also run the independent clinical eval dataset (requires LLM)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force offline/heuristic mode (no LLM)",
    )
    return parser.parse_args()


def run_dataset(path: Path, evaluator: ClinicalEvaluator, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {path}")
    print(f"{'='*60}")

    data = json.loads(path.read_text(encoding="utf-8"))
    # Strip the _note metadata key if present
    data.pop("_note", None)
    request = EvaluationRequest.model_validate(data)
    result = evaluator.evaluate(request)

    print(f"  total_cases:            {result.total_cases}")
    print(f"  exact_field_accuracy:   {result.exact_field_accuracy:.4f}")
    print(f"  risk_accuracy:          {result.risk_accuracy:.4f}")
    print(f"  conditions  F1:         {result.conditions_f1.f1:.4f}  "
          f"(P={result.conditions_f1.precision:.3f} R={result.conditions_f1.recall:.3f})")
    print(f"  medications F1:         {result.medications_f1.f1:.4f}  "
          f"(P={result.medications_f1.precision:.3f} R={result.medications_f1.recall:.3f})")
    print(f"  hallucination_violations: {result.hallucination_violations}")
    print(f"  failures:               {len(result.failures)}")

    if result.failures:
        print("\n  --- Failures ---")
        for f in result.failures[:5]:
            print(f"  case {f['case_index']}: {f['field_results']}")
        if len(result.failures) > 5:
            print(f"  ... and {len(result.failures) - 5} more")


def main() -> None:
    load_environment()
    args = parse_args()

    if args.offline:
        agent = ClinicalExtractionAgent(llm_client=OfflineLLMClient())
    else:
        agent = ClinicalExtractionAgent()

    evaluator = ClinicalEvaluator(agent=agent)

    run_dataset(args.dataset, evaluator, "Heuristic Consistency Eval")

    if args.clinical:
        clinical_path = ROOT / "examples" / "clinical_eval_cases.json"
        run_dataset(clinical_path, evaluator, "Independent Clinical Eval (LLM-required)")


if __name__ == "__main__":
    main()
