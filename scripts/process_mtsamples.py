"""
Process MTSamples Kaggle dataset into Sanare eval cases.

Usage:
    # Step 1: download the dataset
    python -c "import kagglehub; print(kagglehub.dataset_download('tboyle10/medicaltranscriptions'))"

    # Step 2: run this script against the downloaded path
    python scripts/process_mtsamples.py <path_to_dataset> [--limit 100] [--offline] [--out examples/mtsamples_eval.json]

    # Step 3: open the review file, correct any wrong expected values, then use as eval input
    python scripts/run_eval.py examples/mtsamples_eval.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RELEVANT_SPECIALTIES = {
    "Cardiovascular / Pulmonary",
    "Internal Medicine",
    "Nephrology",
    "Endocrinology",
    "Neurology",
    "Hematology - Oncology",
    "Gastroenterology",
    "Pulmonology",
    "Rheumatology",
    "General Medicine",
    "Emergency Room Reports",
    "Consult - History and Phy.",
    "SOAP / Chart / Progress Notes",
}

# Max characters to take from a transcription as the input note
_NOTE_MAX_CHARS = 400


def _excerpt(text: str) -> str:
    """Take the first meaningful paragraph of a transcription as a note excerpt."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # Prefer the first 1-3 sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    excerpt = ""
    for s in sentences:
        if len(excerpt) + len(s) > _NOTE_MAX_CHARS:
            break
        excerpt = (excerpt + " " + s).strip()
    return excerpt or text[:_NOTE_MAX_CHARS].strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process MTSamples into Sanare eval cases")
    p.add_argument("dataset_path", type=Path, help="Path returned by kagglehub.dataset_download()")
    p.add_argument("--limit", type=int, default=100, help="Max cases to process (default 100)")
    p.add_argument("--offline", action="store_true", help="Use heuristic only, no LLM")
    p.add_argument("--out", type=Path, default=ROOT / "examples" / "mtsamples_eval.json",
                   help="Output path (default examples/mtsamples_eval.json)")
    p.add_argument("--specialty", action="append", dest="specialties",
                   help="Filter to specific specialty (can repeat). Default: all relevant.")
    return p.parse_args()


def main() -> None:
    try:
        import pandas as pd
    except ImportError:
        print("pandas required: pip install pandas")
        sys.exit(1)

    args = parse_args()

    # Find the CSV file in the dataset directory
    csv_files = list(args.dataset_path.glob("*.csv"))
    if not csv_files:
        print(f"No CSV found in {args.dataset_path}")
        sys.exit(1)
    csv_path = csv_files[0]
    print(f"Loading {csv_path} ...")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")

    # Strip leading/trailing whitespace from specialty column (CSV artifact)
    df["medical_specialty"] = df["medical_specialty"].str.strip()

    # Filter to relevant specialties
    target_specialties = set(args.specialties) if args.specialties else RELEVANT_SPECIALTIES
    df = df[df["medical_specialty"].isin(target_specialties)].copy()
    print(f"After specialty filter: {len(df)} rows")

    # Drop rows with empty transcription
    df = df[df["transcription"].notna() & (df["transcription"].str.strip() != "")]
    df = df.head(args.limit)
    print(f"Processing {len(df)} cases...")

    # Import Sanare pipeline
    from sanare.config import load_environment
    load_environment()

    if args.offline:
        from sanare.llm_client import LLMUnavailableError
        from sanare.agent import ClinicalExtractionAgent

        class OfflineLLMClient:
            def complete_json(self, _s: str, _u: str) -> str:
                raise LLMUnavailableError("offline")

        agent = ClinicalExtractionAgent(llm_client=OfflineLLMClient())
    else:
        from sanare.agent import ClinicalExtractionAgent
        agent = ClinicalExtractionAgent()

    cases = []
    for i, (_, row) in enumerate(df.iterrows()):
        note = _excerpt(str(row.get("transcription", "")))
        if not note:
            continue

        try:
            result = agent.analyze(note)
            case = {
                "_specialty": row.get("medical_specialty", ""),
                "_sample_name": row.get("sample_name", ""),
                "_review_status": "pending",  # mark as 'verified' after manual review
                "text": note,
                "expected": result.model_dump(mode="json"),
            }
            cases.append(case)
            print(f"  [{i+1}/{len(df)}] {row.get('medical_specialty', '')} — risk={result.risk_level}")
        except Exception as e:
            print(f"  [{i+1}/{len(df)}] ERROR: {e}")

    output = {
        "_note": (
            "MTSamples-derived eval cases. Fields prefixed with _ are metadata. "
            "Set _review_status='verified' after manually checking each expected output. "
            "Remove _ fields before using as EvaluationRequest input."
        ),
        "cases": cases,
    }

    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote {len(cases)} cases to {args.out}")
    print("Next: open the file, verify expected values, set _review_status='verified', then strip _ fields.")


if __name__ == "__main__":
    main()
