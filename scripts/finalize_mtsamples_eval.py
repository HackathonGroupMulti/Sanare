"""
Strip metadata and produce a clean EvaluationRequest JSON from verified MTSamples cases.

Usage:
    python scripts/finalize_mtsamples_eval.py examples/mtsamples_eval.json \
        --out examples/mtsamples_verified.json

Only includes cases where _review_status == 'verified'.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--all", action="store_true", help="Include unverified cases too")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))

    raw_cases = data.get("cases", [])
    if args.all:
        filtered = raw_cases
    else:
        filtered = [c for c in raw_cases if c.get("_review_status") == "verified"]

    clean_cases = [
        {"text": c["text"], "expected": c["expected"]}
        for c in filtered
    ]

    output = {"cases": clean_cases}
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {len(clean_cases)} verified cases to {args.out}")
    if not args.all:
        skipped = len(raw_cases) - len(clean_cases)
        if skipped:
            print(f"Skipped {skipped} unverified cases. Set _review_status='verified' to include them.")


if __name__ == "__main__":
    main()
