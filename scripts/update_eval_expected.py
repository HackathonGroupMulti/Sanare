"""Regenerate next_step expected values in eval_cases.json to match current infer_next_step logic."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sanare.risk import infer_next_step


def main() -> None:
    path = ROOT / "examples" / "eval_cases.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    updated = 0
    for case in data["cases"]:
        exp = case["expected"]
        new_next_step = infer_next_step(exp["risk_level"], exp["conditions"], case["text"])
        if new_next_step != exp["next_step"]:
            print(f"  {case['text']!r}")
            print(f"    {exp['next_step']!r} -> {new_next_step!r}")
            exp["next_step"] = new_next_step
            updated += 1

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nUpdated {updated} / {len(data['cases'])} cases.")


if __name__ == "__main__":
    main()
