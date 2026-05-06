from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sanare.pipeline import ClinicalPipeline
from sanare.config import load_environment
from sanare.schemas import EvaluationRequest

try:
    from fastmcp import FastMCP
except ImportError:
    FastMCP = None


load_environment()

if FastMCP:
    mcp = FastMCP(
        name="Sanare",
        instructions=(
            "Sanare extracts structured clinical facts from unstructured notes, "
            "de-identifies PHI, infers risk, and returns JSON/FHIR-ready output."
        ),
    )
else:
    mcp = None

pipeline = ClinicalPipeline()


def _require_mcp() -> Any:
    if not mcp:
        raise RuntimeError("Install fastmcp to run the Sanare MCP server: pip install fastmcp")
    return mcp


if mcp:

    @mcp.tool
    def analyze_clinical_note(text: str) -> dict[str, Any]:
        """Extract schema-valid clinical JSON from an unstructured note."""
        return pipeline.analyze(text).model_dump(mode="json")


    @mcp.tool
    def analyze_clinical_note_fhir(text: str) -> dict[str, Any]:
        """Extract clinical facts and return a FHIR R4 Bundle."""
        return pipeline.analyze_fhir(text).model_dump(mode="json")


    @mcp.tool
    def get_analysis_run(run_id: str) -> dict[str, Any]:
        """Fetch a prior Sanare analysis run by run_id."""
        record = pipeline.get_run(run_id)
        if not record:
            return {"error": "run not found", "run_id": run_id}
        return record.model_dump(mode="json")


    @mcp.tool
    def evaluate_clinical_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
        """Evaluate extraction accuracy against golden clinical cases."""
        from sanare.evaluation import ClinicalEvaluator

        request = EvaluationRequest.model_validate({"cases": cases})
        return ClinicalEvaluator(agent=pipeline.agent).evaluate(request).model_dump(mode="json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Sanare MCP server")
    parser.add_argument("--transport", default="http", choices=["stdio", "http", "sse"])
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "9000")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = _require_mcp()
    if args.transport == "stdio":
        server.run()
        return
    server.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

