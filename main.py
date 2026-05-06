import asyncio
import json
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from sanare.config import load_environment
from sanare.evaluation import ClinicalEvaluator
from sanare.mcp_app import get_mcp, get_pipeline
from sanare.nvidia import nvidia_runtime_report
from sanare.security import OptionalApiKeyMiddleware
from sanare.schemas import (
    AnalyzeEnvelope,
    AnalyzeRequest,
    CapabilityReport,
    ClinicalAnalysis,
    EvaluationRequest,
    EvaluationResult,
    FhirEnvelope,
    NvidiaRuntimeReport,
    RunRecord,
)

VERSION = "0.3.0"

load_environment()

app = FastAPI(title="Sanare Clinical Agent API", version=VERSION)
app.add_middleware(OptionalApiKeyMiddleware)
pipeline = get_pipeline()
evaluator = ClinicalEvaluator(agent=pipeline.agent)

_mcp = get_mcp()
if _mcp:
    app.mount("/mcp", _mcp.http_app())

_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/", include_in_schema=False)
def demo_ui() -> FileResponse:
    return FileResponse(os.path.join(_static_dir, "index.html"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/capabilities", response_model=CapabilityReport)
def capabilities() -> CapabilityReport:
    return CapabilityReport(
        name="Sanare",
        version=VERSION,
        endpoints=[
            "POST /analyze",
            "POST /analyze/full",
            "POST /analyze/fhir",
            "GET /analyze/stream",
            "POST /evaluate",
            "GET /runs/{run_id}",
        ],
        mcp_tools=[
            "analyze_clinical_note",
            "analyze_clinical_note_fhir",
            "get_analysis_run",
            "evaluate_clinical_cases",
        ],
        fhir_resources=["Patient", "Condition", "Observation", "MedicationStatement"],
        optional_integrations=[
            "OpenAI",
            "Gemini",
            "vLLM",
            "Presidio",
            "Postgres JSONB",
            "Qdrant",
            "OpenTelemetry",
            "Temporal",
            "MCP HTTP",
            "NVIDIA NIM",
            "NVIDIA GPU-accelerated biomedical NER (SANARE_ENABLE_NER=1)",
            "NVIDIA Dynamo-ready inference",
            "NeMo Guardrails-ready safety layer",
        ],
        security={
            "api_key_env": "SANARE_API_KEY",
            "accepted_headers": ["x-sanare-api-key", "authorization: Bearer <token>"],
            "enabled": bool(os.getenv("SANARE_API_KEY")),
        },
    )


@app.get("/nvidia/status", response_model=NvidiaRuntimeReport)
def nvidia_status() -> NvidiaRuntimeReport:
    return nvidia_runtime_report()


@app.post("/analyze", response_model=ClinicalAnalysis)
def analyze(request: AnalyzeRequest) -> ClinicalAnalysis:
    return pipeline.analyze_plain(request.text)


@app.post("/analyze/full", response_model=AnalyzeEnvelope)
def analyze_full(request: AnalyzeRequest) -> AnalyzeEnvelope:
    return pipeline.analyze(request.text)


@app.post("/analyze/fhir", response_model=FhirEnvelope)
def analyze_fhir(request: AnalyzeRequest) -> FhirEnvelope:
    return pipeline.analyze_fhir(request.text)


@app.get("/analyze/stream")
async def analyze_stream(text: str, request: Request) -> StreamingResponse:
    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    async def event_gen():
        loop = asyncio.get_event_loop()

        yield _sse({"stage": "deidentifying"})
        redacted = await loop.run_in_executor(None, pipeline.deidentifier.redact, text)
        yield _sse({"stage": "deidentified", "phi_count": len(redacted.phi_entities)})

        yield _sse({"stage": "extracting"})

        # Send keep-alive pings while LLM processes so Render's proxy doesn't buffer
        analysis_future = loop.run_in_executor(None, pipeline.agent.analyze, redacted.text)
        while not analysis_future.done():
            yield ": ping\n\n"
            await asyncio.sleep(0.5)
        analysis = await analysis_future

        yield _sse({"stage": "extracted"})

        fields = analysis.model_dump()
        # Deduplicate: remove bare drug/condition name if a longer form already present
        for list_key in ("conditions", "medications"):
            items = fields.get(list_key) or []
            cleaned = []
            for item in items:
                dominated = any(
                    other != item and item.lower() in other.lower()
                    for other in items
                )
                if not dominated:
                    cleaned.append(item)
            fields[list_key] = cleaned

        for key, value in fields.items():
            yield _sse({"field": key, "value": value})
            await asyncio.sleep(0.4)

        if pipeline.ner is not None:
            yield _sse({"stage": "ner_running"})
            ner_entities = await loop.run_in_executor(None, pipeline._run_ner, redacted.text, analysis)
            yield _sse({"ner_entities": [e.model_dump() for e in ner_entities]})

        yield _sse({"stage": "done"})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/evaluate", response_model=EvaluationResult)
def evaluate(request: EvaluationRequest) -> EvaluationResult:
    return evaluator.evaluate(request)


@app.get("/runs/{run_id}", response_model=RunRecord)
def get_run(run_id: str) -> RunRecord:
    record = pipeline.get_run(run_id)
    if not record:
        raise HTTPException(status_code=404, detail="run not found")
    return record
