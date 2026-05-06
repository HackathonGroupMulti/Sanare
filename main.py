import os

from fastapi import FastAPI, HTTPException

from sanare.config import load_environment
from sanare.evaluation import ClinicalEvaluator
from sanare.nvidia import nvidia_runtime_report
from sanare.pipeline import ClinicalPipeline
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

VERSION = "0.2.0"

load_environment()

app = FastAPI(title="Sanare Clinical Agent API", version=VERSION)
app.add_middleware(OptionalApiKeyMiddleware)
pipeline = ClinicalPipeline()
evaluator = ClinicalEvaluator(agent=pipeline.agent)


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


@app.post("/evaluate", response_model=EvaluationResult)
def evaluate(request: EvaluationRequest) -> EvaluationResult:
    return evaluator.evaluate(request)


@app.get("/runs/{run_id}", response_model=RunRecord)
def get_run(run_id: str) -> RunRecord:
    record = pipeline.get_run(run_id)
    if not record:
        raise HTTPException(status_code=404, detail="run not found")
    return record
