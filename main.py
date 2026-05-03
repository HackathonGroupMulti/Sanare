from fastapi import FastAPI, HTTPException

from iasis.evaluation import ClinicalEvaluator
from iasis.pipeline import ClinicalPipeline
from iasis.schemas import (
    AnalyzeEnvelope,
    AnalyzeRequest,
    ClinicalAnalysis,
    EvaluationRequest,
    EvaluationResult,
    FhirEnvelope,
    RunRecord,
)

app = FastAPI(title="IASIS Clinical Agent API", version="0.1.0")
pipeline = ClinicalPipeline()
evaluator = ClinicalEvaluator(agent=pipeline.agent)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
