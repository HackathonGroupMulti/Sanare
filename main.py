import asyncio
import json
import os
import queue as stdlib_queue

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

_mcp = get_mcp()
_mcp_asgi = _mcp.http_app(path="/") if _mcp else None

app = FastAPI(
    title="Sanare Clinical Agent API",
    version=VERSION,
    lifespan=_mcp_asgi.lifespan if _mcp_asgi else None,
)
app.add_middleware(OptionalApiKeyMiddleware)
pipeline = get_pipeline()
evaluator = ClinicalEvaluator(agent=pipeline.agent)

if _mcp_asgi:
    app.mount("/mcp", _mcp_asgi)

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

    def _extract_codes(bundle: dict) -> list[dict]:
        _system_labels = {
            "http://hl7.org/fhir/sid/icd-10-cm": "ICD-10",
            "http://www.nlm.nih.gov/research/umls/rxnorm": "RxNorm",
        }
        codes = []
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            rtype = resource.get("resourceType")
            if rtype in ("Condition", "Observation"):
                text_val = resource.get("code", {}).get("text", "")
                for c in resource.get("code", {}).get("coding", []):
                    codes.append({
                        "term": text_val,
                        "system": _system_labels.get(c.get("system", ""), c.get("system", "")),
                        "code": c.get("code"),
                        "display": c.get("display"),
                    })
            elif rtype == "MedicationStatement":
                text_val = resource.get("medicationCodeableConcept", {}).get("text", "")
                for c in resource.get("medicationCodeableConcept", {}).get("coding", []):
                    codes.append({
                        "term": text_val,
                        "system": _system_labels.get(c.get("system", ""), c.get("system", "")),
                        "code": c.get("code"),
                        "display": c.get("display"),
                    })
        return codes

    async def event_gen():
        loop = asyncio.get_event_loop()

        yield _sse({"stage": "deidentifying"})
        redacted = await loop.run_in_executor(None, pipeline.deidentifier.redact, text)
        yield _sse({"stage": "deidentified", "phi_entities": redacted.phi_entities})

        evidence_hits = await loop.run_in_executor(None, pipeline.retriever.search, redacted.text)
        evidence = [h.text for h in evidence_hits]

        yield _sse({"stage": "extracting"})

        # Stream tokens from LLM via thread-safe queue
        token_q: stdlib_queue.Queue = stdlib_queue.Queue()

        def _run_streaming():
            try:
                for item in pipeline.agent.analyze_streaming(redacted.text, evidence):
                    token_q.put(("token" if isinstance(item, str) else "result", item))
            except Exception as exc:
                token_q.put(("error", str(exc)))
            finally:
                token_q.put(("sentinel", None))

        loop.run_in_executor(None, _run_streaming)

        analysis = None
        while True:
            try:
                kind, value = token_q.get_nowait()
            except stdlib_queue.Empty:
                yield ": ping\n\n"
                await asyncio.sleep(0.05)
                continue
            if kind == "token":
                yield _sse({"token": value})
            elif kind == "result":
                analysis = value
            elif kind == "error":
                yield _sse({"error": value})
                return
            elif kind == "sentinel":
                break

        if analysis is None:
            yield _sse({"error": "extraction failed"})
            return

        yield _sse({"stage": "extracted"})

        # Build FHIR bundle to get codes
        bundle = pipeline.fhir_mapper.bundle(str(id(analysis)), analysis)
        codes = _extract_codes(bundle)
        if codes:
            yield _sse({"codes": codes})
            await asyncio.sleep(0.1)

        # Grounding check — flag terms not found in the original note
        note_lower = redacted.text.lower()

        def _grounded(term: str) -> bool:
            import re as _re
            return bool(_re.search(rf"\b{_re.escape(term.lower())}\b", note_lower))

        fields = analysis.model_dump()
        for list_key in ("conditions", "medications"):
            items = fields.get(list_key) or []
            cleaned = [item for item in items if not any(
                other != item and item.lower() in other.lower() for other in items
            )]
            fields[list_key] = cleaned

        for key, value in fields.items():
            extra: dict = {}
            if isinstance(value, list):
                extra["grounded"] = {item: _grounded(item) for item in value}
            yield _sse({"field": key, "value": value, **extra})
            await asyncio.sleep(0.35)

        if pipeline.ner is not None:
            yield _sse({"stage": "ner_running"})
            ner_entities = await loop.run_in_executor(None, pipeline._run_ner, redacted.text, analysis)
            yield _sse({"ner_entities": [e.model_dump() for e in ner_entities]})

        yield _sse({"fhir_bundle": bundle})
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
