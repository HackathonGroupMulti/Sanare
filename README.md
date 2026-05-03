# IASIS Agent System

Clinical AI extraction infrastructure for converting de-identified clinical notes into validated JSON and FHIR R4 resources.

## Stack

- FastAPI + Pydantic v2 API contracts
- OpenAI, Gemini, or vLLM OpenAI-compatible structured output
- Presidio PHI de-identification with regex fallback
- FHIR R4 `Condition`, `MedicationStatement`, and `Observation` mapping
- ICD-10-CM and RxNorm terminology normalization
- Postgres JSONB run persistence with in-memory fallback
- Qdrant retrieval hook for clinical guideline search
- OpenTelemetry tracing with no-op fallback
- Temporal workflow/worker skeleton for durable execution
- pytest golden-case evaluation endpoint

## Core Run

```powershell
pip install -r requirements-core.txt
uvicorn main:app --reload
```

Core mode runs without external services. It uses deterministic local extraction when no LLM key or vLLM endpoint is configured.

## Full Infra

```powershell
pip install -r requirements.txt
copy .env.example .env
docker compose up postgres qdrant temporal temporal-ui
uvicorn main:app --reload
```

GPU vLLM profile:

```powershell
docker compose --profile gpu up vllm
```

Environment:

```powershell
$env:LLM_PROVIDER="vllm"
$env:LLM_BASE_URL="http://localhost:8001/v1"
$env:LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
$env:DATABASE_URL="postgresql://iasis:iasis@localhost:5432/iasis"
$env:QDRANT_URL="http://localhost:6333"
$env:IASIS_ENABLE_OTEL="1"
```

OpenAI alternative:

```powershell
$env:OPENAI_API_KEY="your_key"
$env:LLM_PROVIDER="openai"
$env:LLM_MODEL="gpt-4o-mini"
```

## Endpoints

```text
GET  /health
POST /analyze
POST /analyze/full
POST /analyze/fhir
POST /evaluate
GET  /runs/{run_id}
```

Analyze:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/analyze `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"text":"54M HTN chest pain on lisinopril SOB exertion"}'
```

FHIR:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/analyze/fhir `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"text":"54M HTN chest pain on lisinopril SOB exertion"}'
```

Evaluation:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/evaluate `
  -Method Post `
  -ContentType "application/json" `
  -InFile examples/evaluate_request.json
```

## Temporal Worker

```powershell
$env:TEMPORAL_ADDRESS="localhost:7233"
python scripts/temporal_worker.py
```

## Test

```powershell
pytest
```
