# Sanare

Clinical AI extraction infrastructure for converting de-identified clinical notes into validated JSON and FHIR R4 resources.

## Stack

- FastAPI + Pydantic v2 API contracts
- OpenAI, Gemini, vLLM, or NVIDIA NIM OpenAI-compatible structured output
- Presidio PHI de-identification with regex fallback
- FHIR R4 `Condition`, `MedicationStatement`, and `Observation` mapping
- ICD-10-CM and RxNorm terminology normalization
- Postgres JSONB run persistence with in-memory fallback
- Qdrant retrieval hook for clinical guideline search
- OpenTelemetry tracing with no-op fallback
- Temporal workflow/worker skeleton for durable execution
- pytest golden-case evaluation endpoint
- optional API-key protection for deployed demos
- GitHub Actions CI

## Core Run

```powershell
pip install -r requirements-core.txt
uvicorn main:app --reload
```

Core mode runs without external services. It uses deterministic local extraction when no LLM key or vLLM endpoint is configured.

API key protection is disabled by default. To enable it:

```powershell
$env:SANARE_API_KEY="change-me-for-deployed-demos"
```

Then call protected endpoints with either:

```text
x-sanare-api-key: change-me-for-deployed-demos
```

or:

```text
authorization: Bearer change-me-for-deployed-demos
```

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
$env:DATABASE_URL="postgresql://sanare:sanare@localhost:5432/sanare"
$env:QDRANT_URL="http://localhost:6333"
$env:SANARE_ENABLE_OTEL="1"
$env:SANARE_API_KEY="change-me-for-deployed-demos"
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
GET  /capabilities
GET  /nvidia/status
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

Golden-case eval runner:

```powershell
python scripts/run_eval.py examples/eval_cases.json
```

## NVIDIA-Backed Proof

Verified with a real NVIDIA NIM-compatible hosted request:

```text
Provider: NVIDIA NIM-compatible endpoint
Base URL: https://integrate.api.nvidia.com/v1
Model: meta/llama-3.1-8b-instruct
Input: 54M HTN chest pain on lisinopril SOB exertion
```

Sanare normalized the NVIDIA model output through its deterministic enforcement layer:

```json
{
  "patient_summary": "54-year-old male with hypertension, chest pain, shortness of breath and on lisinopril",
  "conditions": ["hypertension", "chest pain", "shortness of breath"],
  "medications": ["lisinopril"],
  "risk_level": "moderate",
  "next_step": "cardiology referral"
}
```

Golden-case evaluation:

```text
Eval cases: 100 synthetic clinical notes
Field accuracy: 100%
Risk accuracy: 100%
Hallucination violations: 0
```

## Temporal Worker

```powershell
$env:TEMPORAL_ADDRESS="localhost:7233"
python scripts/temporal_worker.py
```

## MCP Server

Run Sanare as an MCP server for agent platforms that support MCP tools:

```powershell
pip install -r requirements.txt
python scripts/sanare_mcp_server.py --transport http --host 0.0.0.0 --port 9000
```

Local MCP endpoint:

```text
http://127.0.0.1:9000/mcp/
```

For hosted agent builders, expose or deploy this endpoint over HTTPS, then add it as an MCP server:

```text
https://your-sanare-domain.com/mcp/
```

Exposed tools:

- `analyze_clinical_note`
- `analyze_clinical_note_fhir`
- `get_analysis_run`
- `evaluate_clinical_cases`

Detailed agent-builder setup:

```text
docs/CONNECT_AGENT.md
```

NVIDIA deployment path:

```text
docs/NVIDIA_DEPLOYMENT.md
```

## Test

```powershell
pytest
```

