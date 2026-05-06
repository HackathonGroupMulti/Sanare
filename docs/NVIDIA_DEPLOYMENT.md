# NVIDIA Deployment Path

Sanare can run against hosted NVIDIA NIM endpoints or a self-hosted NIM container.

## Hosted NVIDIA NIM

```powershell
$env:LLM_PROVIDER="nvidia"
$env:NVIDIA_API_KEY="nvapi-your-key"
$env:NVIDIA_NIM_BASE_URL="https://integrate.api.nvidia.com/v1"
$env:NVIDIA_MODEL="meta/llama-3.1-8b-instruct"
uvicorn main:app --reload
```

Sanare routes extraction through the OpenAI-compatible wrapper while preserving schema enforcement, PHI redaction, FHIR mapping, and repair.

## Local NIM On NVIDIA GPU

Requirements:

- NVIDIA GPU with enough VRAM for the selected model
- NVIDIA driver
- Docker with NVIDIA Container Toolkit
- NGC API key

Run:

```powershell
$env:NGC_API_KEY="your-ngc-key"
docker compose -f docker-compose.yml -f docker-compose.nvidia.yml --profile nvidia up
```

Inside Docker, Sanare calls:

```text
http://nim-llm:8000/v1/chat/completions
```

The NIM endpoint is exposed on the host as:

```text
http://localhost:8001/v1
```

## Runtime Check

```powershell
Invoke-RestMethod http://127.0.0.1:8000/nvidia/status
```

This reports `nvidia-smi` availability, visible CUDA devices, GPU names, memory, and active NIM base URL.

## NVIDIA Resume Positioning

Use this only after you demo it with NVIDIA NIM:

```text
Built Sanare, a clinical AI extraction service using FastAPI, MCP, NVIDIA NIM-compatible inference, FHIR R4 mapping, PHI redaction, Postgres audit logging, and golden-case evaluation for schema-valid clinical note structuring.
```

## Future NVIDIA Enhancements

- NeMo Guardrails as a policy layer before and after extraction
- NVIDIA Dynamo for distributed inference serving when traffic justifies it
- TensorRT-LLM-backed NIM for optimized latency
- Triton Inference Server for non-LLM clinical models
