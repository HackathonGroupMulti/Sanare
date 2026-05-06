# Connect Sanare To An Agent Builder

Sanare exposes two integration surfaces:

- FastAPI for direct HTTP calls
- MCP HTTP for agent-tool registration

Use MCP for the agent-builder UI shown in the screenshots.

## 1. Run Sanare Locally

```powershell
pip install -r requirements.txt
python scripts/sanare_mcp_server.py --transport http --host 0.0.0.0 --port 9000
```

Local MCP endpoint:

```text
http://127.0.0.1:9000/mcp/
```

## 2. Expose It For Cloud Agent Builders

If the agent builder runs in the cloud, it cannot reach `127.0.0.1`.

For a quick demo:

```powershell
ngrok http 9000
```

Use the HTTPS URL from ngrok:

```text
https://your-ngrok-subdomain.ngrok-free.app/mcp/
```

For a cleaner demo, deploy the MCP server to Render, Railway, Fly.io, Cloud Run, or a small VM.

Start command:

```text
python scripts/sanare_mcp_server.py --transport http --host 0.0.0.0
```

The script reads the hosting provider's `PORT` environment variable automatically.

## 3. Add MCP Server In The Agent UI

In the Tools tab:

```text
Name: Sanare
URL: https://your-public-sanare-url.com/mcp/
Auth: None for local demo
```

Then keep these toggles enabled:

```text
Disable Embedded Tools: On
Disable Community MCP Server: On
```

Sanare should be the only tool source for this clinical extraction agent.

## 4. Expected Tools

The agent should discover:

```text
analyze_clinical_note
analyze_clinical_note_fhir
get_analysis_run
evaluate_clinical_cases
```

## 5. Prompt Addition

Add this to the agent System Prompt:

```text
When clinical text needs structured extraction, call the Sanare MCP tool analyze_clinical_note.

When FHIR output is requested, call analyze_clinical_note_fhir.

Do not answer clinical extraction tasks from memory when the Sanare tool is available.
```

## 6. Smoke Test

Ask the agent:

```text
Analyze: 54M HTN chest pain on lisinopril SOB exertion
```

Expected shape:

```json
{
  "run_id": "...",
  "analysis": {
    "patient_summary": "54-year-old male with hypertension, chest pain, shortness of breath and on lisinopril",
    "conditions": ["hypertension", "chest pain", "shortness of breath"],
    "medications": ["lisinopril"],
    "risk_level": "moderate",
    "next_step": "cardiology referral"
  },
  "phi_entities": [],
  "fhir": {
    "resourceType": "Bundle"
  }
}
