from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_analyze_example_payload() -> None:
    response = client.post(
        "/analyze",
        json={"text": "54M HTN chest pain on lisinopril SOB exertion"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["conditions"] == ["hypertension", "chest pain", "shortness of breath"]
    assert body["medications"] == ["lisinopril"]
    assert body["risk_level"] == "moderate"
    assert body["next_step"] == "cardiology referral"


def test_analyze_rejects_empty_text() -> None:
    response = client.post("/analyze", json={"text": ""})

    assert response.status_code == 422


def test_capabilities_reports_sanare_stack() -> None:
    response = client.get("/capabilities")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Sanare"
    assert "MCP HTTP" in body["optional_integrations"]
    assert "NVIDIA NIM" in body["optional_integrations"]
    assert "analyze_clinical_note" in body["mcp_tools"]


def test_nvidia_status_reports_runtime_shape() -> None:
    response = client.get("/nvidia/status")

    assert response.status_code == 200
    body = response.json()
    assert body["recommended_provider"] == "nvidia"
    assert "nim_base_url" in body


def test_optional_api_key_protects_analysis(monkeypatch) -> None:
    monkeypatch.setenv("SANARE_API_KEY", "test-secret")

    blocked = client.post("/analyze", json={"text": "54M HTN"})
    assert blocked.status_code == 401

    allowed = client.post(
        "/analyze",
        json={"text": "54M HTN"},
        headers={"x-sanare-api-key": "test-secret"},
    )
    assert allowed.status_code == 200


def test_analyze_fhir_returns_bundle() -> None:
    response = client.post(
        "/analyze/fhir",
        json={"text": "54M HTN chest pain on lisinopril SOB exertion"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"]
    assert body["bundle"]["resourceType"] == "Bundle"
    resource_types = {entry["resource"]["resourceType"] for entry in body["bundle"]["entry"]}
    assert {"Patient", "Condition", "Observation", "MedicationStatement"}.issubset(resource_types)


def test_analyze_full_redacts_phi_and_persists_run() -> None:
    response = client.post(
        "/analyze/full",
        json={"text": "MRN ABC123 54M HTN chest pain on lisinopril SOB exertion"},
    )

    assert response.status_code == 200
    body = response.json()
    assert "MRN" in body["phi_entities"]

    run_response = client.get(f"/runs/{body['run_id']}")
    assert run_response.status_code == 200
    run = run_response.json()
    assert run["status"] == "completed"
    assert "<MRN>" in run["input_text_redacted"]


def test_evaluate_endpoint_scores_cases() -> None:
    response = client.post(
        "/evaluate",
        json={
            "cases": [
                {
                    "text": "54M HTN chest pain on lisinopril SOB exertion",
                    "expected": {
                        "patient_summary": "54-year-old male with hypertension, chest pain, shortness of breath and on lisinopril",
                        "conditions": ["hypertension", "chest pain", "shortness of breath"],
                        "medications": ["lisinopril"],
                        "risk_level": "moderate",
                        "next_step": "cardiology referral",
                    },
                }
            ]
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total_cases"] == 1
    assert body["risk_accuracy"] == 1.0

