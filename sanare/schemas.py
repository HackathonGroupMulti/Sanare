from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

RiskLevel = Literal["low", "moderate", "high"]
RunStatus = Literal["completed", "failed"]


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1)


class ClinicalAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_summary: str
    conditions: list[str]
    medications: list[str]
    risk_level: RiskLevel
    next_step: str

    @field_validator("patient_summary", "next_step")
    @classmethod
    def require_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field cannot be empty")
        return cleaned

    @field_validator("conditions", "medications")
    @classmethod
    def clean_string_list(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for value in values:
            item = value.strip().lower()
            if item and item not in seen:
                cleaned.append(item)
                seen.add(item)
        return cleaned


class CodedConcept(BaseModel):
    text: str
    system: str
    code: str
    display: str


class DeidentifiedNote(BaseModel):
    text: str
    phi_entities: list[str] = Field(default_factory=list)


class AnalyzeEnvelope(BaseModel):
    run_id: str
    analysis: ClinicalAnalysis
    phi_entities: list[str]
    fhir: dict[str, Any]


class FhirEnvelope(BaseModel):
    run_id: str
    bundle: dict[str, Any]


class RunRecord(BaseModel):
    run_id: str
    status: RunStatus
    input_text_redacted: str
    analysis: ClinicalAnalysis | None = None
    fhir: dict[str, Any] | None = None
    phi_entities: list[str] = Field(default_factory=list)
    error: str | None = None


class EvaluationCase(BaseModel):
    text: str = Field(..., min_length=1)
    expected: ClinicalAnalysis


class EvaluationRequest(BaseModel):
    cases: list[EvaluationCase] = Field(..., min_length=1)


class EvaluationResult(BaseModel):
    total_cases: int
    exact_field_accuracy: float
    risk_accuracy: float
    hallucination_violations: int
    failures: list[dict[str, Any]]


class CapabilityReport(BaseModel):
    name: str
    version: str
    endpoints: list[str]
    mcp_tools: list[str]
    fhir_resources: list[str]
    optional_integrations: list[str]
    security: dict[str, Any]


class NvidiaRuntimeReport(BaseModel):
    nvidia_smi_available: bool
    cuda_visible_devices: str | None = None
    driver_version: str | None = None
    gpus: list[dict[str, Any]]
    recommended_provider: str
    nim_base_url: str
    notes: list[str]

