from __future__ import annotations

import types
from typing import Any

from sanare.pipeline import ClinicalPipeline
from sanare.schemas import EvaluationRequest
from sanare.sharp import SharpContext, enrich_note_with_context, fetch_patient_context

_PO_FHIR_EXTENSION = "ai.promptopinion/fhir-context"
_PO_SCOPES = [
    {"name": "patient/Patient.rs", "required": True},
    {"name": "patient/Condition.rs"},
    {"name": "patient/MedicationStatement.rs"},
]

try:
    from fastmcp import FastMCP
    _mcp: Any = FastMCP(
        name="Sanare",
        instructions=(
            "Sanare extracts structured clinical facts from unstructured notes, "
            "de-identifies PHI, infers risk, and returns JSON/FHIR-ready output. "
            "Accepts optional SHARP context to fetch existing patient FHIR data and "
            "link results to a live EHR session."
        ),
    )

    # Declare Prompt Opinion FHIR extension in MCP capabilities
    _orig_caps = _mcp._mcp_server.get_capabilities

    def _patched_caps(self, notification_options, experimental_capabilities):
        caps = _orig_caps(notification_options, experimental_capabilities)
        existing = getattr(caps, "extensions", None) or {}
        caps.extensions = {**existing, _PO_FHIR_EXTENSION: {"scopes": _PO_SCOPES}}
        return caps

    _mcp._mcp_server.get_capabilities = types.MethodType(
        _patched_caps, _mcp._mcp_server
    )

except ImportError:
    _mcp = None

_pipeline = ClinicalPipeline()


def _sharp_from_headers() -> SharpContext:
    """Read Prompt Opinion's FHIR context headers injected at call time."""
    try:
        from fastmcp.server.http import _current_http_request
        request = _current_http_request.get()
        if request is None:
            return SharpContext()
        h = request.headers
        return SharpContext(
            patient_id=h.get("x-patient-id"),
            fhir_base_url=h.get("x-fhir-server-url"),
            fhir_access_token=h.get("x-fhir-access-token"),
        )
    except Exception:
        return SharpContext()


def _parse_sharp(sharp_context: dict[str, Any] | None) -> SharpContext:
    """Prefer explicit sharp_context param; fall back to PO FHIR headers."""
    if sharp_context:
        return SharpContext.model_validate(sharp_context)
    return _sharp_from_headers()


def _sharp_envelope(result: dict[str, Any], ctx: SharpContext) -> dict[str, Any]:
    out = dict(result)
    if ctx.patient_id or ctx.fhir_base_url:
        out["sharp_context"] = ctx.model_dump(exclude_none=True)
    return out


def _rewrite_patient_id(bundle: dict[str, Any], real_patient_id: str) -> None:
    pid = real_patient_id.split("/")[-1]
    fhir_ref = f"Patient/{real_patient_id}" if "/" not in real_patient_id else real_patient_id
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Patient":
            resource["id"] = pid
            entry["fullUrl"] = f"urn:uuid:{pid}"
        subject = resource.get("subject", {})
        if "reference" in subject and subject["reference"].startswith("Patient/"):
            subject["reference"] = fhir_ref


if _mcp:

    @_mcp.tool
    def analyze_clinical_note(
        text: str,
        sharp_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extract schema-valid clinical JSON from an unstructured note.

        SHARP/FHIR context is read automatically from Prompt Opinion headers
        (X-Patient-ID, X-FHIR-Server-URL, X-FHIR-Access-Token) or can be
        passed explicitly via the sharp_context parameter.
        """
        ctx = _parse_sharp(sharp_context)
        patient_ctx = fetch_patient_context(ctx)
        enriched = enrich_note_with_context(text, patient_ctx)
        result = _pipeline.analyze(enriched).model_dump(mode="json")
        result["fhir_patient_id"] = ctx.patient_id
        result["fhir_encounter_id"] = ctx.encounter_id
        return _sharp_envelope(result, ctx)

    @_mcp.tool
    def analyze_clinical_note_fhir(
        text: str,
        sharp_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extract clinical facts and return a FHIR R4 Bundle.

        SHARP/FHIR context is read automatically from Prompt Opinion headers
        or passed explicitly. The Bundle patient resource is linked to the
        real patient_id when context is available.
        """
        ctx = _parse_sharp(sharp_context)
        patient_ctx = fetch_patient_context(ctx)
        enriched = enrich_note_with_context(text, patient_ctx)
        result = _pipeline.analyze_fhir(enriched).model_dump(mode="json")
        if ctx.patient_id and "bundle" in result:
            _rewrite_patient_id(result["bundle"], ctx.patient_id)
        return _sharp_envelope(result, ctx)

    @_mcp.tool
    def fetch_patient_summary(sharp_context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fetch a patient's existing conditions and medications from a FHIR server.

        FHIR context is read automatically from Prompt Opinion headers or
        passed explicitly via sharp_context.
        """
        ctx = _parse_sharp(sharp_context)
        patient_ctx = fetch_patient_context(ctx)
        return _sharp_envelope({"patient_id": ctx.patient_id, **patient_ctx}, ctx)

    @_mcp.tool
    def get_analysis_run(run_id: str) -> dict[str, Any]:
        """Fetch a prior Sanare analysis run by run_id."""
        record = _pipeline.get_run(run_id)
        if not record:
            return {"error": "run not found", "run_id": run_id}
        return record.model_dump(mode="json")

    @_mcp.tool
    def evaluate_clinical_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
        """Evaluate extraction accuracy against golden clinical cases."""
        from sanare.evaluation import ClinicalEvaluator
        request = EvaluationRequest.model_validate({"cases": cases})
        return ClinicalEvaluator(agent=_pipeline.agent).evaluate(request).model_dump(mode="json")


def get_mcp() -> Any:
    return _mcp


def get_pipeline() -> ClinicalPipeline:
    return _pipeline
