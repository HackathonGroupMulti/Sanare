from __future__ import annotations

from typing import Any

from sanare.pipeline import ClinicalPipeline
from sanare.schemas import EvaluationRequest
from sanare.sharp import SharpContext, enrich_note_with_context, fetch_patient_context

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
except ImportError:
    _mcp = None

_pipeline = ClinicalPipeline()


def _parse_sharp(sharp_context: dict[str, Any] | None) -> SharpContext:
    return SharpContext.model_validate(sharp_context) if sharp_context else SharpContext()


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

        Accepts an optional SHARP context dict with patient_id, fhir_base_url,
        fhir_access_token, encounter_id, scope, and user. When present, fetches
        existing FHIR conditions and medications to enrich the extraction.
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

        When SHARP context is provided the Bundle patient resource is linked
        to the real patient_id from the EHR session.
        """
        ctx = _parse_sharp(sharp_context)
        patient_ctx = fetch_patient_context(ctx)
        enriched = enrich_note_with_context(text, patient_ctx)
        result = _pipeline.analyze_fhir(enriched).model_dump(mode="json")
        if ctx.patient_id and "bundle" in result:
            _rewrite_patient_id(result["bundle"], ctx.patient_id)
        return _sharp_envelope(result, ctx)

    @_mcp.tool
    def fetch_patient_summary(sharp_context: dict[str, Any]) -> dict[str, Any]:
        """Fetch a patient's existing conditions and medications from a FHIR server.

        Requires sharp_context with patient_id, fhir_base_url, and fhir_access_token.
        Returns existing_conditions, existing_medications, and echoes the sharp_context
        for downstream agent chaining.
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
