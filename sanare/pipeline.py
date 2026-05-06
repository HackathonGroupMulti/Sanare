from __future__ import annotations

from uuid import uuid4

from sanare.agent import ClinicalExtractionAgent
from sanare.deidentify import ClinicalDeidentifier
from sanare.fhir import FhirMapper
from sanare.retrieval import ClinicalRetriever, build_retriever
from sanare.schemas import AnalyzeEnvelope, ClinicalAnalysis, FhirEnvelope, RunRecord
from sanare.storage import RunStore, build_run_store
from sanare.tracing import Tracer


class ClinicalPipeline:
    def __init__(
        self,
        agent: ClinicalExtractionAgent | None = None,
        deidentifier: ClinicalDeidentifier | None = None,
        fhir_mapper: FhirMapper | None = None,
        run_store: RunStore | None = None,
        retriever: ClinicalRetriever | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.agent = agent or ClinicalExtractionAgent()
        self.deidentifier = deidentifier or ClinicalDeidentifier()
        self.fhir_mapper = fhir_mapper or FhirMapper()
        self.run_store = run_store or build_run_store()
        self.retriever = retriever or build_retriever()
        self.tracer = tracer or Tracer()

    def analyze(self, text: str) -> AnalyzeEnvelope:
        run_id = str(uuid4())
        with self.tracer.span("clinical_pipeline.analyze", run_id=run_id) as span:
            try:
                redacted = self.deidentifier.redact(text)
                span.set_attribute("phi.entity_count", len(redacted.phi_entities))

                evidence = self.retriever.search(redacted.text)
                span.set_attribute("retrieval.hit_count", len(evidence))

                analysis = self.agent.analyze(redacted.text)
                bundle = self.fhir_mapper.bundle(run_id, analysis)

                record = RunRecord(
                    run_id=run_id,
                    status="completed",
                    input_text_redacted=redacted.text,
                    analysis=analysis,
                    fhir=bundle,
                    phi_entities=redacted.phi_entities,
                )
                self.run_store.save(record)
                return AnalyzeEnvelope(run_id=run_id, analysis=analysis, phi_entities=redacted.phi_entities, fhir=bundle)
            except Exception as exc:
                span.record_exception(exc)
                self.run_store.save(
                    RunRecord(run_id=run_id, status="failed", input_text_redacted="[failed]", error=str(exc))
                )
                raise

    def analyze_plain(self, text: str) -> ClinicalAnalysis:
        return self.analyze(text).analysis

    def analyze_fhir(self, text: str) -> FhirEnvelope:
        envelope = self.analyze(text)
        return FhirEnvelope(run_id=envelope.run_id, bundle=envelope.fhir)

    def get_run(self, run_id: str) -> RunRecord | None:
        return self.run_store.get(run_id)

