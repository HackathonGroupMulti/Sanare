from __future__ import annotations

from uuid import uuid4

from sanare.agent import ClinicalExtractionAgent
from sanare.deidentify import ClinicalDeidentifier
from sanare.fhir import FhirMapper
from sanare.ner import BiomedicalNER, build_ner
from sanare.retrieval import ClinicalRetriever, build_retriever
from sanare.schemas import (
    AnalyzeEnvelope,
    ClinicalAnalysis,
    FhirEnvelope,
    NEREntity,
    RunRecord,
)
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
        ner: BiomedicalNER | None = None,
    ) -> None:
        self.agent = agent or ClinicalExtractionAgent()
        self.deidentifier = deidentifier or ClinicalDeidentifier()
        self.fhir_mapper = fhir_mapper or FhirMapper()
        self.run_store = run_store or build_run_store()
        self.retriever = retriever or build_retriever()
        self.tracer = tracer or Tracer()
        self.ner = ner if ner is not None else build_ner()

    def analyze(self, text: str) -> AnalyzeEnvelope:
        run_id = str(uuid4())
        with self.tracer.span("clinical_pipeline.analyze", run_id=run_id) as span:
            try:
                redacted = self.deidentifier.redact(text)
                span.set_attribute("phi.entity_count", len(redacted.phi_entities))

                evidence = self.retriever.search(redacted.text)
                span.set_attribute("retrieval.hit_count", len(evidence))

                analysis = self.agent.analyze(redacted.text, [h.text for h in evidence])
                ner_entities = self._run_ner(redacted.text, analysis)
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
                return AnalyzeEnvelope(
                    run_id=run_id,
                    analysis=analysis,
                    phi_entities=redacted.phi_entities,
                    fhir=bundle,
                    ner_entities=ner_entities,
                )
            except Exception as exc:
                span.record_exception(exc)
                self.run_store.save(
                    RunRecord(run_id=run_id, status="failed", input_text_redacted="[failed]", error=str(exc))
                )
                raise

    def _run_ner(self, text: str, analysis: ClinicalAnalysis) -> list[NEREntity]:
        if self.ner is None:
            return []

        ml_conditions = dict(self.ner.conditions(text))
        ml_medications = dict(self.ner.medications(text))
        llm_conditions = set(analysis.conditions)
        llm_medications = set(analysis.medications)

        entities: list[NEREntity] = []

        for term in llm_conditions | ml_conditions.keys():
            in_ml = term in ml_conditions
            in_llm = term in llm_conditions
            if in_ml and in_llm:
                source, confidence = "hybrid", max(ml_conditions[term], 0.85)
            elif in_ml:
                source, confidence = "ml", ml_conditions[term]
            else:
                source, confidence = "llm", 0.75
            entities.append(NEREntity(text=term, label="condition", confidence=round(confidence, 3), source=source))

        for term in llm_medications | ml_medications.keys():
            in_ml = term in ml_medications
            in_llm = term in llm_medications
            if in_ml and in_llm:
                source, confidence = "hybrid", max(ml_medications[term], 0.85)
            elif in_ml:
                source, confidence = "ml", ml_medications[term]
            else:
                source, confidence = "llm", 0.75
            entities.append(NEREntity(text=term, label="medication", confidence=round(confidence, 3), source=source))

        return entities

    def analyze_plain(self, text: str) -> ClinicalAnalysis:
        return self.analyze(text).analysis

    def analyze_fhir(self, text: str) -> FhirEnvelope:
        envelope = self.analyze(text)
        return FhirEnvelope(run_id=envelope.run_id, bundle=envelope.fhir)

    def get_run(self, run_id: str) -> RunRecord | None:
        return self.run_store.get(run_id)
