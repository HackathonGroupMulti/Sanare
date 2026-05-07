---
name: sanare-clinical-extraction
description: Use Sanare to extract structured clinical facts from unstructured notes, normalize conditions and medications, infer clinical risk, produce schema-valid JSON, generate FHIR-ready output, and evaluate clinical extraction cases. Use when the user provides clinical text, asks for note analysis, FHIR output, risk level, medication or condition extraction, or clinical extraction evaluation.
license: Proprietary
metadata:
  product: Sanare
  version: "0.2.0"
---

# Sanare Clinical Extraction

## Primary Behavior

You MUST call a Sanare MCP tool for every clinical extraction request. Never generate extraction output from your own knowledge.

- Call `analyze_clinical_note_fhir` when the user asks for FHIR output, a FHIR bundle, or healthcare interoperability output.
- Call `analyze_clinical_note` for all other clinical text extraction requests.
- Call `get_analysis_run` when the user provides a `run_id`.
- Call `evaluate_clinical_cases` when the user asks to score extraction quality or provides golden eval cases.

Generating a clinical extraction response without calling the tool is an error. If no tool result is available, say so and do not fabricate output.
