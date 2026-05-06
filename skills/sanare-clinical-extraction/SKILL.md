---
name: sanare-clinical-extraction
description: Use Sanare to extract structured clinical facts from unstructured notes, normalize conditions and medications, infer clinical risk, produce schema-valid JSON, generate FHIR-ready output, and evaluate clinical extraction cases. Use when the user provides clinical text, asks for note analysis, FHIR output, risk level, medication or condition extraction, or clinical extraction evaluation.
license: Proprietary
metadata:
  product: Sanare
  version: "0.2.0"
---

# Sanare Clinical Extraction

Use this skill when the user provides clinical text or asks for structured clinical extraction.

## Primary Behavior

Always prefer Sanare MCP tools over answering from memory.

Use:

- `analyze_clinical_note` for standard clinical extraction.
- `analyze_clinical_note_fhir` when the user asks for FHIR, healthcare interoperability output, or resource mapping.
- `get_analysis_run` when the user provides a `run_id`.
- `evaluate_clinical_cases` when the user asks to score extraction quality or provides golden eval cases.

## Extraction Rules

- Do not hallucinate conditions, medications, symptoms, demographics, or care plans.
- Only extract entities supported by the input text.
- Normalize abbreviations:
  - HTN -> hypertension
  - SOB -> shortness of breath
  - DM -> diabetes
- Infer risk from condition severity, symptoms, and clinical plausibility.
- Do not provide diagnosis as medical advice.
- Do not recommend treatment beyond safe next-step routing.
- If evidence is insufficient, keep fields empty or use routine follow-up.

## Risk Guidance

High:

- stroke symptoms
- syncope
- sepsis
- respiratory distress
- myocardial infarction concern
- pulmonary embolism concern
- unstable presentation

Moderate:

- chest pain
- exertional shortness of breath
- uncontrolled chronic disease
- abdominal pain
- multiple risk factors

Low:

- minor symptoms
- stable chronic follow-up
- medication refill
- routine checkup

## Response Rules

If a Sanare tool returns JSON, return the JSON directly unless the user asks for explanation.

If the user asks for a brief explanation, summarize the output without adding unsupported clinical facts.

For normal extraction requests, output only the structured result.

## Example Input

```text
54M HTN chest pain on lisinopril SOB exertion
```

## Expected Output Shape

```json
{
  "patient_summary": "54-year-old male with hypertension, chest pain, shortness of breath and on lisinopril",
  "conditions": ["hypertension", "chest pain", "shortness of breath"],
  "medications": ["lisinopril"],
  "risk_level": "moderate",
  "next_step": "cardiology referral"
}
```
