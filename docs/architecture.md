# Architecture and design notes

This document expands on the design decisions behind each stage. The guiding
principle throughout is that a clinical pipeline should be **auditable,
local-first, and validated at every boundary**, even in a demonstration.

## Data flow

A single clinical note flows through six stages. The orchestrator
(`pipeline.run_pipeline`) is the only place that knows the full order; every
other module depends only on the shared schema (`schema.py`), so stages can be
tested and swapped independently.

```
note text
   │  EventExtractor.extract()
   ▼
list[ClinicalEvent]            # date, event_type, description, source_text
   │  OntologyMapper.harmonize()
   ▼
list[ClinicalEvent] (+codes)   # canonical, icd10, snomed, loinc
   │  build_timeline()
   ▼
PatientTimeline                # sorted, de-duplicated
   │  featurize_timeline() → RiskModel
   ▼
risk score ∈ [0, 1]
   │  explain_prediction()
   ▼
attributions + SHAP + HTML
```

## Harmonization is type-aware

Matching is restricted to the event's own category: a response event is only
compared against response concepts, a treatment against treatment concepts, a
lab against observation concepts (which carry LOINC codes), and so on. An exact
synonym match is tried first, with Sentence-BERT embeddings as the fallback.
This avoids a whole class of errors where a long sentence embeds close to an
unrelated concept.

## Stage 2 — extraction: why LLM *and* rules

Large language models are excellent at reading messy clinical prose, but they
are non-deterministic and occasionally return malformed output. A purely
rule-based system is the opposite: predictable and auditable, but brittle on
unseen phrasing.

The design uses both. The LLM (via local Ollama) is tried first; its output is
parsed leniently (models wrap JSON in fences) but **validated strictly** through
Pydantic, with malformed rows dropped. If Ollama is unavailable or fails after
retries, a deterministic sentence-level parser takes over. This is the standard
way to de-risk an LLM in a setting where a silent error is unacceptable: keep a
transparent floor underneath the capable-but-unpredictable layer.

## Stage 3 — harmonization: free text to standard codes

The same clinical concept appears under many surface forms ("lung cancer",
"NSCLC", "carcinoma of the lung"). Harmonization collapses these onto a canonical
concept and attaches standardized codes (ICD-10, SNOMED CT) and, for responses,
the RECIST 1.1 category.

Two matching modes share one interface:

- **Embedding mode** (Sentence-BERT, `all-MiniLM-L6-v2`): cosine similarity
  between the event text and the canonical concept names. Handles paraphrase.
- **Lexical mode** (fallback): normalized longest-synonym containment. Needs no
  model download, so CI and the offline demo always work.

The similarity score that justified each mapping is preserved on the event, which
is the first piece of "why did the system decide this".

## Stage 5 — temporal model

Each event becomes a feature vector: a one-hot event type, the (capped,
normalized) time gap since the previous event, a RECIST-style response score, and
a stage flag. The sequence is fed to a `TransformerEncoder` with a learned
positional embedding, mean-pooled over real (non-padded) positions, and passed to
a small MLP head that outputs a risk logit.

The model is trained on synthetic sequences whose labels come from a known latent
rule (progression and short gaps raise risk; good responses lower it). Because the
rule is recoverable, the trained model is a meaningful object for the
explainability stage to interrogate. This is explicitly a methods demonstration,
not a clinical claim.

## Stage 6 — explainability

Two views serve two audiences:

1. **Leave-one-out per-event attribution.** Remove each event, re-run the model,
   measure the change in risk. Large positive change means that event drove risk
   up. This maps every contribution back to a specific source sentence, which is
   the basis for output-to-document traceability.
2. **SHAP on a trajectory summary.** A compact tabular summary (event counts,
   mean gap, response score) gives globally consistent feature attributions.
   SHAP is optional; the pipeline skips it cleanly if it is not installed.

## Testing strategy

The test suite (`tests/test_pipeline.py`) avoids any network call or model
download, so it runs anywhere. It checks: schema validation, date normalization,
rule-based extraction completeness, synonym collapse in harmonization, timeline
sorting and de-duplication, the event graph structure, featurization shape, and a
smoke test that the model trains and emits a valid probability.

## Extending the project

- Add concepts by appending `Concept` rows in `concepts.py`.
- Swap the embedding model via the `ONCO_EMBED_MODEL` environment variable.
- Point at a real Ollama host with `ONCO_OLLAMA_HOST` / `ONCO_OLLAMA_MODEL`.
- Replace the synthetic dataset in `temporal_model.py` with real (de-identified,
  ethically approved) trajectories to move from demonstration toward research.
