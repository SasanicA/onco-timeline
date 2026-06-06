# Tutorials: the logic, step by step

These two single-file scripts exist to make the logic of the pipeline easy to
read. They are not used by the main project at runtime; they are here for
anyone who wants to understand how the pieces fit together before looking at the
full code. Everything here runs on sample data only and is for learning, not for
any clinical use.

Read them in this order:

### 1. `01_pipeline_rules_only.py`
The whole pipeline with the simplest possible code and no dependencies. It uses
plain keyword rules for extraction, a small dictionary for the standard codes,
and a transparent points formula for the risk score. Run it with:

```
python tutorial/01_pipeline_rules_only.py
```

### 2. `02_pipeline_with_models.py`
The same pipeline, but two steps are upgraded to real models: Sentence-BERT for
matching text to concepts by meaning, and a small PyTorch Transformer for the
risk score (trained on the spot on synthetic sequences). Run it with:

```
python tutorial/02_pipeline_with_models.py
```

After these two, the full project in `src/onco_timeline/` is the same shape with
production details added: a local LLM option for extraction, the type-aware
harmonizer, the timeline export, the explainability report, and the GUI.
