"""Unit tests.

These cover the deterministic parts of the pipeline (extraction rules,
harmonization, timeline building, schema validation) plus a smoke test that the
temporal model trains and produces a probability. They do not require Ollama or
any downloaded weights, so they run anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from onco_timeline.schema import ClinicalEvent, EventType, PatientTimeline
from onco_timeline.extraction import EventExtractor, _normalize_date
from onco_timeline.harmonization import OntologyMapper
from onco_timeline.timeline import build_timeline, to_graph
from onco_timeline.temporal_model import (
    featurize_timeline, generate_synthetic_dataset, predict_risk, train_model,
    FEATURE_DIM,
)

SAMPLE = (
    "Patient diagnosed with Stage III NSCLC in January 2023.\n"
    "Started Cisplatin chemotherapy in March 2023.\n"
    "Tumor size reduced by 40 percent in September 2023.\n"
    "Disease progression observed in January 2024."
)


# --- schema -----------------------------------------------------------
def test_date_validation_pads_year():
    ev = ClinicalEvent(date="2023", event_type=EventType.DIAGNOSIS, description="x")
    assert ev.date == "2023-01"


def test_date_validation_rejects_garbage():
    with pytest.raises(Exception):
        ClinicalEvent(date="not-a-date", event_type=EventType.OTHER, description="x")


# --- date normalization ----------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("January 2023", "2023-01"),
    ("Mar 2023", "2023-03"),
    ("03/2023", "2023-03"),
    ("2024-01", "2024-01"),
])
def test_normalize_date(text, expected):
    assert _normalize_date(text) == expected


# --- extraction (rule-based) -----------------------------------------
def test_rule_extraction_finds_all_events():
    extractor = EventExtractor(prefer_llm=False)
    events = extractor.extract(SAMPLE)
    types = {e.event_type for e in events}
    assert EventType.DIAGNOSIS in types
    assert EventType.TREATMENT in types
    assert EventType.RESPONSE in types
    assert EventType.PROGRESSION in types
    assert len(events) == 4


def test_rule_extraction_captures_stage():
    extractor = EventExtractor(prefer_llm=False)
    events = extractor.extract(SAMPLE)
    diag = [e for e in events if e.event_type == EventType.DIAGNOSIS][0]
    assert diag.stage == "III"


# --- harmonization (lexical fallback) --------------------------------
def test_harmonization_maps_nsclc_to_icd10():
    mapper = OntologyMapper(use_embeddings=False)
    events = EventExtractor(prefer_llm=False).extract(SAMPLE)
    events = mapper.harmonize(events)
    diag = [e for e in events if e.event_type == EventType.DIAGNOSIS][0]
    assert diag.canonical == "Non-small cell lung cancer"
    assert diag.icd10 == "C34.9"


def test_harmonization_synonyms_collapse():
    mapper = OntologyMapper(use_embeddings=False)
    r1 = mapper.match("lung cancer")
    r2 = mapper.match("NSCLC")
    assert r1.concept is not None and r2.concept is not None
    assert r1.concept.canonical == r2.concept.canonical


# --- timeline ---------------------------------------------------------
def test_timeline_is_sorted_and_deduped():
    extractor = EventExtractor(prefer_llm=False)
    events = extractor.extract(SAMPLE)
    # duplicate one event; build_timeline should drop it
    events.append(events[0].model_copy())
    tl = build_timeline("p1", events)
    dates = [e.date for e in tl.sorted_events()]
    assert dates == sorted(dates)
    assert len(tl.events) == 4  # dedup removed the copy


def test_timeline_graph_is_a_chain():
    extractor = EventExtractor(prefer_llm=False)
    tl = build_timeline("p1", extractor.extract(SAMPLE))
    g = to_graph(tl)
    assert g.number_of_nodes() == 4
    assert g.number_of_edges() == 3


# --- temporal model ---------------------------------------------------
def test_featurize_shape():
    extractor = EventExtractor(prefer_llm=False)
    tl = build_timeline("p1", extractor.extract(SAMPLE))
    feats = featurize_timeline(tl)
    assert feats.shape[1] == FEATURE_DIM
    assert feats.shape[0] == 4


def test_synthetic_dataset_balanced_enough():
    data = generate_synthetic_dataset(200, seed=0)
    labels = [s.label for s in data]
    # both classes should appear
    assert 0.0 in labels and 1.0 in labels


def test_model_trains_and_predicts_probability():
    model = train_model(epochs=2, verbose=False)
    feats = np.zeros((3, FEATURE_DIM), dtype=np.float32)
    risk = predict_risk(model, feats)
    assert 0.0 <= risk <= 1.0


# --- harmonization: type-aware + LOINC (added) -----------------------
def test_harmonization_is_type_aware():
    """A response event must map to a response concept, not some treatment."""
    mapper = OntologyMapper(use_embeddings=False)
    ev = ClinicalEvent(
        date="2023-06", event_type=EventType.RESPONSE,
        description="Restaging imaging showed a partial response with the tumor reduced in size",
    )
    mapper.harmonize([ev])
    assert ev.canonical == "Partial response"


def test_observation_maps_to_loinc():
    mapper = OntologyMapper(use_embeddings=False)
    ev = ClinicalEvent(
        date="2021-10", event_type=EventType.OBSERVATION,
        description="Serum PSA was elevated",
    )
    mapper.harmonize([ev])
    assert ev.canonical == "Prostate specific antigen"
    assert ev.loinc == "2857-1"


def test_extractor_picks_up_lab_observation():
    note = "Serum CEA was elevated in December 2021."
    events = EventExtractor(prefer_llm=False).extract(note)
    assert len(events) == 1
    assert events[0].event_type == EventType.OBSERVATION
