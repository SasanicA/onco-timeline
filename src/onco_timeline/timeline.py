"""Module 4: build a patient timeline from extracted, harmonized events.

Takes the loose list of events and turns it into a PatientTimeline: sorted by
date, de-duplicated, and exportable to JSON / CSV. It also builds a directed
NetworkX graph (event -> next event) which is a convenient structure both for
visualization and for feeding a sequence model.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import ClinicalEvent, PatientTimeline


def build_timeline(patient_id: str, events: list[ClinicalEvent]) -> PatientTimeline:
    """Assemble and clean a timeline.

    De-duplication: two events on the same date with the same type and canonical
    concept are almost certainly the same real-world event mentioned twice, so we
    keep the higher-confidence one.
    """
    seen: dict[tuple, ClinicalEvent] = {}
    for ev in events:
        key = (ev.date, ev.event_type, ev.canonical or ev.description)
        if key not in seen or ev.confidence > seen[key].confidence:
            seen[key] = ev
    timeline = PatientTimeline(patient_id=patient_id, events=list(seen.values()))
    return timeline


def to_graph(timeline: PatientTimeline):
    """Return a directed NetworkX graph connecting consecutive events."""
    import networkx as nx

    g = nx.DiGraph()
    ordered = timeline.sorted_events()
    for i, ev in enumerate(ordered):
        g.add_node(i, **{
            "date": ev.date,
            "type": ev.event_type.value,
            "label": ev.canonical or ev.description,
            "icd10": ev.icd10,
        })
        if i > 0:
            g.add_edge(i - 1, i)
    return g


def export_json(timeline: PatientTimeline, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(timeline.model_dump_json(indent=2), encoding="utf-8")
    return path


def export_csv(timeline: PatientTimeline, path: str | Path) -> Path:
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(timeline.to_records())
    df.to_csv(path, index=False)
    return path
