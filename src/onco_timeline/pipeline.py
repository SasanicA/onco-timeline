"""End-to-end pipeline orchestration.

One function, run_pipeline(), wires the modules together for a single note so
that demo.py, the tests, and the GUI all share exactly the same code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import get_settings
from .explain import explain_prediction
from .extraction import EventExtractor
from .harmonization import OntologyMapper
from .report import export_timeline_html
from .schema import PatientTimeline
from .temporal_model import RiskModel, featurize_timeline, predict_risk, train_model
from .timeline import build_timeline, export_csv, export_json


@dataclass
class PipelineResult:
    timeline: PatientTimeline
    risk: float
    explanation: dict
    json_path: Path | None = None
    csv_path: Path | None = None
    timeline_html: Path | None = None
    explanation_html: Path | None = None


def run_pipeline(patient_id: str, note_text: str, model: RiskModel | None = None,
                 extractor: EventExtractor | None = None,
                 mapper: OntologyMapper | None = None,
                 write_outputs: bool = True) -> PipelineResult:
    """Run the full chain for one note: extract -> harmonize -> timeline ->
    predict -> explain, optionally writing all artifacts to the output dir."""
    settings = get_settings()
    extractor = extractor or EventExtractor()
    mapper = mapper or OntologyMapper()
    model = model or train_model(verbose=False)

    events = extractor.extract(note_text)
    events = mapper.harmonize(events)
    timeline = build_timeline(patient_id, events)
    feats = featurize_timeline(timeline)
    risk = predict_risk(model, feats)

    result = PipelineResult(timeline=timeline, risk=risk, explanation={})

    if write_outputs:
        out = settings.output_dir
        out.mkdir(parents=True, exist_ok=True)
        result.json_path = export_json(timeline, out / f"{patient_id}_timeline.json")
        result.csv_path = export_csv(timeline, out / f"{patient_id}_timeline.csv")

        # Explain first, so we can put the per-event contributions onto the chart.
        result.explanation = explain_prediction(
            model, timeline, html_path=out / f"{patient_id}_explanation.html")
        result.explanation_html = out / f"{patient_id}_explanation.html"

        # Map each event's contribution by its position in the sorted timeline.
        contributions = {a.index: a.contribution
                         for a in result.explanation["attributions"]}
        result.timeline_html = export_timeline_html(
            timeline, out / f"{patient_id}_timeline.html", contributions=contributions)
    else:
        result.explanation = explain_prediction(model, timeline)

    return result
