"""Module 6: explainability / attribution.

Two complementary views, because "explainability" in a clinical setting means
different things to different readers:

  1. Per-event attribution (occlusion): we remove each event from the sequence,
     re-run the model, and measure how much the risk score changes. Large drop =>
     that event was pushing risk up. This ties directly back to source events,
     which is the "link model outputs to their source documents" task. It always
     works (no extra dependency) and is trivial to explain to a clinician.

  2. SHAP on a tabular summary of the trajectory (counts, mean gap, response
     score). This gives globally-consistent feature attributions. SHAP is
     optional; if it is not installed we skip it gracefully.

Both feed a self-contained HTML report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .schema import EventType, PatientTimeline
from .temporal_model import EVENT_ORDER, RiskModel, featurize_timeline, predict_risk


@dataclass
class EventAttribution:
    index: int
    label: str
    date: str
    contribution: float  # signed change in risk attributable to this event


def occlusion_attribution(model: RiskModel, timeline: PatientTimeline) -> tuple[float, list[EventAttribution]]:
    """Leave-one-event-out attribution over the real timeline."""
    events = timeline.sorted_events()
    feats = featurize_timeline(timeline)
    base = predict_risk(model, feats)

    attributions: list[EventAttribution] = []
    for i, ev in enumerate(events):
        reduced = np.delete(feats, i, axis=0)
        if reduced.shape[0] == 0:
            reduced = np.zeros((1, feats.shape[1]), dtype=np.float32)
        without = predict_risk(model, reduced)
        # contribution = how much risk drops when we remove this event
        attributions.append(
            EventAttribution(
                index=i,
                label=ev.canonical or ev.description,
                date=ev.date,
                contribution=round(base - without, 4),
            )
        )
    attributions.sort(key=lambda a: abs(a.contribution), reverse=True)
    return base, attributions


# --- tabular summary used by SHAP -------------------------------------
SUMMARY_FEATURES = ["n_treatment", "n_progression", "n_response", "mean_gap", "response_score"]


def summarize(features: np.ndarray) -> np.ndarray:
    type_block = features[:, : len(EVENT_ORDER)]
    return np.array([
        type_block[:, EVENT_ORDER.index(EventType.TREATMENT)].sum(),
        type_block[:, EVENT_ORDER.index(EventType.PROGRESSION)].sum(),
        type_block[:, EVENT_ORDER.index(EventType.RESPONSE)].sum(),
        features[:, len(EVENT_ORDER)].mean(),
        features[:, len(EVENT_ORDER) + 1].sum(),
    ], dtype=np.float32)


def shap_summary(model: RiskModel, timeline: PatientTimeline, background_n: int = 50):
    """Return (feature_names, shap_values) or None if SHAP is unavailable."""
    try:
        import shap  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"[explain] SHAP not available ({exc}); skipping SHAP view.")
        return None

    from .temporal_model import generate_synthetic_dataset

    # Build a background distribution of summary vectors from synthetic data.
    bg = np.stack([summarize(s.features) for s in generate_synthetic_dataset(background_n)])
    target = summarize(featurize_timeline(timeline)).reshape(1, -1)

    # A prediction function over summary vectors. We approximate the sequence
    # model with a quick surrogate so SHAP can perturb tabular features: fit a
    # tiny linear probe from summaries -> model risk on the background set.
    from sklearn.linear_model import LogisticRegression

    bg_risks = []
    for s in generate_synthetic_dataset(background_n):
        bg_risks.append(predict_risk(model, s.features))
    bg_risks = np.array(bg_risks)
    probe = LogisticRegression(max_iter=500)
    # guard against a degenerate all-same-label background
    labels = (bg_risks > bg_risks.mean()).astype(int)
    if labels.min() == labels.max():
        return None
    probe.fit(bg, labels)

    explainer = shap.LinearExplainer(probe, bg)
    values = explainer.shap_values(target)[0]
    return SUMMARY_FEATURES, np.asarray(values, dtype=float)


def explain_prediction(model: RiskModel, timeline: PatientTimeline,
                       html_path: str | Path | None = None) -> dict:
    """Run both attribution views and (optionally) write an HTML report."""
    base_risk, attribs = occlusion_attribution(model, timeline)
    shap_result = shap_summary(model, timeline)

    result = {
        "risk": base_risk,
        "attributions": attribs,
        "shap": shap_result,
    }

    if html_path is not None:
        _write_html(result, timeline, Path(html_path))
        result["html_path"] = str(html_path)
    return result


def _write_html(result: dict, timeline: PatientTimeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    risk = result["risk"]
    rows = "".join(
        f"<tr><td>{a.date}</td><td>{a.label}</td>"
        f"<td style='text-align:right'>{a.contribution:+.3f}</td></tr>"
        for a in result["attributions"]
    )
    shap_block = ""
    if result["shap"] is not None:
        names, vals = result["shap"]
        shap_rows = "".join(
            f"<tr><td>{n}</td><td style='text-align:right'>{v:+.3f}</td></tr>"
            for n, v in zip(names, vals)
        )
        shap_block = f"""
        <h2>SHAP feature attribution (trajectory summary)</h2>
        <table><tr><th>Feature</th><th>SHAP value</th></tr>{shap_rows}</table>"""

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>OncoTimeline explanation - {timeline.patient_id}</title>
<style>
 body{{font-family:system-ui,Arial,sans-serif;margin:2rem;color:#1a2238;}}
 h1{{color:#2b3a67}} table{{border-collapse:collapse;margin:1rem 0;width:100%}}
 td,th{{border:1px solid #d0d7e2;padding:6px 10px;font-size:14px}}
 th{{background:#eef2f8;text-align:left}}
 .risk{{font-size:1.6rem;font-weight:700;color:{'#b00020' if risk > 0.5 else '#0a7d33'}}}
</style></head><body>
<h1>Risk explanation - patient {timeline.patient_id}</h1>
<p>Predicted trajectory risk: <span class="risk">{risk:.1%}</span></p>
<h2>Per-event attribution (leave-one-out)</h2>
<p>How much the risk score drops when each event is removed. Positive means the
event pushed risk upward.</p>
<table><tr><th>Date</th><th>Event</th><th>Contribution</th></tr>{rows}</table>
{shap_block}
<p style="color:#777;font-size:12px;margin-top:2rem">Demonstration on synthetic /
example data. Not a validated clinical model.</p>
</body></html>"""
    path.write_text(html, encoding="utf-8")
