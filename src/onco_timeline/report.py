"""Module 7: timeline visualization with Plotly.

The chart is a swimmer-style timeline: one horizontal lane per event type, a
marker for each event, and a short clinical code on each marker (TNM stage at
diagnosis, the RECIST category at response/progression, the drug name for
treatment, the marker name for a lab). The full sentence sits in the hover
tooltip, so the chart stays readable while keeping the detail one hover away.

If per-event risk contributions are passed in, each marker also gets an up or
down arrow with how much that event moved the risk. That puts the explanation
directly onto the timeline.
"""

from __future__ import annotations

from pathlib import Path

from .schema import PatientTimeline

_COLORS = {
    "diagnosis": "#2b3a67",
    "stage": "#5c6bc0",
    "treatment": "#26a69a",
    "observation": "#8e6fc4",
    "response": "#0a7d33",
    "progression": "#b00020",
    "other": "#9e9e9e",
}

_RECIST = {
    "Complete response": "CR",
    "Partial response": "PR",
    "Stable disease": "SD",
    "Progressive disease": "PD",
}

# vertical lane order (first item sits at the bottom of the y-axis)
_LANES = ["diagnosis", "stage", "treatment", "observation", "response", "progression"]


def _short_label(ev) -> str:
    """A compact tag to print on the marker, in clinical shorthand."""
    et = ev.event_type.value
    if et == "diagnosis":
        bits = []
        if ev.stage:
            bits.append(f"Stage {ev.stage}")
        if ev.icd10:
            bits.append(ev.icd10)
        return " | ".join(bits) if bits else (ev.canonical or "Diagnosis")
    if et in ("response", "progression"):
        return _RECIST.get(ev.canonical or "", ev.canonical or et)
    if et == "treatment":
        return (ev.canonical or "Treatment").split()[0]  # first word, e.g. "Cisplatin"
    if et == "observation":
        return ev.canonical or "Lab"
    return ev.canonical or et


def timeline_figure(timeline: PatientTimeline, contributions: dict | None = None):
    """Build the timeline figure.

    contributions: optional dict {event_index: risk_change}, where event_index is
    the position of the event in timeline.sorted_events(). When given, an up/down
    arrow and the value are added under each marker label.
    """
    import plotly.graph_objects as go

    events = timeline.sorted_events()
    xs = [ev.date for ev in events]
    ys = [ev.event_type.value for ev in events]

    # With many points the labels above the markers overlap and become
    # unreadable, so past a threshold we drop the text and keep the hover only.
    show_text = len(events) <= 10

    fig = go.Figure()

    # faint connecting line first, so the markers draw on top of it
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color="#cdd5e6", width=1), hoverinfo="skip", showlegend=False,
    ))

    for i, ev in enumerate(events):
        full = ev.canonical or ev.description
        code = f" ({ev.icd10 or ev.loinc})" if (ev.icd10 or ev.loinc) else ""
        label = _short_label(ev)
        if contributions is not None and i in contributions:
            c = contributions[i]
            if abs(c) >= 0.005:                       # no arrow for ~zero effect
                arrow = "\u25b2" if c > 0 else "\u25bc"   # up / down triangle
                label = f"{label}<br>{arrow}{abs(c):.2f}"
        fig.add_trace(go.Scatter(
            x=[ev.date], y=[ev.event_type.value],
            mode="markers+text" if show_text else "markers",
            marker=dict(size=15, color=_COLORS.get(ev.event_type.value, "#9e9e9e"),
                        line=dict(width=1.5, color="white")),
            text=[label] if show_text else None,
            textposition="top center" if i % 2 == 0 else "bottom center",
            textfont=dict(size=11),
            hovertext=[f"{ev.date}: {full}{code}"], hoverinfo="text",
            showlegend=False,
        ))

    present = [lane for lane in _LANES if lane in ys]
    n = max(len(present), 1)
    fig.update_layout(
        title=dict(text=f"Patient {timeline.patient_id} - clinical trajectory",
                   font=dict(size=15)),
        template="plotly_white",
        height=280,                                   # fits the panel; no scrollbar
        margin=dict(l=28, r=24, t=46, b=40),
        xaxis=dict(title="Date", type="category", tickangle=0),
        # A little headroom above the top lane for its label, but tighter than
        # before so the lanes sit closer together.
        yaxis=dict(title="", categoryorder="array", categoryarray=present,
                   range=[-0.6, n - 1 + 1.8]),
    )
    return fig


def export_timeline_html(timeline: PatientTimeline, path: str | Path,
                         contributions: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = timeline_figure(timeline, contributions=contributions)
    # include_plotlyjs=True embeds the whole library so the chart works offline
    # and inside the embedded browser in the GUI.
    fig.write_html(str(path), include_plotlyjs=True, config={"responsive": True})
    # Remove the default page margins so the embedded browser does not show a
    # scrollbar from a few stray pixels of overflow.
    html = path.read_text(encoding="utf-8")
    html = html.replace(
        "<body>", '<body style="margin:0;padding:0;overflow:hidden;background:#fff">', 1)
    path.write_text(html, encoding="utf-8")
    return path
