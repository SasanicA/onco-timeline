"""OncoTimeline: turn unstructured oncology notes into structured, code-mapped
patient timelines, then run a small temporal model with explainability on top.

This package is deliberately split into small, independently testable modules so
each stage of the pipeline can be inspected on its own:

    notes -> extraction -> harmonization -> timeline -> temporal model -> explain

Nothing here touches real patient data. The sample notes are synthetic and the
temporal model is trained on synthetic sequences, so the whole thing runs offline
on a laptop and is safe to publish.
"""

from .config import Settings, get_settings
from .schema import ClinicalEvent, EventType, PatientTimeline
from .note_loader import load_notes, load_note_text
from .extraction import EventExtractor
from .harmonization import OntologyMapper
from .timeline import build_timeline
from .temporal_model import RiskModel, generate_synthetic_dataset, train_model
from .explain import explain_prediction

__all__ = [
    "Settings",
    "get_settings",
    "ClinicalEvent",
    "EventType",
    "PatientTimeline",
    "load_notes",
    "load_note_text",
    "EventExtractor",
    "OntologyMapper",
    "build_timeline",
    "RiskModel",
    "generate_synthetic_dataset",
    "train_model",
    "explain_prediction",
]

__version__ = "0.1.0"
