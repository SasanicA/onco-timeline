"""Data model for the whole pipeline.

We use Pydantic v2 for two reasons:
  1. It validates the JSON that comes back from the LLM, so a malformed model
     response becomes a clean, catchable error instead of a mystery crash later.
  2. It gives us free serialization to/from dict and JSON for export.

Keeping the schema in one place means every module agrees on what an "event"
is, which avoids the classic bug where the extractor and the timeline builder
quietly disagree about field names.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class EventType(str, Enum):
    """The oncology event categories we extract from free text.

    These mirror the ones named in the job description: diagnoses, staging,
    treatments and disease course (response / progression).
    """

    DIAGNOSIS = "diagnosis"
    STAGE = "stage"
    TREATMENT = "treatment"
    RESPONSE = "response"
    PROGRESSION = "progression"
    OBSERVATION = "observation"
    OTHER = "other"


# Accepts "2023-01" or "2023" (month optional). We normalize everything to
# YYYY-MM so the timeline can sort reliably.
_DATE_RE = re.compile(r"^\d{4}(-\d{2})?$")


class ClinicalEvent(BaseModel):
    """A single dated event lifted out of a clinical note."""

    date: str = Field(..., description="Event date, normalized to YYYY-MM (or YYYY).")
    event_type: EventType = Field(..., description="One of the EventType values.")
    description: str = Field(..., description="Short human-readable label.")

    # Optional structured fields. Not every event has every field; a treatment
    # has a drug name, a diagnosis has a disease, and so on.
    disease: Optional[str] = None
    stage: Optional[str] = None

    # Where this came from, so we can later trace a model output back to the
    # exact sentence (this is the seed of the "attribution" task).
    source_text: Optional[str] = None

    # Canonical concept + codes are filled in by the OntologyMapper.
    canonical: Optional[str] = None
    icd10: Optional[str] = None
    snomed: Optional[str] = None
    loinc: Optional[str] = None  # for lab / observation events

    # Confidence in [0, 1]. The rule-based extractor uses 1.0 for exact matches;
    # the LLM path can report its own.
    confidence: float = 1.0

    @field_validator("date")
    @classmethod
    def _check_date(cls, v: str) -> str:
        v = v.strip()
        if not _DATE_RE.match(v):
            raise ValueError(f"date must be YYYY-MM or YYYY, got {v!r}")
        # pad bare years to YYYY-01 so all dates are comparable
        return v if "-" in v else f"{v}-01"

    def sort_key(self) -> str:
        """Used to order events chronologically."""
        return self.date


class PatientTimeline(BaseModel):
    """An ordered set of events for one patient."""

    patient_id: str
    events: list[ClinicalEvent] = Field(default_factory=list)

    def sorted_events(self) -> list[ClinicalEvent]:
        return sorted(self.events, key=lambda e: e.sort_key())

    def has_progression(self) -> bool:
        return any(e.event_type == EventType.PROGRESSION for e in self.events)

    def to_records(self) -> list[dict]:
        """Flat list of dicts, convenient for pandas / CSV export."""
        return [e.model_dump() for e in self.sorted_events()]
