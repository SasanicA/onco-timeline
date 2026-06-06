"""Module 2: extract structured oncology events from free text.

Two extraction backends share one interface:

  * LLM backend (Ollama): builds a strict prompt, calls a local model
    (gemma3 / llama3 / mistral), retries on failure, and validates the returned
    JSON against our Pydantic schema. Local + offline because clinical text is
    sensitive and should not leave the machine.

  * Rule-based fallback: a deterministic parser for the common note patterns.
    It exists so the demo always produces output even with no model installed,
    and so the unit tests have something reproducible to check.

The extractor tries the LLM first (if reachable) and falls back automatically.
That mirrors how you would actually de-risk a clinical pipeline: a fast,
auditable rule layer underneath a more capable but less predictable model.
"""

from __future__ import annotations

import calendar
import json
import re
from typing import Optional

import requests

from .config import Settings, get_settings
from .schema import ClinicalEvent, EventType

# Map month names (and 3-letter abbreviations) to two-digit numbers.
_MONTHS = {m.lower(): f"{i:02d}" for i, m in enumerate(calendar.month_name) if m}
_MONTHS.update({m.lower(): f"{i:02d}" for i, m in enumerate(calendar.month_abbr) if m})

# "January 2023", "Jan 2023", "03/2023", "2023-03", "March, 2023"
_DATE_PATTERNS = [
    re.compile(r"\b(" + "|".join(_MONTHS) + r")[,\.]?\s+(\d{4})\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,2})/(\d{4})\b"),
    re.compile(r"\b(\d{4})-(\d{2})\b"),
]


def _normalize_date(text: str) -> Optional[str]:
    """Pull the first date out of a span of text and return it as YYYY-MM."""
    for pat in _DATE_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        a, b = m.group(1), m.group(2)
        if a.lower() in _MONTHS:                 # "January 2023"
            return f"{b}-{_MONTHS[a.lower()]}"
        if pat.pattern.startswith(r"\b(\d{1,2})/"):  # "03/2023"
            return f"{b}-{int(a):02d}"
        return f"{a}-{b}"                          # "2023-03"
    return None


# --- The strict prompt handed to the local LLM -------------------------
_SYSTEM_PROMPT = """You are a clinical information extraction system for oncology.
Read the clinical note and extract every dated oncological event.

Return ONLY a JSON array. No prose, no markdown fences. Each element MUST have:
  "date": string "YYYY-MM"
  "event_type": one of "diagnosis","stage","treatment","response","progression"
  "description": short label
Optional: "disease", "stage".

Example output:
[{"date":"2023-01","event_type":"diagnosis","disease":"NSCLC","description":"Diagnosed with Stage III NSCLC"}]
"""


class EventExtractor:
    """Extracts ClinicalEvents from note text, LLM-first with a rule fallback."""

    def __init__(self, settings: Settings | None = None, prefer_llm: bool = True):
        self.settings = settings or get_settings()
        self.prefer_llm = prefer_llm

    # -- public API -----------------------------------------------------
    def extract(self, text: str) -> list[ClinicalEvent]:
        """Return a list of validated ClinicalEvents for one note."""
        if self.prefer_llm and self._ollama_available():
            try:
                events = self._extract_with_llm(text)
                if events:
                    return events
            except Exception as exc:  # noqa: BLE001 - we want to fall back on anything
                print(f"[extractor] LLM extraction failed ({exc}); using rule-based fallback.")
        return self._extract_with_rules(text)

    # -- Ollama backend -------------------------------------------------
    def _ollama_available(self) -> bool:
        try:
            r = requests.get(f"{self.settings.ollama_host}/api/tags", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def _extract_with_llm(self, text: str) -> list[ClinicalEvent]:
        prompt = f"{_SYSTEM_PROMPT}\n\nClinical note:\n{text}\n\nJSON:"
        last_err: Exception | None = None

        for attempt in range(self.settings.ollama_retries + 1):
            try:
                resp = requests.post(
                    f"{self.settings.ollama_host}/api/generate",
                    json={
                        "model": self.settings.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        # low temperature: we want extraction, not creativity
                        "options": {"temperature": 0.0},
                    },
                    timeout=self.settings.ollama_timeout_s,
                )
                resp.raise_for_status()
                raw = resp.json().get("response", "")
                return self._parse_json_events(raw)
            except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
                last_err = exc
                print(f"[extractor] attempt {attempt + 1} failed: {exc}")

        raise RuntimeError(f"LLM extraction exhausted retries: {last_err}")

    @staticmethod
    def _parse_json_events(raw: str) -> list[ClinicalEvent]:
        """Be forgiving about what the model returns, strict about validation.

        Models love to wrap JSON in ```json fences or add a sentence before it,
        so we grab the first [...] block, parse it, then validate each row
        through Pydantic and drop anything that does not conform.
        """
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            raise ValueError("no JSON array found in model output")
        data = json.loads(match.group(0))

        events: list[ClinicalEvent] = []
        for row in data:
            try:
                events.append(ClinicalEvent(**row))
            except Exception as exc:  # noqa: BLE001 - skip malformed rows, keep going
                print(f"[extractor] skipping invalid row {row}: {exc}")
        return events

    # -- Rule-based backend --------------------------------------------
    def _extract_with_rules(self, text: str) -> list[ClinicalEvent]:
        """Deterministic parser for the common note shapes.

        It walks the note sentence by sentence, finds a date in each sentence,
        and assigns an event_type from simple keyword cues. This is intentionally
        transparent: every output can be traced to the sentence that produced it.
        """
        events: list[ClinicalEvent] = []
        sentences = re.split(r"(?<=[.\n])\s+", text)

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            date = _normalize_date(sent)
            if date is None:
                continue
            low = sent.lower()

            event_type = EventType.OTHER
            stage = None
            disease = None

            _LAB_TOKENS = ("psa", "cea", "ldh", "ca 19-9", "ca19-9",
                           "ca 125", "ca-125", "tumor marker", "tumour marker")

            if any(k in low for k in ("diagnos", "presented with", "found to have")):
                event_type = EventType.DIAGNOSIS
            elif "progress" in low or "new metasta" in low:
                event_type = EventType.PROGRESSION
            elif any(k in low for k in ("reduced", "shrink", "response", "remission",
                                        "stable disease", "no evidence")):
                event_type = EventType.RESPONSE
            elif any(k in low for k in _LAB_TOKENS):
                event_type = EventType.OBSERVATION
            elif any(k in low for k in ("chemo", "cisplatin", "carboplatin", "docetaxel",
                                        "radiation", "radiotherapy", "surgery", "resection", "excision",
                                        "pembrolizumab", "immunotherapy", "started",
                                        "treatment")):
                event_type = EventType.TREATMENT

            # Stage like "Stage III" / "stage 3" / "stage IIIa"
            stage_m = re.search(r"stage\s+([0-4ivx]+a?b?c?)", low)
            if stage_m:
                stage = stage_m.group(1).upper()
                # A staging statement at diagnosis time stays a diagnosis event,
                # but we keep the stage value on it.
                if event_type == EventType.OTHER:
                    event_type = EventType.STAGE

            # crude disease pickup for the description
            for d in ("nsclc", "lung cancer", "breast cancer", "colorectal",
                      "colon cancer", "myocardial infarction", "heart attack"):
                if d in low:
                    disease = d.upper() if d == "nsclc" else d.title()
                    break

            if event_type == EventType.OTHER:
                # a dated sentence with no recognised cue is not worth a row
                continue

            events.append(
                ClinicalEvent(
                    date=date,
                    event_type=event_type,
                    description=sent.rstrip("."),
                    disease=disease,
                    stage=stage,
                    source_text=sent,
                    confidence=1.0,
                )
            )
        return events
