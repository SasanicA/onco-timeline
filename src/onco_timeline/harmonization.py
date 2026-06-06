"""Module 3: semantic harmonization.

Free text says "lung cancer", "NSCLC", "carcinoma of the lung", all the same
clinical concept. This module maps each event's surface text onto a canonical
concept from concepts.py and attaches its standard codes (ICD-10, SNOMED CT,
LOINC, RECIST category).

Two important design choices:

  1. Matching is restricted to the event's own category. A "response" event is
     only compared against response concepts, a "treatment" event only against
     treatment concepts, and so on. Without this, a long sentence like
     "Restaging imaging showed a partial response" could land on an unrelated
     concept just because the whole sentence happened to embed close to it.

  2. For each event we try an exact synonym match first (fast, predictable,
     auditable), and only fall back to Sentence-BERT embeddings when no synonym
     is found. Embeddings handle paraphrase; the synonym pass keeps the obvious
     cases exact. If sentence-transformers or its weights are unavailable, the
     code stays in synonym-only mode, so the demo and the tests always run.

Either way the event ends up with canonical / icd10 / snomed / loinc filled in,
and we keep the similarity score that justified the mapping. That score is the
first "why did the system decide this" signal, which the explainability module
later builds on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .concepts import CONCEPTS, Concept, concepts_in
from .config import Settings, get_settings
from .schema import ClinicalEvent, EventType

# Which concept categories each event type is allowed to map to.
_CATEGORY_FOR_EVENT = {
    EventType.DIAGNOSIS: {"diagnosis"},
    EventType.STAGE: {"diagnosis"},
    EventType.TREATMENT: {"treatment"},
    EventType.RESPONSE: {"response"},
    EventType.PROGRESSION: {"response"},
    EventType.OBSERVATION: {"observation"},
    EventType.OTHER: None,  # no restriction
}


@dataclass
class MatchResult:
    concept: Optional[Concept]
    score: float
    method: str  # "lexical" | "embedding" | "none"


class OntologyMapper:
    """Maps event text to canonical concepts and their codes."""

    def __init__(self, settings: Settings | None = None, use_embeddings: bool = True):
        self.settings = settings or get_settings()
        self._model = None
        self._concept_embeddings = None  # numpy array aligned with CONCEPTS
        if use_embeddings:
            self._try_load_model()

    # -- model loading (best effort) -----------------------------------
    def _try_load_model(self) -> None:
        """Load Sentence-BERT if available; otherwise stay in lexical mode."""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self.settings.embedding_model)
            names = [c.canonical for c in CONCEPTS]
            self._concept_embeddings = self._model.encode(
                names, convert_to_numpy=True, normalize_embeddings=True
            )
            print(f"[harmonizer] using embeddings: {self.settings.embedding_model}")
        except Exception as exc:  # noqa: BLE001 - any failure -> lexical fallback
            print(f"[harmonizer] embeddings unavailable ({exc}); using lexical matching.")
            self._model = None

    @property
    def mode(self) -> str:
        return "embedding" if self._model is not None else "lexical"

    # -- matching -------------------------------------------------------
    def match(self, text: str, categories: set[str] | None = None) -> MatchResult:
        """Find the best concept for a piece of text, within the given categories.

        Exact synonyms are tried first; embeddings are the fallback.
        """
        candidates = concepts_in(categories)

        # 1) exact synonym containment, longest synonym wins
        low = text.lower()
        best: Optional[Concept] = None
        best_len = 0
        for c in candidates:
            for syn in c.synonyms:
                if syn in low and len(syn) > best_len:
                    best, best_len = c, len(syn)
        if best is not None:
            return MatchResult(best, 0.95, "lexical")

        # 2) embedding fallback (only if a model is loaded)
        if self._model is not None:
            return self._match_embedding(text, candidates)

        return MatchResult(None, 0.0, "none")

    def _match_embedding(self, text: str, candidates: list[Concept]) -> MatchResult:
        import numpy as np

        idx = [CONCEPTS.index(c) for c in candidates]
        if not idx:
            return MatchResult(None, 0.0, "embedding")
        sub = self._concept_embeddings[idx]
        vec = self._model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = sub @ vec  # cosine similarity (vectors are normalized)
        local_best = int(np.argmax(sims))
        score = float(sims[local_best])
        if score < self.settings.similarity_threshold:
            return MatchResult(None, score, "embedding")
        return MatchResult(candidates[local_best], score, "embedding")

    # -- pipeline entry point ------------------------------------------
    def harmonize(self, events: list[ClinicalEvent]) -> list[ClinicalEvent]:
        """Attach canonical concept + codes to each event."""
        for ev in events:
            categories = _CATEGORY_FOR_EVENT.get(ev.event_type, None)

            result = MatchResult(None, 0.0, "none")
            for query in (ev.disease, ev.description):
                if not query:
                    continue
                candidate = self.match(query, categories)
                if candidate.concept is not None:
                    result = candidate
                    break

            if result.concept is not None:
                ev.canonical = result.concept.canonical
                ev.icd10 = result.concept.icd10
                ev.snomed = result.concept.snomed
                ev.loinc = result.concept.loinc
                ev.confidence = round(
                    min(ev.confidence, result.score) if result.score else ev.confidence, 3
                )
        return events
