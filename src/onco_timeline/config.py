"""Central configuration.

Everything that you might reasonably want to tweak from the outside lives here,
so the rest of the code never hardcodes a model name or a folder path. Values can
be overridden with environment variables (handy for CI or for pointing at a real
Ollama host), but the defaults are chosen so that `python demo.py` just works.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


# The project root is two levels up from this file: src/onco_timeline/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Frozen so it behaves like a read-only config object."""

    # --- Data locations -------------------------------------------------
    patients_dir: Path = PROJECT_ROOT / "data" / "patients"
    output_dir: Path = PROJECT_ROOT / "examples" / "output"

    # --- LLM (Ollama) extraction ---------------------------------------
    # If Ollama is running locally we use it; otherwise the extractor falls back
    # to a deterministic rule-based parser so the demo never hard-fails.
    ollama_host: str = os.environ.get("ONCO_OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.environ.get("ONCO_OLLAMA_MODEL", "gemma3")
    ollama_timeout_s: float = float(os.environ.get("ONCO_OLLAMA_TIMEOUT", "60"))
    ollama_retries: int = int(os.environ.get("ONCO_OLLAMA_RETRIES", "2"))

    # --- Semantic harmonization ----------------------------------------
    # all-MiniLM-L6-v2 is small (~80 MB) and fast. If sentence-transformers or
    # the weights are unavailable, harmonization falls back to normalized
    # lexical matching against the same synonym table.
    embedding_model: str = os.environ.get(
        "ONCO_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    similarity_threshold: float = float(os.environ.get("ONCO_SIM_THRESHOLD", "0.45"))

    # --- Temporal model -------------------------------------------------
    model_dim: int = 64
    model_heads: int = 4
    model_layers: int = 2
    max_events: int = 32  # longest event sequence the model will look at
    seed: int = 42

    def ensure_dirs(self) -> None:
        """Create output directories if they do not exist yet."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (so we build it only once)."""
    return Settings()
