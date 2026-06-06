"""Module 1: load clinical notes from disk.

Nothing clever here on purpose. It reads .txt files from data/patients/ and
hands back (patient_id, text) pairs. The patient_id is just the file stem, which
keeps the rest of the pipeline from caring about file paths.
"""

from __future__ import annotations

from pathlib import Path

from .config import get_settings


def load_note_text(path: str | Path) -> str:
    """Read a single note file as UTF-8 text."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No note at {path}")
    return path.read_text(encoding="utf-8").strip()


def load_notes(patients_dir: str | Path | None = None) -> list[tuple[str, str]]:
    """Load every .txt note in the patients directory.

    Returns a list of (patient_id, note_text), sorted by patient_id so runs are
    reproducible. Raises if the directory is empty, because a silent empty run is
    more confusing than a clear error.
    """
    directory = Path(patients_dir) if patients_dir else get_settings().patients_dir
    if not directory.exists():
        raise FileNotFoundError(f"Patients directory not found: {directory}")

    notes: list[tuple[str, str]] = []
    for txt in sorted(directory.glob("*.txt")):
        notes.append((txt.stem, load_note_text(txt)))

    if not notes:
        raise ValueError(f"No .txt notes found in {directory}")
    return notes
