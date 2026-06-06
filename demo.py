"""One-click demo: `python demo.py`.

Trains the temporal model once on synthetic data, then runs the full pipeline on
every sample note in data/patients/ and writes JSON, CSV, an interactive timeline
HTML, and an explanation HTML into examples/output/.

Works with zero external services: if Ollama is not running it uses the
rule-based extractor; if Sentence-BERT weights are not present it uses lexical
harmonization. So this script always produces output.
"""

from __future__ import annotations

from onco_timeline.config import get_settings
from onco_timeline.extraction import EventExtractor
from onco_timeline.harmonization import OntologyMapper
from onco_timeline.note_loader import load_notes
from onco_timeline.pipeline import run_pipeline
from onco_timeline.temporal_model import train_model


def main() -> None:
    settings = get_settings()
    settings.ensure_dirs()

    print("=" * 64)
    print("OncoTimeline demo")
    print("=" * 64)

    # Build the shared components once so we do not reload models per note.
    print("\n[1/3] Training temporal model on synthetic data ...")
    model = train_model(verbose=True)

    print("\n[2/3] Loading extractor + harmonizer ...")
    extractor = EventExtractor()
    mapper = OntologyMapper()
    print(f"      extraction backend : {'Ollama' if extractor._ollama_available() else 'rule-based'}")
    print(f"      harmonization mode : {mapper.mode}")

    print("\n[3/3] Processing notes ...\n")
    for patient_id, text in load_notes():
        result = run_pipeline(patient_id, text, model=model,
                              extractor=extractor, mapper=mapper)
        print(f"--- {patient_id} ---------------------------------------------")
        for ev in result.timeline.sorted_events():
            code = ev.icd10 or ev.loinc or ev.snomed
            code = f"  [{code}]" if code else ""
            concept = f" -> {ev.canonical}" if ev.canonical else ""
            print(f"  {ev.date}  {ev.event_type.value:<11}{ev.description[:48]}{concept}{code}")
        print(f"  predicted trajectory risk: {result.risk:.1%}")
        top = result.explanation['attributions'][0]
        print(f"  top driver: {top.label} ({top.date}), contribution {top.contribution:+.3f}")
        print(f"  artifacts: {result.json_path.name}, {result.csv_path.name}, "
              f"{result.timeline_html.name}, {result.explanation_html.name}\n")

    print(f"All artifacts written to: {settings.output_dir}")


if __name__ == "__main__":
    main()
