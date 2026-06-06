"""Optional PySide6 GUI (Module 9): wide dashboard layout.

Layout:
    Row 1  : Clinical note (left) with Browse + Run buttons beside it (right)
    Row 2  : Timeline table | Why (attribution) table | Risk (centered)
    Row 3  : Trajectory chart (full width)

The Run button doubles as a status light: "Loading model..." (grey, disabled)
while the model prepares, "Run pipeline" (green) when ready, "Processing..."
(amber, disabled) during a run. Browse loads a .txt note from data/patients/.

Run it with:  python -m onco_timeline.gui
"""

from __future__ import annotations

# --- Quiet the harmless HuggingFace / Torch startup noise ------------
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import logging
import warnings
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*nested tensors.*")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

import sys

from .config import get_settings
from .pipeline import PipelineResult, run_pipeline


def _build_app():  # pragma: no cover - GUI code is not exercised in CI
    from PySide6.QtCore import QThread, Signal, QUrl, Qt
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout,
        QHBoxLayout, QWidget, QLabel, QTableWidget, QTableWidgetItem,
        QHeaderView, QFileDialog,
    )

    try:
        from PySide6.QtWebEngineWidgets import QWebEngineView
        HAS_WEB = True
    except Exception:
        HAS_WEB = False

    class ModelLoader(QThread):
        ready = Signal(object)
        failed = Signal(str)

        def run(self):
            try:
                from .temporal_model import train_model
                from .extraction import EventExtractor
                from .harmonization import OntologyMapper
                model = train_model(verbose=False)
                extractor = EventExtractor()
                mapper = OntologyMapper()
                self.ready.emit((model, extractor, mapper))
            except Exception as exc:  # noqa: BLE001
                self.failed.emit(str(exc))

    class Worker(QThread):
        done = Signal(object)
        failed = Signal(str)

        def __init__(self, note_text, model, extractor, mapper):
            super().__init__()
            self.note_text = note_text
            self.model = model
            self.extractor = extractor
            self.mapper = mapper

        def run(self):
            try:
                result = run_pipeline(
                    "gui_patient", self.note_text,
                    model=self.model, extractor=self.extractor,
                    mapper=self.mapper, write_outputs=True,
                )
                self.done.emit(result)
            except Exception as exc:  # noqa: BLE001
                self.failed.emit(str(exc))

    def _label(text):
        lab = QLabel(text)
        lab.setStyleSheet("font-size:14px;font-weight:700;color:#2b3a67;")
        return lab

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("OncoTimeline")
            self.resize(1280, 820)

            self.model = None
            self.extractor = None
            self.mapper = None
            self.worker = None

            page = QWidget()
            outer = QVBoxLayout(page)

            # Row 1: note (left) + Browse/Run buttons (right) -------------
            outer.addWidget(_label("Clinical note  (or Browse a .txt from data/patients)"))
            top = QHBoxLayout()
            self.note_edit = QTextEdit()
            self.note_edit.setPlainText(
                "Patient diagnosed with Stage III NSCLC in January 2023.\n"
                "Started Cisplatin chemotherapy in March 2023.\n"
                "Tumor size reduced by 40 percent in September 2023.\n"
                "Disease progression observed in January 2024."
            )
            self.note_edit.setMaximumHeight(96)
            top.addWidget(self.note_edit, 8)

            btn_col = QVBoxLayout()
            self.browse_btn = QPushButton("Browse...")
            self.browse_btn.setMinimumHeight(30)
            self.browse_btn.clicked.connect(self._browse)
            self.run_btn = QPushButton()
            self.run_btn.setMinimumHeight(62)
            self.run_btn.clicked.connect(self._run)
            btn_col.addWidget(self.browse_btn)
            btn_col.addWidget(self.run_btn)
            top.addLayout(btn_col, 2)
            outer.addLayout(top)

            # The Run button starts as a "loading" indicator.
            self._set_run_state("Loading model...", "#9e9e9e", enabled=False)

            # Row 2: timeline | attribution | risk (centered) ------------
            mid = QHBoxLayout()

            tl_col = QVBoxLayout()
            tl_col.addWidget(_label("Timeline (the trajectory)"))
            self.timeline_table = QTableWidget(0, 6)
            self.timeline_table.setHorizontalHeaderLabels(
                ["Date", "Type", "Concept", "ICD-10", "SNOMED", "LOINC"])
            self._set_widths(self.timeline_table, wide_col=2)
            tl_col.addWidget(self.timeline_table)
            mid.addLayout(tl_col, 5)

            ex_col = QVBoxLayout()
            ex_col.addWidget(_label("Why? (per-event attribution)"))
            self.explain_table = QTableWidget(0, 3)
            self.explain_table.setHorizontalHeaderLabels(
                ["Date", "Event", "Contribution"])
            self._set_widths(self.explain_table, wide_col=1)
            ex_col.addWidget(self.explain_table)
            mid.addLayout(ex_col, 5)

            rk_col = QVBoxLayout()
            rk_col.addWidget(_label("Trajectory risk"))
            rk_col.addStretch(1)
            self.risk_label = QLabel("Run to see\nthe risk.")
            self.risk_label.setAlignment(Qt.AlignCenter)
            self.risk_label.setStyleSheet("font-size:34px;font-weight:800;")
            rk_col.addWidget(self.risk_label, alignment=Qt.AlignCenter)
            rk_col.addStretch(1)
            mid.addLayout(rk_col, 2)

            outer.addLayout(mid, 4)

            # Row 3: chart (full width) ----------------------------------
            self.web = None
            if HAS_WEB:
                outer.addWidget(_label("Trajectory chart"))
                self.web = QWebEngineView()
                self.web.setMinimumHeight(330)
                outer.addWidget(self.web, 5)

            self.setCentralWidget(page)

            self.statusBar().showMessage("Preparing model on synthetic data, please wait ...")
            self.loader = ModelLoader()
            self.loader.ready.connect(self._on_ready)
            self.loader.failed.connect(self._startup_failed)
            self.loader.start()

        # -- helpers ----------------------------------------------------
        def _set_run_state(self, text, bg, enabled):
            """Update the Run button's label, colour, and enabled state."""
            self.run_btn.setText(text)
            self.run_btn.setEnabled(enabled)
            self.run_btn.setStyleSheet(
                f"font-weight:700; color:white; background-color:{bg};"
                f"border:none; border-radius:6px;")

        @staticmethod
        def _set_widths(table, wide_col):
            table.setMinimumHeight(180)
            header = table.horizontalHeader()
            for c in range(table.columnCount()):
                mode = (QHeaderView.Stretch if c == wide_col
                        else QHeaderView.ResizeToContents)
                header.setSectionResizeMode(c, mode)

        def _browse(self):
            start_dir = str(get_settings().patients_dir)
            path, _ = QFileDialog.getOpenFileName(
                self, "Open a clinical note", start_dir,
                "Text files (*.txt);;All files (*)")
            if not path:
                return
            try:
                with open(path, encoding="utf-8") as f:
                    self.note_edit.setPlainText(f.read())
                self.statusBar().showMessage(f"Loaded {path}")
            except Exception as exc:  # noqa: BLE001
                self.statusBar().showMessage(f"Could not open file: {exc}")

        def _startup_failed(self, message):
            self._set_run_state("Load failed", "#b00020", enabled=False)
            self.statusBar().showMessage(f"Startup error: {message}")

        def _on_ready(self, bundle):
            self.model, self.extractor, self.mapper = bundle
            self._set_run_state("Run pipeline", "#0a7d33", enabled=True)  # green = go
            mode = getattr(self.mapper, "mode", "?")
            self.statusBar().showMessage(
                f"Ready. Harmonization mode: {mode}. Edit or Browse a note, then Run.")

        def _run(self):
            if self.model is None:
                return
            self._set_run_state("Processing...", "#e08600", enabled=False)  # amber = busy
            self.statusBar().showMessage("Processing note ...")
            self.worker = Worker(self.note_edit.toPlainText(),
                                 self.model, self.extractor, self.mapper)
            self.worker.done.connect(self._show)
            self.worker.failed.connect(self._error)
            self.worker.start()

        def _error(self, message):
            self._set_run_state("Run pipeline", "#0a7d33", enabled=True)
            self.statusBar().showMessage(f"Error: {message}")

        def _show(self, result: PipelineResult):
            events = result.timeline.sorted_events()
            self.timeline_table.setRowCount(len(events))
            for r, ev in enumerate(events):
                for c, val in enumerate([ev.date, ev.event_type.value,
                                         ev.canonical or "", ev.icd10 or "",
                                         ev.snomed or "", ev.loinc or ""]):
                    self.timeline_table.setItem(r, c, QTableWidgetItem(str(val)))

            if self.web is not None and result.timeline_html is not None:
                try:
                    self.web.load(QUrl.fromLocalFile(str(result.timeline_html.resolve())))
                except Exception:
                    pass

            color = "#b00020" if result.risk > 0.5 else "#0a7d33"
            self.risk_label.setText(f"{result.risk:.1%}")
            self.risk_label.setStyleSheet(
                f"font-size:38px;font-weight:800;color:{color};")

            attribs = result.explanation["attributions"]
            self.explain_table.setRowCount(len(attribs))
            for r, a in enumerate(attribs):
                for c, val in enumerate([a.date, a.label, f"{a.contribution:+.3f}"]):
                    self.explain_table.setItem(r, c, QTableWidgetItem(str(val)))

            self._set_run_state("Run pipeline", "#0a7d33", enabled=True)
            self.statusBar().showMessage("Done. Outputs also written to examples/output/.")

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app, win


def main():  # pragma: no cover
    app, _win = _build_app()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    main()
