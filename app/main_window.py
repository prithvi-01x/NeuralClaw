"""
app/main_window.py — NeuralClaw Dashboard Window (Phase I stub)

Phase H: This module exists as a minimal placeholder so that imports
from app/ do not fail. The full 7-tab dashboard is implemented in Phase I.

Phase I tabs (from roadmap):
    1. Today     — daily briefing, weather, calendar
    2. Tasks     — scheduler history with status + output links
    3. Reports   — file browser for ~/neuralclaw/reports/
    4. Memory    — semantic search over ChromaDB
    5. Skills    — loaded skills with metadata + invocation counts
    6. Logs      — tailed structured log with filters
    7. Settings  — voice model, wake sensitivity, LLM provider, trust level
"""

from __future__ import annotations

from PyQt6.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt


class MainWindow(QMainWindow):
    """
    Phase H stub — displays a placeholder message.
    Replaced by the full 7-tab implementation in Phase I.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("NeuralClaw Dashboard")
        self.setMinimumSize(900, 600)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        label = QLabel(
            "<h2>NeuralClaw Dashboard</h2>"
            "<p style='color: #888;'>Coming in Phase I — Days 105–124</p>"
            "<p>The full dashboard includes:<br>"
            "Today · Tasks · Reports · Memory · Skills · Logs · Settings</p>"
        )
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(label)

    def show_and_raise(self) -> None:
        """Show the window and bring it to the front."""
        self.show()
        self.raise_()
        self.activateWindow()