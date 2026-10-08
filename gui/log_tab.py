# gui/log_tab.py
import os
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QPlainTextEdit)

MAX_LINES = 3000


class LogTab(QWidget):
    """Живой просмотр подробного лога сессии (файл logs/session_<ts>.log)."""

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        log_path = getattr(controller.session_log, "path", None)
        self.path_label = QLabel(f"Файл лога: {log_path if log_path else 'логирование выключено'}")
        self.path_label.setStyleSheet("color: #a0aab5; font-size: 12px;")
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        top.addWidget(self.path_label)
        top.addStretch()

        open_btn = QPushButton("📂 Открыть папку логов")
        open_btn.setStyleSheet("background-color: #455a64; color: white; padding: 8px 16px; border-radius: 6px;")
        open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        open_btn.clicked.connect(self._open_logs_dir)
        top.addWidget(open_btn)

        clear_btn = QPushButton("🧹 Очистить")
        clear_btn.setStyleSheet("background-color: #455a64; color: white; padding: 8px 16px; border-radius: 6px;")
        clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_btn.clicked.connect(lambda: self.view.clear())
        top.addWidget(clear_btn)
        layout.addLayout(top)

        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(MAX_LINES)
        self.view.setStyleSheet(
            "background-color: #14171a; color: #d4d4d4; font-family: Consolas, monospace; "
            "font-size: 12px; border: 1px solid #2d3740; border-radius: 6px;"
        )
        layout.addWidget(self.view)

        controller.log_signal.connect(self.append)

    def append(self, message: str) -> None:
        self.view.appendPlainText(message)

    def _open_logs_dir(self) -> None:
        logs_dir = Path(__file__).resolve().parent.parent / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(logs_dir))  # type: ignore[attr-defined]
        except Exception:
            pass
