# src/session_log.py
"""
Логи сессии и метрики задержек (Phase 0.7).

Файл: ``logs/session_<timestamp>.log``. Текстовые реплики пишутся только если
включён ``log_text``. Метрики собираются в памяти и по запросу пишутся итогом.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class Metrics:
    """Простые счётчики и наблюдения задержек (мс)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = {}
        self._observations: Dict[str, List[float]] = {}

    def inc(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def observe(self, name: str, ms: float) -> None:
        with self._lock:
            self._observations.setdefault(name, []).append(float(ms))

    def counters(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def stats(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            result = {}
            for name, values in self._observations.items():
                if not values:
                    continue
                ordered = sorted(values)
                result[name] = {
                    "count": len(ordered),
                    "avg_ms": sum(ordered) / len(ordered),
                    "p50_ms": ordered[len(ordered) // 2],
                    "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
                    "max_ms": ordered[-1],
                }
            return result

    def summary(self) -> str:
        parts = []
        for name, value in sorted(self.counters().items()):
            parts.append(f"{name}={value}")
        for name, st in sorted(self.stats().items()):
            parts.append(
                f"{name}: n={st['count']} avg={st['avg_ms']:.0f}ms p95={st['p95_ms']:.0f}ms"
            )
        return "; ".join(parts) if parts else "нет данных"


class SessionLog:
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        enabled: bool = True,
        log_text: bool = True,
        prefix: str = "session",
        on_message=None,
    ):
        self.enabled = bool(enabled)
        self.log_text = bool(log_text)
        self.on_message = on_message
        self.metrics = Metrics()
        self.path: Optional[Path] = None
        self._logger: Optional[logging.Logger] = None

        if self.enabled:
            base = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent / "logs"
            base.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.path = base / f"{prefix}_{stamp}.log"
            self._logger = logging.getLogger(f"artrans.session.{id(self)}")
            self._logger.setLevel(logging.DEBUG)
            self._logger.propagate = False
            handler = logging.FileHandler(self.path, encoding="utf-8")
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
            )
            self._logger.addHandler(handler)
            self.info(f"Сессия начата: {self.path}")

    # ------------------------------------------------------------------ logging
    def _write(self, level: int, message: str) -> None:
        if self._logger is not None:
            self._logger.log(level, message)
        if self.on_message is not None:
            try:
                self.on_message(f"{logging.getLevelName(level)[0]}: {message}")
            except Exception:
                pass

    def debug(self, message: str) -> None:
        self._write(logging.DEBUG, message)

    def info(self, message: str) -> None:
        self._write(logging.INFO, message)

    def warn(self, message: str) -> None:
        self._write(logging.WARNING, message)

    def error(self, message: str) -> None:
        self._write(logging.ERROR, message)

    def event(self, kind: str, **fields) -> None:
        if "text" in fields and not self.log_text:
            fields = {k: v for k, v in fields.items() if k != "text"}
        payload = " ".join(f"{k}={v!r}" for k, v in fields.items())
        self._write(logging.INFO, f"[event:{kind}] {payload}".rstrip())

    def text_event(self, kind: str, text: str, **fields) -> None:
        if self.log_text:
            self.event(kind, text=text, **fields)
        else:
            self.event(kind, **fields)

    def flush_summary(self) -> str:
        summary = self.metrics.summary()
        self.info(f"ИТОГИ: {summary}")
        return summary

    def close(self) -> None:
        if self._logger is not None:
            self.flush_summary()
            self.info("Сессия завершена.")
            for handler in list(self._logger.handlers):
                handler.flush()
                handler.close()
                self._logger.removeHandler(handler)
