# src/noise.py
"""
Подавление шума для аудио-канала.

Бэкенды (авто-выбор, см. план Phase 0.5):
- ``rnnoise``      — реалтайм-подавитель на 48 кГц (пакет ``pyrnnoise``);
- ``noisereduce``  — спектральный фолбэк (пакет ``noisereduce``);
- ``none``         — шумодав недоступен/выключен, аудио не меняется.

Модуль не ломает импорт приложения, если пакеты не установлены: тогда
``active_backend == 'none'`` и ``process()`` возвращает вход как есть.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from scipy.signal import resample_poly

_BACKENDS = ("rnnoise", "noisereduce", "none")


def _try_import_rnnoise():
    try:
        from pyrnnoise import RNNoise  # type: ignore

        return RNNoise
    except Exception:
        return None


def _try_import_noisereduce():
    try:
        import noisereduce as nr  # type: ignore

        return nr
    except Exception:
        return None


def available_backends() -> dict:
    """Доступность опциональных бэкендов (для логов и диагностики)."""
    return {
        "rnnoise": _try_import_rnnoise() is not None,
        "noisereduce": _try_import_noisereduce() is not None,
    }


class NoiseReducer:
    def __init__(
        self,
        sample_rate: int = 16000,
        backend: str = "auto",
        prop_decrease: float = 0.8,
        enabled: bool = True,
    ):
        self.sample_rate = int(sample_rate)
        self.requested_backend = backend
        self.prop_decrease = float(prop_decrease)
        self.enabled = bool(enabled)
        self.active_backend = "none"
        self._rnnoise = None
        self._noisereduce = None
        if self.enabled:
            self._select_backend()

    # ---------------------------------------------------------------- backend
    def _select_backend(self) -> None:
        order = {
            "auto": ("rnnoise", "noisereduce"),
            "rnnoise": ("rnnoise",),
            "noisereduce": ("noisereduce",),
            "none": (),
        }.get(self.requested_backend, ("rnnoise", "noisereduce"))

        for name in order:
            if name == "rnnoise":
                cls = _try_import_rnnoise()
                if cls is not None:
                    try:
                        self._rnnoise = cls(sample_rate=48000)
                        self.active_backend = "rnnoise"
                        return
                    except Exception:
                        self._rnnoise = None
            elif name == "noisereduce":
                nr = _try_import_noisereduce()
                if nr is not None:
                    self._noisereduce = nr
                    self.active_backend = "noisereduce"
                    return

        self.active_backend = "none"

    @property
    def is_active(self) -> bool:
        return self.enabled and self.active_backend != "none"

    # ---------------------------------------------------------------- process
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Очищает моно float32 аудио при частоте self.sample_rate."""
        if not self.is_active or audio is None or len(audio) == 0:
            return audio

        try:
            if self.active_backend == "rnnoise":
                return self._process_rnnoise(audio)
            if self.active_backend == "noisereduce":
                return self._process_noisereduce(audio)
        except Exception:
            # Ошибка шумодава не должна ронять канал — отдаём исходное аудио.
            return audio
        return audio

    def _process_rnnoise(self, audio: np.ndarray) -> np.ndarray:
        # RNNoise работает на 48 кГц.
        up = 3 if self.sample_rate == 16000 else 1
        work = resample_poly(audio.astype(np.float32), up, 1) if up != 1 else audio.astype(np.float32)

        denoiser = self._rnnoise
        result = denoiser.denoise(work)
        if isinstance(result, tuple):
            result = result[0]
        result = np.asarray(result, dtype=np.float32)

        if up != 1:
            result = resample_poly(result, 1, up)
        if len(result) < len(audio):
            result = np.pad(result, (0, len(audio) - len(result)))
        return result[: len(audio)].astype(np.float32)

    def _process_noisereduce(self, audio: np.ndarray) -> np.ndarray:
        reduced = self._noisereduce.reduce_noise(
            y=audio.astype(np.float32),
            sr=self.sample_rate,
            stationary=True,
            prop_decrease=self.prop_decrease,
        )
        return np.asarray(reduced, dtype=np.float32)
