# src/audio_router.py
"""
Независимое воспроизведение перевода на разные устройства (Phase 0.2).

Вместо глобального ``sd.play``/``sd.wait`` (который сериализует все каналы на
один поток) роутер держит по одному постоянному ``sd.OutputStream`` на каждое
устройство и свою FIFO-очередь. Падение/недоступность одного устройства не
останавливает второй канал.
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, Dict, List, Optional

import numpy as np

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - окружение без audio
    sd = None

from math import gcd

from scipy.signal import resample_poly

_RATE_CANDIDATES = (22050, 24000, 48000, 44100, 16000)


def _resample_int16(samples: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    if orig_rate == target_rate or len(samples) == 0:
        return samples
    g = gcd(int(orig_rate), int(target_rate))
    up, down = int(target_rate) // g, int(orig_rate) // g
    data = samples.astype(np.float32) / 32768.0
    res = resample_poly(data, up, down)
    return np.clip(res * 32768.0, -32768, 32767).astype(np.int16)


class AudioOutputChannel:
    """Постоянный OutputStream + очередь на одно устройство."""

    def __init__(self, device_id: int, on_error: Optional[Callable[[str], None]] = None):
        self.device_id = device_id
        self._on_error = on_error
        self._queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stream = None
        self._running = False
        self._rate: Optional[int] = None
        self._lock = threading.Lock()
        self.errors = 0
        self.played = 0

    # ------------------------------------------------------------------ helpers
    def _report(self, message: str) -> None:
        self.errors += 1
        if self._on_error:
            try:
                self._on_error(message)
            except Exception:
                pass

    def _negotiate_rate(self, requested: int) -> Optional[int]:
        rates: List[int] = []
        for rate in (requested, *_RATE_CANDIDATES):
            if rate not in rates:
                rates.append(rate)
        try:
            dev = sd.query_devices(self.device_id)
        except Exception as exc:
            self._report(f"Устройство вывода {self.device_id} недоступно: {exc}")
            return None

        for rate in rates:
            try:
                sd.check_output_settings(device=self.device_id, samplerate=rate, channels=1)
                return rate
            except Exception:
                continue

        default_rate = int(dev.get("default_samplerate") or 0)
        if default_rate:
            return default_rate
        self._report(f"Не удалось подобрать частоту для устройства {self.device_id}")
        return None

    # -------------------------------------------------------------------- start
    def start(self, requested_rate: int = 22050) -> bool:
        if self._running:
            return True
        if sd is None:
            self._report("sounddevice недоступен")
            return False
        if self.device_id is None:
            self._report("Индекс устройства вывода не задан")
            return False

        device_rate = self._negotiate_rate(requested_rate)
        if device_rate is None:
            return False

        try:
            self._stream = sd.OutputStream(
                device=self.device_id,
                samplerate=device_rate,
                channels=1,
                dtype="int16",
            )
            self._stream.start()
        except Exception as exc:
            self._report(f"Не удалось открыть поток вывода {self.device_id}: {exc}")
            self._stream = None
            return False

        self._rate = device_rate
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def _run(self) -> None:
        while self._running:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            samples, src_rate = item
            try:
                if src_rate != self._rate:
                    samples = _resample_int16(samples, src_rate, self._rate)
                self._stream.write(np.ascontiguousarray(samples.reshape(-1, 1)))
                self.played += 1
            except Exception as exc:
                self._report(f"Ошибка воспроизведения на устройстве {self.device_id}: {exc}")
                break

    # ------------------------------------------------------------------- submit
    def submit(self, samples: np.ndarray, sample_rate: int) -> bool:
        if not self._running:
            if not self.start(sample_rate):
                return False
        self._queue.put((np.asarray(samples, dtype=np.int16), int(sample_rate)))
        return True

    def stop(self, timeout: float = 1.0) -> None:
        self._running = False
        self._queue.put(None)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


class AudioRouter:
    """Маршрутизация PCM16-аудио на несколько устройств независимо."""

    def __init__(self, on_error: Optional[Callable[[str], None]] = None):
        self._on_error = on_error
        self._channels: Dict[int, AudioOutputChannel] = {}
        self._lock = threading.Lock()
        self.dropped = 0

    def _channel(self, device_id: int) -> AudioOutputChannel:
        with self._lock:
            channel = self._channels.get(device_id)
            if channel is None:
                channel = AudioOutputChannel(device_id, self._on_error)
                self._channels[device_id] = channel
            return channel

    def route(self, samples: np.ndarray, sample_rate: int, device_id: int) -> bool:
        """Ставит аудио в очередь устройства. True — принято, False — потеряно."""
        if device_id is None:
            self.dropped += 1
            return False
        channel = self._channel(device_id)
        if channel.submit(samples, sample_rate):
            return True
        self.dropped += 1
        return False

    def stop(self, timeout: float = 1.0) -> None:
        with self._lock:
            channels = list(self._channels.values())
            self._channels.clear()
        for channel in channels:
            channel.stop(timeout=timeout)

    def health(self) -> Dict[int, Dict[str, int]]:
        with self._lock:
            return {
                device_id: {"played": ch.played, "errors": ch.errors}
                for device_id, ch in self._channels.items()
            }
