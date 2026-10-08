# src/segmentation.py
"""
Гибридная сегментация речи для двустороннего переводчика.

Правила (см. plans/multilingual-translator-roadmap.md, раздел «Сегментация»):
- короткая пауза (< merge_pause_ms) НЕ завершает реплику — аудио склеивается;
- жёсткий флеш при паузе >= merge_pause_ms, либо при длине сегмента >= max_segment_ms;
- при жёстком резе по лимиту следующие сегменты стартуют с перекрытием
  chunk_overlap_ms, а текст на стыке дедуплицируется через dedupe_overlap();
- тишина не создаёт реплик: без речевых фреймов сегмент не выдаётся;
- на stop() вызывается flush(), чтобы последняя фраза не терялась.

Модуль не зависит от sounddevice/моделей — только numpy, поэтому легко тестируется.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np


@dataclass
class Segment:
    """Готовый к распознаванию фрагмент речи."""

    audio: np.ndarray            # моно float32 при sample_rate
    sample_rate: int
    forced: bool = False         # True — отрез по лимиту длины (есть перекрытие с продолжением)
    final: bool = False          # True — выдан по stop()/flush()
    speech_ms: float = 0.0       # суммарная длительность речевых фреймов
    duration_ms: float = 0.0     # длительность всего фрагмента


class SpeechSegmenter:
    """Stateful-сегментатор: кормится парами (аудио-блок, is_speech)."""

    def __init__(
        self,
        sample_rate: int,
        merge_pause_ms: int = 2500,
        max_segment_ms: int = 20000,
        chunk_overlap_ms: int = 800,
        min_speech_ms: int = 500,
        min_audio_length_sec: float = 0.4,
        preroll_ms: int = 200,
        tail_pad_ms: int = 150,
    ):
        if sample_rate <= 0:
            raise ValueError("sample_rate должен быть > 0")
        self.sample_rate = int(sample_rate)
        self.merge_pause_ms = float(merge_pause_ms)
        self.max_segment_ms = float(max_segment_ms)
        self.chunk_overlap_ms = float(chunk_overlap_ms)
        self.min_speech_ms = float(min_speech_ms)
        self.min_audio_length_sec = float(min_audio_length_sec)
        self.preroll_ms = float(preroll_ms)
        self.tail_pad_ms = float(tail_pad_ms)

        self._preroll: Deque[np.ndarray] = deque()
        self._preroll_ms = 0.0
        self._buffer: List[np.ndarray] = []
        self._buffer_ms = 0.0
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._in_speech = False

    # ------------------------------------------------------------------ utils
    def _ms(self, block: np.ndarray) -> float:
        return len(block) / self.sample_rate * 1000.0

    def _push_preroll(self, block: np.ndarray) -> None:
        self._preroll.append(block)
        self._preroll_ms += self._ms(block)
        while self._preroll and self._preroll_ms - self._ms(self._preroll[0]) >= self.preroll_ms:
            self._preroll_ms -= self._ms(self._preroll.popleft())

    def _start_segment(self, block: np.ndarray) -> None:
        self._buffer = list(self._preroll)
        self._buffer_ms = self._preroll_ms
        self._preroll.clear()
        self._preroll_ms = 0.0
        self._buffer.append(block)
        self._buffer_ms += self._ms(block)
        block_ms = self._ms(block)
        self._speech_ms = block_ms
        self._silence_ms = 0.0
        self._in_speech = True

    def _state_snapshot_ms(self) -> float:
        return self._buffer_ms

    def _reset(self) -> None:
        self._buffer = []
        self._buffer_ms = 0.0
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._in_speech = False

    # ------------------------------------------------------------------- feed
    def feed(self, block: np.ndarray, is_speech: bool) -> Optional[Segment]:
        """Добавляет блок аудио. Возвращает Segment, если реплику пора отдать."""
        if block is None or len(block) == 0:
            return None
        block = np.asarray(block, dtype=np.float32)
        block_ms = self._ms(block)

        if not self._in_speech:
            if not is_speech:
                self._push_preroll(block)
                return None
            self._start_segment(block)
        else:
            self._buffer.append(block)
            self._buffer_ms += block_ms
            if is_speech:
                self._speech_ms += block_ms
                self._silence_ms = 0.0
            else:
                self._silence_ms += block_ms

        if self._silence_ms >= self.merge_pause_ms:
            return self._emit(forced=False, final=False, trim_tail=True)

        if self._buffer_ms >= self.max_segment_ms:
            return self._emit(forced=True, final=False, trim_tail=False)

        return None

    # ------------------------------------------------------------------ flush
    def flush(self) -> Optional[Segment]:
        """Принудительно отдаёт текущую реплику (остановка приложения)."""
        if not self._in_speech or not self._buffer:
            self._reset()
            self._preroll.clear()
            self._preroll_ms = 0.0
            return None
        return self._emit(forced=False, final=True, trim_tail=True)

    # ------------------------------------------------------------------- emit
    def _emit(self, forced: bool, final: bool, trim_tail: bool) -> Optional[Segment]:
        if not self._buffer:
            self._reset()
            return None

        audio = np.concatenate(self._buffer)
        if trim_tail:
            trim = int(max(0.0, self._silence_ms - self.tail_pad_ms) / 1000.0 * self.sample_rate)
            if trim > 0 and trim < len(audio):
                audio = audio[:-trim]

        duration = len(audio) / self.sample_rate
        speech_ms = self._speech_ms
        keep_overlap = forced
        overlap = None
        if keep_overlap and self.chunk_overlap_ms > 0:
            n = int(self.chunk_overlap_ms / 1000.0 * self.sample_rate)
            if n > 0 and n < len(audio):
                overlap = audio[-n:].copy()

        too_short = (
            speech_ms < self.min_speech_ms
            or duration < self.min_audio_length_sec
        )

        self._reset()
        if too_short:
            # Короткий/шумовой фрагмент отбрасываем, но продолжение по лимиту не рвём.
            if keep_overlap and overlap is not None:
                self._resume(overlap)
            return None

        segment = Segment(
            audio=audio,
            sample_rate=self.sample_rate,
            forced=forced,
            final=final,
            speech_ms=speech_ms,
            duration_ms=duration * 1000.0,
        )

        if keep_overlap and overlap is not None:
            self._resume(overlap)
        return segment

    def _resume(self, overlap: np.ndarray) -> None:
        """Продолжает реплику после жёсткого реза, начиная с перекрытия."""
        self._buffer = [overlap]
        self._buffer_ms = self._ms(overlap)
        self._speech_ms = self._buffer_ms
        self._silence_ms = 0.0
        self._in_speech = True

    def reset(self) -> None:
        self._reset()
        self._preroll.clear()
        self._preroll_ms = 0.0


def _tokens(text: str) -> List[str]:
    return text.split()


def dedupe_overlap(prev_text: str, new_text: str, max_words: int = 30) -> str:
    """
    Убирает дублирование текста на стыке перекрывающихся сегментов.

    Ищет наибольший суффикс prev_text (по словам), совпадающий с префиксом
    new_text, и отрезает его. Если совпадения по словам нет — пробует
    посимвольно (частично распознанное слово).
    """
    if not prev_text or not new_text:
        return new_text

    prev = _tokens(prev_text)
    new = _tokens(new_text)
    limit = min(len(prev), len(new), max_words)
    for k in range(limit, 0, -1):
        if [w.lower() for w in prev[-k:]] == [w.lower() for w in new[:k]]:
            return " ".join(new[k:]).strip()

    # Посимвольный фолбэк на случай обрыва внутри слова.
    a = prev_text.strip()
    b = new_text.strip()
    max_chars = min(len(a), len(b), 60)
    for k in range(max_chars, 2, -1):
        if a[-k:].lower() == b[:k].lower():
            return b[k:].strip()

    return new_text
