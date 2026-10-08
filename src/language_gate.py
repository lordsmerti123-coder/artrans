# src/language_gate.py
"""
Жёсткий языковой гейт для аудио-петли.

Канал принимает только текст на своём языке: так отсекаются реплики второго
собеседника, попавшие в микрофон (эхо/перекрёстная запись), и «мусор» ASR.

Проверка двухуровневая:
1) скрипт (алфавит) текста против ожидаемого языка — быстро и без модели;
2) уверенность Lingua (если язык есть в детекторе) — как дополнительный фильтр.

В ручном режиме (`manual_language_mode`) гейт становится мягче: отбрасываются
только явно чужеродные строки, короткие реплики не режутся.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# Диапазоны Unicode для основных алфавитов
_SCRIPT_RANGES: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "latin": ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF)),
    "cyrillic": ((0x0400, 0x04FF), (0x0500, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)),
    "armenian": ((0x0530, 0x058F), (0xFB13, 0xFB17)),
    "cjk": (
        (0x3040, 0x30FF),   # хирагана/катакана
        (0x3400, 0x4DBF),
        (0x4E00, 0x9FFF),
        (0xF900, 0xFAFF),
    ),
    "greek": ((0x0370, 0x03FF), (0x1F00, 0x1FFF)),
    "arabic": ((0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF)),
    "hebrew": ((0x0590, 0x05FF),),
    "devanagari": ((0x0900, 0x097F),),
}

# Для каждого языка — допустимые алфавиты. Латиница допускается как «гость»
# (заимствования, латинские аббревиатуры), поэтому она почти везде разрешена.
_COMPATIBLE_SCRIPTS: Dict[str, Tuple[str, ...]] = {
    # Языки, для которых есть перевод и голос синтеза.
    "ru": ("cyrillic", "latin"),
    "en": ("latin",),
    "fr": ("latin",),
    "es": ("latin",),
    "hy": ("armenian", "latin"),
    "zh": ("cjk", "latin"),
    # Языки с настроенным алфавитом, но без пары перевода.
    "kk": ("cyrillic", "latin"),
    "ky": ("cyrillic", "latin"),
    "uz": ("latin", "cyrillic"),
    "tl": ("latin",),
}


@dataclass
class GateResult:
    accepted: bool
    reason: str
    detected: Optional[str] = None
    confidence: float = 0.0
    script: Optional[str] = None


class LanguageGate:
    def __init__(
        self,
        languages: Optional[Sequence[str]] = None,
        threshold: float = 0.3,
        min_chars: int = 3,
        enabled: bool = True,
        detector=None,
    ):
        self.enabled = enabled
        self.threshold = float(threshold)
        self.min_chars = int(min_chars)
        self.languages = list(languages) if languages else ["ru", "en"]
        self.detector = detector
        if self.detector is None:
            try:
                from src.language_detector import LanguageDetector

                self.detector = LanguageDetector(self.languages)
            except Exception:
                self.detector = None

    # ------------------------------------------------------------------ script
    @staticmethod
    def dominant_script(text: str) -> Tuple[Optional[str], int]:
        counts: Dict[str, int] = {}
        for ch in text:
            code = ord(ch)
            for script, ranges in _SCRIPT_RANGES.items():
                if any(lo <= code <= hi for lo, hi in ranges):
                    counts[script] = counts.get(script, 0) + 1
                    break
        if not counts:
            return None, 0
        script = max(counts, key=counts.get)
        return script, counts[script]

    # ------------------------------------------------------------------- check
    def check(
        self,
        text: str,
        expected_lang: str,
        strict: bool = True,
    ) -> GateResult:
        if not self.enabled:
            return GateResult(True, "disabled")

        stripped = (text or "").strip()
        if not stripped:
            return GateResult(False, "empty")

        if len(stripped) < self.min_chars:
            return GateResult(True, "too_short")

        script, script_count = self.dominant_script(stripped)
        compatible = _COMPATIBLE_SCRIPTS.get(expected_lang)
        if script is not None and compatible is not None and script_count >= self.min_chars:
            if script not in compatible:
                return GateResult(False, "script_mismatch", script=script)

        detected = None
        confidence = 0.0
        if self.detector is not None and expected_lang in self.languages:
            try:
                detected, confidence = self.detector.detect_with_confidence(stripped)
            except Exception:
                detected, confidence = None, 0.0

            # В строгом режиме достаточно порога; в ручном — только явно чужой язык.
            required = self.threshold if strict else min(0.9, max(self.threshold * 2, 0.5))
            if detected is not None and detected != expected_lang and confidence >= required:
                return GateResult(
                    False, "lingua_mismatch", detected=detected, confidence=confidence, script=script
                )

        return GateResult(True, "ok", detected=detected, confidence=confidence, script=script)

    def accept(
        self,
        text: str,
        expected_lang: str,
        strict: bool = True,
    ) -> bool:
        return self.check(text, expected_lang, strict=strict).accepted
