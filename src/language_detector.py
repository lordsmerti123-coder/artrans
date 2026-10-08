# src/language_detector.py
"""
Определение языка текста с использованием библиотеки Lingua.
Установка: pip install lingua-language-detector
"""

from typing import Optional, List
from lingua import Language, LanguageDetectorBuilder


class LanguageDetector:
    """Детектор языка на основе Lingua."""

    # Словарь соответствий кодов и объектов Language.
    # Список должен совпадать с языками перевода (src/nmt.py) и с
    # голосами синтеза (models/tts/piper/), иначе канал не заработает.
    LANG_MAP = {
        "ru": Language.RUSSIAN,
        "en": Language.ENGLISH,
        "fr": Language.FRENCH,
        "es": Language.SPANISH,
        "hy": Language.ARMENIAN,
        "zh": Language.CHINESE,
    }

    def __init__(self, languages: Optional[List[str]] = None):
        if languages is None:
            languages = ["ru", "en"]
        self.language_codes = languages
        self._build_detector(languages)

    def _build_detector(self, codes: List[str]):
        """Создаёт детектор для указанных языков."""
        lang_objects = []
        for code in codes:
            lang = self.LANG_MAP.get(code)
            if lang is not None:
                lang_objects.append(lang)
            else:
                print(f"⚠️ Язык '{code}' не поддерживается, пропускаем.")

        if not lang_objects:
            print("⚠️ Нет поддерживаемых языков, используем все доступные.")
            self.detector = LanguageDetectorBuilder.from_all_languages().build()
        else:
            self.detector = LanguageDetectorBuilder.from_languages(*lang_objects).build()

        print(f"✅ LanguageDetector инициализирован для {len(lang_objects)} языков.")

    def detect(self, text: str) -> Optional[str]:
        if not text or not text.strip():
            return None
        result = self.detector.detect_language_of(text)
        if result is not None:
            # Преобразуем объект Language в код
            for code, lang in self.LANG_MAP.items():
                if lang == result:
                    return code
            # Fallback: попробуем взять iso код
            try:
                return result.iso_code_639_1.name.lower()
            except AttributeError:
                return None
        return None

    def detect_with_confidence(self, text: str) -> tuple[Optional[str], float]:
        if not text or not text.strip():
            return None, 0.0
        confidence_values = self.detector.compute_language_confidence_values(text)
        if not confidence_values:
            return None, 0.0
        best = max(confidence_values, key=lambda x: x.value)
        if best.value > 0.3:
            lang_obj = best.language
            for code, lang in self.LANG_MAP.items():
                if lang == lang_obj:
                    return code, best.value
            try:
                return lang_obj.iso_code_639_1.name.lower(), best.value
            except AttributeError:
                return None, 0.0
        return None, 0.0

    def is_language(self, text: str, target_lang: str, threshold: float = 0.3) -> bool:
        if not text or not text.strip():
            return False
        detected, confidence = self.detect_with_confidence(text)
        if detected is None:
            return False
        return detected == target_lang and confidence >= threshold


_default_detector = None

def get_detector(languages: Optional[List[str]] = None) -> LanguageDetector:
    global _default_detector
    if _default_detector is None:
        _default_detector = LanguageDetector(languages)
    return _default_detector