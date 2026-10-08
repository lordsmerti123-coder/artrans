"""
Живая проверка определения языка и фильтра.

Прогоняет фразы на разных языках и показывает, что детектор отвечает
и как ведёт себя фильтр алфавита.

Запуск:
    python tests/test_language_live.py
"""

import sys
from pathlib import Path

# Консоль Windows по умолчанию не выводит кириллицу и эмодзи —
# переключаем потоки на UTF-8, как это делает app.py.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT = Path(r"D:\ARtrans")
sys.path.insert(0, str(PROJECT))

from src.language_detector import LanguageDetector
from src.language_gate import LanguageGate

# Фразы для проверки. Коды — те, что реально подключены и заявлены.
SAMPLES = [
    ("ru", "Сегодня хорошая погода, давай пойдём гулять в парк"),
    ("en", "The weather is nice today, let us go for a walk"),
    ("de", "Das Wetter ist heute schön, gehen wir spazieren"),
    ("fr", "Le temps est beau aujourd'hui, allons nous promener"),
    ("zh", "今天天气很好，我们去公园散步吧"),
    ("kk", "Бүгін ауа райы жақсы, серуендеуге барайық"),
    ("en", "This is a short one"),
    ("ru", "ок"),
]


def main() -> None:
    """Прогоняет фразы через детектор и фильтр."""
    detector = LanguageDetector(["ru", "en"])
    gate = LanguageGate(["ru", "en"], threshold=0.3, min_chars=3)

    print("детектор настроен на:", detector.language_codes)
    print()
    print(f"{'ожидался':10} {'определён':10} {'уверенность':12} фраза")
    print("-" * 78)

    for expected, phrase in SAMPLES:
        code = detector.detect(phrase)
        _, confidence = detector.detect_with_confidence(phrase)
        short = phrase[:38] + ("…" if len(phrase) > 38 else "")
        print(f"{expected:10} {str(code):10} {confidence:<12.2f} {short}")

    print()
    print("=== ФИЛЬТР АЛФАВИТА (канал ru принимает только русский) ===")
    for expected, phrase in SAMPLES:
        result = gate.check(phrase, "ru")
        verdict = "принято" if result.accepted else "отсеяно"
        reason = result.reason[:44]
        print(f"  {verdict:8} {reason}")


if __name__ == "__main__":
    main()
