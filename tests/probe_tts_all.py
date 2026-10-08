"""
Живая проверка синтеза речи на всех языках канала.

Загружает голоса Piper и озвучивает по фразе на каждом языке.
Проверяет, что голос выдаёт звук, а не пустоту.

Запуск:
    python tests/probe_tts_all.py
"""

import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT = Path(r"D:\ARtrans")
sys.path.insert(0, str(PROJECT))

from src.model_manager import ModelManager

# Фразы для проверки синтеза. Для каждого языка — простая фраза,
# которую голос должен уверенно произнести.
PHRASES = {
    "ru": "Сегодня хорошая погода, пойдём гулять",
    "en": "The weather is nice today",
    "fr": "Il fait beau aujourd'hui",
    "es": "Hoy hace buen tiempo",
    "hy": "Այսօր լավ եղանակ է",
    "zh": "今天天气很好",
}


def main() -> None:
    """Синтезирует фразу на каждом языке и сообщает длину звука."""
    manager = ModelManager()
    print()

    results = []
    for lang, phrase in PHRASES.items():
        try:
            model = manager.get_tts_model(lang)
            audio = model.synthesize_to_bytes(phrase)
            size = len(audio) if audio else 0
            results.append((lang, "OK" if size > 1000 else "слишком мало данных", size))
        except Exception as error:
            results.append((lang, f"ошибка: {type(error).__name__}", 0))

    print("=== ИТОГ СИНТЕЗА ===")
    for lang, status, size in results:
        mark = "✅" if status == "OK" else "❌"
        print(f"  {mark} {lang}: {status}, байт звука: {size}")


if __name__ == "__main__":
    main()
