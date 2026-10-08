"""
Проверка: что произойдёт, если добавить третий язык.

Показывает, достаточно ли добавить код в LANG_MAP или требуются
дополнительные правки. Ничего не меняет — только проверяет.
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


def main() -> None:
    """Пробует собрать детектор с немецким языком."""
    from src.language_detector import LanguageDetector

    print("=== ДЕТЕКТОР: 2 языка (как сейчас) ===")
    two = LanguageDetector(["ru", "en"])
    for phrase, note in [
        ("Das Wetter ist heute schön", "немецкая фраза"),
        ("Сегодня хорошая погода", "русская фраза"),
    ]:
        print(f"  {note:16} -> {two.detect(phrase)}")

    print()
    print("=== ДЕТЕКТОР: запрос третьего языка 'de' ===")
    try:
        three = LanguageDetector(["ru", "en", "de"])
        print("  создан без ошибок")
        for phrase, note in [
            ("Das Wetter ist heute schön", "немецкая фраза"),
            ("Сегодня хорошая погода", "русская фраза"),
        ]:
            print(f"  {note:16} -> {three.detect(phrase)}")
    except Exception as error:
        print(f"  ошибка: {type(error).__name__}: {error}")

    print()
    print("=== ФИЛЬТР: язык 'de' в списке алфавитов ===")
    from src.language_gate import _COMPATIBLE_SCRIPTS
    if "de" in _COMPATIBLE_SCRIPTS:
        print("  настроен:", _COMPATIBLE_SCRIPTS["de"])
    else:
        print("  НЕ настроен — нужен алфавит, иначе фильтр не пропустит текст")
        print("  для немецкого подойдёт ('latin',) — та же логика, что у 'en'")


if __name__ == "__main__":
    main()
