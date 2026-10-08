"""
Проверка смены языковой пары через контроллер.

Проверяет, что выбор языка сохраняется в конфигурацию, фильтр языка
пересобирается и канал начинает работать с новой парой.

Запуск:
    python tests/probe_lang_change.py
"""

import json
import shutil
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT = Path(r"D:\ARtrans")
sys.path.insert(0, str(PROJECT))

CONFIG = PROJECT / "config.json"
BACKUP = PROJECT / "config.json.test-backup"


def main() -> None:
    """Меняет языковую пару и проверяет результат."""
    shutil.copy2(CONFIG, BACKUP)

    try:
        from PyQt6.QtWidgets import QApplication
        app = QApplication(sys.argv)

        from gui.controller import Controller

        controller = Controller()

        print("=== ДО СМЕНЫ ===")
        print(f"  канал ru -> {controller.config.get('ru_tgt_lang')}")
        print(f"  фильтр знает языки: {controller.language_gate.languages}")

        # Меняем: русский канал переводит на французский.
        controller.change_language("ru", "ru", "fr")

        print("\n=== ПОСЛЕ СМЕНЫ на французский ===")
        print(f"  канал ru -> {controller.config.get('ru_tgt_lang')}")
        print(f"  фильтр знает языки: {controller.language_gate.languages}")

        # Проверяем, что записалось в файл.
        with open(CONFIG, encoding="utf-8") as handle:
            saved = json.load(handle)
        print(f"  в config.json: ru_tgt_lang = {saved.get('ru_tgt_lang')}")

        # Проверяем фильтр: французский текст должен приниматься каналом.
        print("\n=== проверка фильтра на французском тексте ===")
        result = controller.language_gate.check(
            "Il fait beau aujourd'hui", "ru"
        )
        print(f"  французская фраза: {'принята' if result.accepted else 'отсеяна'}"
              f" ({result.reason})")

        # Смена на тот же язык должна отклоняться.
        print("\n=== защита от одинаковой пары ===")
        controller.change_language("ru", "fr", "fr")
        print(f"  после попытки ru->fr и fr->fr: ru_tgt_lang = "
              f"{controller.config.get('ru_tgt_lang')}")

    finally:
        shutil.copy2(BACKUP, CONFIG)
        BACKUP.unlink()
        print("\nконфигурация восстановлена")


if __name__ == "__main__":
    main()
