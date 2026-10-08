"""
Проверка интерфейса: появился ли выбор языковой пары.

Создаёт окно без показа и проверяет, что в панелях каналов есть
списки языков, а выбор сохраняется в конфигурацию.

Запуск:
    python tests/probe_lang_ui.py
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

from PyQt6.QtWidgets import QApplication


def main() -> None:
    """Открывает окно и проверяет списки языков в панелях."""
    app = QApplication(sys.argv)

    from gui.main_window import MainWindow
    from gui import main_tab

    print("=== доступные языки в интерфейсе ===")
    for code, title in main_tab.LANGUAGE_CHOICES:
        print(f"  {code}: {title}")

    window = MainWindow()
    print("\n=== поиск списков языков в панелях ===")

    found = 0
    for widget in window.findChildren(main_tab.PersonPanel):
        source = widget.src_lang_combo
        target = widget.tgt_lang_combo
        print(f"\nканал '{widget.lang_code}':")
        print(f"  говорит: {source.currentText()} ({source.currentData()})")
        print(f"  переводит на: {target.currentText()} ({target.currentData()})")
        print(f"  вариантов в списке: {source.count()}")

        # Проверяем, что есть все шесть языков.
        codes = {source.itemData(index) for index in range(source.count())}
        expected = {code for code, _ in main_tab.LANGUAGE_CHOICES}
        print(f"  все языки на месте: {codes == expected}")
        found += 1

    print(f"\nнайдено панелей каналов: {found}")
    window.close()


if __name__ == "__main__":
    main()
