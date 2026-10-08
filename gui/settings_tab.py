# gui/settings_tab.py
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QTableWidget, QTableWidgetItem, QComboBox,
                             QLineEdit, QPushButton, QCheckBox, QGroupBox,
                             QMessageBox)
from PyQt6.QtCore import Qt
from pathlib import Path

DEFAULT_TTS_DIR = str(Path(__file__).resolve().parent.parent / "models" / "tts" / "piper")
DEFAULT_TRANSLATOR_GGUF = "D:/models/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"


class SettingsTab(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        layout = QVBoxLayout(self)

        # Пресеты (оставляем)
        preset_group = QGroupBox("Пресеты моделей")
        preset_layout = QVBoxLayout(preset_group)
        preset_hlayout = QHBoxLayout()
        preset_hlayout.addWidget(QLabel("Выберите пресет:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Быстрый (экономия ресурсов)", "Сбалансированный", "Мощный (высокое качество)"])
        preset_hlayout.addWidget(self.preset_combo)
        preset_layout.addLayout(preset_hlayout)
        apply_preset_btn = QPushButton("Применить пресет")
        apply_preset_btn.clicked.connect(self.apply_preset)
        preset_layout.addWidget(apply_preset_btn)
        layout.addWidget(preset_group)

        # Модели
        models_group = QGroupBox("Модели для каждого этапа")
        models_layout = QVBoxLayout(models_group)
        self.models_table = QTableWidget(3, 4)
        self.models_table.setHorizontalHeaderLabels(["Компонент", "Текущая модель", "Доступные модели", "Путь"])
        self.models_table.horizontalHeader().setStretchLastSection(True)

        components = [
            ("Распознавание (STT)", "multilingual_ctc", ["v3_ctc", "v3_rnnt", "multilingual_ctc", "multilingual_large_ctc"]),
            ("Перевод (NMT)", "llama-gguf", ["llama-gguf"]),
            ("Синтез речи (TTS)", "Piper", ["Piper"])
        ]

        for i, (comp, curr, avail) in enumerate(components):
            self.models_table.setItem(i, 0, QTableWidgetItem(comp))
            self.models_table.setItem(i, 1, QTableWidgetItem(curr))
            combo = QComboBox()
            combo.addItems(avail)
            combo.setCurrentText(curr)
            self.models_table.setCellWidget(i, 2, combo)
            path_edit = QLineEdit()
            if comp == "Перевод (NMT)":
                path_edit.setPlaceholderText("Путь к GGUF-файлу")
                path_edit.setText(DEFAULT_TRANSLATOR_GGUF)
            elif comp == "Синтез речи (TTS)":
                path_edit.setPlaceholderText("Путь к папке с голосами Piper")
                path_edit.setText(DEFAULT_TTS_DIR)
            self.models_table.setCellWidget(i, 3, path_edit)

        models_layout.addWidget(self.models_table)

        apply_btn = QPushButton("Применить изменения (перезагрузит модели)")
        apply_btn.clicked.connect(self.apply_settings)
        models_layout.addWidget(apply_btn)
        layout.addWidget(models_group)

        # Логи
        log_group = QGroupBox("Настройки логирования")
        log_layout = QHBoxLayout(log_group)
        self.log_enabled = QCheckBox("Вести логи")
        self.log_enabled.setChecked(True)
        log_layout.addWidget(self.log_enabled)
        self.log_text = QCheckBox("Текстовые логи")
        self.log_text.setChecked(True)
        log_layout.addWidget(self.log_text)
        self.log_audio = QCheckBox("Аудио-логи (WAV)")
        self.log_audio.setChecked(False)
        log_layout.addWidget(self.log_audio)
        layout.addWidget(log_group)

        self.load_settings()

    def load_settings(self):
        config = self.controller.config
        if 'log_enabled' in config:
            self.log_enabled.setChecked(config['log_enabled'])
        if 'log_text' in config:
            self.log_text.setChecked(config['log_text'])
        if 'log_audio' in config:
            self.log_audio.setChecked(config['log_audio'])
        if 'preset' in config:
            index = self.preset_combo.findText(config['preset'])
            if index >= 0:
                self.preset_combo.setCurrentIndex(index)

        models = config.get('models', {})
        translator_path = models.get('translator_path', '')
        tts_path = models.get('tts_path', '')
        if not tts_path and models.get('tts_ru'):
            tts_path = str(Path(models['tts_ru']).parent.parent)
        if translator_path:
            for i in range(self.models_table.rowCount()):
                if self.models_table.item(i, 0).text() == "Перевод (NMT)":
                    path_edit = self.models_table.cellWidget(i, 3)
                    if path_edit:
                        path_edit.setText(translator_path)
                    break
        if tts_path:
            for i in range(self.models_table.rowCount()):
                if self.models_table.item(i, 0).text() == "Синтез речи (TTS)":
                    path_edit = self.models_table.cellWidget(i, 3)
                    if path_edit:
                        path_edit.setText(tts_path)
                    break

    def apply_preset(self):
        preset_name = self.preset_combo.currentText()
        presets = {
            "Быстрый (экономия ресурсов)": {"stt": "v3_ctc", "translator": "llama-gguf"},
            "Сбалансированный": {"stt": "multilingual_ctc", "translator": "llama-gguf"},
            "Мощный (высокое качество)": {"stt": "multilingual_large_ctc", "translator": "llama-gguf"}
        }
        selected = presets.get(preset_name, {})
        if selected:
            for i, comp in enumerate(["Распознавание (STT)", "Перевод (NMT)"]):
                key = ["stt", "translator"][i]
                new_model = selected.get(key, "")
                if new_model:
                    combo = self.models_table.cellWidget(i, 2)
                    if combo is not None:
                        index = combo.findText(new_model)
                        if index >= 0:
                            combo.setCurrentIndex(index)
                    self.models_table.setItem(i, 1, QTableWidgetItem(new_model))
            QMessageBox.information(self, "Пресет применён",
                f"Выбран пресет '{preset_name}'. Нажмите 'Применить изменения' для перезагрузки моделей.")

    def apply_settings(self):
        models = {}
        for i in range(self.models_table.rowCount()):
            component = self.models_table.item(i, 0).text()
            combo = self.models_table.cellWidget(i, 2)
            selected_model = combo.currentText()
            path_edit = self.models_table.cellWidget(i, 3)
            local_path = path_edit.text().strip()

            if component == "Распознавание (STT)":
                models['stt'] = selected_model
            elif component == "Перевод (NMT)":
                models['translator_path'] = local_path
            elif component == "Синтез речи (TTS)":
                models['tts_path'] = local_path

        config_updates = {
            'log_enabled': self.log_enabled.isChecked(),
            'log_text': self.log_text.isChecked(),
            'log_audio': self.log_audio.isChecked(),
            'preset': self.preset_combo.currentText(),
            'models': models
        }

        self.controller.apply_settings(config_updates)
        QMessageBox.information(self, "Настройки применены",
            "Модели будут перезагружены. Это может занять некоторое время.")