# gui/main_tab.py
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QTextEdit, QLabel, QComboBox, QSlider, QFrame,
                             QGraphicsOpacityEffect, QSplitter, QCheckBox)
from PyQt6.QtCore import Qt, QPropertyAnimation, QPoint, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QPixmap
import sounddevice as sd
import numpy as np
import threading
from pathlib import Path

from gui.vu_meter import VUMeter

VIRTUAL_DEVICES = ["cable", "vb-audio", "voicemeeter", "stereo mix", "virtual", "wave", "what u hear"]

# Языки, доступные для выбора в канале. Список совпадает с языками,
# для которых есть перевод (src/nmt.py) и голос синтеза
# (models/tts/piper/<код>/voice.onnx).
LANGUAGE_CHOICES = [
    ("ru", "Русский"),
    ("en", "English"),
    ("fr", "Français"),
    ("es", "Español"),
    ("hy", "Հայերեն"),
    ("zh", "中文"),
]

def is_real_device(device_name):
    name_lower = device_name.lower()
    for kw in VIRTUAL_DEVICES:
        if kw in name_lower:
            return False
    return True

class MicMeter(QObject):
    level_signal = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.stream = None
        self._alive = True

    def start_meter(self, device_id):
        self.stop_meter()
        if device_id is None or device_id < 0:
            return
        self._alive = True
        try:
            def audio_callback(indata, frames, time_info, status):
                if not self._alive:
                    return
                if indata.size > 0:
                    rms = np.sqrt(np.mean(indata**2))
                    if rms > 1e-10:
                        db = 20 * np.log10(rms)
                    else:
                        db = -100
                    level = (db + 60) / 60
                    level = max(0.0, min(1.0, level))
                    self.level_signal.emit(level)

            self.stream = sd.InputStream(device=device_id, channels=1, callback=audio_callback)
            self.stream.start()
        except Exception as e:
            print(f"⚠️ Не удалось запустить измеритель громкости: {e}")

    def stop_meter(self):
        self._alive = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
            self.stream = None
            self.level_signal.emit(0.0)


class FloatingText(QLabel):
    def __init__(self, full_text, parent_panel, is_left):
        super().__init__("", parent_panel)
        self.full_text = full_text
        self.current_text = ""
        self.char_index = 0

        self.setWordWrap(True)
        self.setMinimumWidth(150)
        self.setMaximumWidth(250)

        self.setStyleSheet("""
            background-color: rgba(43, 48, 54, 240);
            color: #00ffcc;
            border-radius: 12px;
            padding: 10px 14px;
            font-family: Consolas;
            font-size: 14px;
            border: 1px solid #00ffcc;
        """)

        avatar_rect = parent_panel.avatar_label.geometry()
        cx = avatar_rect.center().x()
        cy = avatar_rect.center().y()

        if is_left:
            self.start_x = cx + 80
        else:
            self.start_x = cx - 280

        self.start_y = cy - 50
        self.move(self.start_x, self.start_y)
        self.show()
        self.raise_()

        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(1.0)

        self.type_timer = QTimer(self)
        self.type_timer.timeout.connect(self._type_next_char)
        self.type_timer.start(15)

    def _type_next_char(self):
        if self.char_index < len(self.full_text):
            self.current_text += self.full_text[self.char_index]
            self.setText(self.current_text)
            self.adjustSize()
            self.char_index += 1
        else:
            self.type_timer.stop()
            QTimer.singleShot(1500, self._start_fly_and_fade)

    def _start_fly_and_fade(self):
        self.anim_pos = QPropertyAnimation(self, b"pos")
        self.anim_pos.setDuration(1500)
        self.anim_pos.setStartValue(QPoint(self.start_x, self.start_y))
        self.anim_pos.setEndValue(QPoint(self.start_x, self.start_y - 60))

        self.anim_op = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.anim_op.setDuration(1500)
        self.anim_op.setStartValue(1.0)
        self.anim_op.setEndValue(0.0)

        self.anim_pos.start()
        self.anim_op.start()
        self.anim_op.finished.connect(self.deleteLater)


class PersonPanel(QWidget):
    def __init__(self, lang_name, lang_code, is_left, image_path, controller):
        super().__init__()
        self.lang_code = lang_code
        self.is_left = is_left
        self.controller = controller
        self.is_muted = False

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # --- ПАНЕЛЬ УСТРОЙСТВ ---
        dev_frame = QFrame()
        dev_frame.setStyleSheet("background-color: #1e2227; border-radius: 8px;")
        dev_layout = QVBoxLayout(dev_frame)

        header = QLabel(lang_name)
        header.setStyleSheet("color: white; font-weight: bold; font-size: 16px; padding: 5px;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dev_layout.addWidget(header)

        style_combo = "background-color: #2b3036; color: white; border-radius: 4px; padding: 4px; border: 1px solid #455a64;"

        # Выбор языковой пары канала: на каком языке говорит собеседник
        # у этого микрофона и на какой язык переводить.
        lang_row = QHBoxLayout()
        src_label = QLabel("Говорит:")
        src_label.setStyleSheet("color: #a0aab5; font-size: 12px;")
        lang_row.addWidget(src_label)

        self.src_lang_combo = QComboBox()
        self.src_lang_combo.setStyleSheet(style_combo)
        for code, title in LANGUAGE_CHOICES:
            self.src_lang_combo.addItem(title, code)
        self.src_lang_combo.currentIndexChanged.connect(lambda idx: self._on_language_changed())
        lang_row.addWidget(self.src_lang_combo, 1)

        arrow = QLabel("→")
        arrow.setStyleSheet("color: #6b7684; font-size: 13px; padding: 0 4px;")
        lang_row.addWidget(arrow)

        self.tgt_lang_combo = QComboBox()
        self.tgt_lang_combo.setStyleSheet(style_combo)
        for code, title in LANGUAGE_CHOICES:
            self.tgt_lang_combo.addItem(title, code)
        self.tgt_lang_combo.currentIndexChanged.connect(lambda idx: self._on_language_changed())
        lang_row.addWidget(self.tgt_lang_combo, 1)

        dev_layout.addLayout(lang_row)
        self._load_language_pair()

        # Блок микрофона
        mic_label = QLabel("🎤 Микрофон:")
        mic_label.setStyleSheet("color: #a0aab5; font-size: 12px;")
        dev_layout.addWidget(mic_label)
        self.mic_combo = QComboBox()
        self.mic_combo.setStyleSheet(style_combo)
        self.mic_combo.currentIndexChanged.connect(lambda idx: self._on_mic_changed())
        dev_layout.addWidget(self.mic_combo)

        # VU-метр вместо QProgressBar
        self.vu_meter = VUMeter(orientation=Qt.Orientation.Vertical)
        self.vu_meter.setFixedHeight(60)
        self.vu_meter.setFixedWidth(20)
        meter_layout = QHBoxLayout()
        meter_layout.addStretch()
        meter_layout.addWidget(self.vu_meter)
        meter_layout.addStretch()
        dev_layout.addLayout(meter_layout)

        # Блок динамика
        spk_label = QLabel("🔊 Динамик:")
        spk_label.setStyleSheet("color: #a0aab5; font-size: 12px; margin-top: 5px;")
        dev_layout.addWidget(spk_label)
        spk_layout = QHBoxLayout()
        self.spk_combo = QComboBox()
        self.spk_combo.setStyleSheet(style_combo)
        self.spk_combo.currentIndexChanged.connect(lambda idx: self._on_spk_changed())

        self.test_spk_btn = QPushButton("🎵 Тест")
        self.test_spk_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.test_spk_btn.setStyleSheet("background-color: #455a64; color: white; border-radius: 4px; padding: 5px 10px; font-weight: bold;")
        self.test_spk_btn.clicked.connect(self._test_speaker)

        spk_layout.addWidget(self.spk_combo)
        spk_layout.addWidget(self.test_spk_btn)
        dev_layout.addLayout(spk_layout)

        # Блок уровня громкости перевода
        vol_layout = QHBoxLayout()
        vol_label = QLabel("Громкость перевода:")
        vol_label.setStyleSheet("color: #a0aab5; font-size: 12px;")
        vol_layout.addWidget(vol_label)
        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setRange(0, 150)
        self.vol_slider.setValue(100)
        self.vol_slider.valueChanged.connect(self._on_vol_changed)
        vol_layout.addWidget(self.vol_slider)
        dev_layout.addLayout(vol_layout)

        main_layout.addWidget(dev_frame)

        # Блок шумодава
        noise_layout = QHBoxLayout()
        self.noise_check = QCheckBox("Шумоподавление")
        self.noise_check.setStyleSheet("color: #a0aab5; font-size: 12px;")
        self.noise_check.setChecked(
            bool((controller.config.get('noise_reduction', {}) or {}).get(lang_code, False))
        )
        self.noise_check.stateChanged.connect(self._on_noise_changed)
        noise_layout.addWidget(self.noise_check)
        noise_layout.addStretch()
        dev_layout.addLayout(noise_layout)

        # --- АВАТАР ---
        avatar_layout = QHBoxLayout()
        self.avatar_label = QLabel()
        self.avatar_label.setFixedSize(250, 250)
        self.avatar_label.setStyleSheet("background-color: #14171a; border-radius: 125px; border: 3px solid #2d3740;")
        self.avatar_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(250, 250, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
            self.avatar_label.setPixmap(scaled)
        else:
            self.avatar_label.setText("Нет фото")
            self.avatar_label.setStyleSheet("color: #455a64;")

        avatar_layout.addStretch()
        avatar_layout.addWidget(self.avatar_label)
        avatar_layout.addStretch()
        main_layout.addLayout(avatar_layout)

        # --- КНОПКА МИКРОФОНА ---
        btn_layout = QHBoxLayout()
        self.mouth_btn = QPushButton("🎤 Микрофон Вкл")
        self.mouth_btn.setFixedSize(160, 50)
        self.mouth_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mouth_btn.setStyleSheet(self._btn_style(active=True))
        self.mouth_btn.clicked.connect(self.toggle_mute)

        btn_layout.addStretch()
        btn_layout.addWidget(self.mouth_btn)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)
        main_layout.addStretch()

        self.mic_meter_obj = MicMeter()
        self.mic_meter_obj.level_signal.connect(self.vu_meter.set_level)

    def _btn_style(self, active=True):
        color = "#00ffcc" if active else "#ff4444"
        bg = "#2b3036" if active else "#3a1c1c"
        return f"""
            QPushButton {{
                background-color: {bg}; color: {color};
                border-radius: 25px; border: 2px solid {color};
                font-size: 15px; font-weight: bold;
            }}
            QPushButton:hover {{ background-color: #3b424a; }}
        """

    def toggle_mute(self):
        self.is_muted = not self.is_muted
        self.mouth_btn.setText("🔇 Микрофон Выкл" if self.is_muted else "🎤 Микрофон Вкл")
        self.mouth_btn.setStyleSheet(self._btn_style(active=not self.is_muted))
        self.controller.set_mute(self.lang_code, self.is_muted)

    def _test_speaker(self):
        dev_id = self.spk_combo.currentData()
        if dev_id is not None:
            threading.Thread(target=self.controller.test_output_device, args=(dev_id,), daemon=True).start()

    def _on_vol_changed(self, val):
        self.controller.set_volume(self.lang_code, val)

    def _on_noise_changed(self, state):
        self.controller.set_noise_reduction(self.lang_code, self.noise_check.isChecked())

    def _on_mic_changed(self):
        dev_id = self.mic_combo.currentData()
        if dev_id is not None:
            self.controller.config[f'{self.lang_code}_input'] = dev_id
            self.controller.save_config()
            self.controller.reconfigure_pipeline(self.lang_code, dev_id)
            self._restart_meter()

    def _on_spk_changed(self):
        dev_id = self.spk_combo.currentData()
        if dev_id is not None:
            self.controller.config[f'{self.lang_code}_output'] = dev_id
            self.controller.save_config()
            self.controller.reconfigure_output(self.lang_code, dev_id)

    def _load_language_pair(self) -> None:
        """Показывает в списках языковую пару, сохранённую для канала."""
        config = self.controller.config
        source = config.get(f'{self.lang_code}_src_lang', self.lang_code)
        target = config.get(f'{self.lang_code}_tgt_lang', 'en')

        for combo, code in ((self.src_lang_combo, source), (self.tgt_lang_combo, target)):
            index = combo.findData(code)
            if index >= 0:
                # Блокируем сигналы: иначе выбор вызовет сохранение
                # прямо во время построения интерфейса.
                combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(False)

    def _on_language_changed(self) -> None:
        """Сохраняет выбранную пару языков и передаёт её контроллеру."""
        source = self.src_lang_combo.currentData()
        target = self.tgt_lang_combo.currentData()
        if source is None or target is None:
            return
        self.controller.change_language(self.lang_code, source, target)

    def _restart_meter(self):
        dev_id = self.mic_combo.currentData()
        if dev_id is not None:
            self.mic_meter_obj.start_meter(dev_id)

    def stop_meter(self):
        self.mic_meter_obj.stop_meter()

    def update_devices(self, devices):
        in_devs = []
        out_devs = []
        for dev in devices:
            name = dev['name']
            # Добавляем метку для Bluetooth-устройств
            from src.audio_devices import is_bluetooth_device
            if is_bluetooth_device(name):
                display_name = f"🔵 {name}"
            else:
                display_name = name
            if is_real_device(name):
                if dev['max_input_channels'] > 0:
                    in_devs.append((dev['index'], display_name))
                if dev['max_output_channels'] > 0:
                    out_devs.append((dev['index'], display_name))

        # Сортируем: сначала Bluetooth, потом остальные (по наличию эмодзи)
        in_devs.sort(key=lambda x: (0 if '🔵' in x[1] else 1, x[1].lower()))
        out_devs.sort(key=lambda x: (0 if '🔵' in x[1] else 1, x[1].lower()))

        self.mic_combo.blockSignals(True)
        self.spk_combo.blockSignals(True)

        self.mic_combo.clear()
        self.spk_combo.clear()

        for idx, name in in_devs:
            self.mic_combo.addItem(name[:40], idx)
        for idx, name in out_devs:
            self.spk_combo.addItem(name[:40], idx)

        cfg = self.controller.config
        mic_id = cfg.get(f'{self.lang_code}_input')
        spk_id = cfg.get(f'{self.lang_code}_output')

        if mic_id is not None:
            idx = self.mic_combo.findData(mic_id)
            if idx >= 0:
                self.mic_combo.setCurrentIndex(idx)
        if spk_id is not None:
            idx = self.spk_combo.findData(spk_id)
            if idx >= 0:
                self.spk_combo.setCurrentIndex(idx)

        self.mic_combo.blockSignals(False)
        self.spk_combo.blockSignals(False)

        self._restart_meter()

    def show_floating_text(self, text):
        FloatingText(text, self, self.is_left)


class MainTab(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.controller.chat_signal.connect(self.handle_chat_signal)
        self.controller.devices_updated.connect(self.refresh_devices)   # подписка на обновление
        self.setStyleSheet("background-color: #14171a;")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0,0,0,0)
        main_layout.setSpacing(0)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setStyleSheet("QSplitter::handle { background-color: #2d3740; width: 2px; }")

        self.persons_widget = QWidget()
        persons_layout = QHBoxLayout(self.persons_widget)
        persons_layout.setContentsMargins(0,0,0,0)

        base_dir = Path(__file__).parent.parent
        left_img = str(base_dir / "left_profile.png")
        right_img = str(base_dir / "right_profile.png")

        # Панели названы по собеседникам, а не по языкам: язык выбирается
        # списками внутри панели и может быть любым из доступных.
        self.ru_panel = PersonPanel("Собеседник 1", "ru", True, left_img, self.controller)
        self.en_panel = PersonPanel("Собеседник 2", "en", False, right_img, self.controller)

        persons_layout.addWidget(self.ru_panel)
        persons_layout.addWidget(self.en_panel)

        self.main_splitter.addWidget(self.persons_widget)

        self.chat_widget = QFrame()
        self.chat_widget.setStyleSheet("background-color: #1e2227;")
        chat_layout = QVBoxLayout(self.chat_widget)

        header_layout = QHBoxLayout()
        history_label = QLabel("История перевода")
        history_label.setStyleSheet("color: #00ffcc; font-weight: bold; font-size: 16px;")
        header_layout.addWidget(history_label)
        header_layout.addStretch()

        close_btn = QPushButton("✖")
        close_btn.setStyleSheet("color: #ff4444; background: transparent; font-weight: bold; font-size: 16px;")
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.toggle_chat)
        header_layout.addWidget(close_btn)

        chat_layout.addLayout(header_layout)

        self.chat_text = QTextEdit()
        self.chat_text.setReadOnly(True)
        self.chat_text.setStyleSheet("background-color: #14171a; color: #d4d4d4; font-size: 14px; border: 1px solid #2d3740; border-radius: 6px;")
        chat_layout.addWidget(self.chat_text)

        self.main_splitter.addWidget(self.chat_widget)
        self.chat_widget.hide()

        main_layout.addWidget(self.main_splitter)

        bottom_bar = QFrame()
        bottom_bar.setFixedHeight(70)
        bottom_bar.setStyleSheet("background-color: #1e2227; border-top: 2px solid #2d3740;")
        b_layout = QHBoxLayout(bottom_bar)

        self.start_btn = QPushButton("▶ Запустить переводчик")
        self.start_btn.setStyleSheet("background-color: #00ffcc; color: #1e2227; font-weight: bold; padding: 12px 24px; border-radius: 6px; font-size: 14px;")
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.clicked.connect(self.toggle_running)

        self.chat_btn = QPushButton("💬 Открыть чат")
        self.chat_btn.setStyleSheet("background-color: #455a64; color: white; padding: 12px 24px; border-radius: 6px; font-size: 14px;")
        self.chat_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chat_btn.clicked.connect(self.toggle_chat)

        self.refresh_btn = QPushButton("🔄 Обновить устройства")
        self.refresh_btn.setStyleSheet("background-color: #455a64; color: white; padding: 12px 24px; border-radius: 6px; font-size: 14px;")
        self.refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refresh_btn.clicked.connect(self.refresh_devices)

        # Новая кнопка автонастройки
        self.auto_btn = QPushButton("🔧 Автонастройка")
        self.auto_btn.setStyleSheet("background-color: #00aaff; color: white; padding: 12px 24px; border-radius: 6px; font-size: 14px;")
        self.auto_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_btn.clicked.connect(self.run_auto_setup)

        # Ручной режим языка (обход авто-детекции, см. план Phase 0.3)
        self.manual_lang_check = QCheckBox("Ручной язык")
        self.manual_lang_check.setStyleSheet("color: #d4d4d4; font-size: 13px;")
        self.manual_lang_check.setChecked(bool(getattr(self.controller, 'manual_language_mode', False)))
        self.manual_lang_check.setCursor(Qt.CursorShape.PointingHandCursor)
        self.manual_lang_check.stateChanged.connect(self._on_manual_language_changed)

        b_layout.addStretch()
        b_layout.addWidget(self.start_btn)
        b_layout.addWidget(self.chat_btn)
        b_layout.addWidget(self.refresh_btn)
        b_layout.addWidget(self.auto_btn)
        b_layout.addWidget(self.manual_lang_check)
        b_layout.addStretch()

        main_layout.addWidget(bottom_bar)

        self._refresh_devices()

    def toggle_chat(self):
        if self.chat_widget.isVisible():
            self.chat_widget.hide()
            self.chat_btn.setText("💬 Открыть чат")
        else:
            self.chat_widget.show()
            self.chat_btn.setText("💬 Скрыть чат")
            self.main_splitter.setSizes([int(self.width() * 0.7), int(self.width() * 0.3)])

    def _on_manual_language_changed(self, state):
        self.controller.set_manual_language_mode(self.manual_lang_check.isChecked())

    def toggle_running(self):
        if self.controller.is_running():
            self.controller.stop_all()
            self.start_btn.setText("▶ Запустить переводчик")
            self.start_btn.setStyleSheet("background-color: #00ffcc; color: #1e2227; font-weight: bold; padding: 12px 24px; border-radius: 6px; font-size: 14px;")
            self.ru_panel._restart_meter()
            self.en_panel._restart_meter()
        else:
            self.ru_panel.mic_meter_obj.stop_meter()
            self.en_panel.mic_meter_obj.stop_meter()

            self.controller.start_all()
            self.start_btn.setText("⏹ Остановить переводчик")
            self.start_btn.setStyleSheet("background-color: #ff4444; color: white; font-weight: bold; padding: 12px 24px; border-radius: 6px; font-size: 14px;")

    def handle_chat_signal(self, text):
        if text.startswith("[ru]"):
            panel = self.ru_panel
            content = text[4:].strip()
        elif text.startswith("[en]"):
            panel = self.en_panel
            content = text[4:].strip()
        else:
            return

        if content.startswith("RECOGNIZED:"):
            msg = content.replace("RECOGNIZED:", "").strip()
            panel.show_floating_text(msg)
        elif content.startswith("TRANSLATED:"):
            msg = content.replace("TRANSLATED:", "").strip()
            self.chat_text.append(f"<b style='color:#00ffcc;'>{panel.lang_code.upper()}:</b> {msg}")

    def _refresh_devices(self):
        devices = sd.query_devices()
        self.ru_panel.update_devices(devices)
        self.en_panel.update_devices(devices)

    def refresh_devices(self):
        self._refresh_devices()

    def stop_all_meters(self):
        self.ru_panel.stop_meter()
        self.en_panel.stop_meter()

    # ---------- НОВЫЙ МЕТОД ДЛЯ АВТОНАСТРОЙКИ ----------
    def run_auto_setup(self):
        self.controller.start_auto_setup()