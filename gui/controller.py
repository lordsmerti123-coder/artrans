# gui/controller.py
import json
import threading
import sys
import time
import queue
import numpy as np
import os
import re
from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QMessageBox, QInputDialog, QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

BASE_DIR = Path(__file__).resolve().parent.parent

from src.model_manager import ModelManager
from src.audio_pipeline import AudioPipeline, AudioSegmentTask
from src.audio_devices import AudioDeviceManager, is_bluetooth_device
from src.audio_router import AudioRouter
from src.language_gate import LanguageGate
from src.segmentation import dedupe_overlap
from src.session_log import SessionLog

DEFAULT_SEGMENTATION = {
    "merge_pause_ms": 2500,
    "max_segment_ms": 20000,
    "chunk_overlap_ms": 800,
    "min_speech_ms": 500,
    "min_audio_length_sec": 0.4,
}

DEFAULT_LANGUAGE_GATE = {
    "enabled": True,
    "threshold": 0.3,
    "min_chars": 3,
}

# Языки канала. Ключи используются в config.json (ru_input, ru_src_lang
# и так далее), поэтому менять их нельзя — иначе потеряются настройки
# устройств у тех, кто уже пользуется программой.
CHANNEL_CODES = ("ru", "en")

# Языки, доступные для перевода. Совпадает с языками в src/nmt.py
# и с голосами в models/tts/piper/.
LANGUAGE_CHOICES = [
    ("ru", "Русский"),
    ("en", "English"),
    ("fr", "Français"),
    ("es", "Español"),
    ("hy", "Հայերեն"),
    ("zh", "中文"),
]


class NLLBTranslator:
    def __init__(self, model_name="facebook/nllb-200-distilled-1.3B"):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Загрузка NLLB ({model_name}) на {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self._lock = threading.Lock()
        print("NLLB загружена.")

    def translate(self, text, src_lang, tgt_lang):
        if not text:
            return ""
        import torch
        lang_codes = {
            "ru": "rus_Cyrl",
            "en": "eng_Latn",
            "zh": "zho_Hans",
            "hy": "hye_Armn",
            "fr": "fra_Latn",
            "es": "spa_Latn"
        }
        src = lang_codes.get(src_lang, src_lang)
        tgt = lang_codes.get(tgt_lang, tgt_lang)

        with self._lock:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True
                )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class Controller(QObject):
    log_signal = pyqtSignal(str)
    chat_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, str)
    settings_updated = pyqtSignal()
    devices_updated = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.config = {}
        self.running = False
        self.mute_state = {"ru": False, "en": False}
        self.volume_state = {"ru": 100, "en": 100}
        self.pipelines = {}
        self.process_threads = {}
        self.threads = []

        self.model_manager = ModelManager(models_root=str(BASE_DIR / "models"))
        self.asr_model = None
        self.nmt_model = None
        self.tts_ru = None
        self.tts_en = None

        self.audio_router = AudioRouter(on_error=self._on_router_error)
        self.language_gate = None

        self.bluetooth_mode = False
        self.manual_language_mode = False

        self.load_config()

        self.session_log = SessionLog(
            base_dir=BASE_DIR / "logs",
            enabled=bool(self.config.get('log_enabled', True)),
            log_text=bool(self.config.get('log_text', True)),
            on_message=self.log_signal.emit,
        )
        self.metrics = self.session_log.metrics

        self._apply_models_config()
        self._init_language_gate()

    def load_config(self):
        config_path = Path(__file__).parent.parent / "config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.config = {
                'ru_input': None,
                'en_input': None,
                'ru_output': None,
                'en_output': None,
                'ru_src_lang': 'ru',
                'ru_tgt_lang': 'en',
                'en_src_lang': 'en',
                'en_tgt_lang': 'ru',
                'noise_reduction': {'ru': False, 'en': False},
                'bluetooth_mode': False,
                'manual_language_mode': False,
                'segmentation': dict(DEFAULT_SEGMENTATION),
                'language_gate': dict(DEFAULT_LANGUAGE_GATE),
                'models': {
                    'stt': 'multilingual_ctc',
                    'translator_type': 'gguf',
                    'translator_path': '',
                    'tts_ru': 'models/tts/piper/ru/voice.onnx',
                    'tts_en': 'models/tts/piper/en/voice.onnx'
                }
            }
        self.bluetooth_mode = self.config.get('bluetooth_mode', False)
        self.manual_language_mode = self.config.get('manual_language_mode', False)
        self.config.setdefault('segmentation', dict(DEFAULT_SEGMENTATION))
        self.config.setdefault('language_gate', dict(DEFAULT_LANGUAGE_GATE))
        self.save_config()

    @staticmethod
    def _resolve_path(path: str) -> str:
        """Превращает относительный путь в абсолютный относительно корня проекта."""
        if not path:
            return path
        p = Path(path).expanduser()
        if p.is_absolute():
            return str(p)
        return str((BASE_DIR / p).resolve())

    def _apply_models_config(self):
        """Передаёт настройки моделей из config.json в ModelManager."""
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

            m = self.config.get('models', {}) or {}
            stt = m.get('stt') or 'multilingual_ctc'
            translator_path = m.get('translator_path') or ''
            translator_type = m.get('translator_type') or ('gguf' if translator_path.lower().endswith('.gguf') else 'nllb')

            tts_root = m.get('tts_path')
            tts_ru = m.get('tts_ru') or (f"{tts_root}/ru/voice.onnx" if tts_root else 'models/tts/piper/ru/voice.onnx')
            tts_en = m.get('tts_en') or (f"{tts_root}/en/voice.onnx" if tts_root else 'models/tts/piper/en/voice.onnx')

            cfg = {
                'asr': {'type': 'gigaam', 'name': stt},
                'nmt': {'type': translator_type, 'path': translator_path, 'device': device},
                'tts': {
                    'ru': self._resolve_path(tts_ru),
                    'en': self._resolve_path(tts_en),
                    'use_cuda': False
                }
            }
            self.model_manager.set_config(cfg)
        except Exception as e:
            print(f"⚠️ Не удалось применить конфигурацию моделей: {e}")

    def save_config(self):
        config_path = Path(__file__).parent.parent / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------------- helpers
    def _on_router_error(self, message: str) -> None:
        self.metrics.inc("playback_errors")
        self.session_log.warn(f"[audio] {message}")
        self.log_signal.emit(f"⚠️ {message}")

    def _init_language_gate(self) -> None:
        """Собирает фильтр языка по текущим парам каналов.

        Фильтру важен полный набор языков: он отсекает текст, который
        не относится ни к одному из каналов.
        """
        gate_cfg = self.config.get('language_gate', {}) or {}

        languages = []
        for code in self.channel_codes:
            for key in (f'{code}_src_lang', f'{code}_tgt_lang'):
                value = self.config.get(key)
                if value and value not in languages:
                    languages.append(value)

        self.language_gate = LanguageGate(
            languages=languages or ["ru", "en"],
            threshold=gate_cfg.get('threshold', DEFAULT_LANGUAGE_GATE['threshold']),
            min_chars=gate_cfg.get('min_chars', DEFAULT_LANGUAGE_GATE['min_chars']),
            enabled=gate_cfg.get('enabled', True),
        )

    def _segmentation_config(self) -> dict:
        cfg = dict(DEFAULT_SEGMENTATION)
        cfg.update(self.config.get('segmentation', {}) or {})
        return cfg

    def set_manual_language_mode(self, enabled: bool) -> None:
        self.manual_language_mode = bool(enabled)
        self.config['manual_language_mode'] = bool(enabled)
        self.save_config()
        mode = "ручной" if enabled else "авто"
        self.session_log.info(f"Режим языка: {mode}")
        self.log_signal.emit(f"🌐 Режим языка: {mode}")

    def set_bluetooth_mode(self, enabled):
        self.bluetooth_mode = enabled
        self.config['bluetooth_mode'] = enabled
        self.save_config()
        self.log_signal.emit(f"🔵 Bluetooth режим: {'вкл' if enabled else 'выкл'}")

    def is_running(self):
        return self.running

    @property
    def channel_codes(self):
        """Коды каналов, которые ведёт контроллер."""
        return CHANNEL_CODES

    def change_language(self, lang_code, src_lang, tgt_lang):
        """Сохраняет языковую пару канала и перезапускает его при работе."""
        if lang_code not in self.channel_codes:
            return
        if src_lang == tgt_lang:
            self.log_signal.emit("⚠️ Язык говорящего и язык перевода совпадают")
            return

        self.config[f'{lang_code}_src_lang'] = src_lang
        self.config[f'{lang_code}_tgt_lang'] = tgt_lang
        self.save_config()

        # Фильтр языка настроен на конкретный список — пересобираем его,
        # иначе новая пара не пройдёт проверку.
        self._init_language_gate()

        names = {code: title for code, title in LANGUAGE_CHOICES}
        self.log_signal.emit(
            f"🌐 Канал {lang_code}: {names.get(src_lang, src_lang)} → "
            f"{names.get(tgt_lang, tgt_lang)}"
        )
        self.session_log.info(
            f"Языковая пара канала {lang_code}: {src_lang} → {tgt_lang}"
        )
        self.log_signal.emit(f"🌐 Язык {lang_code}: {src_lang} → {tgt_lang}")

    def set_noise_reduction(self, lang_code, enabled):
        if 'noise_reduction' not in self.config:
            self.config['noise_reduction'] = {'ru': False, 'en': False}
        self.config['noise_reduction'][lang_code] = bool(enabled)
        self.save_config()
        state = 'вкл' if enabled else 'выкл'
        self.session_log.event("noise_reduction", channel=lang_code, enabled=bool(enabled))
        self.log_signal.emit(f"🎛️ Шумоподавление для {lang_code}: {state}")
        if self.running:
            self.reconfigure_pipeline(lang_code, self.config.get(f'{lang_code}_input'), force=True)

    def set_mute(self, lang_code, muted):
        self.mute_state[lang_code] = muted
        self.log_signal.emit(f"{'🔇 Выключен' if muted else '🎤 Включен'} микрофон для {lang_code}")

    def set_volume(self, lang_code, volume):
        self.volume_state[lang_code] = volume

    def test_output_device(self, device_id):
        import sounddevice as sd
        try:
            rates = [44100, 48000, 16000, 22050, 8000]
            for rate in rates:
                try:
                    duration = 0.5
                    t = np.linspace(0, duration, int(rate * duration))
                    wave = 0.3 * np.sin(2 * np.pi * 440 * t)
                    sd.play(wave, samplerate=rate, device=device_id)
                    sd.wait()
                    self.log_signal.emit(f"✅ Тест звука: устройство {device_id}")
                    return True
                except Exception:
                    continue
            return False
        except Exception as e:
            return False

    def test_input_device(self, device_id):
        success, result = AudioPipeline.test_microphone(device_id)
        if success:
            self.log_signal.emit(f"✅ Микрофон {device_id} работает")
            return True
        return False

    def start_all(self):
        if self.running:
            return
        self.running = True
        self.status_signal.emit("Запуск...")
        threading.Thread(target=self._init_models_and_start, daemon=True).start()

    def stop_all(self):
        if not self.running:
            return
        self.running = False
        self.log_signal.emit("⏹️ Остановка...")
        # stop() пайплайна делает финальный флеш последней фразы в processing_queue.
        for pipe in self.pipelines.values():
            if pipe and hasattr(pipe, 'stop'):
                pipe.stop()

        for lang, thread in self.process_threads.items():
            if thread and thread.is_alive():
                thread.join(timeout=5.0)
        self.process_threads.clear()
        self.pipelines.clear()

        self.audio_router.stop()
        self.session_log.flush_summary()
        self.status_signal.emit("Остановлен")

    def _init_models_and_start(self):
        try:
            if not self.running: return
            self.progress_signal.emit(0, "Инициализация...")
            self.asr_model = self.model_manager.get_asr_model()
            self.nmt_model = self.model_manager.get_nmt_model()

            # Голос синтеза берётся по языку перевода каждого канала:
            # переводят на французский — звучит французский голос.
            self.tts_voices = {}
            for code in self.channel_codes:
                target = self.config.get(f'{code}_tgt_lang', 'en')
                if target not in self.tts_voices:
                    self.tts_voices[target] = self.model_manager.get_tts_model(target)

            self.progress_signal.emit(80, "Запуск пайплайнов...")
            if self.running:
                self._start_pipelines()
        except Exception as e:
            self.session_log.error(f"Ошибка инициализации: {e}")
            self.log_signal.emit(f"❌ Ошибка: {e}")
            self.running = False

    def _make_pipeline(self, lang, device_id):
        noise_cfg = self.config.get('noise_reduction', {}) or {}
        return AudioPipeline(
            device_id=device_id,
            asr_model=self.asr_model,
            asr_lang=self.config.get(f'{lang}_src_lang', lang),
            vad_threshold=self.config.get('vad_threshold', 0.8),
            segmentation=self._segmentation_config(),
            denoise=bool(noise_cfg.get(lang, False)),
            noise_backend=self.config.get('noise_backend', 'auto'),
        )

    def _prepare_playback(self, audio_bytes, lang_name):
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        vol_percent = self.volume_state.get(lang_name, 100)
        if vol_percent != 100:
            multiplier = vol_percent / 100.0
            audio = np.clip(audio * multiplier, -32768, 32767).astype(np.int16)
        if self.bluetooth_mode:
            silence = np.zeros(int(0.5 * 22050), dtype=np.int16)
            audio = np.concatenate((silence, audio, silence))
        return audio

    def _channel_worker(self, pipeline, translator, tts, src_lang, tgt_lang, lang_name):
        """STT -> языковой гейт -> MT -> TTS -> вывод на устройство получателя."""
        prev_text = ""
        prev_forced = False

        def handle(task):
            nonlocal prev_text, prev_forced
            path = getattr(task, 'path', task)
            forced = bool(getattr(task, 'forced', False))

            if self.mute_state.get(lang_name, False):
                try: os.remove(path)
                except OSError: pass
                return

            t_start = time.time()
            text = pipeline.transcribe_file(path)
            stt_ms = (time.time() - t_start) * 1000.0
            try: os.remove(path)
            except OSError: pass
            if not text or not text.strip():
                return

            self.metrics.inc("segments_recognized")
            self.metrics.observe("stt_ms", stt_ms)

            # Склейка перекрытия после жёсткого реза по лимиту длины.
            if prev_forced and prev_text:
                text = dedupe_overlap(prev_text, text)
                if not text:
                    return

            result = self.language_gate.check(
                text, src_lang, strict=not self.manual_language_mode
            )
            if not result.accepted:
                self.metrics.inc("gate_rejects")
                self.session_log.event(
                    "gate_reject", channel=lang_name, reason=result.reason,
                    detected=result.detected, script=result.script,
                )
                return

            prev_text = text
            prev_forced = forced

            self.session_log.text_event(
                "recognized", text, channel=lang_name, forced=forced,
                stt_ms=round(stt_ms),
            )
            self.chat_signal.emit(f"[{lang_name}] RECOGNIZED: {text}")

            t_mt = time.time()
            translated = translator.translate(text, src_lang, tgt_lang)
            mt_ms = (time.time() - t_mt) * 1000.0
            if not translated or not translated.strip():
                return

            self.metrics.observe("mt_ms", mt_ms)
            self.session_log.text_event(
                "translated", translated, channel=lang_name, mt_ms=round(mt_ms)
            )
            self.chat_signal.emit(f"[{lang_name}] TRANSLATED: {text} → {translated}")

            t_tts = time.time()
            audio_bytes = tts.synthesize_to_bytes(translated)
            tts_ms = (time.time() - t_tts) * 1000.0
            if not audio_bytes:
                return
            self.metrics.observe("tts_ms", tts_ms)

            output_device = self.config.get(f'{tgt_lang}_output')
            # Громкость берём у панели получателя перевода (её слушатель).
            audio = self._prepare_playback(audio_bytes, tgt_lang)
            if self.audio_router.route(audio, 22050, output_device):
                self.metrics.inc("segments_played")
                self.metrics.observe("e2e_ms", (time.time() - t_start) * 1000.0)
                self.session_log.event(
                    "segment", channel=lang_name, tts_ms=round(tts_ms),
                    total_ms=round((time.time() - t_start) * 1000.0),
                )
            else:
                self.session_log.warn(
                    f"Канал {lang_name}: не удалось воспроизвести на устройстве {output_device}"
                )

        while self.running:
            try:
                task = pipeline.processing_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                handle(task)
            except Exception as e:
                self.session_log.error(f"Ошибка пайплайна {lang_name}: {e}")
                self.log_signal.emit(f"❌ Ошибка пайплайна {lang_name}: {e}")

        # Дренаж очереди: финальный сегмент, отданный при stop().
        while True:
            try:
                task = pipeline.processing_queue.get_nowait()
            except queue.Empty:
                break
            try:
                handle(task)
            except Exception as e:
                self.session_log.error(f"Ошибка финального сегмента {lang_name}: {e}")

    def _start_pipelines(self):
        if not self.running: return

        ru_in = self.config.get('ru_input')
        en_in = self.config.get('en_input')
        ru_out = self.config.get('ru_output')
        en_out = self.config.get('en_output')

        if ru_in is None or en_in is None or ru_out is None or en_out is None:
            self.session_log.error("Устройства не настроены полностью")
            self.log_signal.emit("❌ Устройства не настроены полностью.")
            self.running = False
            return

        # Пайплайны собираются по кодам каналов. Язык берётся из конфига
        # при запуске канала ниже.
        started = {}
        for code in self.channel_codes:
            device_id = self.config.get(f'{code}_input')
            self.pipelines[code] = self._make_pipeline(code, device_id)
            started[code] = self.pipelines[code].start()

        for lang, pipe in self.pipelines.items():
            denoiser = getattr(pipe, '_denoiser', None)
            self.session_log.info(
                f"Канал {lang}: устройство={self.config.get(f'{lang}_input')} "
                f"rate={getattr(pipe, '_actual_sample_rate', None)} "
                f"channels={getattr(pipe, 'channels', None)} "
                f"started={started.get(lang)} "
                f"denoise={getattr(denoiser, 'active_backend', 'off')}"
            )
            noise_cfg = self.config.get('noise_reduction', {}) or {}
            if noise_cfg.get(lang) and not getattr(denoiser, 'is_active', False):
                self.session_log.warn(
                    f"Канал {lang}: шумодав запрошен, но недоступен "
                    f"(не установлены rnnoise/noisereduce)"
                )

        self.session_log.event(
            "start", manual_language_mode=self.manual_language_mode,
            bluetooth_mode=self.bluetooth_mode, segmentation=self._segmentation_config(),
            ru_in=ru_in, en_in=en_in, ru_out=ru_out, en_out=en_out,
        )

        # Каждый канал переводит на свой целевой язык и озвучивает его
        # голосом. Голоса загружены заранее в _init_models_and_start.
        for code in self.channel_codes:
            if code not in self.pipelines:
                continue

            src = self.config.get(f'{code}_src_lang', code)
            tgt = self.config.get(f'{code}_tgt_lang', 'en' if code == 'ru' else 'ru')
            voice = getattr(self, 'tts_voices', {}).get(tgt)

            if voice is None:
                self.session_log.warn(
                    f"Канал {code}: нет голоса для языка '{tgt}', канал пропущен"
                )
                continue

            thread = threading.Thread(
                target=self._channel_worker,
                args=(self.pipelines[code], self.nmt_model, voice, src, tgt, code),
                daemon=True,
            )
            thread.start()
            self.process_threads[code] = thread

        self.status_signal.emit("Переводчик работает")

    def reconfigure_pipeline(self, lang, new_input_device, force: bool = False):
        if new_input_device is None:
            return
        old_device = self.config.get(f'{lang}_input')
        if old_device == new_input_device and not force:
            return

        self.config[f'{lang}_input'] = new_input_device
        self.save_config()

        if not self.running:
            self.log_signal.emit(f"🔄 Устройство {lang} обновлено (не активно)")
            return

        old_pipe = self.pipelines.get(lang)
        if old_pipe:
            old_pipe.stop()
            time.sleep(0.3)

        old_thread = self.process_threads.get(lang)
        if old_thread and old_thread.is_alive():
            old_thread.join(timeout=2.0)

        src_lang = self.config.get(f'{lang}_src_lang', lang)
        tgt_lang = self.config.get(f'{lang}_tgt_lang', 'en' if lang == 'ru' else 'ru')

        new_pipe = self._make_pipeline(lang, new_input_device)
        new_pipe.start()
        self.pipelines[lang] = new_pipe

        tts = self.tts_en if lang == 'ru' else self.tts_ru
        new_thread = threading.Thread(
            target=self._channel_worker,
            args=(new_pipe, self.nmt_model, tts, src_lang, tgt_lang, lang),
            daemon=True,
        )
        new_thread.start()
        self.process_threads[lang] = new_thread

        self.log_signal.emit(f"🔄 Микрофон {lang} переключён на {new_input_device}")

    def reconfigure_output(self, lang, new_output_device):
        if new_output_device is None:
            return
        old_device = self.config.get(f'{lang}_output')
        if old_device == new_output_device:
            return

        self.config[f'{lang}_output'] = new_output_device
        self.save_config()
        self.log_signal.emit(f"🔄 Динамик {lang} переключён на {new_output_device}")

    def apply_settings(self, updates):
        self.config.update(updates)
        self.save_config()
        self._apply_models_config()
        if self.running:
            self.stop_all()
        time.sleep(0.5)
        self.start_all()
        self.settings_updated.emit()
        self.devices_updated.emit()

    # ---------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ АВТОНАСТРОЙКИ ----------
    def _normalize_device_name(self, name: str) -> str:
        """
        Извлекает основное имя гарнитуры из полного имени устройства.
        Пример: 'Output 1 (... ;(QCY-T1C))' -> 'QCY-T1C'
        """
        # Ищем последние скобки
        match = re.search(r'\(([^)]+)\)', name)
        if match:
            return match.group(1).strip()
        # Если нет скобок, убираем путь до последнего разделителя
        parts = re.split(r'[;/\\]', name)
        if parts:
            last = parts[-1].strip()
            # Если там есть запятая, берём после запятой
            if ',' in last:
                last = last.split(',')[-1].strip()
            return last
        return name.strip()

    def _pyaudio_to_sounddevice(self, pyaudio_idx):
        """
        Сопоставляет индекс PyAudio с индексом sounddevice по нормализованному имени.
        """
        # Получаем все устройства PyAudio и sounddevice с принудительным обновлением
        pyaudio_devices = self._get_current_devices()
        sd_devices = self._get_sounddevice_devices()
        
        # Находим устройство в PyAudio по индексу
        pyaudio_dev = next((d for d in pyaudio_devices if d['index'] == pyaudio_idx), None)
        if not pyaudio_dev:
            print(f"  ❌ Устройство с PyAudio индексом {pyaudio_idx} не найдено")
            return None
        
        pyaudio_name = self._normalize_device_name(pyaudio_dev['name'])
        print(f"  Сопоставление PyAudio индекса {pyaudio_idx} (норм. имя '{pyaudio_name}') с sounddevice...")
        
        # Ищем в sounddevice по нормализованному имени
        for sd_dev in sd_devices:
            sd_name = self._normalize_device_name(sd_dev['name'])
            if sd_name == pyaudio_name:
                print(f"    ✅ Точное совпадение: sounddevice индекс {sd_dev['index']}")
                return sd_dev['index']
            # Проверяем вхождение (один содержит другой)
            if pyaudio_name in sd_name or sd_name in pyaudio_name:
                print(f"    ⚠️ Частичное совпадение: sounddevice индекс {sd_dev['index']} (норм. имя '{sd_name}')")
                return sd_dev['index']
        
        print(f"    ❌ Совпадений не найдено")
        return None

    # ---------- АВТОНАСТРОЙКА ----------
    def start_auto_setup(self):
        """Интерактивный мастер автонастройки Bluetooth-гарнитур."""
        print("\n=== ЗАПУСК АВТОНАСТРОЙКИ ===")
        all_devices = self._get_current_devices()
        print("Все доступные устройства (через PyAudio):")
        for dev in all_devices:
            bt = "🔵" if is_bluetooth_device(dev['name']) else "  "
            print(f"  {bt} [{dev['index']}] {dev['name']} (вход:{dev['max_input_channels']}, выход:{dev['max_output_channels']})")

        dialog = QDialog()
        dialog.setWindowTitle("Автонастройка устройств")
        dialog.setModal(True)
        dialog.resize(550, 350)

        layout = QVBoxLayout(dialog)

        instruction_label = QLabel()
        instruction_label.setWordWrap(True)
        layout.addWidget(instruction_label)

        status_label = QLabel()
        status_label.setWordWrap(True)
        layout.addWidget(status_label)

        btn_layout = QHBoxLayout()
        find_btn = QPushButton("Найти устройство")
        manual_btn = QPushButton("Выбрать вручную")
        cancel_btn = QPushButton("Отмена")
        btn_layout.addWidget(find_btn)
        btn_layout.addWidget(manual_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        step = 0
        assigned = {'en': None, 'ru': None}  # каждый: {'name': str, 'input': int (pyaudio), 'output': int (pyaudio)}
        initial_devices = all_devices

        def update_ui():
            if step == 0:
                instruction_label.setText(
                    "Шаг 1 из 2: Настройка для <b>английского</b> языка.\n"
                    "Подключите Bluetooth-гарнитуру для англоговорящего.\n"
                    "Затем нажмите «Найти устройство»."
                )
                status_label.setText("Ожидание подключения…")
                find_btn.setText("Найти устройство")
                find_btn.setEnabled(True)
                manual_btn.setEnabled(True)
            elif step == 1:
                instruction_label.setText(
                    "Шаг 2 из 2: Настройка для <b>русского</b> языка.\n"
                    "Подключите вторую Bluetooth-гарнитуру для русскоговорящего.\n"
                    "Затем нажмите «Найти устройство»."
                )
                status_label.setText("Ожидание подключения…")
                find_btn.setText("Найти устройство")
                find_btn.setEnabled(True)
                manual_btn.setEnabled(True)
            elif step == 2:
                instruction_label.setText("✅ Настройка завершена!")
                en = assigned['en']
                ru = assigned['ru']
                status_label.setText(
                    f"Английский: {en['name']} (вход {en['input']}, выход {en['output']})\n"
                    f"Русский: {ru['name']} (вход {ru['input']}, выход {ru['output']})"
                )
                find_btn.setText("Готово")
                find_btn.setEnabled(True)
                manual_btn.setEnabled(False)
                find_btn.clicked.disconnect()
                find_btn.clicked.connect(dialog.accept)

        def find_bluetooth_headset(devices):
            """Ищет в списке устройств Bluetooth-гарнитуру (группу вход+выход)."""
            bt_devices = [d for d in devices if is_bluetooth_device(d['name'])]
            if not bt_devices:
                return None

            groups = {}
            for dev in bt_devices:
                norm_name = self._normalize_device_name(dev['name'])
                if norm_name not in groups:
                    groups[norm_name] = {'inputs': [], 'outputs': []}
                if dev['max_input_channels'] > 0:
                    groups[norm_name]['inputs'].append(dev)
                if dev['max_output_channels'] > 0:
                    groups[norm_name]['outputs'].append(dev)

            for name, group in groups.items():
                if group['inputs'] and group['outputs']:
                    return {
                        'name': name,
                        'input': group['inputs'][0]['index'],
                        'output': group['outputs'][0]['index']
                    }
            return None

        def on_find():
            nonlocal step, initial_devices
            if step == 2:
                dialog.accept()
                return

            find_btn.setEnabled(False)
            status_label.setText("🔍 Поиск новой Bluetooth-гарнитуры…")
            dialog.repaint()

            current_devices = self._get_current_devices()
            initial_indices = {dev['index'] for dev in initial_devices}
            new_devices = [dev for dev in current_devices if dev['index'] not in initial_indices]

            print("\n--- Поиск новых устройств (PyAudio, обновлённый список) ---")
            print(f"Было устройств: {len(initial_devices)}, стало: {len(current_devices)}")
            print("Новые устройства:")
            for dev in new_devices:
                print(f"  [{dev['index']}] {dev['name']} (вх:{dev['max_input_channels']}, вых:{dev['max_output_channels']})")

            headset = find_bluetooth_headset(new_devices)
            if headset:
                print(f"Найдена Bluetooth-гарнитура: {headset['name']} (вход {headset['input']}, выход {headset['output']})")
                status_label.setText(f"Найдена гарнитура: {headset['name']}. Тестирование…")
                dialog.repaint()

                # Сопоставляем индексы с sounddevice по имени
                sd_input = self._pyaudio_to_sounddevice(headset['input'])
                sd_output = self._pyaudio_to_sounddevice(headset['output'])
                if sd_input is None or sd_output is None:
                    status_label.setText("❌ Не удалось сопоставить индексы устройств. Попробуйте вручную.")
                    find_btn.setEnabled(True)
                    return

                mic_ok = self.test_input_device(sd_input)
                spk_ok = self.test_output_device(sd_output)
                if mic_ok and spk_ok:
                    lang = 'en' if step == 0 else 'ru'
                    assigned[lang] = headset
                    status_label.setText(f"✅ Гарнитура {headset['name']} успешно настроена для {lang}.")
                    step += 1
                    initial_devices = current_devices
                    update_ui()
                else:
                    status_label.setText("❌ Тест устройства не пройден. Попробуйте другую гарнитуру или выберите вручную.")
                    find_btn.setEnabled(True)
            else:
                status_label.setText(
                    "⚠️ Новой Bluetooth-гарнитуры не обнаружено.\n"
                    "Убедитесь, что гарнитура подключена и имеет микрофон и динамик.\n"
                    "Нажмите «Найти устройство» снова или выберите вручную."
                )
                find_btn.setEnabled(True)

        def on_manual():
            nonlocal step
            all_devs = self._get_current_devices()
            headsets = []
            groups = {}
            for dev in all_devs:
                if not is_bluetooth_device(dev['name']):
                    continue
                norm_name = self._normalize_device_name(dev['name'])
                if norm_name not in groups:
                    groups[norm_name] = {'inputs': [], 'outputs': []}
                if dev['max_input_channels'] > 0:
                    groups[norm_name]['inputs'].append(dev)
                if dev['max_output_channels'] > 0:
                    groups[norm_name]['outputs'].append(dev)
            for name, group in groups.items():
                if group['inputs'] and group['outputs']:
                    headsets.append({
                        'name': name,
                        'input': group['inputs'][0]['index'],
                        'output': group['outputs'][0]['index']
                    })

            if not headsets:
                QMessageBox.warning(dialog, "Нет устройств", "Не найдено Bluetooth-гарнитур с микрофоном и динамиком.")
                return

            items = [f"{h['name']} (вх:{h['input']}, вых:{h['output']})" for h in headsets]
            item, ok = QInputDialog.getItem(dialog, "Выбор гарнитуры",
                                            "Выберите Bluetooth-гарнитуру для текущего языка:",
                                            items, 0, False)
            if ok and item:
                selected_name = item.split(' (вх:')[0]
                for h in headsets:
                    if h['name'] == selected_name:
                        lang = 'en' if step == 0 else 'ru'
                        assigned[lang] = h
                        status_label.setText(f"✅ Вручную выбрана гарнитура {h['name']} для {lang}.")
                        step += 1
                        update_ui()
                        break

        find_btn.clicked.connect(on_find)
        manual_btn.clicked.connect(on_manual)
        cancel_btn.clicked.connect(dialog.reject)

        update_ui()
        if dialog.exec() == QDialog.DialogCode.Accepted:
            ru = assigned.get('ru')
            en = assigned.get('en')
            if ru and en:
                # Получаем индексы sounddevice
                ru_sd_input = self._pyaudio_to_sounddevice(ru['input'])
                ru_sd_output = self._pyaudio_to_sounddevice(ru['output'])
                en_sd_input = self._pyaudio_to_sounddevice(en['input'])
                en_sd_output = self._pyaudio_to_sounddevice(en['output'])
                if None in (ru_sd_input, ru_sd_output, en_sd_input, en_sd_output):
                    QMessageBox.warning(None, "Ошибка", "Не удалось сопоставить индексы устройств. Попробуйте вручную.")
                    return
                self.config['ru_input'] = ru_sd_input
                self.config['ru_output'] = ru_sd_output
                self.config['en_input'] = en_sd_input
                self.config['en_output'] = en_sd_output
                self.save_config()
                if self.running:
                    self.stop_all()
                    self.start_all()
                self.devices_updated.emit()
                QMessageBox.information(None, "Успех", "Устройства настроены успешно!")
            else:
                QMessageBox.warning(None, "Ошибка", "Не удалось настроить оба устройства.")
        print("=== АВТОНАСТРОЙКА ЗАВЕРШЕНА ===\n")

    def _get_current_devices(self):
        """
        Получение свежего списка устройств через PyAudio с пересозданием объекта.
        Это гарантирует, что мы видим подключённые после старта программы устройства.
        """
        import pyaudio
        p = pyaudio.PyAudio()
        devices = []
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            devices.append({
                'index': i,
                'name': info['name'],
                'max_input_channels': int(info['maxInputChannels']),
                'max_output_channels': int(info['maxOutputChannels']),
            })
        p.terminate()
        return devices

    def _get_sounddevice_devices(self):
        """
        Получение свежего списка устройств через sounddevice с принудительной
        переинициализацией PortAudio.
        """
        import sounddevice as sd
        try:
            sd._terminate()
        except Exception:
            pass
        sd._initialize()
        devices = sd.query_devices()
        result = []
        for dev in devices:
            idx = dev.get('index', devices.index(dev))
            result.append({
                'index': idx,
                'name': dev['name'],
                'max_input_channels': dev['max_input_channels'],
                'max_output_channels': dev['max_output_channels']
            })
        return result