# src/model_manager.py
"""
Централизованное управление моделями: загрузка, кеширование, выгрузка.
Все модели загружаются локально (первая загрузка может скачать из интернета,
но затем сохраняются в локальных папках).
"""

import os
import sys
import threading
import time
import torch  # обязательный импорт
from pathlib import Path
from typing import Optional, Dict, Any, Union

MODEL_TYPE_ASR = "asr"
MODEL_TYPE_NMT = "nmt"
MODEL_TYPE_TTS = "tts"

class ModelManager:
    """Менеджер моделей с ленивой загрузкой и возможностью выгрузки."""

    def __init__(self, models_root: Optional[str] = None):
        if models_root is None:
            models_root = str(Path(__file__).resolve().parent.parent / "models")
        self.models_root = Path(models_root)
        self._models: Dict[str, Any] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._loading: Dict[str, bool] = {}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"ModelManager: устройство вычислений — {self._device}")

        self.asr_path = self.models_root / "asr"
        self.tts_ru_path = self.models_root / "tts" / "piper" / "ru" / "voice.onnx"
        self.tts_en_path = self.models_root / "tts" / "piper" / "en" / "voice.onnx"

        # Конфигурация по умолчанию (переопределяется из config.json через set_config)
        self.config = {
            "asr": {
                "type": "gigaam",
                "name": "multilingual_ctc"
            },
            "nmt": {
                "type": "gguf",
                "path": "",
                "device": self._device
            },
            "tts": {
                "ru": str(self.tts_ru_path),
                "en": str(self.tts_en_path),
                "use_cuda": False
            }
        }

    def _get_lock(self, key: str) -> threading.Lock:
        if key not in self._locks:
            self._locks[key] = threading.Lock()
        return self._locks[key]

    def _is_loading(self, key: str) -> bool:
        return self._loading.get(key, False)

    def _set_loading(self, key: str, value: bool):
        self._loading[key] = value

    # ---------- ASR ----------
    def get_asr_model(self, force_reload: bool = False):
        key = "asr"
        if key in self._models and not force_reload:
            return self._models[key]

        with self._get_lock(key):
            if key in self._models and not force_reload:
                return self._models[key]
            if self._is_loading(key):
                while self._is_loading(key):
                    time.sleep(0.1)
                return self._models.get(key)

            self._set_loading(key, True)
            try:
                model = self._load_asr_model()
                self._models[key] = model
                return model
            finally:
                self._set_loading(key, False)

    def _load_asr_model(self):
        import gigaam
        name = self.config["asr"].get("name", "multilingual_ctc")
        download_root = str(self.asr_path)
        print(f"🎙️ Загрузка ASR (GigaAM '{name}') из {download_root} ...")
        # Если файл <name>.ckpt уже лежит в download_root — он будет использован без загрузки.
        # Иначе модель будет скачана и проверена по контрольной сумме.
        model = gigaam.load_model(name, download_root=download_root)
        model.eval()
        print("✅ GigaAM ASR загружена.")
        return model

    def unload_asr(self):
        key = "asr"
        if key in self._models:
            del self._models[key]
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("🗑️ ASR модель выгружена.")

    # ---------- NMT ----------
    def get_nmt_model(self, force_reload: bool = False):
        key = "nmt"
        if key in self._models and not force_reload:
            return self._models[key]

        with self._get_lock(key):
            if key in self._models and not force_reload:
                return self._models[key]
            if self._is_loading(key):
                while self._is_loading(key):
                    time.sleep(0.1)
                return self._models.get(key)

            self._set_loading(key, True)
            try:
                model = self._load_nmt_model()
                self._models[key] = model
                return model
            finally:
                self._set_loading(key, False)

    def _load_nmt_model(self):
        nmt_cfg = self.config["nmt"]
        nmt_type = nmt_cfg.get("type", "nllb")
        path = nmt_cfg.get("path", "") or ""
        device = nmt_cfg.get("device") or self._device

        if nmt_type == "gguf":
            if not path or not os.path.isfile(path):
                raise FileNotFoundError(
                    f"GGUF-файл переводчика не найден: '{path}'. "
                    f"Укажите путь к модели в config.json (models.translator_path)."
                )
            from src.nmt import Translator
            print(f"🧠 Загрузка GGUF модели: {path}")
            model = Translator(
                model_path=path,
                device=device,
                n_gpu_layers=-1 if device == "cuda" else 0,
                n_ctx=2048
            )
            return model
        else:
            if not path:
                path = "facebook/nllb-200-distilled-1.3B"
            print(f"🧠 Загрузка NLLB модели: {path}")
            from gui.controller import NLLBTranslator
            model = NLLBTranslator(model_name=path)
            return model

    def unload_nmt(self):
        key = "nmt"
        if key in self._models:
            model = self._models[key]
            if hasattr(model, "stop"):
                model.stop()
            del self._models[key]
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("🗑️ NMT модель выгружена.")

    # ---------- TTS ----------
    def get_tts_model(self, lang: str, force_reload: bool = False):
        key = f"tts_{lang}"
        if key in self._models and not force_reload:
            return self._models[key]

        with self._get_lock(key):
            if key in self._models and not force_reload:
                return self._models[key]
            if self._is_loading(key):
                while self._is_loading(key):
                    time.sleep(0.1)
                return self._models.get(key)

            self._set_loading(key, True)
            try:
                model = self._load_tts_model(lang)
                self._models[key] = model
                return model
            finally:
                self._set_loading(key, False)

    def _load_tts_model(self, lang: str):
        """Загружает голос синтеза для указанного языка.

        Путь ищется в настройках, иначе берётся стандартный
        models/tts/piper/<код>/voice.onnx. Так добавляется любой язык,
        для которого положен голос.

        @param lang: код языка голоса.
        @returns: загруженная модель синтеза.
        """
        from src.tts import TTS

        tts_cfg = self.config.get("tts", {}) or {}
        voice_path = tts_cfg.get(lang) or str(
            self.models_root / "tts" / "piper" / lang / "voice.onnx"
        )

        if not Path(voice_path).exists():
            raise ValueError(
                f"Голос для языка '{lang}' не найден: {voice_path}"
            )

        use_cuda = tts_cfg.get("use_cuda", False)
        print(f"🧠 Загрузка TTS ({lang}): {voice_path}")
        return TTS(voice_path, use_cuda=use_cuda)

    def unload_tts(self, lang: str):
        key = f"tts_{lang}"
        if key in self._models:
            del self._models[key]
            import gc
            gc.collect()
            print(f"🗑️ TTS ({lang}) модель выгружена.")

    def unload_all(self):
        self.unload_asr()
        self.unload_nmt()
        self.unload_tts("ru")
        self.unload_tts("en")
        print("🧹 Все модели выгружены.")

    def set_config(self, config: dict):
        """Обновляет конфигурацию моделей и выгружает старые при необходимости."""
        reload_needed = False
        for key in ["asr", "nmt", "tts"]:
            if key in config and config[key] != self.config.get(key):
                reload_needed = True
                self.config[key] = config[key]
        if reload_needed and self._models:
            print("⚙️ Конфигурация моделей обновлена, выгружаем загруженные модели.")
            self.unload_all()