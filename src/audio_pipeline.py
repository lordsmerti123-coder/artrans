# src/audio_pipeline.py
import math
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from scipy.signal import resample_poly

from src.noise import NoiseReducer
from src.segmentation import Segment, SpeechSegmenter

VAD_CHUNK = 512
TARGET_RATE = 16000


@dataclass
class AudioSegmentTask:
    """Задание на распознавание: файл + метаданные сегмента."""

    path: str
    duration: float = 0.0
    forced: bool = False
    final: bool = False
    index: int = 0
    created_at: float = field(default_factory=time.time)

    def __fspath__(self):
        return self.path


class AudioPipeline:
    def __init__(self,
                 device_id: Optional[int] = None,
                 target_sample_rate: int = TARGET_RATE,
                 vad_threshold: float = 0.8,
                 silence_timeout_ms: int = 1500,
                 min_speech_duration_ms: int = 500,
                 min_audio_length_sec: float = 0.4,
                 asr_model: Optional[Any] = None,
                 asr_lang: Optional[str] = None,
                 segmentation: Optional[dict] = None,
                 denoise: bool = False,
                 noise_backend: str = "auto"):
        self.device_id = device_id
        self.target_sample_rate = target_sample_rate
        self.vad_threshold = vad_threshold
        self.silence_timeout_ms = silence_timeout_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_audio_length_sec = min_audio_length_sec
        self.running = False
        self.processing_queue = queue.Queue()
        self._thread = None
        self._vad_model = None
        self._actual_sample_rate = None
        self.channels = 1
        self.asr_model = asr_model
        self.asr_lang = asr_lang
        self._asr_lock = threading.Lock()

        self.denoise = bool(denoise)
        self.noise_backend = noise_backend
        self._denoiser: Optional[NoiseReducer] = None

        # Гибридная сегментация: конфиг + параметры для обратной совместимости.
        self.segmentation_config = dict(segmentation or {})
        merge_default = max(self.silence_timeout_ms, 2500) if self.silence_timeout_ms else 2500
        self.merge_pause_ms = float(self.segmentation_config.get("merge_pause_ms", merge_default))
        self.max_segment_ms = float(self.segmentation_config.get("max_segment_ms", 20000))
        self.chunk_overlap_ms = float(self.segmentation_config.get("chunk_overlap_ms", 800))
        self.min_speech_ms = float(
            self.segmentation_config.get("min_speech_ms", self.min_speech_duration_ms)
        )
        self.min_audio_length_sec = float(
            self.segmentation_config.get("min_audio_length_sec", self.min_audio_length_sec)
        )
        self._segmenter: Optional[SpeechSegmenter] = None
        self._segment_index = 0
        self.segments_emitted = 0

        # Состояние стримингового ресемплинга в 16 кГц для VAD.
        self._vad_buf = np.zeros(0, dtype=np.float32)
        self._raw_carry = np.zeros(0, dtype=np.float32)
        self._last_vad = False

        if device_id is None:
            print("⚠️ AudioPipeline: device_id не указан.")
        else:
            print(f"Загрузка Silero VAD для устройства {device_id}...")
            try:
                from silero_vad import load_silero_vad
                self._vad_model = load_silero_vad(onnx=True)
                print("Silero VAD загружен.")
            except Exception as e:
                print(f"❌ Ошибка загрузки VAD: {e}")

    def set_asr_model(self, model):
        self.asr_model = model

    # ------------------------------------------------------------------ startup
    def start(self):
        if self.device_id is None:
            print("❌ Невозможно запустить запись: device_id = None.")
            return False
        if self.running:
            print("⚠️ Запись уже запущена.")
            return True
        if self._vad_model is None:
            print("❌ VAD модель не загружена.")
            return False

        try:
            dev_info = sd.query_devices(self.device_id)
            if dev_info['max_input_channels'] == 0:
                print(f"❌ Устройство {self.device_id} не имеет входных каналов.")
                return False
            rates = [16000, 48000, 44100, 22050, 8000, 11025]
            found = False
            for rate in rates:
                for channels in [1, 2]:
                    if channels > dev_info['max_input_channels']:
                        continue
                    try:
                        sd.check_input_settings(device=self.device_id, samplerate=rate, channels=channels)
                        self._actual_sample_rate = rate
                        self.channels = channels
                        found = True
                        break
                    except Exception:
                        continue
                if found:
                    break
            if not found:
                default_rate = int(dev_info['default_samplerate'])
                default_channels = min(dev_info['max_input_channels'], 2)
                self._actual_sample_rate = default_rate
                self.channels = default_channels
                print(f"⚠️ Использую дефолтные настройки: частота {default_rate} Гц, каналов {default_channels}")
            else:
                print(f"✅ Используется частота {self._actual_sample_rate} Гц, каналов {self.channels} для устройства {self.device_id}")
        except Exception as e:
            print(f"❌ Ошибка получения информации об устройстве {self.device_id}: {e}")
            return False

        # Сегментатор и шумодав создаются под фактическую частоту потока.
        self._segmenter = SpeechSegmenter(
            sample_rate=self._actual_sample_rate,
            merge_pause_ms=self.merge_pause_ms,
            max_segment_ms=self.max_segment_ms,
            chunk_overlap_ms=self.chunk_overlap_ms,
            min_speech_ms=self.min_speech_ms,
            min_audio_length_sec=self.min_audio_length_sec,
        )
        if self.denoise:
            self._denoiser = NoiseReducer(
                sample_rate=self.target_sample_rate,
                backend=self.noise_backend,
                enabled=True,
            )
            print(f"🎛️ Шумодав: {self._denoiser.active_backend}")

        self._vad_buf = np.zeros(0, dtype=np.float32)
        self._raw_carry = np.zeros(0, dtype=np.float32)

        time.sleep(0.5)
        self.running = True
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        print(f"✅ Запись с устройства {self.device_id} запущена (частота {self._actual_sample_rate} Гц, каналов {self.channels})")
        return True

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        # Финальный флеш: последняя фраза не должна потеряться.
        if self._segmenter is not None:
            segment = self._segmenter.flush()
            if segment is not None:
                segment.final = True
                self._emit_segment(segment)
        print(f"⏹️ Запись с устройства {self.device_id} остановлена")

    # --------------------------------------------------------------- resampling
    def _to_16k(self, block: np.ndarray) -> np.ndarray:
        """Точный потоковый ресемплинг блока в 16 кГц (без накопления дрейфа)."""
        rate = int(self._actual_sample_rate or TARGET_RATE)
        if rate == TARGET_RATE:
            return block.astype(np.float32, copy=False)

        g = math.gcd(TARGET_RATE, rate)
        up, down = TARGET_RATE // g, rate // g
        k = max(1, math.ceil(VAD_CHUNK / up))
        chunk_native = down * k

        self._raw_carry = np.concatenate([self._raw_carry, block.astype(np.float32, copy=False)])
        out = []
        while len(self._raw_carry) >= chunk_native:
            piece = self._raw_carry[:chunk_native]
            self._raw_carry = self._raw_carry[chunk_native:]
            out.append(resample_poly(piece, up, down).astype(np.float32))
        if not out:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(out)

    def _vad_decision(self, raw_block: np.ndarray) -> bool:
        """VAD всегда включён: вход ресемплируется в 16 кГц, а не отключается."""
        if self._vad_model is None:
            return False
        samples16 = self._to_16k(raw_block)
        if len(samples16):
            self._vad_buf = np.concatenate([self._vad_buf, samples16])
        probs = []
        while len(self._vad_buf) >= VAD_CHUNK:
            chunk = self._vad_buf[:VAD_CHUNK]
            self._vad_buf = self._vad_buf[VAD_CHUNK:]
            prob = self._vad_model(torch.from_numpy(chunk), TARGET_RATE).item()
            probs.append(prob)
        if probs:
            self._last_vad = max(probs) > self.vad_threshold
        return self._last_vad

    # ---------------------------------------------------------------- recording
    def _record_loop(self):
        block_size = VAD_CHUNK

        def callback(indata, frames, time_info, status):
            if status:
                print(f"⚠️ Статус записи: {status}")
            raw = indata[:, 0].astype(np.float32, copy=True)
            is_speech = self._vad_decision(raw)
            segment = self._segmenter.feed(raw, is_speech) if self._segmenter else None
            if segment is not None:
                self._emit_segment(segment)

        try:
            with sd.InputStream(device=self.device_id, samplerate=self._actual_sample_rate,
                                channels=self.channels, blocksize=block_size, callback=callback):
                while self.running:
                    sd.sleep(10)
        except Exception as e:
            print(f"❌ Ошибка в потоке записи: {e}")
            self.running = False

    def _emit_segment(self, segment: Segment):
        audio = segment.audio
        if segment.sample_rate != self.target_sample_rate:
            audio = librosa.resample(
                audio, orig_sr=segment.sample_rate, target_sr=self.target_sample_rate
            )
        audio = np.asarray(audio, dtype=np.float32)

        if self._denoiser is not None and self._denoiser.is_active:
            audio = self._denoiser.process(audio)

        duration = len(audio) / self.target_sample_rate
        self._segment_index += 1
        self.segments_emitted += 1
        temp_file = f"temp_{self.device_id}_{int(time.time() * 1000)}_{self._segment_index}.wav"
        try:
            sf.write(temp_file, audio, self.target_sample_rate, subtype="PCM_16")
            task = AudioSegmentTask(
                path=temp_file,
                duration=duration,
                forced=segment.forced,
                final=segment.final,
                index=self._segment_index,
            )
            self.processing_queue.put(task)
            print(
                f"📁 Сегмент #{self._segment_index}: {duration:.2f} с "
                f"(forced={segment.forced}, final={segment.final}) -> {temp_file}"
            )
        except Exception as e:
            print(f"❌ Ошибка сохранения аудио: {e}")

    # ---------------------------------------------------------------------- ASR
    def _call_asr(self, audio_path: str, lang: Optional[str] = None) -> str:
        if self.asr_model is None:
            raise ValueError("ASR модель не загружена")
        if not hasattr(self.asr_model, 'transcribe'):
            raise AttributeError("Модель не имеет метода transcribe")

        import inspect

        sig = inspect.signature(self.asr_model.transcribe)
        params = sig.parameters

        if 'language' in params and lang is not None:
            return self.asr_model.transcribe(audio_path, language=lang)
        return self.asr_model.transcribe(audio_path)

    def transcribe_file(self, audio_path: str) -> str:
        if self.asr_model is None:
            print("⚠️ Модель ASR не установлена. Невозможно распознать.")
            return ""

        with self._asr_lock:
            try:
                info = sf.info(audio_path)
                if info.duration > 25.0:
                    print(f"⏳ Файл длиной {info.duration:.1f}с, используем transcribe_longform()...")
                    if hasattr(self.asr_model, 'transcribe_longform'):
                        result = self.asr_model.transcribe_longform(audio_path)
                        if isinstance(result, list):
                            return " ".join([item.get('text', '') for item in result])
                        if isinstance(result, str):
                            return result
                        return str(result)
                    print("⚠️ longform не доступен, разбиваем на чанки вручную")
                    return self._transcribe_long(audio_path, chunk_duration=20.0, overlap=2.0)

                data, sr = sf.read(audio_path)
                if sr != self.target_sample_rate:
                    data = librosa.resample(data, orig_sr=sr, target_sr=self.target_sample_rate)
                    temp_resampled = f"temp_resampled_{int(time.time())}.wav"
                    sf.write(temp_resampled, data, self.target_sample_rate)
                    result = self._call_asr(temp_resampled, self.asr_lang)
                    os.remove(temp_resampled)
                else:
                    result = self._call_asr(audio_path, self.asr_lang)

                return self._extract_text(result)
            except Exception as e:
                print(f"❌ Ошибка распознавания: {e}")
                return ""

    def _transcribe_long(self, audio_path: str, chunk_duration: float, overlap: float) -> str:
        try:
            data, sr = sf.read(audio_path)
        except Exception as e:
            print(f"❌ Ошибка чтения файла {audio_path}: {e}")
            return ""

        if sr != self.target_sample_rate:
            data = librosa.resample(data, orig_sr=sr, target_sr=self.target_sample_rate)
            sr = self.target_sample_rate

        chunk_samples = int(chunk_duration * sr)
        overlap_samples = int(overlap * sr)
        step = chunk_samples - overlap_samples
        total = len(data)
        texts = []
        for start in range(0, total, step):
            end = min(start + chunk_samples, total)
            chunk = data[start:end]
            if len(chunk) < 0.5 * sr:
                continue
            temp_chunk = f"temp_chunk_{start}_{int(time.time())}.wav"
            try:
                sf.write(temp_chunk, chunk, sr)
                result = self._call_asr(temp_chunk, self.asr_lang)
                text = self._extract_text(result)
                if text:
                    texts.append(text)
            except Exception as e:
                print(f"⚠️ Ошибка распознавания чанка {start}: {e}")
            finally:
                if os.path.exists(temp_chunk):
                    os.remove(temp_chunk)
        return " ".join(texts)

    @staticmethod
    def _extract_text(result):
        if result is None:
            return ""
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, tuple):
            return result[0].strip() if result else ""
        if hasattr(result, 'text'):
            return result.text.strip()
        if isinstance(result, dict) and 'text' in result:
            return result['text'].strip()
        return str(result).strip()

    @staticmethod
    def test_microphone(device_id, duration=2):
        try:
            dev_info = sd.query_devices(device_id)
            if dev_info['max_input_channels'] == 0:
                return False, "Устройство не имеет входных каналов"
            rates = [16000, 8000, 44100, 48000, 22050, 11025]
            working_rate = None
            channels = 1
            for rate in rates:
                for ch in [1, 2]:
                    if ch > dev_info['max_input_channels']:
                        continue
                    try:
                        sd.check_input_settings(device=device_id, samplerate=rate, channels=ch)
                        working_rate = rate
                        channels = ch
                        break
                    except Exception:
                        continue
                if working_rate is not None:
                    break
            if working_rate is None:
                working_rate = int(dev_info['default_samplerate'])
                channels = min(dev_info['max_input_channels'], 2)
                print(f"⚠️ Использую дефолтные: частота {working_rate} Гц, каналов {channels}")

            print(f"🔴 Запись тестового аудио с устройства {device_id} (частота {working_rate} Гц, каналов {channels})...")
            recording = sd.rec(int(duration * working_rate), samplerate=working_rate,
                               channels=channels, device=device_id, blocking=True)
            temp_file = f"test_mic_{device_id}.wav"
            sf.write(temp_file, recording, working_rate)
            print(f"✅ Тестовая запись сохранена в {temp_file}")
            return True, temp_file
        except Exception as e:
            return False, str(e)
