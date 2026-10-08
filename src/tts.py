# src/tts.py
import numpy as np
import soundfile as sf
from piper import PiperVoice
from typing import Optional
import io

class TTS:
    def __init__(self, voice_path: str, use_cuda: bool = False):
        print(f"Загрузка TTS голоса: {voice_path}")
        self.voice = PiperVoice.load(
            voice_path,
            config_path=voice_path + ".json",
            use_cuda=use_cuda
        )
        print("TTS голос загружен.")

    def synthesize(self, text: str, output_path: str = "output.wav") -> str:
        if not text or not text.strip():
            return ""

        audio_chunks = []
        for chunk in self.voice.synthesize(text):
            if hasattr(chunk, 'audio_int16_bytes'):
                audio_chunks.append(chunk.audio_int16_bytes)
            elif isinstance(chunk, bytes):
                audio_chunks.append(chunk)

        if not audio_chunks:
            print("Нет аудиоданных для синтеза")
            return ""

        audio_bytes = b''.join(audio_chunks)
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        sf.write(output_path, audio, 22050, subtype='PCM_16')
        print(f"Аудио сохранено в {output_path}")
        return output_path

    def synthesize_to_bytes(self, text: str) -> bytes:
        """Синтез речи и возврат в виде байтов (для стриминга)."""
        if not text or not text.strip():
            return b''

        audio_chunks = []
        for chunk in self.voice.synthesize(text):
            if hasattr(chunk, 'audio_int16_bytes'):
                audio_chunks.append(chunk.audio_int16_bytes)
            elif isinstance(chunk, bytes):
                audio_chunks.append(chunk)

        return b''.join(audio_chunks)

    def unload(self):
        self.voice = None