import pytest
import os
import numpy as np
from src.tts import TTS

# Path to the Russian voice found during exploration
RUSSIAN_VOICE = os.path.join("D:\\ARtrans\\models\\tts\\piper\\ru", "voice.onnx")
ENGLISH_VOICE = os.path.join("D:\\ARtrans\\models\\tts\\piper\\en", "voice.onnx")

def test_tts_initialization_ru():
    """Test if Russian TTS initializes correctly."""
    tts = TTS(RUSSIAN_VOICE)
    assert tts.voice is not None
    tts.unload()

def test_tts_initialization_en():
    """Test if English TTS initializes correctly."""
    tts = TTS(ENGLISH_VOICE)
    assert tts.voice is not None
    tts.unload()

def test_tts_synthesize_to_file():
    """Test synthesis to a file."""
    tts = TTS(RUSSIAN_VOICE)
    text = "Привет, это тест синтеза речи."
    output_file = "test_output_ru.wav"
    
    try:
        result_path = tts.synthesize(text, output_path=output_file)
        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 0
    finally:
        if os.path.exists(output_file):
            os.remove(output_file)
        tts.unload()

def test_tts_synthesize_to_bytes():
    """Test synthesis to bytes."""
    tts = TTS(ENGLISH_VOICE)
    text = "Hello, this is a test of speech synthesis."
    
    try:
        audio_bytes = tts.synthesize_to_bytes(text)
        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) > 0
    finally:
        tts.unload()

def test_tts_empty_text():
    """Test synthesis with empty text."""
    tts = TTS(RUSSIAN_VOICE)
    
    # Test synthesize (file)
    output_file = "test_empty.wav"
    try:
        result = tts.synthesize("", output_path=output_file)
        assert result == ""
        if os.path.exists(output_file):
            os.remove(output_file)
    finally:
        tts.unload()
        
    # Test synthesize_to_bytes
    audio_bytes = tts.synthesize_to_bytes("   ")
    assert audio_bytes == b''
    tts.unload()

if __name__ == "__main__":
    pytest.main([__file__])
