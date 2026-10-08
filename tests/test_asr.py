import pytest
import os
import numpy as np
from src.model_manager import ModelManager
from src.audio_pipeline import AudioPipeline

# Use the same path as the project
MODELS_ROOT = os.path.join(os.getcwd(), "models")

@pytest.fixture
def model_manager():
    return ModelManager(models_root=MODELS_ROOT)

@pytest.fixture
def audio_pipeline():
    # Using a dummy device_id or a real one if possible. 
    # For testing purposes, we might need to handle the device error.
    # However, we'll try to initialize it.
    return AudioPipeline(device_id=None)

def test_asr_model_loading(model_manager):
    """Test if ASR model can be retrieved from ModelManager."""
    # This might attempt to download if not present, which is slow.
    # But we want to test the manager logic.
    try:
        asr_model = model_manager.get_asr_model()
        assert asr_model is not None
    except Exception as e:
        pytest.fail(f"ASR model loading failed: {e}")
    finally:
        model_manager.unload_asr()

def test_audio_pipeline_initialization():
    """Test AudioPipeline initialization."""
    pipeline = AudioPipeline(device_id=None)
    assert pipeline.device_id is None
    assert pipeline.asr_model is None

def test_audio_pipeline_transcribe_file(model_manager, tmp_path):
    """Test transcription of a file using AudioPipeline."""
    # 1. Create a dummy wav file
    audio_file = tmp_path / "test_audio.wav"
    fs = 16000
    duration = 2
    t = np.linspace(0, duration, fs * duration, False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    import soundfile as sf
    sf.write(str(audio_file), audio, fs)

    # 2. Setup pipeline with a mocked or real ASR model
    # Since we want to test the pipeline's logic, we'll mock the ASR model's transcribe method
    class MockASR:
        def transcribe(self, audio_path, language=None):
            return "test transcription"
        def __enter__(self): return self
        def __exit__(self, *args): pass

    mock_asr = MockASR()
    pipeline = AudioPipeline(device_id=None, asr_model=mock_asr)
    
    # 3. Run transcription
    result = pipeline.transcribe_file(str(audio_file))
    assert result == "test transcription"

def test_audio_pipeline_longform_fallback(model_manager, tmp_path):
    """Test the fallback to longform if transcribe is not available or file is long."""
    audio_file = tmp_path / "long_audio.wav"
    fs = 16000
    duration = 30  # > 25 с — включается ветка transcribe_longform()
    t = np.linspace(0, duration, fs * duration, False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    import soundfile as sf
    sf.write(str(audio_file), audio, fs)

    class MockASRLong:
        def transcribe(self, audio_path, language=None):
            return "short result"
        def transcribe_longform(self, audio_path):
            return [{"text": "longform result"}]
    
    mock_asr = MockASRLong()
    pipeline = AudioPipeline(device_id=None, asr_model=mock_asr)
    
    result = pipeline.transcribe_file(str(audio_file))
    assert result == "longform result"

if __name__ == "__main__":
    pytest.main([__file__])
