import numpy as np

from src.noise import NoiseReducer, available_backends


def test_available_backends_shape():
    backends = available_backends()
    assert set(backends) == {"rnnoise", "noisereduce"}
    assert all(isinstance(v, bool) for v in backends.values())


def test_disabled_reducer_passthrough():
    audio = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
    reducer = NoiseReducer(sample_rate=16000, enabled=False)
    assert reducer.active_backend == "none"
    assert reducer.is_active is False
    out = reducer.process(audio)
    assert np.array_equal(out, audio)


def test_backend_none_passthrough():
    audio = np.ones(800, dtype=np.float32)
    reducer = NoiseReducer(sample_rate=16000, backend="none")
    assert reducer.active_backend == "none"
    assert np.array_equal(reducer.process(audio), audio)


def test_auto_backend_is_consistent_with_availability():
    reducer = NoiseReducer(sample_rate=16000, backend="auto")
    if reducer.active_backend == "rnnoise":
        assert available_backends()["rnnoise"]
    elif reducer.active_backend == "noisereduce":
        assert available_backends()["noisereduce"]
    else:
        assert not available_backends()["rnnoise"] and not available_backends()["noisereduce"]


def test_process_keeps_length_and_dtype_when_active():
    reducer = NoiseReducer(sample_rate=16000, backend="auto")
    if not reducer.is_active:
        return  # опциональные пакеты не установлены — проверять нечего
    audio = (np.sin(np.linspace(0, 40, 16000)) * 0.1).astype(np.float32)
    out = reducer.process(audio)
    assert len(out) == len(audio)
    assert out.dtype == np.float32
