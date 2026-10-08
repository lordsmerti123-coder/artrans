import time

import numpy as np
import pytest

import src.audio_router as audio_router
from src.audio_router import AudioRouter


class FakeStream:
    def __init__(self, device, samplerate, channels, dtype):
        self.device = device
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.written = []

    def start(self):
        pass

    def write(self, data):
        self.written.append(np.asarray(data).copy())

    def stop(self):
        pass

    def close(self):
        pass


class FakeSD:
    def __init__(self, supported_rate=22050, failing_devices=()):
        self.supported_rate = supported_rate
        self.failing_devices = set(failing_devices)
        self.streams = []

    def query_devices(self, device=None):
        return {"default_samplerate": self.supported_rate}

    def check_output_settings(self, device=None, samplerate=None, channels=None):
        if device in self.failing_devices:
            raise RuntimeError(f"device {device} unavailable")
        if samplerate != self.supported_rate:
            raise RuntimeError(f"unsupported rate {samplerate}")

    def OutputStream(self, device=None, samplerate=None, channels=None, dtype=None):
        if device in self.failing_devices:
            raise RuntimeError(f"cannot open device {device}")
        stream = FakeStream(device, samplerate, channels, dtype)
        self.streams.append(stream)
        return stream


@pytest.fixture
def fake_sd(monkeypatch):
    fake = FakeSD()
    monkeypatch.setattr(audio_router, "sd", fake)
    return fake


def wait_for(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_routes_audio_and_reports_health(fake_sd):
    router = AudioRouter()
    audio = (np.ones(2205, dtype=np.float32) * 1000).astype(np.int16)
    assert router.route(audio, 22050, 1) is True

    assert wait_for(lambda: router.health().get(1, {}).get("played", 0) >= 1)
    assert fake_sd.streams[0].samplerate == 22050
    router.stop()


def test_channels_are_independent(fake_sd):
    router = AudioRouter()
    audio = np.ones(1000, dtype=np.int16)
    assert router.route(audio, 22050, 1) is True
    assert router.route(audio, 22050, 2) is True

    assert wait_for(lambda: router.health().get(1, {}).get("played", 0) >= 1)
    assert wait_for(lambda: router.health().get(2, {}).get("played", 0) >= 1)
    assert len(fake_sd.streams) == 2
    router.stop()


def test_resamples_when_device_rate_differs(monkeypatch):
    fake = FakeSD(supported_rate=22050)
    monkeypatch.setattr(audio_router, "sd", fake)
    router = AudioRouter()
    audio = np.ones(1600, dtype=np.int16)  # 0.1 c при 16 кГц
    assert router.route(audio, 16000, 1) is True
    assert wait_for(lambda: router.health().get(1, {}).get("played", 0) >= 1)
    assert fake.streams[0].samplerate == 22050
    assert len(fake.streams[0].written[0]) > 1600
    router.stop()


def test_unavailable_device_reports_error(monkeypatch):
    fake = FakeSD(failing_devices={99})
    monkeypatch.setattr(audio_router, "sd", fake)
    errors = []
    router = AudioRouter(on_error=errors.append)
    assert router.route(np.ones(100, dtype=np.int16), 22050, 99) is False
    assert router.dropped == 1
    assert wait_for(lambda: len(errors) >= 1)
    router.stop()


def test_none_device_is_dropped(fake_sd):
    router = AudioRouter()
    assert router.route(np.ones(100, dtype=np.int16), 22050, None) is False
    assert router.dropped == 1
    router.stop()
