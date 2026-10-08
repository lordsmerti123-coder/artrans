# scripts/bt_diagnose.py
"""
Диагностика аудиоустройств для двустороннего переводчика (Phase 0.1).

Собирает:
- список устройств sounddevice и PyAudio (index, name, host API, каналы, дефолтная частота);
- пометку Bluetooth и host API (основной контроллер vs USB-донгл);
- проверку check_input_settings / check_output_settings на 8/16/44.1/48 кГц;
- при --hardware-tests: тестовый тон 440 Гц и запись 2 с (RMS, клиппинг).

Отчёт пишется в logs/audio_diagnose_<timestamp>.json и .txt. Системные настройки
не изменяются.

Запуск:
    venv\\Scripts\\python.exe scripts\\bt_diagnose.py
    venv\\Scripts\\python.exe scripts\\bt_diagnose.py --hardware-tests
    venv\\Scripts\\python.exe scripts\\bt_diagnose.py --json-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.audio_devices import is_bluetooth_device  # noqa: E402

RATES = [8000, 16000, 22050, 44100, 48000]
LOGS_DIR = ROOT / "logs"


def _hostapi_names(sd) -> dict:
    try:
        return {i: api.get("name", f"hostapi{i}") for i, api in enumerate(sd.query_hostapis())}
    except Exception:
        return {}


def collect_sounddevice(sd) -> list:
    devices = []
    hostapis = _hostapi_names(sd)
    try:
        raw = sd.query_devices()
    except Exception as exc:
        return [{"error": str(exc)}]

    for idx, dev in enumerate(raw):
        hostapi_idx = dev.get("hostapi")
        devices.append({
            "index": idx,
            "name": dev.get("name"),
            "hostapi_index": hostapi_idx,
            "hostapi": hostapis.get(hostapi_idx, "unknown"),
            "max_input_channels": int(dev.get("max_input_channels", 0)),
            "max_output_channels": int(dev.get("max_output_channels", 0)),
            "default_samplerate": dev.get("default_samplerate"),
            "is_bluetooth": bool(is_bluetooth_device(str(dev.get("name", "")))),
        })
    return devices


def collect_pyaudio() -> list:
    try:
        import pyaudio
    except Exception as exc:
        return [{"error": f"pyaudio недоступен: {exc}"}]

    p = pyaudio.PyAudio()
    devices = []
    try:
        hostapis = {}
        for i in range(p.get_host_api_count()):
            info = p.get_host_api_info_by_index(i)
            hostapis[i] = info.get("name", f"hostapi{i}")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            devices.append({
                "index": i,
                "name": info.get("name"),
                "hostapi_index": info.get("hostApi"),
                "hostapi": hostapis.get(info.get("hostApi"), "unknown"),
                "max_input_channels": int(info.get("maxInputChannels", 0)),
                "max_output_channels": int(info.get("maxOutputChannels", 0)),
                "default_sample_rate": info.get("defaultSampleRate"),
                "is_bluetooth": bool(is_bluetooth_device(str(info.get("name", "")))),
            })
    finally:
        p.terminate()
    return devices


def check_settings(sd, device_index: int, kind: str) -> dict:
    result = {}
    checker = sd.check_input_settings if kind == "input" else sd.check_output_settings
    for rate in RATES:
        key = f"{rate}"
        try:
            checker(device=device_index, samplerate=rate, channels=1)
            result[key] = "ok"
        except Exception as exc:
            result[key] = f"fail: {exc}"
    return result


def tone_test(sd, device_index: int) -> dict:
    outcome = {"ok": False, "rate": None, "error": None}
    for rate in RATES:
        try:
            sd.check_output_settings(device=device_index, samplerate=rate, channels=1)
        except Exception:
            continue
        try:
            duration = 0.5
            t = np.linspace(0, duration, int(rate * duration), endpoint=False)
            wave = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
            stream = sd.OutputStream(device=device_index, samplerate=rate, channels=1, dtype="int16")
            stream.start()
            stream.write(wave.reshape(-1, 1))
            stream.stop()
            stream.close()
            outcome.update(ok=True, rate=rate)
            return outcome
        except Exception as exc:
            outcome["error"] = str(exc)
    if outcome["error"] is None:
        outcome["error"] = "нет поддерживаемой частоты"
    return outcome


def mic_test(sd, device_index: int, duration: float = 2.0) -> dict:
    outcome = {"ok": False, "rate": None, "rms": None, "peak": None, "error": None}
    for rate in RATES:
        try:
            sd.check_input_settings(device=device_index, samplerate=rate, channels=1)
        except Exception:
            continue
        try:
            recording = sd.rec(int(duration * rate), samplerate=rate, channels=1,
                               device=device_index, blocking=True)
            data = np.asarray(recording, dtype=np.float32).reshape(-1)
            outcome.update(
                ok=True,
                rate=rate,
                rms=float(np.sqrt(np.mean(data ** 2))) if len(data) else 0.0,
                peak=float(np.max(np.abs(data))) if len(data) else 0.0,
            )
            return outcome
        except Exception as exc:
            outcome["error"] = str(exc)
    if outcome["error"] is None:
        outcome["error"] = "нет поддерживаемой частоты"
    return outcome


def build_report(hardware_tests: bool) -> dict:
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hardware_tests": hardware_tests,
        "sounddevice": None,
        "pyaudio": None,
    }

    try:
        import sounddevice as sd
    except Exception as exc:
        report["sounddevice_error"] = str(exc)
        return report

    report["default_input"] = sd.default.device[0] if sd.default.device else None
    report["default_output"] = sd.default.device[1] if sd.default.device else None
    devices = collect_sounddevice(sd)
    report["sounddevice"] = devices
    report["pyaudio"] = collect_pyaudio()

    checks = []
    for dev in devices:
        if "error" in dev:
            continue
        entry = {"index": dev["index"], "name": dev["name"]}
        if dev["max_input_channels"] > 0:
            entry["input_rates"] = check_settings(sd, dev["index"], "input")
            if hardware_tests:
                entry["mic_test"] = mic_test(sd, dev["index"])
        if dev["max_output_channels"] > 0:
            entry["output_rates"] = check_settings(sd, dev["index"], "output")
            if hardware_tests:
                entry["tone_test"] = tone_test(sd, dev["index"])
        checks.append(entry)
    report["checks"] = checks

    hostapis = sorted({d.get("hostapi") for d in devices if d.get("hostapi")})
    report["hostapis"] = hostapis
    report["bluetooth_devices"] = [d["name"] for d in devices if d.get("is_bluetooth")]
    report["summary"] = {
        "device_count": len(devices),
        "hostapi_count": len(hostapis),
        "bluetooth_count": len(report["bluetooth_devices"]),
        "separate_controllers": len(hostapis) > 1,
    }
    return report


def render_text(report: dict) -> str:
    lines = [f"Аудио-диагностика: {report['timestamp']}", "=" * 60]
    s = report.get("summary", {})
    lines.append(
        f"Устройств: {s.get('device_count')}, host API: {s.get('hostapi_count')}, "
        f"Bluetooth: {s.get('bluetooth_count')}, независимые контроллеры: {s.get('separate_controllers')}"
    )
    lines.append("")
    lines.append("Устройства sounddevice:")
    for dev in report.get("sounddevice") or []:
        if "error" in dev:
            lines.append(f"  ERROR: {dev['error']}")
            continue
        bt = " [BT]" if dev.get("is_bluetooth") else ""
        lines.append(
            f"  [{dev['index']}] {dev['name']}{bt} | hostapi={dev['hostapi']} | "
            f"in={dev['max_input_channels']} out={dev['max_output_channels']} | "
            f"default={dev['default_samplerate']}"
        )
    lines.append("")
    lines.append("Проверка частот (sounddevice, mono):")
    for entry in report.get("checks", []):
        parts = [f"  [{entry['index']}] {entry['name']}"]
        if "input_rates" in entry:
            parts.append("    IN : " + ", ".join(f"{r}:{v}" for r, v in entry["input_rates"].items()))
        if "output_rates" in entry:
            parts.append("    OUT: " + ", ".join(f"{r}:{v}" for r, v in entry["output_rates"].items()))
        if "mic_test" in entry:
            mt = entry["mic_test"]
            parts.append(f"    MIC: ok={mt['ok']} rate={mt['rate']} rms={mt['rms']} peak={mt['peak']} {mt.get('error') or ''}")
        if "tone_test" in entry:
            tt = entry["tone_test"]
            parts.append(f"    TONE: ok={tt['ok']} rate={tt['rate']} {tt.get('error') or ''}")
        lines.extend(parts)
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Диагностика аудиоустройств BabelDuo")
    parser.add_argument("--hardware-tests", action="store_true",
                        help="выполнить реальный тестовый тон и запись 2 с")
    parser.add_argument("--json-only", action="store_true", help="печатать только JSON")
    args = parser.parse_args(argv)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = build_report(args.hardware_tests)

    json_path = LOGS_DIR / f"audio_diagnose_{stamp}.json"
    txt_path = LOGS_DIR / f"audio_diagnose_{stamp}.txt"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    text = render_text(report)
    txt_path.write_text(text, encoding="utf-8")

    if args.json_only:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(text)
        print()
        print(f"Отчёт: {json_path}")
        print(f"Отчёт: {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
