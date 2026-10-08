import json
import os
import pyaudio
import sounddevice as sd
from pathlib import Path
from typing import Dict, List, Optional

CONFIG_FILE = str(Path(__file__).resolve().parent.parent / "config.json")

# Ключевые слова для определения Bluetooth-устройств
BLUETOOTH_KEYWORDS = [
    'bluetooth', 'hands-free', 'headset', 'a2dp', 'wireless',
    'bt', 'bth', 'voice', 'stereo', 'headphone', 'headset'
]

def is_bluetooth_device(name: str) -> bool:
    """Определяет, является ли устройство Bluetooth-гарнитурой по имени."""
    name_lower = name.lower()
    return any(kw in name_lower for kw in BLUETOOTH_KEYWORDS)

def get_bluetooth_devices(devices=None):
    """Возвращает список Bluetooth-устройств с входом и выходом."""
    if devices is None:
        devices = AudioDeviceManager.get_devices_sounddevice()
    return [dev for dev in devices if is_bluetooth_device(dev['name']) and
            dev['max_input_channels'] > 0 and dev['max_output_channels'] > 0]


class AudioDeviceManager:
    @staticmethod
    def get_devices_pyaudio() -> List[Dict]:
        """Получение списка устройств через PyAudio (для обратной совместимости)"""
        p = pyaudio.PyAudio()
        devices = []
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            devices.append({
                'index': i,
                'name': info['name'],
                'max_input_channels': int(info['maxInputChannels']),
                'max_output_channels': int(info['maxOutputChannels']),
                'default_sample_rate': int(info['defaultSampleRate'])
            })
        p.terminate()
        return devices

    @staticmethod
    def get_devices_sounddevice() -> List[Dict]:
        """Получение списка устройств через sounddevice (рекомендуемый способ)"""
        devices = sd.query_devices()
        result = []
        for dev in devices:
            # В sounddevice 0.4.5+ есть поле 'index'[reference:19]
            idx = dev.get('index', devices.index(dev))
            result.append({
                'index': idx,
                'name': dev['name'],
                'max_input_channels': dev['max_input_channels'],
                'max_output_channels': dev['max_output_channels'],
                'default_sample_rate': dev['default_samplerate']
            })
        return result

    @staticmethod
    def list_devices():
        """Вывод списка устройств в консоль"""
        devices = AudioDeviceManager.get_devices_sounddevice()
        print("Доступные аудиоустройства:")
        for dev in devices:
            print(f"  [{dev['index']}] {dev['name']} (вход: {dev['max_input_channels']}, выход: {dev['max_output_channels']})")
        return devices

    @staticmethod
    def load_config() -> Dict:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    @staticmethod
    def save_config(config: Dict):
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

    @staticmethod
    def get_default_devices() -> Dict:
        """Получение дефолтных устройств через sounddevice"""
        config = AudioDeviceManager.load_config()
        if config and 'devices' in config:
            return config['devices']

        try:
            default_input = sd.query_devices(kind='input')  # [reference:20]
            default_output = sd.query_devices(kind='output')
            input_idx = default_input.get('index', 0)
            output_idx = default_output.get('index', 0)
        except Exception:
            # fallback на PyAudio
            p = pyaudio.PyAudio()
            input_idx = p.get_default_input_device_info()['index']
            output_idx = p.get_default_output_device_info()['index']
            p.terminate()

        return {
            'pipeline_ru': {'input': input_idx, 'output': output_idx},
            'pipeline_en': {'input': input_idx, 'output': output_idx}
        }