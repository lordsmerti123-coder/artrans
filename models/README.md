# Распознавание речи — GigaAM (STT)
- https://github.com/salute-developers/GigaAM
- Модель `multilingual_ctc` скачивается автоматически при первом распознавании
  в папку `models/asr` (проверяется контрольная сумма). Файл:
  `models/asr/multilingual_ctc.ckpt` (~840 МБ).
- В `config.json` → `models.stt` можно указать другую модель GigaAM
  (`v3_ctc`, `v3_rnnt`, `multilingual_large_ctc` и т.д.) — она будет скачана
  при следующем запуске.

# Перевод — локальная LLM (NMT)
- Используется instruct-LLM в формате GGUF через llama-cpp-python
  (например, `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`).
- Путь задаётся в `config.json` → `models.translator_path`.
- Альтернатива: `models.translator_type = "nllb"` (модель
  `facebook/nllb-200-distilled-1.3B` скачается из интернета при первом запуске).

# Синтез речи — Piper (TTS)
- Голоса Piper: `models/tts/piper/{ru,en}/voice.onnx` + одноимённый `.json`.
- Взять голоса можно из репозиториев rhasspy/piper-voices:
  - ru: `ru_RU-irina-medium`
  - en: `en_US-lessac-medium`
- После скачивания положите файлы `voice.onnx` и `voice.onnx.json` в
  соответствующие папки.
