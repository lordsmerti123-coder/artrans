# src/nmt.py
from llama_cpp import Llama
import queue
import threading
import time
from typing import Optional, Dict, Callable

class Translator:
    """
    Перевод через GGUF модель с очередью запросов.
    Документация llama-cpp-python: https://llama-cpp-python.readthedocs.io/
    """
    def __init__(self, 
                 model_path: str,
                 device: str = "cuda",
                 n_gpu_layers: int = -1,
                 n_ctx: int = 2048):
        
        self.device = device
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        
        # Загрузка модели
        print(f"Загрузка модели перевода: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers if device == "cuda" else 0,
            n_ctx=n_ctx,
            flash_attn=True,
            verbose=False
        )
        print("Модель перевода загружена.")
        
        # Очередь запросов
        self._queue = queue.Queue()
        self._results: Dict[int, queue.Queue] = {}
        self._counter = 0
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        """Рабочий поток — обрабатывает запросы последовательно"""
        while self._running:
            try:
                req_id, text, src_lang, tgt_lang, result_queue = self._queue.get(timeout=0.5)
                if req_id is None:
                    break
                try:
                    result = self._translate_text(text, src_lang, tgt_lang)
                    result_queue.put(result)
                except Exception as e:
                    result_queue.put(f"[Ошибка перевода: {e}]")
            except queue.Empty:
                continue

    def _translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Собственно перевод"""
        if not text or not text.strip():
            return ""

        lang_map = {
            'ru': 'Russian', 'en': 'English', 'fr': 'French',
            'es': 'Spanish', 'zh': 'Chinese', 'hy': 'Armenian'
        }
        src = lang_map.get(source_lang, source_lang)
        tgt = lang_map.get(target_lang, target_lang)

        prompt = f"Translate the following text from {src} to {tgt}. Output only the translation, nothing else:\n\n{text}"

        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": f"You are a professional translator from {src} to {tgt}. Respond only with the translation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=512,
                stop=["\n\n", "###"]
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Ошибка перевода: {e}")
            return text

    def translate(self, text: str, source_lang: str, target_lang: str, timeout: float = 60.0) -> str:
        """Асинхронный перевод через очередь"""
        req_id = self._counter
        self._counter += 1
        result_queue = queue.Queue()
        self._queue.put((req_id, text, source_lang, target_lang, result_queue))
        
        try:
            return result_queue.get(timeout=timeout)
        except queue.Empty:
            return f"[Таймаут перевода: {text}]"

    def stop(self):
        """Остановка рабочего потока и освобождение ресурсов"""
        self._running = False
        self._queue.put((None, None, None, None, None))
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        # Дополнительно можно удалить llm, но он может освободиться при сборке мусора
        self.llm = None