import unittest
import os
import torch
from src.nmt import Translator


MODEL_PATH = os.environ.get(
    "ARTRANS_TEST_GGUF",
    r"D:\models\lmstudio-community\Meta-Llama-3.1-8B-Instruct-GGUF\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
)

RUN_HEAVY = os.environ.get("ARTRANS_RUN_HEAVY_TESTS") == "1"


@unittest.skipUnless(RUN_HEAVY, "тяжёлые тесты MT: задайте ARTRANS_RUN_HEAVY_TESTS=1")
class TestNMT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = MODEL_PATH
        if not os.path.exists(cls.model_path):
            raise unittest.SkipTest(f"GGUF-модель не найдена: {cls.model_path}")

        print(f"\n[TEST] Loading model for testing: {cls.model_path}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.translator = Translator(model_path=cls.model_path, device=device)

    @classmethod
    def tearDownClass(cls):
        print("[TEST] Stopping translator...")
        cls.translator.stop()

    def test_translation_en_to_ru(self):
        print("[TEST] Running test: English to Russian")
        text = "Hello, how are you today?"
        result = self.translator.translate(text, "en", "ru")
        print(f"[TEST] Result: {result}")
        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "")
        # We don't assert exact string because LLM might vary slightly, 
        # but we check it's not empty and not the error message.
        self.assertNotIn("[Ошибка", result)

    def test_translation_ru_to_en(self):
        print("[TEST] Running test: Russian to English")
        text = "Привет, как твои дела?"
        result = self.translator.translate(text, "ru", "en")
        print(f"[TEST] Result: {result}")
        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "")
        self.assertNotIn("[Ошибка", result)

    def test_empty_text(self):
        print("[TEST] Running test: Empty text")
        result = self.translator.translate("", "en", "ru")
        self.assertEqual(result, "")

if __name__ == "__main__":
    unittest.main()
