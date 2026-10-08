from src.language_gate import LanguageGate


def make_gate():
    return LanguageGate(languages=["ru", "en"])


def test_empty_and_short_text():
    gate = make_gate()
    assert gate.accept("", "ru") is False
    assert gate.accept("  ", "ru") is False
    assert gate.accept("да", "ru") is True  # короче min_chars — не режем


def test_ru_channel_rejects_foreign_scripts():
    gate = make_gate()
    assert gate.accept("你好，今天天气很好", "ru") is False
    assert gate.accept("Բարեւ Ձեզ, ինչպես եք", "ru") is False


def test_en_channel_rejects_cyrillic():
    gate = make_gate()
    assert gate.accept("Привет, как дела сегодня", "en") is False


def test_valid_text_accepted():
    gate = make_gate()
    assert gate.accept("Hello, how are you today?", "en") is True
    assert gate.accept("Привет, как твои дела?", "ru") is True


def test_unknown_language_script_is_accepted():
    gate = make_gate()
    # zh вне языков детектора, но скрипт совместим со своим каналом
    assert gate.accept("你好世界", "zh") is True


def test_disabled_gate_accepts_everything():
    gate = LanguageGate(languages=["ru", "en"], enabled=False)
    assert gate.accept("你好世界", "ru") is True


def test_check_reports_reason():
    gate = make_gate()
    result = gate.check("你好世界", "ru")
    assert result.accepted is False
    assert result.reason == "script_mismatch"
    assert result.script == "cjk"
