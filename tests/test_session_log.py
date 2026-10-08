from src.session_log import Metrics, SessionLog


def test_metrics_summary_after_observations():
    metrics = Metrics()
    metrics.inc("segments", 2)
    metrics.observe("stt_ms", 100)
    metrics.observe("stt_ms", 300)

    summary = metrics.summary()
    assert "segments=2" in summary
    assert "stt_ms" in summary
    assert metrics.stats()["stt_ms"]["count"] == 2
    assert metrics.stats()["stt_ms"]["max_ms"] == 300


def test_metrics_summary_empty():
    assert Metrics().summary() == "нет данных"


def test_session_log_disabled_has_no_file():
    log = SessionLog(enabled=False)
    assert log.path is None
    log.info("не пишется")
    assert log.flush_summary() == "нет данных"
    log.close()


def test_session_log_writes_file_and_masks_text(tmp_path):
    messages = []
    log = SessionLog(base_dir=tmp_path, enabled=True, log_text=False, on_message=messages.append)
    log.metrics.inc("segments_played")
    log.text_event("recognized", "секретный текст", channel="ru")
    log.close()

    assert log.path is not None and log.path.exists()
    content = log.path.read_text(encoding="utf-8")
    assert "segments_played=1" in content
    assert "секретный текст" not in content
    assert "event:recognized" in content
    assert any("segments_played=1" in m for m in messages)
