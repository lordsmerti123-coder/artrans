import numpy as np

from src.segmentation import SpeechSegmenter, dedupe_overlap


SR = 16000
BLOCK_MS = 100
BLOCK = int(SR * BLOCK_MS / 1000)


def block(value: float = 0.1) -> np.ndarray:
    return np.full(BLOCK, value, dtype=np.float32)


def collect(segmenter, sequence):
    """sequence — список (is_speech, count) блоков; возвращает список сегментов."""
    out = []
    for is_speech, count in sequence:
        for _ in range(count):
            seg = segmenter.feed(block(), is_speech)
            if seg is not None:
                out.append(seg)
    return out


def test_silence_never_creates_segment():
    seg = SpeechSegmenter(SR)
    assert collect(seg, [(False, 30)]) == []
    assert seg.flush() is None


def test_short_pause_merges_into_one_reply():
    seg = SpeechSegmenter(SR, merge_pause_ms=2500, min_speech_ms=500)
    segments = collect(seg, [(True, 6), (False, 10), (True, 6), (False, 30)])
    assert len(segments) == 1
    assert segments[0].forced is False
    assert segments[0].duration_ms > 2000


def test_long_pause_splits_replies():
    seg = SpeechSegmenter(SR, merge_pause_ms=2500, min_speech_ms=500)
    segments = collect(seg, [(True, 6), (False, 30), (True, 6), (False, 30)])
    assert len(segments) == 2
    assert all(s.forced is False for s in segments)


def test_max_segment_hard_split_with_overlap():
    overlap_ms = 300
    seg = SpeechSegmenter(
        SR, merge_pause_ms=10000, max_segment_ms=1000, chunk_overlap_ms=overlap_ms,
        min_speech_ms=100,
    )
    segments = collect(seg, [(True, 25)])
    assert len(segments) >= 2
    assert segments[0].forced is True
    overlap_n = int(overlap_ms / 1000 * SR)
    assert np.allclose(segments[1].audio[:overlap_n], segments[0].audio[-overlap_n:])


def test_flush_returns_last_reply():
    seg = SpeechSegmenter(SR, min_speech_ms=300)
    assert collect(seg, [(True, 6)]) == []
    final = seg.flush()
    assert final is not None
    assert final.final is True
    assert final.duration_ms >= 500


def test_too_short_speech_is_dropped():
    seg = SpeechSegmenter(SR, min_speech_ms=500, min_audio_length_sec=0.4)
    assert collect(seg, [(True, 1), (False, 30)]) == []


def test_dedupe_overlap_words_and_chars():
    assert dedupe_overlap("hello world how are you", "how are you today") == "today"
    assert dedupe_overlap("hello wor", "world") == "ld"
    assert dedupe_overlap("", "text") == "text"
    assert dedupe_overlap("text", "other") == "other"
