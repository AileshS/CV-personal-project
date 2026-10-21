"""Diagnostic fixtures test preservation and UI; they make no accuracy claim."""
import numpy as np
import pytest

from playlab.detection_inspector import inspect_frame, overlay, sampled_detections
from playlab.store import Store


def test_ball_candidates_survive_missing_rim_in_saved_chunks(tmp_path, monkeypatch):
    from playlab import analysis
    from tests.test_analysis import fixture_store

    store = fixture_store(tmp_path)
    monkeypatch.setattr(analysis, "iter_frames", lambda *a, **k: iter([(1., np.zeros((100, 160, 3), dtype=np.uint8))]))
    class Detector:
        mode = "pretrained_coco_ball"
        def __init__(self, *a):
            pass
        def reset(self):
            pass
        def __call__(self, frame):
            return {"balls": [{"bbox": [1, 2, 5, 6], "confidence": .42}], "rims": [], "players": []}
    monkeypatch.setattr(analysis, "BasketballDetector", Detector)
    result = analysis.analyze_possession(store, "p", mode="automatic")
    row = store.list("analysis_chunk")[0]["frames"][0]
    assert result["observation_count"] == 0
    assert row["raw_detections"]["balls"][0]["confidence"] == .42
    assert (row["width"], row["height"]) == (160, 100)


def test_overlay_native_scaling_and_hint_has_no_detector_score():
    row = {"width": 100, "height": 50, "raw_detections": {
        "balls": [{"bbox": [10, 5, 20, 10], "confidence": .73}], "rims": []}}
    image, table = overlay(np.zeros((100, 200, 3), dtype=np.uint8), row, [.7, .2, .8, .3])
    assert image.size == (200, 100)
    assert table == [{"Object": "Ball", "Detector score": .73,
                      "Native box (left, top, right, bottom)": [20., 10., 40., 20.]}]


def test_single_frame_inference_bounds_and_gold_isolation(monkeypatch):
    import playlab.detection_inspector as diagnostic
    calls = []
    monkeypatch.setattr(diagnostic, "frame_at", lambda *a, **k: calls.append(a[1]) or np.zeros((50,100,3), dtype=np.uint8))
    class Detector:
        mode = "fixture"
        def reset(self):
            pass
        def __call__(self, frame):
            return {"balls": [], "rims": []}
    game = {"source_path": "unused", "source_offset_seconds": 750}
    clip = {"start": 0., "end": 20.}
    row = inspect_frame(game, clip, 1/30, Detector())
    assert row["time"] == 1/30 and calls == [1/30]
    for at in (-1, 20, float("nan")):
        with pytest.raises(ValueError):
            inspect_frame(game, clip, at, Detector())
    game["split"] = "test"
    with pytest.raises(ValueError, match="locked test"):
        inspect_frame(game, clip, 1, Detector())
    assert calls == [1/30]


def test_saved_samples_have_explicit_timestamps(tmp_path):
    store = Store(tmp_path)
    store.save("quick_analysis_chunk", {"id": "c:f:0", "frames": [
        {"time": 1.2, "raw_detections": {"balls": [], "rims": []}}, {"time": 1.3}]})
    assert [r["time"] for r in sampled_detections(store, {"id": "c"}, {"chunks": 1, "fingerprint": "f"})] == [1.2]
