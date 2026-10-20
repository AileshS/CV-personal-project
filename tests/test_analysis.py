"""Behavior tests use explicit fakes, never pretend to establish game accuracy."""
import numpy as np
import pytest

from playlab import analysis
from playlab.detectors import nearest_defender, team_for_player
from playlab.store import Store


class FakeDetector:
    mode = "test_fixture"
    calls = 0

    def __init__(self, *args):
        pass

    def reset(self):
        pass

    def __call__(self, frame):
        type(self).calls += 1
        return {"balls": [], "rims": [], "players": []}


def fixture_store(tmp_path, duration=5):
    store = Store(tmp_path / "workspace")
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fixture; decoding is replaced explicitly")
    store.save("game", {"id": "g", "source_path": str(source), "duration": duration,
                        "fps": 30, "width": 160, "height": 100})
    store.save("possession", {"id": "p", "game_id": "g", "start": 0, "end": duration,
                              "split": "test", "reviewed": True})
    return store


@pytest.fixture
def fake_video(monkeypatch):
    def iterator(path, start=0, end=5, **kwargs):
        # Number of frames intentionally has no relation to real game accuracy.
        for time in np.arange(start, end, .2):
            yield float(time), np.zeros((100, 160, 3), dtype=np.uint8)
    monkeypatch.setattr(analysis, "iter_frames", iterator)
    monkeypatch.setattr(analysis, "frame_at", lambda *a, **kw: np.zeros((100, 160, 3), dtype=np.uint8))
    monkeypatch.setattr(analysis, "BasketballDetector", FakeDetector)
    FakeDetector.calls = 0


def test_whole_possession_resumes_completed_chunks(tmp_path, fake_video):
    store = fixture_store(tmp_path, duration=45)
    cancelled = [False]
    def progress(fraction, checkpoint):
        if checkpoint.get("completed_chunks") == 1:
            cancelled[0] = True
    with pytest.raises(InterruptedError):
        analysis.analyze_possession(store, "p", progress=progress, cancelled=lambda: cancelled[0])
    saved = store.list("analysis_chunk")
    assert len(saved) == 1
    first_count = FakeDetector.calls
    cancelled[0] = False
    result = analysis.analyze_possession(store, "p")
    assert result["chunks"] == 3
    assert result["frames_processed"] >= 220  # complete 45-second span, no 600-frame truncation
    assert len(store.list("analysis_chunk")) == 3
    assert FakeDetector.calls - first_count < first_count * 1.5


def test_human_result_never_becomes_automatic_prediction(tmp_path, fake_video):
    store = fixture_store(tmp_path)
    store.save("shot", {"id": "s", "game_id": "g", "possession_id": "p",
                        "release": 1.0, "result": "made", "reviewed": True, "split": "test"})
    result = analysis.analyze_possession(store, "p")
    event = result["events"][0]
    assert event["result"] == "unknown"
    assert event["source"] == "prediction" and not event["reviewed"]
    assert event["defender"]["status"] == "unknown"
    assert event["evidence_path"].endswith(".png")
    before = FakeDetector.calls
    shot = store.get("shot", "s")
    store.save("shot", {**shot, "result": "missed"})
    again = analysis.analyze_possession(store, "p")
    assert FakeDetector.calls == before
    assert again["fingerprint"] == result["fingerprint"]
    assert again["events"][0]["result"] == "unknown"
    assert store.get("shot", "s")["result"] == "missed"


def test_corrected_release_invalidates_cache(tmp_path, fake_video):
    store = fixture_store(tmp_path)
    store.save("shot", {"id": "s", "possession_id": "p", "release": 1., "result": "unknown"})
    first = analysis.analyze_possession(store, "p")
    store.save("shot", {"id": "s", "possession_id": "p", "release": 2., "result": "unknown"})
    second = analysis.analyze_possession(store, "p")
    assert first["fingerprint"] != second["fingerprint"]
    assert second["events"][0]["release"] == 2.


def test_nearest_defender_excludes_teammates_and_reports_ties():
    players = [
        {"track_id": "s", "bbox": [0, 0, 20, 50], "team": "a"},
        {"track_id": "mate", "bbox": [1, 0, 21, 50], "team": "a"},
        {"track_id": "opp", "bbox": [30, 0, 50, 50], "team": "b"},
        {"track_id": "ref", "bbox": [0, 0, 20, 50], "team": None},
    ]
    result = nearest_defender(players, "s")
    assert result["track_id"] == "opp"
    assert result["distance_px"] == 30
    assert result["status"] == "approximate"
    players.append({"track_id": "tie", "bbox": [-30, 0, -10, 50], "team": "b"})
    assert nearest_defender(players, "s")["status"] == "unknown"
    assert nearest_defender(players, "missing")["status"] == "unknown"
    players[0]["track_id"] = None
    assert nearest_defender(players, None)["status"] == "unknown"


def test_teams_require_distinct_torso_color_evidence():
    frame = np.full((100, 100, 3), (230, 30, 30), dtype=np.uint8)
    assert team_for_player(frame, [0, 0, 100, 100], {"red": "#e61e1e", "blue": "#1414e6"}) == "red"
    frame[:] = 128
    assert team_for_player(frame, [0, 0, 100, 100], {"red": "#e61e1e", "blue": "#1414e6"}) is None


def test_isolated_ball_near_rim_is_not_a_shot():
    observations = [{"time": 1., "x": 50., "y": 30., "rim_x": 50., "rim_y": 30., "rim_width": 10.}]
    assert analysis.shot_proposals(observations) == []


def test_later_rebound_shot_does_not_enter_first_shot_evidence(tmp_path, fake_video, monkeypatch):
    store = fixture_store(tmp_path)
    for identifier, release in (("first", 1.), ("rebound", 2.)):
        store.save("shot", {"id": identifier, "possession_id": "p", "release": release,
                            "result": "unknown"})
    class FixedObjects(FakeDetector):
        def __call__(self, frame):
            return {"balls": [{"bbox": [10., 10., 14., 14.], "confidence": .9}],
                    "rims": [{"bbox": [20., 20., 30., 23.], "confidence": .9}], "players": []}
    windows = []
    original_features = analysis.compute_features
    def record_features(rows):
        windows.append([r["time"] for r in rows])
        return original_features(rows)
    monkeypatch.setattr(analysis, "BasketballDetector", FixedObjects)
    monkeypatch.setattr(analysis, "compute_features", record_features)
    result = analysis.analyze_possession(store, "p")
    assert len(result["events"]) == 2
    assert windows[0] and max(windows[0]) < 2.
    assert max(windows[1]) > 2.


def test_edit_during_background_analysis_marks_completed_output_stale(tmp_path, fake_video):
    store = fixture_store(tmp_path, duration=25)
    edited = [False]
    def progress(fraction, checkpoint):
        if checkpoint.get("completed_chunks") == 1 and not edited[0]:
            old = store.get("possession", "p")
            store.save("possession", {**old, "start": 1.})
            edited[0] = True
    result = analysis.analyze_possession(store, "p", progress=progress)
    assert result["stale"] is True
    assert result["possession_revision"] < store.get("possession", "p")["revision"]


def test_explicit_gpu_failure_does_not_silently_switch_to_color(tmp_path, fake_video, monkeypatch):
    store = fixture_store(tmp_path)
    def fail(*args, **kwargs):
        raise RuntimeError("CUDA is not available")
    monkeypatch.setattr(analysis, "BasketballDetector", fail)
    with pytest.raises(RuntimeError, match="cuda"):
        analysis.analyze_possession(store, "p", device="cuda")


def test_two_visible_baskets_require_unambiguous_ball_association():
    left = {"bbox": [10., 20., 30., 25.], "confidence": .95}
    right = {"bbox": [110., 20., 130., 25.], "confidence": .5}
    assert analysis._rim({"rims": [left, right], "balls": []}, {}, 160, 100) is None
    assert analysis._rim({"rims": [left, right], "balls": [
        {"bbox": [115., 12., 121., 18.], "confidence": .9}]}, {}, 160, 100) == right


def test_automatic_event_matches_gold_by_source_time_without_human_id(tmp_path, fake_video, monkeypatch):
    from playlab.evaluation import outcome_metrics
    store = fixture_store(tmp_path)
    monkeypatch.setattr(analysis, "shot_proposals", lambda observations: [1.])
    result = analysis.analyze_possession(store, "p")
    gold = [{"id": "human-gold", "game_id": "g", "possession_id": "p", "release": 1.,
             "reviewed": True, "result": "made"}]
    metrics = outcome_metrics(gold, result["events"])
    assert metrics["matched"] == 1
    assert metrics["emitted"] == 0
    assert metrics["false_positive_events"] == 0
