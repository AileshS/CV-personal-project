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
    # Fakes produce no ball, so injected proposals skip the reached-the-rim confirmation.
    monkeypatch.setattr(analysis, "_reached_rim", lambda features: True)
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
    assert result["performance"]["scan_image_size"] == 960
    assert result["performance"]["cpu_threads"] <= 4
    assert result["performance"]["elapsed_seconds"] >= 0
    assert result["performance"]["cached_scan_frames"] > 0
    assert (result["performance"]["inferred_frames"] + result["performance"]["cached_scan_frames"]
            == result["frames_processed"])
    assert result["performance"]["evaluated_frames_per_second"] == pytest.approx(
        result["performance"]["inferred_frames"] / result["performance"]["elapsed_seconds"])
    assert result["frames_processed"] >= 220  # complete 45-second span, no 600-frame truncation
    assert len(store.list("analysis_chunk")) == 3
    assert FakeDetector.calls - first_count == result["performance"]["inferred_frames"]
    assert len(result["diagnostic_frames"]) == result["frames_processed"]


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
    stages = {row["stage"] for row in result["diagnostic_frames"]}
    assert stages == {"scan", "dense_shot_check", "release_players"}
    assert all(set(row["raw_detections"]) >= {"balls", "rims"} for row in result["diagnostic_frames"])
    assert all("players" in row["raw_detections"] for row in result["diagnostic_frames"]
               if row["stage"] == "release_players")


def test_corrected_release_invalidates_cache(tmp_path, fake_video):
    store = fixture_store(tmp_path)
    store.save("shot", {"id": "s", "possession_id": "p", "release": 1., "result": "unknown", "reviewed": True})
    first = analysis.analyze_possession(store, "p")
    store.save("shot", {"id": "s", "possession_id": "p", "release": 2., "result": "unknown", "reviewed": True})
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


def test_permissive_proposals_need_a_close_sighting_or_two_and_a_confirmed_rim_reach():
    def at(time, distance):
        return {"time": time, "x": 50. + 10 * distance, "y": 30., "rim_x": 50., "rim_y": 30., "rim_width": 10.}
    assert analysis.shot_proposals([at(2., 1.2)]) == [2.]          # one sighting close to the rim
    assert analysis.shot_proposals([at(2., 2.2)]) == []            # one farther sighting is not enough
    assert analysis.shot_proposals([at(2., 2.2), at(2.8, 2.0)]) == [2.]
    assert analysis.shot_proposals([at(2., 2.2), at(3.5, 2.0)]) == []
    # The 30 fps re-check confirms a proposal only if the ball got within a rim width.
    assert analysis._reached_rim({"min_rim_distance": .6})
    assert not analysis._reached_rim({"min_rim_distance": 1.8})
    assert not analysis._reached_rim({"min_rim_distance": None})


def test_later_rebound_shot_does_not_enter_first_shot_evidence(tmp_path, fake_video, monkeypatch):
    store = fixture_store(tmp_path)
    for identifier, release in (("first", 1.), ("rebound", 2.)):
        store.save("shot", {"id": identifier, "possession_id": "p", "release": release,
                            "result": "unknown", "reviewed": True})
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


def test_automatic_mode_ignores_human_anchors_and_preserves_assisted_run(tmp_path, fake_video, monkeypatch):
    store = fixture_store(tmp_path)
    store.save("shot", {"id": "s", "game_id": "g", "possession_id": "p", "release": 1.,
                        "result": "made", "reviewed": True})
    monkeypatch.setattr(analysis, "shot_proposals", lambda observations: [2.])
    assisted = analysis.analyze_possession(store, "p")
    automatic = analysis.analyze_possession(store, "p", mode="automatic")
    assert assisted["events"][0]["origin"] == "human_release_anchor"
    assert len(automatic["events"]) == 1
    assert automatic["events"][0]["release"] == 2.
    assert automatic["events"][0]["origin"] == "automatic_rim_approach"
    assert automatic["analysis_mode"] == "automatic"
    assert store.get("analysis", "p")["fingerprint"] == assisted["fingerprint"]
    before = FakeDetector.calls
    store.save("shot", {"id": "s", "game_id": "g", "possession_id": "p", "release": 3.,
                        "result": "missed", "reviewed": True})
    assert analysis.analysis_is_fresh(store, automatic)
    assert not analysis.analysis_is_fresh(store, assisted)
    again = analysis.analyze_possession(store, "p", mode="automatic")
    assert again["fingerprint"] == automatic["fingerprint"] and FakeDetector.calls == before


def test_freshness_checks_source_setup_and_model_contents(tmp_path, fake_video):
    import os
    store = fixture_store(tmp_path)
    weights = tmp_path / "fixture-weights.pt"
    weights.write_bytes(b"AAAA")
    result = analysis.analyze_possession(store, "p", weights=weights, mode="automatic")
    assert analysis.analysis_is_fresh(store, result)
    stamp = weights.stat()
    weights.write_bytes(b"BBBB")
    os.utime(weights, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert not analysis.analysis_is_fresh(store, result)
    weights.write_bytes(b"AAAA")
    os.utime(weights, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert analysis.analysis_is_fresh(store, result)
    store.save("setup", {"id": "g", "rim_box": [.1, .2, .3, .4]})
    assert not analysis.analysis_is_fresh(store, result)
    store.delete("setup", "g")
    assert analysis.analysis_is_fresh(store, result)
    (tmp_path / "input.mp4").write_bytes(b"changed video")
    assert not analysis.analysis_is_fresh(store, result)


def test_assisted_mode_ignores_draft_shots_until_human_review(tmp_path, fake_video, monkeypatch):
    store = fixture_store(tmp_path)
    draft = store.save("shot", {"id": "draft", "game_id": "g", "possession_id": "p", "release": 1.,
                                "result": "made", "reviewed": False, "source": "prediction"})
    monkeypatch.setattr(analysis, "shot_proposals", lambda observations: [2.])
    first = analysis.analyze_possession(store, "p", mode="assisted")
    assert first["dependencies"]["anchors"] == []
    assert len(first["events"]) == 1
    assert first["events"][0]["origin"] == "automatic_rim_approach"
    assert first["events"][0]["release"] == 2.
    assert first["events"][0]["release_confirmed"] is False
    before = FakeDetector.calls
    draft = store.save("shot", {**draft, "release": 3.})
    assert analysis.analysis_is_fresh(store, first)
    again = analysis.analyze_possession(store, "p", mode="assisted")
    assert again["fingerprint"] == first["fingerprint"] and FakeDetector.calls == before
    store.save("shot", {**draft, "reviewed": True, "source": "human"})
    reviewed = analysis.analyze_possession(store, "p", mode="assisted")
    assert reviewed["fingerprint"] != first["fingerprint"]
    assert reviewed["events"][0]["origin"] == "human_release_anchor"
    assert reviewed["events"][0]["release_confirmed"] is True


@pytest.mark.parametrize("changed", ["weights", "model_bundle"])
def test_custom_artifact_changed_during_detector_startup_marks_result_stale(tmp_path, fake_video, monkeypatch, changed):
    store = fixture_store(tmp_path)
    artifact = tmp_path / "custom.pt"
    artifact.write_bytes(b"before startup")
    class ChangingDetector(FakeDetector):
        def __init__(self, *args):
            artifact.write_bytes(b"changed while starting")
    monkeypatch.setattr(analysis, "BasketballDetector", ChangingDetector)
    result = analysis.analyze_possession(store, "p", **{changed: artifact})
    assert result["stale"] is True
    assert not analysis.analysis_is_fresh(store, result)


def test_initial_default_model_download_is_captured_without_marking_result_stale(tmp_path, fake_video, monkeypatch):
    store = fixture_store(tmp_path)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    default = model_dir / "yolo11n.pt"
    class DownloadingDetector(FakeDetector):
        def __init__(self, *args):
            default.write_bytes(b"explicit mock default model download")
    monkeypatch.setenv("PLAYLAB_MODEL_DIR", str(model_dir))
    monkeypatch.setattr(analysis, "_vision_version", lambda: "test-runtime")
    monkeypatch.setattr(analysis, "BasketballDetector", DownloadingDetector)
    result = analysis.analyze_possession(store, "p")
    assert result["stale"] is False
    assert analysis.analysis_is_fresh(store, result)


def test_missing_optional_weights_allow_honest_fallback_export_and_freeze(tmp_path, fake_video, monkeypatch):
    import json

    from playlab.benchmark import predict_dataset
    from playlab.dataset import export_dataset
    from playlab.evaluation import evaluate_locked, freeze_run

    store = fixture_store(tmp_path)
    missing = tmp_path / "offline-models" / "yolo11n.pt"
    monkeypatch.setattr(analysis, "_default_model_path", lambda: missing)
    monkeypatch.setattr(analysis, "_vision_version", lambda: "installed-test-runtime")

    def offline_detector(*args):
        raise RuntimeError("Optional pretrained weights are unavailable offline")

    monkeypatch.setattr(analysis, "BasketballDetector", offline_detector)
    monkeypatch.setattr(analysis, "shot_proposals", lambda observations: [1.0])
    result = analysis.analyze_possession(store, "p", mode="automatic")
    assert result["dependencies"]["artifacts"] == {}
    assert result["dependencies"]["unavailable_artifacts"] == [str(missing)]
    assert analysis.analysis_is_fresh(store, result)
    assert result["events"][0]["predicted_result"] == "unknown"
    store.save("shot", {"id": "gold", "game_id": "g", "possession_id": "p", "release": 1.,
                        "result": "made", "reviewed": True, "split": "test",
                        "context_start": 0., "context_end": 5.})
    dataset = export_dataset(store, tmp_path / "dataset")
    predictions = predict_dataset(store, dataset, tmp_path / "predictions.json")
    assert json.loads(predictions.read_text())["inference_artifacts"] == {}
    freeze = freeze_run(dataset, [predictions], tmp_path / "freeze")
    report = evaluate_locked(dataset, predictions, tmp_path / "report", freeze)
    assert report["eligible"] == 1 and report["emitted"] == 0
    missing.parent.mkdir()
    missing.write_bytes(b"newly available weights")
    assert not analysis.analysis_is_fresh(store, result)


def test_basket_hint_follows_a_camera_pan_and_is_withheld_when_lost():
    rng = np.random.default_rng(0)
    scene = rng.integers(0, 255, (300, 500, 3), dtype=np.uint8)
    hint = [.40, .40, .46, .44]  # normalized rim box on the hint frame
    template = analysis._rim_template(scene, hint)
    panned = np.roll(scene, (-30, 50), axis=(0, 1))
    box = analysis._track_rim(panned, template)
    assert box is not None
    assert box[0] == pytest.approx(hint[0] + 50 / 500, abs=.005)
    assert box[1] == pytest.approx(hint[1] - 30 / 300, abs=.005)
    unrelated = rng.integers(0, 255, (300, 500, 3), dtype=np.uint8)
    assert analysis._track_rim(unrelated, template) is None
    assert analysis._track_rim(np.zeros_like(scene), template) is None


def test_auto_teams_split_two_uniforms_and_leave_officials_unresolved():
    from playlab.detectors import assign_auto_teams
    white, navy, stripes = [250., 128., 128.], [40., 140., 100.], [120., 128., 128.]
    frames = [{"players": [{"torso": white} for _ in range(8)] + [{"torso": navy} for _ in range(7)]
               + [{"torso": stripes}]}]
    assert assign_auto_teams(frames)
    players = frames[0]["players"]
    assert {p["team"] for p in players[:8]} == {players[0]["team"]}
    assert {p["team"] for p in players[8:15]} == {players[8]["team"]} != {players[0]["team"]}
    assert players[15].get("team") is None
    assert not assign_auto_teams([{"players": [{"torso": white} for _ in range(20)]}])


def test_progress_moves_within_a_chunk_and_names_the_stage(tmp_path, fake_video):
    store = fixture_store(tmp_path, duration=20)
    reports = []
    analysis.analyze_possession(store, "p", progress=lambda fraction, checkpoint: reports.append((fraction, checkpoint)))
    detecting = [(f, c) for f, c in reports if c.get("stage") == "detecting"]
    assert len(detecting) >= 5
    fractions = [f for f, _ in detecting]
    assert fractions == sorted(fractions) and 0 < fractions[0] < fractions[-1] < .85
    assert detecting[-1][1]["seconds_total"] == 20
    assert reports[-1][0] == 1.0


def test_rim_crop_keeps_arc_room_and_maps_boxes_back_to_frame():
    crop = analysis._rim_crop([600, 100, 640, 110], 1280, 720)
    left, top, right, bottom = crop
    assert right - left == bottom - top == 240          # 3 rim widths each side
    assert left <= 600 and right >= 640 and top == 0      # clamped at the top edge
    edge = analysis._rim_crop([1260, 700, 1280, 710], 1280, 720)
    assert edge[2] <= 1280 and edge[3] <= 720

    class Cropped:
        def ball_rim(self, frame, imgsz):
            Cropped.seen = (frame.shape, imgsz)
            return {"balls": [{"bbox": [10, 20, 14, 24], "confidence": .9}], "rims": [], "players": []}

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    found = analysis._cropped_ball_rim(Cropped(), frame, crop)
    assert Cropped.seen == ((240, 240, 3), 192)           # 240 px crop at the full-frame 0.75 scale
    assert found["balls"][0]["bbox"] == [10 + left, 20 + top, 14 + left, 24 + top]


def _ball(cx, cy, confidence, size=6.):
    return {"bbox": [cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2], "confidence": confidence}


def test_ball_near_its_previous_position_beats_a_stronger_distractor_elsewhere():
    rim = {"bbox": [70., 20., 90., 25.], "confidence": .9}
    _, last = analysis._ball_observation({"balls": [_ball(40, 50, .8)]}, rim, 0., None, 160)
    # A head across the frame scores higher than the ball, which barely moved.
    row, last = analysis._ball_observation({"balls": [_ball(42, 48, .45), _ball(130, 30, .7)]},
                                           rim, .2, last, 160)
    assert (row["x"], row["y"]) == (42, 48)
    assert analysis._chosen_ball(last, .2)["bbox"] == _ball(42, 48, .45)["bbox"]


def test_ball_prior_follows_velocity_and_expires():
    _, last = analysis._ball_observation({"balls": [_ball(20, 60, .8)]}, None, 0., None, 160)
    _, last = analysis._ball_observation({"balls": [_ball(30, 55, .8)]}, None, .1, last, 160)
    # Predicted near (40, 50): a candidate there wins over one at the old spot.
    _, last = analysis._ball_observation({"balls": [_ball(20, 60, .5), _ball(40, 50, .5)]}, None, .2, last, 160)
    assert (last["x"], last["y"]) == (40, 50)
    # After the prior expires, a lone far detection is accepted and an even split is ambiguous again.
    _, later = analysis._ball_observation({"balls": [_ball(140, 20, .6)]}, None, 1., last, 160)
    assert (later["x"], later["y"]) == (140, 20)
    assert analysis._ball_observation({"balls": [_ball(20, 60, .5), _ball(140, 20, .45)]},
                                      None, 1., last, 160) == (None, last)


def test_two_candidates_near_the_prior_stay_ambiguous_and_far_jumps_are_rejected():
    _, last = analysis._ball_observation({"balls": [_ball(40, 50, .8)]}, None, 0., None, 160)
    assert analysis._ball_observation({"balls": [_ball(41, 50, .5), _ball(44, 52, .45)]},
                                      None, .2, last, 160) == (None, last)
    # A very strong detection far away outranks a weak near one, but a 0.2 s jump that far is rejected.
    assert analysis._ball_observation({"balls": [_ball(42, 50, .15), _ball(140, 20, .95)]},
                                      None, .2, last, 160) == (None, last)
    assert analysis._chosen_ball(last, .2) is None
