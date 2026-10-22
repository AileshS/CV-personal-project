"""Software fixtures only: these checks make no accuracy claim."""
import json
from pathlib import Path

import numpy as np
import pytest

from playlab import analysis
from playlab import quick_analysis as quick
from playlab.store import Store
from playlab.worker import work_once


@pytest.fixture
def lab(tmp_path, monkeypatch):
    store = Store(tmp_path / "store")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake video; decoding replaced explicitly")
    store.save("game", {"id": "g", "source_path": str(source), "duration": 90,
                        "source_offset_seconds": 750, "source_provenance": {"origin": "fixture"}})

    class Detector:
        mode = "fixture_only"
        calls = 0

        def __init__(self, *args):
            pass

        def reset(self):
            pass

        def __call__(self, frame):
            type(self).calls += 1
            return {"balls": [], "rims": [], "players": []}

    def frames(path, start, end, **kwargs):
        for at in np.arange(start, end, .2):
            yield float(at), np.zeros((80, 100, 3), dtype=np.uint8)

    def playback(source, destination, *args, **kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"bounded playback fixture")
        return destination

    monkeypatch.setattr(quick, "clip_video", playback)
    monkeypatch.setattr(analysis, "BasketballDetector", Detector)
    # Fakes produce no ball, so injected proposals skip the reached-the-rim confirmation.
    monkeypatch.setattr(analysis, "_reached_rim", lambda features: True)
    monkeypatch.setattr(analysis, "iter_frames", frames)
    monkeypatch.setattr(analysis, "frame_at", lambda *a, **kw: np.zeros((80, 100, 3), dtype=np.uint8))
    # Real local ball/rim weights must not leak into fixture requests.
    monkeypatch.setattr(analysis, "default_ball_rim_weights", lambda: None)
    return store, Detector


def test_ball_rim_weights_are_discovered_and_recorded(lab, tmp_path, monkeypatch):
    store, _ = lab
    monkeypatch.undo()
    monkeypatch.delenv("PLAYLAB_BALL_RIM_WEIGHTS", raising=False)
    monkeypatch.setenv("PLAYLAB_MODEL_DIR", str(tmp_path / "models"))
    (tmp_path / "models").mkdir()
    assert analysis.default_ball_rim_weights() is None
    weights = tmp_path / "models" / "ball_rim.pt"
    weights.write_bytes(b"ball/rim weights fixture")
    assert analysis.default_ball_rim_weights() == weights.resolve()
    clip = quick.queue_analysis_clip(store, "g", 0, 20)
    assert clip["weights"] == str(weights.resolve())
    assert clip["requested_models"] == {str(weights.resolve()): quick.file_hash(weights)}
    explicit = tmp_path / "other.pt"
    monkeypatch.setenv("PLAYLAB_BALL_RIM_WEIGHTS", str(explicit))
    # An explicit but missing configuration is returned so it fails loudly later.
    assert analysis.default_ball_rim_weights() == explicit.resolve()


@pytest.mark.parametrize("bounds", [(0, 61), (-1, 5), (89, 91), (1, 1), (0, .5), (float("nan"), 5)])
def test_invalid_requests_publish_nothing(lab, bounds):
    store, _ = lab
    with pytest.raises(ValueError):
        quick.queue_analysis_clip(store, "g", *bounds)
    assert store.jobs() == []
    assert store.list("analysis_clip") == []


def test_multiple_attempts_original_times_and_gold_isolation(lab, monkeypatch):
    store, _ = lab
    store.save("shot", {"id": "gold", "game_id": "g", "possession_id": "p", "release": 4, "result": "made"})
    store.save("setup", {"id": "g", "rim_box": [.1, .1, .2, .2]})
    monkeypatch.setattr(analysis, "shot_proposals", lambda observations: [2., 8.])
    before = store.list("shot")
    clip = quick.queue_analysis_clip(store, "g", 0, 20)
    assert quick.queue_analysis_clip(store, "g", 0, 20)["id"] == clip["id"]
    assert work_once(store)
    view = quick.get_analysis_clip(store, clip["id"])
    assert view["status"] == "done" and view["fresh"]
    assert Path(view["playback_path"]).is_file()
    result = view["result"]
    assert [(e["release"], e["local_release"]) for e in result["events"]] == [(752., 2.), (758., 8.)]
    assert all(e["result"] == "unknown" for e in result["events"])
    assert all(Path(e["evidence_path"]).is_file() for e in result["events"])
    assert result["dependencies"]["anchors"] == []
    assert result["dependencies"]["setup"] == {}
    assert store.list("shot") == before
    assert all(not store.list(kind) for kind in ("possession", "analysis", "analysis_chunk", "box", "frame"))
    assert len(store.list("quick_analysis_chunk")) == 1


def test_empty_result_has_context_evidence_and_no_anchor_instruction(lab):
    store, _ = lab
    clip = quick.queue_analysis_clip(store, "g", 0, 20)
    result = quick.perform_analysis_clip(store, {"clip_id": clip["id"]})
    assert result["events"] == [] and result["context_time"] == 760
    assert Path(result["context_evidence_path"]).is_file()
    warnings = " ".join(result["warnings"])
    assert "unknown" in warnings and "COCO" in warnings and "anchor" not in warnings


@pytest.mark.parametrize("change", ["offset", "provenance", "source", "setup", "model"])
def test_freshness_tracks_all_inputs(lab, tmp_path, change):
    store, _ = lab
    model = tmp_path / "weights.pt"
    model.write_bytes(b"fixture model")
    clip = quick.queue_analysis_clip(store, "g", 0, 20, weights=model)
    result = quick.perform_analysis_clip(store, {"clip_id": clip["id"]})
    assert quick.analysis_clip_is_fresh(store, result)
    game = store.get("game", "g")
    if change == "offset":
        store.save("game", {**game, "source_offset_seconds": 250})
    elif change == "provenance":
        store.save("game", {**game, "source_provenance": {"origin": "edited"}})
    elif change == "source":
        Path(game["source_path"]).write_bytes(b"changed source")
    elif change == "model":
        model.write_bytes(b"different model")
    else:
        store.save("analysis_clip", {**clip, "setup": {"rim_box": [.1, .1, .2, .2]}})
    assert not quick.analysis_clip_is_fresh(store, result)


def test_cancel_resume_reuses_completed_chunks(lab):
    store, detector = lab
    clip = quick.queue_analysis_clip(store, "g", 0, 45)
    stop = [False]

    def progress(fraction, checkpoint):
        if checkpoint.get("completed_chunks") == 1:
            stop[0] = True

    with pytest.raises(InterruptedError):
        quick.perform_analysis_clip(store, {"clip_id": clip["id"]}, cancelled=lambda: stop[0], progress=progress)
    assert len(store.list("quick_analysis_chunk")) == 1
    calls = detector.calls
    quick.perform_analysis_clip(Store(store.root), {"clip_id": clip["id"]})
    assert detector.calls - calls < calls * 1.5
    assert len(store.list("quick_analysis_chunk")) == 3


def test_missing_model_fails_honestly_and_job_can_resume(lab, tmp_path):
    store, _ = lab
    weights = tmp_path / "missing.pt"
    clip = quick.queue_analysis_clip(store, "g", 0, 20, weights=weights)
    work_once(store)
    job = next(j for j in store.jobs() if j["id"] == clip["job_id"])
    assert job["status"] == "failed" and "missing" in job["error"]
    assert quick.get_analysis_result(store, clip["id"]) is None
    store.resume_job(job["id"])
    assert next(j for j in store.jobs() if j["id"] == job["id"])["status"] == "queued"


def test_worker_recovery_and_cancel_before_decode(lab):
    store, detector = lab
    clip = quick.queue_analysis_clip(store, "g", 0, 20)
    store.claim_job()
    store.recover_interrupted()
    store.resume_job(clip["job_id"])
    store.cancel_job(clip["job_id"])
    assert not work_once(store)
    assert detector.calls == 0


def test_source_change_after_queue_requires_new_request(lab):
    store, _ = lab
    clip = quick.queue_analysis_clip(store, "g", 0, 20)
    game = store.get("game", "g")
    store.save("game", {**game, "source_offset_seconds": 250})
    with pytest.raises(ValueError, match="changed"):
        quick.perform_analysis_clip(store, {"clip_id": clip["id"]})


def test_queue_transaction_rolls_back_on_failure(lab):
    store, _ = lab
    with store.connect() as db:
        db.execute("CREATE TRIGGER reject_job BEFORE INSERT ON jobs BEGIN SELECT RAISE(ABORT, 'fixture queue failure'); END")
    with pytest.raises(Exception, match="fixture queue failure"):
        quick.queue_analysis_clip(store, "g", 0, 20)
    assert store.list("analysis_clip") == []
    assert store.jobs() == []


def test_locked_partition_rejected_without_requiring_splits(lab):
    store, _ = lab
    game = store.get("game", "g")
    store.save("game", {**game, "split_blocks": [{"start": 40, "end": 80, "split": "test"}]})
    with pytest.raises(ValueError, match="locked test"):
        quick.queue_analysis_clip(store, "g", 35, 45)
    assert quick.queue_analysis_clip(store, "g", 0, 20)


def test_evidence_burns_original_timestamp(lab, monkeypatch):
    store, _ = lab
    texts = []
    draw = analysis.ImageDraw.Draw

    def capture(image):
        painter = draw(image)
        original = painter.text

        def text(xy, value, **kwargs):
            texts.append(value)
            return original(xy, value, **kwargs)

        painter.text = text
        return painter

    monkeypatch.setattr(analysis.ImageDraw, "Draw", capture)
    game = store.get("game", "g")
    store.save("game", {**game, "source_offset_seconds": 250})
    clip = quick.queue_analysis_clip(store, "g", 0, 20)
    quick.perform_analysis_clip(store, {"clip_id": clip["id"]})
    assert any("Source 260.00s" in value for value in texts)
    raw = store.list("quick_analysis")[0]
    assert json.dumps(raw, allow_nan=False)


def test_unconfirmed_release_reports_flagged_opponent_in_image_and_result(lab, monkeypatch):
    store, _ = lab
    monkeypatch.setattr(analysis, "shot_proposals", lambda rows: [2.])
    monkeypatch.setattr(analysis, "_shooter", lambda *a, **kw: "shooter")
    monkeypatch.setattr(analysis, "nearest_defender", lambda *a: {
        "status": "approximate", "track_id": "opponent", "distance": 10})
    captured = []
    original = analysis._evidence

    def evidence(source, at, players, shooter, defender, destination, **kwargs):
        captured.append(defender)
        return original(source, at, players, shooter, defender, destination, **kwargs)

    monkeypatch.setattr(analysis, "_evidence", evidence)
    clip = quick.queue_analysis_clip(store, "g", 0, 20)
    result = quick.perform_analysis_clip(store, {"clip_id": clip["id"]})
    assert captured[0]["status"] == "approximate"
    defender = result["events"][0]["defender"]
    assert defender["status"] == "approximate" and defender["track_id"] == "opponent"
    assert defender["release_unconfirmed"] is True and "estimated release" in defender["reason"]


def test_unavailable_neural_runtime_is_diagnostic_unknown(lab, monkeypatch):
    store, _ = lab

    def unavailable(*args):
        raise RuntimeError("fixture neural runtime unavailable")

    monkeypatch.setattr(analysis, "BasketballDetector", unavailable)
    monkeypatch.setattr(analysis, "shot_proposals", lambda rows: [2.])
    clip = quick.queue_analysis_clip(store, "g", 0, 20)
    result = quick.perform_analysis_clip(store, {"clip_id": clip["id"]})
    assert result["events"][0]["result"] == "unknown"
    assert "Learned detector unavailable" in result["events"][0]["reason"]
    assert any("fixture neural runtime unavailable" in w for w in result["warnings"])


def test_requested_device_failure_is_not_silently_fallback(lab, monkeypatch):
    store, _ = lab

    def unavailable(*args):
        raise RuntimeError("fixture unavailable device")

    monkeypatch.setattr(analysis, "BasketballDetector", unavailable)
    clip = quick.queue_analysis_clip(store, "g", 0, 20, device="cuda")
    with pytest.raises(RuntimeError, match="Requested detector/device"):
        quick.perform_analysis_clip(store, {"clip_id": clip["id"]})
    assert quick.get_analysis_result(store, clip["id"]) is None


def test_setup_request_edit_requires_new_request(lab):
    store, _ = lab
    clip = quick.queue_analysis_clip(store, "g", 0, 20)
    store.save("analysis_clip", {**clip, "setup": {"rim_box": [.1, .1, .2, .2]}})
    with pytest.raises(ValueError, match="Queue a new interval request"):
        quick.perform_analysis_clip(store, {"clip_id": clip["id"]})


def test_default_model_replacement_after_queue_requires_new_request(lab, monkeypatch, tmp_path):
    store, _ = lab
    default = tmp_path / "default.pt"
    default.write_bytes(b"default fixture version one")
    monkeypatch.setattr(analysis, "_default_model_path", lambda: default)
    clip = quick.queue_analysis_clip(store, "g", 0, 20)
    default.write_bytes(b"default fixture version two")
    with pytest.raises(ValueError, match="Queue a new interval request"):
        quick.perform_analysis_clip(store, {"clip_id": clip["id"]})


def test_locked_partition_rejection_reads_no_source_content(lab, monkeypatch):
    store, _ = lab
    game = store.get("game", "g")
    store.save("game", {**game, "split": "test"})

    def forbidden(path):
        pytest.fail("Locked source content must not be read")

    monkeypatch.setattr(quick, "source_fingerprint", forbidden)
    with pytest.raises(ValueError, match="locked test"):
        quick.queue_analysis_clip(store, "g", 0, 20)


def test_default_model_directory_change_requires_new_request(lab, monkeypatch, tmp_path):
    store, _ = lab
    clip = quick.queue_analysis_clip(store, "g", 0, 20)
    monkeypatch.setattr(analysis, "_default_model_path", lambda: tmp_path / "other.pt")
    with pytest.raises(ValueError, match="Queue a new interval request"):
        quick.perform_analysis_clip(store, {"clip_id": clip["id"]})


def test_shared_test_source_never_hashes_locked_head_tail(lab, monkeypatch):
    store, _ = lab
    game = store.get("game", "g")
    store.save("game", {**game, "split_blocks": [{"start": 40, "end": 90, "split": "test"}]})

    def forbidden(path):
        pytest.fail("Hashing a shared source could read held-out bytes")

    monkeypatch.setattr(quick, "source_fingerprint", forbidden)
    clip = quick.queue_analysis_clip(store, "g", 0, 20)
    result = quick.perform_analysis_clip(store, {"clip_id": clip["id"]})
    assert quick.analysis_clip_is_fresh(store, result)
