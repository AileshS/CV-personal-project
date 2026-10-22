"""Queue/lifecycle software fixtures, never basketball accuracy evidence."""
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from playlab import clip_tracking as tracking
from playlab import model_assets, quick_analysis, tracked_video, worker
from playlab.dataset import file_hash
from playlab.store import Store


@pytest.fixture
def lab(tmp_path, monkeypatch):
    store = Store(tmp_path / "data")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"software source; decoder is replaced")
    model = tmp_path / "model.pt"
    model.write_bytes(b"software model; inference is replaced")
    store.save("game", {"id": "g", "source_path": str(source), "duration": 30,
                        "source_offset_seconds": 750, "source_provenance": {"origin": "fixture"}})
    monkeypatch.setattr(model_assets, "default_ball_rim_path", lambda: model)
    monkeypatch.setattr(model_assets, "DEFAULT_MODEL_SHA256", file_hash(model))
    monkeypatch.setattr(model_assets, "ensure_ball_rim_model", lambda: model)
    monkeypatch.setattr(tracking, "_code_identity", lambda: {"fixture": "stable"})
    monkeypatch.delenv("PLAYLAB_NTFY_URL", raising=False)
    renders = []

    def render(source, directory, **kwargs):
        renders.append((source, directory, kwargs))
        directory.mkdir(parents=True)
        result = {"model_sha256": file_hash(kwargs["weights"]), "frame_count": 30,
                  "fps": 30, "duration": kwargs["end"] - kwargs["start"]}
        for key, name in (("video_path", "tracked.mp4"), ("samples_path", "samples.json"),
                          ("report_path", "report.json")):
            path = directory / name
            path.write_bytes(b"software output")
            result[key] = str(path)
        return result

    monkeypatch.setattr(tracked_video, "render_tracked_video", render)
    return store, source, model, renders


@pytest.mark.parametrize("bounds", [(-1, 1), (0, 10.1), (1, 1), (29.9, 30),
                                    (float("nan"), 5), (True, 2), (25, 31)])
def test_rejects_invalid_bounds_without_publishing(lab, bounds):
    store, *_ = lab
    with pytest.raises(ValueError):
        tracking.queue_tracking_clip(store, "g", *bounds)
    assert not store.jobs() and not store.list("tracking_clip")


def test_deduplicates_atomic_queue_and_preserves_gold(lab):
    store, _, _, renders = lab
    gold = store.save("shot", {"id": "human", "result": "made", "reviewed": True})
    clip = tracking.queue_tracking_clip(store, "g", 2, 12)
    assert tracking.queue_tracking_clip(Store(store.root), "g", 2, 12)["id"] == clip["id"]
    assert worker.work_once(store)
    view = tracking.get_tracking_clip(store, clip["id"])
    assert view["status"] == "done" and view["fresh"]
    assert Path(view["video_path"]).is_file()
    assert (clip["original_start"], clip["original_end"]) == (752, 762)
    assert renders[0][2]["source_offset_seconds"] == 750
    assert not view["result"]["reviewed"]
    assert store.get("shot", "human") == gold
    for kind in ("possession", "analysis", "analysis_chunk", "quick_analysis", "setup", "box", "frame", "event"):
        assert not store.list(kind)


def test_queue_transaction_rolls_back_request_when_job_insert_fails(lab):
    store, *_ = lab
    with store.connect() as db:
        db.execute("CREATE TRIGGER reject_tracking_job BEFORE INSERT ON jobs "
                   "BEGIN SELECT RAISE(ABORT, 'queue fixture failure'); END")
    with pytest.raises(Exception, match="queue fixture failure"):
        tracking.queue_tracking_clip(store, "g", 0, 10)
    assert not store.list("tracking_clip") and not store.jobs()


def test_concurrent_identical_clicks_have_one_request_and_job(lab):
    store, *_ = lab
    clips = []
    failures = []

    def click():
        try:
            clips.append(tracking.queue_tracking_clip(Store(store.root), "g", 0, 10))
        except Exception as error:
            failures.append(error)

    threads = [threading.Thread(target=click) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not failures
    assert len({clip["id"] for clip in clips}) == 1
    assert len(store.jobs()) == len(store.list("tracking_clip")) == 1


def test_locked_test_rejects_before_source_content_read(lab, monkeypatch):
    store, *_ = lab
    game = store.get("game", "g")
    store.save("game", {**game, "split": "test"})

    def forbidden(*args):
        pytest.fail("Locked-test source contents were read")

    monkeypatch.setattr(quick_analysis, "source_fingerprint", forbidden)
    with pytest.raises(ValueError, match="locked test"):
        tracking.queue_tracking_clip(store, "g", 0, 10)
    assert not store.jobs()


def test_shared_source_avoids_held_out_head_tail_hashes(lab, monkeypatch):
    store, *_ = lab
    game = store.get("game", "g")
    store.save("game", {**game, "split_blocks": [{"start": 20, "end": 30, "split": "test"}]})

    def forbidden(*args):
        pytest.fail("A shared test source was content-hashed")

    monkeypatch.setattr(quick_analysis, "source_fingerprint", forbidden)
    clip = tracking.queue_tracking_clip(store, "g", 0, 10)
    tracking.perform_tracking_clip(store, {"clip_id": clip["id"]})
    assert tracking.get_tracking_clip(store, clip["id"])["fresh"]


@pytest.mark.parametrize("change", ["source", "model", "offset", "provenance", "code", "video", "samples", "test"])
def test_completed_output_invalidates_when_inputs_or_outputs_change(lab, monkeypatch, change):
    store, source, model, _ = lab
    clip = tracking.queue_tracking_clip(store, "g", 0, 10)
    worker.work_once(store)
    view = tracking.get_tracking_clip(store, clip["id"])
    assert view["fresh"]
    if change == "source":
        source.write_bytes(b"changed source")
    elif change == "model":
        model.write_bytes(b"changed model")
    elif change == "code":
        monkeypatch.setattr(tracking, "_code_identity", lambda: {"fixture": "changed"})
    elif change in {"video", "samples"}:
        key = "video_path" if change == "video" else "samples_path"
        Path(view["result"][key]).write_bytes(b"changed output")
    else:
        game = store.get("game", "g")
        edit = ({"source_offset_seconds": 900} if change == "offset" else
                {"split": "test"} if change == "test" else {"source_provenance": {"origin": "changed"}})
        store.save("game", {**game, **edit})
    stale = tracking.get_tracking_clip(store, clip["id"])
    assert not stale["fresh"] and stale["video_path"] is None and stale["status"] == "stale"


def test_changed_request_is_rejected_before_render(lab):
    store, source, _, renders = lab
    tracking.queue_tracking_clip(store, "g", 0, 10)
    source.write_bytes(b"changed before worker")
    assert worker.work_once(store)
    assert store.jobs()[0]["status"] == "failed"
    assert not renders and not store.list("tracking_result")


def test_cancelled_attempt_is_clean_and_retry_renders_new_output(lab, monkeypatch):
    store, _, _, renders = lab
    clip = tracking.queue_tracking_clip(store, "g", 0, 10)
    original = tracked_video.render_tracked_video
    stop = [False]

    def cancel_after_render(*args, **kwargs):
        result = original(*args, **kwargs)
        stop[0] = True
        return result

    monkeypatch.setattr(tracked_video, "render_tracked_video", cancel_after_render)
    with pytest.raises(InterruptedError):
        tracking.perform_tracking_clip(store, {"clip_id": clip["id"]}, cancelled=lambda: stop[0])
    assert not store.list("tracking_result")
    assert not renders[0][1].exists()
    stop[0] = False
    monkeypatch.setattr(tracked_video, "render_tracked_video", original)
    tracking.perform_tracking_clip(store, {"clip_id": clip["id"]})
    assert len(renders) == 2 and renders[1][1] != renders[0][1]
    assert tracking.get_tracking_clip(store, clip["id"])["fresh"]


def test_default_missing_can_queue_but_custom_missing_cannot(lab, tmp_path, monkeypatch):
    store, _, model, _ = lab
    original = model.read_bytes()
    model.unlink()
    clip = tracking.queue_tracking_clip(store, "g", 0, 10)
    assert not model.exists()
    with pytest.raises(ValueError, match="missing"):
        tracking.queue_tracking_clip(store, "g", 0, 10, weights=tmp_path / "missing.pt")

    def ensure():
        model.write_bytes(original)
        return model

    monkeypatch.setattr(model_assets, "ensure_ball_rim_model", ensure)
    assert worker.work_once(store)
    assert tracking.get_tracking_clip(store, clip["id"])["fresh"]


def test_same_runtime_worker_autostart_is_reserved_and_deduplicated(tmp_path, monkeypatch):
    store = Store(tmp_path / "data")
    calls = []
    monkeypatch.setattr(worker, "_process_alive", lambda pid: pid == 12345)

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=12345, poll=lambda: None)

    monkeypatch.setattr(worker.subprocess, "Popen", popen)
    first = worker.ensure_tracking_worker(store)
    second = worker.ensure_tracking_worker(store)
    assert first["owned"] and not second["owned"] and second["state"] == "starting"
    assert first["recovered"] is False and second["recovered"] is False
    assert len(calls) == 1
    assert calls[0][0] == [worker.sys.executable, "-m", "playlab.cli", "--data", str(store.root), "worker"]
    assert first["log_path"] == str(store.root / "worker-tracking.log")


def test_autostart_reuses_live_worker_and_refuses_unknown_lock(tmp_path, monkeypatch):
    store = Store(tmp_path / "data")
    lock = store.root / ".worker.lock"
    lock.write_text(str(os.getpid()))
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **kw: pytest.fail("Duplicate worker launch"))
    existing = worker.ensure_tracking_worker(store)
    assert existing["state"] == "running" and existing["recovered"] is False
    lock.write_text("unverifiable")
    with pytest.raises(ValueError, match="cannot be verified"):
        worker.ensure_tracking_worker(store)
    assert lock.read_text() == "unverifiable"


def test_autostart_recovers_only_proven_stopped_worker(tmp_path, monkeypatch):
    store = Store(tmp_path / "data")
    (store.root / ".worker.lock").write_text("123")
    job = store.enqueue("track_clip", {"clip_id": "interrupted"})
    store.claim_job()
    monkeypatch.setattr(worker, "_process_alive", lambda pid: False)
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=456, poll=lambda: None))
    started = worker.ensure_tracking_worker(store)
    assert started["recovered"] is True
    assert not (store.root / ".worker.lock").exists()
    assert next(item for item in store.jobs() if item["id"] == job)["status"] == "failed"


def test_starting_reservation_is_handed_to_matching_worker(tmp_path, monkeypatch):
    store = Store(tmp_path / "data")
    reservation = store.root / ".worker.starting"
    reservation.write_text(json.dumps({"pid": os.getpid()}))
    assert worker.run_worker(store, once=True) is False
    assert not reservation.exists() and not (store.root / ".worker.lock").exists()
    reservation.write_text(json.dumps({"pid": 12345}))
    monkeypatch.setattr(worker, "_process_alive", lambda pid: True)
    with pytest.raises(ValueError, match="already starting"):
        worker.run_worker(store, once=True)
    assert reservation.exists()


def test_late_worker_startup_failure_becomes_stopped_and_retryable(tmp_path, monkeypatch):
    store = Store(tmp_path / "data")
    alive = [True]
    calls = []
    monkeypatch.setattr(worker, "_process_alive", lambda pid: alive[0])

    def popen(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(pid=12345, poll=lambda: None)

    monkeypatch.setattr(worker.subprocess, "Popen", popen)
    worker.ensure_tracking_worker(store)
    assert worker.worker_status(store)["state"] == "starting"
    alive[0] = False  # Child failed after Popen.poll() but before acquiring lock.
    assert worker.worker_status(store)["state"] == "stopped"
    worker.ensure_tracking_worker(store)
    assert len(calls) == 2


def test_worker_death_after_start_is_observable_without_automatic_restart(tmp_path, monkeypatch):
    store = Store(tmp_path / "data")
    (store.root / ".worker.lock").write_text("12345")
    alive = [True]
    monkeypatch.setattr(worker, "_process_alive", lambda pid: alive[0])
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **kw: pytest.fail("Status read launched a worker"))
    assert worker.worker_status(store)["state"] == "running"
    alive[0] = False
    assert worker.worker_status(store)["state"] == "stopped"
    assert (store.root / ".worker.lock").exists()
