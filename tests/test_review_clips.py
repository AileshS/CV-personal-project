"""Bounded context never falls back to a coarse parent or locked test."""
import subprocess
import threading

import pytest

from playlab.media import ffmpeg_executable, probe_video, source_fingerprint
from playlab.review import ReviewBatch
from playlab.review_clips import get_review_clip, queue_review_clip
from playlab.store import Store
from playlab.worker import work_once


@pytest.fixture
def context(tmp_path):
    store = Store(tmp_path / "data")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original source")
    game = store.save("game", {"id": "g", "source_path": str(source), "duration": 1000,
        "source_fingerprint": source_fingerprint(source), "split_mode": "game", "split": "train"})
    batch = ReviewBatch(tmp_path, {"batch_hash": "a" * 64,
        "sources": {"g": {**game, "source_size_bytes": source.stat().st_size}},
        "review_items": [{"id": "i", "game_id": "g", "possession_id": "coarse-parent",
                          "start": 20, "end": 40, "split": "train"}],
        "frames": [{"id": "f", "review_item_id": "i"}]})
    return store, batch, source


def fake_clip(source, destination, start, end, *, cancelled):
    assert not cancelled()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"short clip")
    return destination


def test_exact_item_window_durable_queue_and_playback(context, monkeypatch):
    store, batch, _ = context
    calls = []

    def clip(*args, **kwargs):
        calls.append(args)
        return fake_clip(*args, **kwargs)

    monkeypatch.setattr("playlab.review_clips.clip_video", clip)
    assert get_review_clip(store, batch, "f") is None
    queued = queue_review_clip(store, batch, "f")
    assert queue_review_clip(store, batch, "f")["job_id"] == queued["job_id"]
    assert "path" not in queued and queued["status"] == "queued"
    assert work_once(store)
    ready = get_review_clip(Store(store.root), batch, "f")
    assert ready["status"] == "ready"
    assert calls[0][2:] == (20, 40)
    assert ready["path"] == str(store.root / "clips" / f"review-{ready['id']}.mp4")
    assert queue_review_clip(store, batch, "f")["status"] == "ready"
    assert not store.list("possession") and not store.list("frame")


@pytest.mark.parametrize("start,end", [(0, 121), (0, float("inf")), (float("nan"), 30),
                                       (-1, 30), (30, 20), (999, 1001), (True, 30), (0, None)])
def test_reject_invalid_bounds(context, start, end):
    store, batch, _ = context
    with pytest.raises(ValueError):
        queue_review_clip(store, batch, "f", start, end)
    assert not store.jobs()


def test_context_bounds_are_explicit_and_distinct(context):
    store, batch, _ = context
    first = queue_review_clip(store, batch, "f", 10, 50)
    second = queue_review_clip(store, batch, "f")
    assert first["id"] != second["id"]
    assert (first["start"], first["end"]) == (10, 50)


def test_live_worker_cannot_claim_before_companion_record(context, monkeypatch):
    store, batch, _ = context
    worker_store = Store(store.root)
    monkeypatch.setattr("playlab.review_clips.clip_video", fake_clip)
    connect = store.connect
    threads, errors = [], []
    entered = threading.Event()

    def worker():
        entered.set()
        try:
            assert work_once(worker_store)
        except BaseException as exc:
            errors.append(exc)

    def trace(sql):
        if sql.startswith("INSERT INTO jobs"):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
            assert entered.wait(2)

    def traced_connect():
        db = connect()
        db.set_trace_callback(trace)
        return db

    monkeypatch.setattr(store, "connect", traced_connect)
    queue_review_clip(store, batch, "f")
    assert len(threads) == 1
    threads[0].join(5)
    assert not threads[0].is_alive() and not errors
    assert worker_store.jobs()[0]["status"] == "done"
    assert get_review_clip(store, batch, "f")["status"] == "ready"
    assert queue_review_clip(store, batch, "f")["status"] == "ready"


def test_reject_test_and_guards(context):
    store, batch, _ = context
    game = store.get("game", "g")
    game.update(split_mode="blocks", split_blocks=[{"start": 0, "end": 100, "split": "train"},
        {"start": 100, "end": 200, "split": "val"}, {"start": 200, "end": 1000, "split": "test"}])
    game = store.save("game", game)
    batch.manifest["sources"]["g"].update(game)
    for bounds in [(85, 95), (90, 110), (210, 220)]:
        with pytest.raises(ValueError):
            queue_review_clip(store, batch, "f", *bounds)
    batch.items["i"]["split"] = "test"
    with pytest.raises(ValueError):
        queue_review_clip(store, batch, "f", 220, 240)


@pytest.mark.parametrize("change", ["bytes", "policy", "bounds"])
def test_worker_revalidates_before_encoding(context, monkeypatch, change):
    store, batch, source = context
    queue_review_clip(store, batch, "f")
    if change == "bytes":
        source.write_bytes(b"changed! source")  # Same byte count still changes head/tail identity.
    elif change == "policy":
        store.save("game", {**store.get("game", "g"), "split": "test"})
    else:
        job = store.jobs()[0]
        import json
        with store.connect() as db:
            db.execute("UPDATE jobs SET payload=? WHERE id=?",
                       (json.dumps({**job["payload"], "end": 900}), job["id"]))
    monkeypatch.setattr("playlab.review_clips.clip_video", lambda *a, **kw: pytest.fail("encoded"))
    assert work_once(store)
    assert store.jobs()[0]["status"] == "failed"
    assert "path" not in store.list("review_clip")[0]


def test_ready_clip_requires_current_source_and_safe_path(context, monkeypatch):
    store, batch, source = context
    monkeypatch.setattr("playlab.review_clips.clip_video", fake_clip)
    queue_review_clip(store, batch, "f")
    work_once(store)
    ready = get_review_clip(store, batch, "f")
    from pathlib import Path
    path = Path(ready["path"])
    path.unlink()
    path.symlink_to(source)
    with pytest.raises(ValueError, match="Unsafe"):
        get_review_clip(store, batch, "f")
    path.unlink()
    source.write_bytes(b"changed! source")
    with pytest.raises(ValueError, match="source changed"):
        get_review_clip(store, batch, "f")


def test_worker_preserves_cancel_and_progress(context, monkeypatch):
    store, batch, _ = context
    queued = queue_review_clip(store, batch, "f")

    def cancel_during_clip(*args, **kwargs):
        store.cancel_job(queued["job_id"])
        assert kwargs["cancelled"]()
        raise InterruptedError("Cancelled during encoding")

    monkeypatch.setattr("playlab.review_clips.clip_video", cancel_during_clip)
    assert work_once(store)
    job = store.jobs()[0]
    assert job["status"] == "cancelled" and job["progress"] == .05
    assert "path" not in get_review_clip(store, batch, "f")


def test_actual_worker_encodes_only_temporary_fixture(context):
    store, batch, source = context
    subprocess.run([ffmpeg_executable(), "-v", "error", "-y", "-f", "lavfi", "-i",
                    "color=c=blue:s=64x48:r=10:d=3", "-c:v", "libx264", str(source)], check=True)
    game = store.save("game", {**store.get("game", "g"), "duration": 3,
                             "source_fingerprint": source_fingerprint(source)})
    batch.manifest["sources"]["g"].update(game, source_size_bytes=source.stat().st_size)
    batch.items["i"].update(start=.5, end=1.5)
    queue_review_clip(store, batch, "f")
    assert work_once(store)
    ready = get_review_clip(store, batch, "f")
    assert ready["status"] == "ready"
    assert .9 <= probe_video(ready["path"])["duration"] <= 1.1
