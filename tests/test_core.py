from concurrent.futures import ThreadPoolExecutor

import pytest

from playlab.domain import (
    assign_game_split,
    assign_split_blocks,
    merge_possessions,
    save_possession,
    split_possession,
    validate_annotation,
)
from playlab.store import Store
from playlab.worker import work_once


def game(store, **extra):
    return store.save("game", {"id": "g", "duration": 200, "name": "Test", **extra})


def test_concurrent_queue_deduplicates_and_claims_once(tmp_path):
    store = Store(tmp_path)
    with ThreadPoolExecutor(8) as pool:
        ids = list(pool.map(lambda _: store.enqueue("index", {"game_id": "g"}), range(20)))
    assert len(set(ids)) == 1
    with ThreadPoolExecutor(8) as pool:
        claims = list(pool.map(lambda _: store.claim_job(), range(8)))
    assert sum(c is not None for c in claims) == 1


def test_revision_conflict_preserves_last_saved_value(tmp_path):
    store = Store(tmp_path)
    old = game(store)
    store.save("game", {**old, "name": "New"})
    with pytest.raises(ValueError, match="changed elsewhere"):
        store.save("game", {**old, "name": "Stale"}, expected_revision=old["revision"])
    assert store.get("game", "g")["name"] == "New"


def test_split_merge_moves_human_records_and_invalidates_analysis(tmp_path):
    store = Store(tmp_path)
    game(store)
    p = save_possession(store, "g", 10, 40, split="train")
    store.save("analysis", {"id": p["id"], "events": [{"result": "missed"}]})
    shot = store.save("shot", {"game_id": "g", "possession_id": p["id"], "release": 30,
                              "result": "made", "reviewed": True, "split": "train"})
    left, right = split_possession(store, p["id"], 25)
    assert store.get("shot", shot["id"])["possession_id"] == right["id"]
    assert store.get("analysis", p["id"])["stale"]
    merged = merge_possessions(store, [left["id"], right["id"]])
    assert (merged["start"], merged["end"]) == (10, 40)
    assert store.get("shot", shot["id"])["possession_id"] == merged["id"]


def test_split_assignment_and_guard_gaps(tmp_path):
    store = Store(tmp_path)
    game(store)
    assign_split_blocks(store, "g", [{"start": 0, "end": 100, "split": "train"},
                                    {"start": 100, "end": 200, "split": "test"}])
    assert save_possession(store, "g", 50, 80)["split"] == "train"
    assert save_possession(store, "g", 95, 105)["split"] == "unassigned"
    assign_game_split(store, "g", "val")
    assert store.get("game", "g")["split_mode"] == "game"
    assert store.get("game", "g")["split_blocks"] == []
    assert save_possession(store, "g", 115, 125)["split"] == "val"


def test_labels_cannot_fall_outside_possession_or_conflict_with_negative(tmp_path):
    store = Store(tmp_path)
    game(store)
    p = save_possession(store, "g", 10, 40)
    base = {"game_id": "g", "possession_id": p["id"], "reviewed": True}
    with pytest.raises(ValueError, match="inside"):
        validate_annotation(store, "shot", {**base, "release": 41, "result": "made"})
    store.save("frame", validate_annotation(store, "frame", {**base, "time": 20, "negative": True}))
    with pytest.raises(ValueError, match="negative"):
        validate_annotation(store, "box", {**base, "time": 20, "label": "ball", "box": [.1,.1,.2,.2]})


def test_failure_is_durable_and_creates_one_attention_request(tmp_path, monkeypatch):
    monkeypatch.delenv("PLAYLAB_NTFY_URL", raising=False)
    store = Store(tmp_path)
    job = store.enqueue("unsupported", {})
    assert work_once(store)
    assert store.jobs()[0]["status"] == "failed"
    assert len(store.list("attention")) == 1
    assert not store.list("attention")[0]["notification"]["delivered"]
    store.resume_job(job)
    work_once(store)
    assert len(store.list("attention")) == 1


def test_running_jobs_require_explicit_recovery(tmp_path):
    store = Store(tmp_path)
    job = store.enqueue("index", {"game_id": "g"})
    store.claim_job()
    with pytest.raises(ValueError, match="Stop the worker"):
        store.resume_job(job)
    store.recover_interrupted()
    store.resume_job(job)
    assert store.jobs()[0]["status"] == "queued"


def test_invalid_phone_setup_never_breaks_jobs_or_loses_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAYLAB_NTFY_URL", "http://invalid")
    store = Store(tmp_path)
    store.enqueue("unsupported", {})
    assert work_once(store)
    assert store.jobs()[0]["status"] == "failed"
    request = store.list("attention")[0]
    assert not request["notification"]["delivered"]
    assert "invalid" in request["notification"]["reason"]
