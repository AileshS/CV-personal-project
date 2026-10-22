import os

import pytest

from playlab.store import Store
from playlab.worker import _lifecycle_guard, _process_alive, recover_worker, run_worker, worker_status


def test_status_reports_missing_and_dead_worker(tmp_path, monkeypatch):
    store = Store(tmp_path)
    assert worker_status(store) == {"state": "stopped", "pid": None}
    (tmp_path / ".worker.lock").write_text("123")
    monkeypatch.setattr("playlab.worker._process_alive", lambda pid: False)
    assert worker_status(store) == {"state": "stopped", "pid": 123}


@pytest.mark.parametrize("alive", [True, None])
def test_recovery_preserves_potentially_live_worker_and_running_job(tmp_path, monkeypatch, alive):
    store = Store(tmp_path)
    store.enqueue("index", {"game_id": "g"})
    store.claim_job()
    lock = tmp_path / ".worker.lock"
    lock.write_text("123")
    monkeypatch.setattr("playlab.worker._process_alive", lambda pid: alive)
    with pytest.raises(ValueError, match="Stop the worker"):
        recover_worker(store)
    assert lock.read_text() == "123"
    assert store.jobs()[0]["status"] == "running"


def test_explicit_dead_worker_recovery_keeps_queue_and_marks_interrupted(tmp_path, monkeypatch):
    store = Store(tmp_path)
    running = store.enqueue("index", {"game_id": "g"})
    store.claim_job()
    queued = store.enqueue("index", {"game_id": "g2"})
    (tmp_path / ".worker.lock").write_text("123")
    monkeypatch.setattr("playlab.worker._process_alive", lambda pid: False)
    recover_worker(store)
    statuses = {job["id"]: job["status"] for job in store.jobs()}
    assert statuses == {running: "failed", queued: "queued"}
    assert not (tmp_path / ".worker.lock").exists()


def test_worker_removes_lock_after_keyboard_interrupt(tmp_path, monkeypatch):
    store = Store(tmp_path)
    def interrupt(_store):
        raise KeyboardInterrupt
    monkeypatch.setattr("playlab.worker.work_once", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_worker(store)
    assert not (tmp_path / ".worker.lock").exists()


def test_current_process_is_alive():
    assert _process_alive(os.getpid()) is True


def test_malformed_lock_stays_unknown_and_cannot_be_recovered(tmp_path):
    store = Store(tmp_path)
    (tmp_path / ".worker.lock").write_text("unfinished")
    assert worker_status(store)["state"] == "unknown"
    with pytest.raises(ValueError, match="cannot be verified"):
        recover_worker(store)


def test_recovery_and_startup_share_exclusive_guard(tmp_path):
    store = Store(tmp_path)
    store.enqueue("index", {"game_id": "g"})
    store.claim_job()
    with _lifecycle_guard(store):
        with pytest.raises(ValueError, match="already in progress"):
            recover_worker(store)
        with pytest.raises(ValueError, match="already in progress"):
            run_worker(store, once=True)
        assert store.jobs()[0]["status"] == "running"
        assert not (tmp_path / ".worker.lock").exists()
    recover_worker(store)
    assert store.jobs()[0]["status"] == "failed"


def test_worker_cannot_start_during_recovery_status_check(tmp_path, monkeypatch):
    store = Store(tmp_path)
    def checked_status(_store):
        with pytest.raises(ValueError, match="already in progress"):
            run_worker(store, once=True)
        return {"state": "stopped", "pid": None}
    monkeypatch.setattr("playlab.worker.worker_status", checked_status)
    recover_worker(store)
    assert not (tmp_path / ".worker.lock").exists()
