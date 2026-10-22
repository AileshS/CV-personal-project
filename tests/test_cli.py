from playlab.cli import execute, parser
from playlab.store import Store


def test_queued_analysis_keeps_automatic_mode_for_worker(tmp_path):
    args = parser().parse_args(["--data", str(tmp_path), "analyze", "p", "--mode", "automatic", "--queue"])
    result = execute(args)
    job, = Store(tmp_path).jobs()
    assert job["id"] == result["job_id"]
    assert job["payload"]["mode"] == "automatic"


def test_worker_forwards_requested_release_mode(tmp_path, monkeypatch):
    from playlab.worker import work_once

    calls = []
    monkeypatch.setattr("playlab.analysis.analyze_possession", lambda store, possession_id, **kwargs:
                        calls.append((possession_id, kwargs["mode"])) or {"id": "analysis"})
    monkeypatch.delenv("PLAYLAB_NTFY_URL", raising=False)
    store = Store(tmp_path)
    store.enqueue("analyze", {"possession_id": "p", "mode": "automatic"})
    assert work_once(store)
    assert calls == [("p", "automatic")]
    assert store.jobs()[0]["status"] == "done"
