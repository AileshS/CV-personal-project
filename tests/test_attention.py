"""Notification failures must preserve human requests and worker progress."""
import pytest

from playlab.attention import request_attention
from playlab.store import Store


@pytest.mark.parametrize("topic", ["https://[invalid/topic", "not-a-url", "http://example.test/topic",
                                  "https://example.test:bad/topic", "https://example.test:70000/topic"])
def test_malformed_topic_never_loses_attention_request(tmp_path, monkeypatch, topic):
    monkeypatch.setenv("PLAYLAB_NTFY_URL", topic)
    monkeypatch.setenv("PLAYLAB_NTFY_TOKEN", "test-token-not-sent")
    store = Store(tmp_path)
    request = request_attention(store, "labels", "Review the pending label batch.")
    assert request["resolved"] is False
    assert request["notification"]["delivered"] is False
    assert len(store.list("attention")) == 1
    assert request_attention(store, "labels", "Review the pending label batch.")["id"] == request["id"]
    assert len(store.list("attention")) == 1


def test_invalid_port_never_stops_worker_failure_handling(tmp_path, monkeypatch):
    from playlab.worker import work_once

    monkeypatch.setenv("PLAYLAB_NTFY_URL", "https://example.test:bad/topic")
    store = Store(tmp_path)
    store.enqueue("unsupported-a", {})
    store.enqueue("unsupported-b", {})
    assert work_once(store)
    assert work_once(store)
    assert all(job["status"] == "failed" for job in store.jobs())
    assert len(store.list("attention")) == 2
    assert all(not item["notification"]["delivered"] for item in store.list("attention"))
