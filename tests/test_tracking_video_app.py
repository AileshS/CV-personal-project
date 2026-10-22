"""Simple tracking product controls, durable queues and playback presentation."""
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from playlab.store import Store

APP = Path(__file__).parents[1] / "app.py"


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAYLAB_DATA", str(tmp_path / "data"))
    store = Store(tmp_path / "data")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"artificial UI source; no model execution")
    game = store.save("game", {"id": "g", "name": "Existing video", "source_path": str(source),
                              "duration": 50., "source_offset_seconds": 750., "split": "train",
                              "width": 320, "height": 180, "fps": 30.})
    calls = []
    monkeypatch.setattr("playlab.tracking_video_ui.ensure_tracking_worker", lambda store:
                        calls.append(store.root) or {"state": "running", "pid": 123, "owned": False})
    monkeypatch.setattr("playlab.tracking_video_ui.worker_status", lambda store: {"state": "running", "pid": 123})
    return store, game, calls


def app():
    at = AppTest.from_file(str(APP), default_timeout=15).run()
    assert not at.exception
    return at


def test_default_empty_screen_is_one_simple_video_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAYLAB_DATA", str(tmp_path / "data"))
    at = app()
    assert at.checkbox(key="advanced_tools").value is False
    assert not at.selectbox and not at.number_input
    assert any("Upload or open a video" in item.value for item in at.info)
    assert not at.exception
    assert not Store(tmp_path / "data").jobs()
    text = " ".join(item.value for item in at.markdown)
    for absent in ("possession", "label", "bundle", "made/missed", "unknown"):
        assert absent not in text.lower()


def test_existing_source_analyze_is_ten_seconds_without_labels(tracking):
    store, game, calls = tracking
    at = app()
    assert at.number_input(key="tracking_start_g").value == 750.
    assert not calls and not store.jobs()
    assert not at.selectbox(key="tracking_video_choice").disabled
    at.button(key="tracking_analyze").click().run()
    assert not at.exception
    request, = store.list("tracking_clip")
    assert (request["start"], request["end"], request["original_start"], request["original_end"]) == (0, 10, 750, 760)
    assert store.jobs()[0]["task"] == "track_clip"
    assert len(calls) == 1
    at.run()
    assert not at.exception and len(calls) == 1 and len(store.jobs()) == 1
    for kind in ("possession", "shot", "box", "frame", "event", "setup", "analysis", "analysis_clip"):
        assert not store.list(kind)
    assert store.get("game", game["id"]) == game


def test_last_interval_shortens_at_eof_and_different_selection_hides_old_result(tracking):
    store, _, calls = tracking
    at = app()
    at.number_input(key="tracking_start_g").set_value(796.).run()
    assert any("shortened" in item.value for item in at.caption)
    at.button(key="tracking_analyze").click().run()
    request, = store.list("tracking_clip")
    assert request["start"] == 46. and request["end"] == 50.
    assert len(calls) == 1
    at.number_input(key="tracking_start_g").set_value(750.).run()
    assert not at.get("progress")


def test_local_open_preserves_existing_offset_and_annotations(tracking):
    store, game, _ = tracking
    store.save("shot", {"id": "human", "game_id": "g", "reviewed": True})
    before = store.get("shot", "human")
    at = app()
    at.text_input(key="tracking_local_path").set_value(game["source_path"])
    at.button(key="tracking_open_local").click().run()
    assert not at.exception
    assert at.number_input(key="tracking_start_g").value == 750.
    assert store.get("game", game["id"]) == game
    assert store.get("shot", "human") == before
    assert len(store.list("game")) == 1


def test_upload_storage_is_content_addressed_and_import_preserves_provenance(tracking, tmp_path):
    from playlab.tracking_video_ui import open_tracking_source, save_video_upload

    store, game, _ = tracking
    content = Path(game["source_path"]).read_bytes()
    path = save_video_upload(store, "../../somewhere/video.mp4", content)
    assert path.parent == store.root / "uploads"
    assert save_video_upload(store, "duplicate.mp4", content) == path
    # Existing content with known source mapping is reused without a new zero offset.
    from playlab.media import source_fingerprint
    game = store.save("game", {**game, "source_fingerprint": source_fingerprint(game["source_path"])})
    assert open_tracking_source(store, path) == game
    assert open_tracking_source(store, game["source_path"]) == game
    with pytest.raises(ValueError, match="empty"):
        save_video_upload(store, "empty.mp4", b"")
    with pytest.raises(ValueError, match="Choose an MP4"):
        save_video_upload(store, "payload.py", b"invalid")
    assert not (tmp_path / "somewhere").exists()


def test_preview_decodes_one_frame_only_when_requested(tracking, monkeypatch):
    calls = []
    monkeypatch.setattr("playlab.tracking_video_ui._starting_frame", lambda *args:
                        calls.append(args[1]) or np.zeros((180, 320, 3), dtype=np.uint8))
    at = app()
    assert not calls
    at.checkbox(key="tracking_show_preview_g").check().run()
    assert not at.exception and calls == [0.]
    assert not tracking[0].jobs()


def test_preview_rejects_known_test_interval_before_any_decoding(tracking, monkeypatch):
    store, game, _ = tracking
    store.save("game", {**game, "split_mode": "game", "split": "test"})
    monkeypatch.setattr("playlab.tracking_video_ui._starting_frame", lambda *args:
                        pytest.fail("Protected test source was decoded"))
    at = app()
    at.checkbox(key="tracking_show_preview_g").check().run()
    assert not at.exception
    assert any("test" in item.value.lower() for item in at.error)
    assert not store.jobs()


def test_completed_video_is_playable_downloadable_and_stale_is_hidden(tracking, monkeypatch, tmp_path):
    import streamlit as st

    from playlab.media import make_fixture

    store, _, _ = tracking
    at = app()
    at.button(key="tracking_analyze").click().run()
    request, = store.list("tracking_clip")
    video = make_fixture(tmp_path / "tracked.mp4", duration=1)
    state = {**request, "job": {"status": "done", "progress": 1}, "fresh": True, "video_path": str(video),
             "result": {"frame_count": 30, "duration": 1.}}
    monkeypatch.setattr("playlab.tracking_video_ui.get_tracking_clip", lambda *args: state)
    videos, downloads = [], []
    original_video, original_download = st.video, st.download_button

    def capture_video(data, **kwargs):
        videos.append(data)
        return original_video(data, **kwargs)

    def capture_download(label, **kwargs):
        downloads.append((label, kwargs.get("mime")))
        return original_download(label, **kwargs)

    monkeypatch.setattr(st, "video", capture_video)
    monkeypatch.setattr(st, "download_button", capture_download)
    at.run()
    assert not at.exception
    assert videos == [str(video)]
    assert downloads == [("Download tracked video", "video/mp4")]
    assert any("detector scores" in item.value for item in at.caption)
    state["fresh"] = False
    at.run()
    assert not at.exception and len(videos) == 1 and len(downloads) == 1
    assert any("new analysis" in item.value for item in at.info)


def test_start_failure_is_actionable_and_retried_only_by_user(tracking, monkeypatch):
    calls = []

    def fail(store):
        calls.append(1)
        raise ValueError("Worker status cannot be verified; preserve its lock.")

    monkeypatch.setattr("playlab.tracking_video_ui.ensure_tracking_worker", fail)
    store, _, _ = tracking
    at = app()
    at.button(key="tracking_analyze").click().run()
    request, = store.list("tracking_clip")
    assert not at.exception and calls == [1]
    assert any("preserve its lock" in item.value for item in at.error)
    at.run()
    assert not at.exception and calls == [1]
    at.button(key=f"tracking_start_retry_{request['id']}").click().run()
    assert not at.exception and calls == [1, 1]


def test_dead_worker_after_launch_has_explicit_retry_without_rerun_spawning(tracking, monkeypatch):
    store, _, calls = tracking
    at = app()
    at.button(key="tracking_analyze").click().run()
    request, = store.list("tracking_clip")
    monkeypatch.setattr("playlab.tracking_video_ui.worker_status", lambda store: {"state": "stopped", "pid": None})
    at.run()
    assert not at.exception and len(calls) == 1
    assert any("worker stopped" in item.value for item in at.error)
    at.button(key=f"tracking_start_retry_{request['id']}").click().run()
    assert not at.exception and len(calls) == 2


def test_one_retry_requeues_interrupted_running_job_after_proven_recovery(tracking, monkeypatch):
    store, _, _ = tracking
    at = app()
    at.button(key="tracking_analyze").click().run()
    request, = store.list("tracking_clip")
    store.update_job(request["job_id"], status="running")
    monkeypatch.setattr("playlab.tracking_video_ui.worker_status", lambda store: {"state": "stopped", "pid": None})
    recovered = []

    def ensure(store):
        recovered.append(1)
        store.recover_interrupted()
        return {"state": "starting", "recovered": True}

    monkeypatch.setattr("playlab.tracking_video_ui.ensure_tracking_worker", ensure)
    at.run()
    at.button(key=f"tracking_start_retry_{request['id']}").click().run()
    assert not at.exception
    assert recovered == [1] and store.jobs()[0]["status"] == "queued"
    assert store.jobs()[0]["error"] is None


def test_retry_never_requeues_a_live_or_unverifiable_running_job(tracking, monkeypatch):
    store, _, _ = tracking
    at = app()
    at.button(key="tracking_analyze").click().run()
    request, = store.list("tracking_clip")
    store.update_job(request["job_id"], status="running")
    monkeypatch.setattr("playlab.tracking_video_ui.worker_status", lambda store: {"state": "unknown", "pid": None})

    def ensure(store):
        raise ValueError("Existing worker cannot be verified.")

    monkeypatch.setattr("playlab.tracking_video_ui.ensure_tracking_worker", ensure)
    at.run()
    at.button(key=f"tracking_start_retry_{request['id']}").click().run()
    assert not at.exception
    assert store.jobs()[0]["status"] == "running"
    assert any("cannot be verified" in item.value for item in at.error)


def test_failed_job_has_retry_and_advanced_nav_is_opt_in(tracking):
    store, _, calls = tracking
    at = app()
    at.button(key="tracking_analyze").click().run()
    request, = store.list("tracking_clip")
    store.update_job(request["job_id"], status="failed", error="Video encoder could not finish.")
    at.run()
    assert any("encoder" in item.value for item in at.error)
    at.button(key=f"tracking_retry_{request['id']}").click().run()
    assert not at.exception and store.jobs()[0]["status"] == "queued"
    assert len(calls) == 2
    at.checkbox(key="advanced_tools").check().run()
    assert not at.exception and at.selectbox(key="workspace").value == "Quick analysis"
    at.checkbox(key="advanced_tools").uncheck().run()
    assert not at.exception and at.button(key="tracking_analyze")
