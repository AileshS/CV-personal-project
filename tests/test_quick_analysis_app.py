"""Product flow on artificial metadata; no model execution or accuracy claims."""
import hashlib
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from playlab.quick_import import import_quick_source
from playlab.store import Store

APP = Path(__file__).parents[1] / "app.py"
LABELING = Path(__file__).parents[1] / "labeling_app.py"


@pytest.fixture
def quick(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAYLAB_DATA", str(tmp_path / "data"))
    source = tmp_path / "source.mp4"
    source.write_bytes(b"artificial source; never decoded")
    store = Store(tmp_path / "data")
    game = store.save("game", {"id": "g", "name": "Clip", "source_path": str(source),
                               "duration": 50., "source_offset_seconds": 750., "split": "train",
                               "width": 320, "height": 180, "fps": 30.})
    # Simulate a background worker claiming jobs; in-process runs are tested explicitly.
    monkeypatch.setattr("playlab.quick_analysis_ui.run_job_now", lambda *a, **kw: False)
    return store, game


def browser(app=APP):
    at = AppTest.from_file(str(app), default_timeout=15).run()
    assert not at.exception
    return at


def test_empty_workspace_starts_at_quick_import(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAYLAB_DATA", str(tmp_path))
    at = browser()
    assert not at.toggle and not at.sidebar.selectbox  # Quick analysis is the whole site
    assert not [s for s in at.selectbox if s.key == "workspace"]
    assert any("Open a local video to choose" in message.value for message in at.info)
    assert not Store(tmp_path).jobs()


def test_open_new_local_video_from_empty_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAYLAB_DATA", str(tmp_path / "data"))
    source = tmp_path / "new.mp4"
    source.write_bytes(b"metadata fixture")
    monkeypatch.setattr("playlab.quick_import.probe_video", lambda path: {"duration": 50.})
    monkeypatch.setattr("playlab.quick_analysis_ui.run_job_now", lambda *a, **kw: False)
    at = browser(LABELING)  # offset/provenance fields live in the labeling app
    at.toggle(key="advanced").set_value(True).run()
    at.text_input(key="quick_source").set_value(str(source))
    at.text_input(key="quick_offset").set_value("750")
    at.button(key="quick_open").click().run()
    assert not at.exception
    store = Store(tmp_path / "data")
    game = store.list("game")[0]
    assert at.number_input(key=f"quick_start_{game['id']}").value == 750.
    assert not store.jobs()
    at.button(key="quick_analyze").click().run()
    assert not at.exception
    assert len(store.jobs()) == 1
    assert not store.list("possession")


def test_zero_labels_original_interval_queues_without_inference(quick, monkeypatch):
    store, game = quick
    monkeypatch.setattr("playlab.analysis.analyze_possession", lambda *a, **kw: pytest.fail("UI ran inference"))
    at = browser()
    assert at.number_input(key="quick_start_g").value == 750.
    assert at.number_input(key="quick_duration_g").value == 20.
    assert not store.jobs()
    at.number_input(key="quick_start_g").set_value(790.).run()
    at.button(key="quick_analyze").click().run()
    assert not at.exception
    clip = store.list("analysis_clip")[0]
    assert (clip["start"], clip["end"], clip["original_start"], clip["original_end"]) == (40., 50., 790., 800.)
    assert len(store.jobs()) == 1
    at.run()
    assert len(store.jobs()) == 1
    for kind in ("possession", "shot", "box", "frame", "setup", "analysis"):
        assert store.list(kind) == []
    assert store.get("game", game["id"])["split"] == "train"


def test_failure_cancel_resume_visible(quick):
    store, _ = quick
    at = browser()
    at.button(key="quick_analyze").click().run()
    at.button(key="quick_cancel").click().run()
    assert not at.exception and store.jobs()[0]["status"] == "cancelled"
    at.button(key="quick_resume").click().run()
    job = store.jobs()[0]
    assert job["status"] == "queued"
    store.update_job(job["id"], status="failed", error="Detector weights unavailable")
    at.button(key="quick_refresh").click().run()
    assert any("Detector weights unavailable" in message.value for message in at.error)
    assert any("No result is available" in message.value for message in at.info)


def test_individual_results_evidence_and_stale_suppression(quick, monkeypatch, tmp_path):
    import streamlit as st
    from PIL import Image

    images = []
    captions = []
    original_image = st.image

    def capture_image(image, **kwargs):
        images.append(image)
        captions.append(kwargs.get("caption", ""))
        return original_image(image, **kwargs)

    monkeypatch.setattr(st, "image", capture_image)
    store, _ = quick
    at = browser()
    at.button(key="quick_analyze").click().run()
    image = tmp_path / "evidence.png"
    Image.new("RGB", (10, 10)).save(image)
    result = {"fresh": True, "job": {"status": "done", "progress": 1.},
              "result": {"events": [{"result": "unknown", "release": 760., "reason": "Rim is unresolved",
                                      "evidence_path": str(image)},
                                     {"result": "missed", "release": 768.}], "warnings": []}}
    monkeypatch.setattr("playlab.quick_analysis_ui.get_analysis_clip", lambda *args: result)
    at.run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Attempt 1 · unknown" in text and "Attempt 2 · missed" in text
    assert "0:12:40.00" in text and "Rim is unresolved" in text
    assert str(image) in images
    result["result"] = {"events": [], "context_evidence_path": str(image), "context_time": 750.,
                        "warnings": ["Pretrained COCO has no rim class"]}
    at.run()
    assert not at.exception
    assert any("no dependable shot attempts" in m.value for m in at.info)
    assert any("no confirmed release" in caption for caption in captions)
    result["fresh"] = False
    at.run()
    assert any("stale" in warning.value for warning in at.warning)
    assert not any("Attempt 1" in m.value for m in at.markdown)


def test_safe_import_provenance_idempotence_and_conflict(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"metadata fixture")
    monkeypatch.setattr("playlab.quick_import.probe_video", lambda p: {"duration": 50.})
    store = Store(tmp_path / "data")
    provenance = {"clip_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "split": "train",
                  "original_source_start_seconds": 750., "original_source_end_seconds": 800.,
                  "clip_duration_seconds": 50.}
    manifest = tmp_path / "provenance.json"
    manifest.write_text(json.dumps(provenance))
    game = import_quick_source(store, source, provenance_path=manifest)
    store.save("shot", {"id": "gold", "game_id": game["id"], "reviewed": True})
    assert import_quick_source(store, source, provenance_path=manifest) == game
    with pytest.raises(ValueError, match="different metadata"):
        import_quick_source(store, source, offset=0)
    assert store.get("game", game["id"]) == game
    assert store.get("shot", "gold")["reviewed"]
    with pytest.raises(ValueError, match="finite"):
        import_quick_source(store, source, offset=float("nan"))
    provenance["clip_sha256"] = "0" * 64
    manifest.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="SHA-256"):
        import_quick_source(store, source, provenance_path=manifest)


def test_frame_inspector_old_result_supports_single_frame_without_reanalysis(quick, monkeypatch):
    import numpy as np

    import playlab.detection_inspector_ui as ui
    store, _ = quick
    at = browser(LABELING)  # the frame inspector is a labeling-app tool
    at.button(key="quick_analyze").click().run()
    clip = store.list("analysis_clip")[0]
    monkeypatch.setattr(ui, "_native_frame", lambda *a: np.zeros((180, 320, 3), dtype=np.uint8))
    calls = []
    def predict(*args):
        calls.append(args[2])
        return {"time": args[2], "width": 320, "height": 180, "detector_mode": "pretrained_coco_ball",
                "raw_detections": {"balls": [{"bbox": [10, 20, 30, 40], "confidence": .75}], "rims": []}}
    monkeypatch.setattr(ui, "_predict_frame", predict)
    key = f"inspect_{clip['id']}"
    at.toggle(key="advanced").set_value(True).run()
    at.checkbox(key=key+"_show").check().run()
    assert not at.exception
    assert calls == [0.]
    at.button(key=key+"_detect").click().run()
    assert not at.exception and calls[-1] == 0.
    assert any("unsupported" in m.value for m in at.info)
    at.button(key=key+"_next").click().run()
    assert not at.exception and calls[-1] == pytest.approx(1 / 30)
    assert at.slider(key=key+"_frame").value == 1
    assert any("unsupported" in m.value for m in at.info)
    assert len(store.jobs()) == 1
    assert store.list("box") == [] and store.list("frame") == []


def test_analyze_runs_in_process_and_shows_results(quick, monkeypatch):
    store, _ = quick
    import playlab.quick_analysis as quick_analysis
    from playlab import worker

    monkeypatch.setattr("playlab.quick_analysis_ui.run_job_now", worker.run_job_now)
    calls = []

    def fake_perform(store, payload, cancelled=None, progress=None):
        calls.append(payload["clip_id"])
        progress(.5, {"stage": "fixture"})
        return {}

    monkeypatch.setattr(quick_analysis, "perform_analysis_clip", fake_perform)
    at = browser()
    at.button(key="quick_analyze").click().run()
    assert not at.exception
    job = store.jobs()[0]
    assert calls == [store.list("analysis_clip")[0]["id"]]
    assert job["status"] == "done" and job["progress"] == 1


def test_interrupted_in_process_run_stays_resumable(quick):
    store, _ = quick
    from playlab import quick_analysis, worker

    clip = quick_analysis.queue_analysis_clip(store, "g", 0, 10)

    class PageLeft(BaseException):
        pass

    def leave(fraction, checkpoint=None):
        raise PageLeft

    original = worker.perform
    try:
        worker.perform = lambda store, job, on_progress=None: on_progress(.2, {})
        with pytest.raises(PageLeft):
            worker.run_job_now(store, clip["job_id"], on_progress=leave)
    finally:
        worker.perform = original
    job = store.jobs()[0]
    assert job["status"] == "cancelled" and "resume" in job["error"]
    assert not worker.run_job_now(store, clip["job_id"])  # only queued jobs are claimed
    store.resume_job(job["id"])
    assert store.jobs()[0]["status"] == "queued"
