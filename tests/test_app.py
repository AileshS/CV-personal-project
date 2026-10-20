"""Review-desk workflows: real SQLite persistence, no model or network downloads."""
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from playlab.store import Store

APP = Path(__file__).parents[1] / "app.py"


@pytest.fixture
def desk(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAYLAB_DATA", str(tmp_path / "data"))
    store = Store(tmp_path / "data")
    source = tmp_path / "game.mp4"
    source.write_bytes(b"synthetic fixture; decoding is mocked")
    store.save("game", {"id": "game1", "name": "Review fixture", "source_path": str(source),
                        "duration": 120.0, "width": 320, "height": 180, "fps": 30.0})
    store.save("possession", {"id": "pos1", "game_id": "game1", "start": 10.0, "end": 30.0,
                              "reviewed": False, "split": "unassigned"})
    import playlab.media
    monkeypatch.setattr(playlab.media, "frame_at", lambda *args, **kwargs: np.zeros((180, 320, 3), dtype=np.uint8))
    return store


def app():
    at = AppTest.from_file(str(APP), default_timeout=15).run()
    assert not at.exception
    return at


def by_label(widgets, label):
    return next(w for w in widgets if w.label == label)


def test_empty_desk_has_onboarding_without_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAYLAB_DATA", str(tmp_path))
    at = app()
    assert any("Start with Import game" in message.value for message in at.info)
    assert not Store(tmp_path).jobs()


def test_import_is_queued_only_on_submit(desk):
    at = app()
    at.selectbox(key="workspace").set_value("Import game").run()
    at.text_input(key="import_source").set_value("https://www.youtube.com/watch?v=fixture")
    at.text_input(key="import_name").set_value("A full game")
    at.button(key="queue_import").click().run()
    assert not at.exception
    assert len(desk.jobs()) == 1
    assert desk.jobs()[0]["payload"]["name"] == "A full game"
    at.run()
    assert len(desk.jobs()) == 1
    assert not desk.list("analysis")


def test_possession_review_and_split_assignment_persist(desk):
    at = app()
    at.number_input(key="edit_start_pos1_1").set_value(11.0)
    at.number_input(key="edit_end_pos1_1").set_value(29.0)
    at.selectbox(key="split_pos1_1").set_value("train")
    at.checkbox(key="checked_pos1_1").check()
    at.button(key="save_possession").click().run()
    assert not at.exception
    possession = desk.get("possession", "pos1")
    assert (possession["start"], possession["end"], possession["split"], possession["reviewed"]) == (11, 29, "train", True)


def test_human_shot_does_not_overwrite_model_prediction(desk):
    desk.save("analysis", {"id": "auto", "possession_id": "pos1", "events": [{"release": 20.0, "result": "missed"}]})
    at = app()
    by_label(at.number_input, "Release time (game seconds)").set_value(20.0)
    by_label(at.selectbox, "Shot result").set_value("made")
    by_label(at.text_input, "Shooter track ID (blank if unclear)").set_value("7")
    by_label(at.text_input, "Closest opponent track ID (blank if unclear)").set_value("4")
    at.checkbox(key="confirm_shot_pos1_1_new").check()
    at.button(key="save_shot").click().run()
    assert not at.exception
    shot = desk.list("shot")[0]
    assert shot["result"] == "made" and shot["defender_id"] == "4" and shot["reviewed"]
    assert shot["split"] == "unassigned"
    assert desk.get("analysis", "auto")["events"][0]["result"] == "missed"


def test_unconfirmed_shot_is_not_saved(desk):
    at = app()
    at.button(key="save_shot").click().run()
    assert not at.exception
    assert any("Review the attempt" in e.value for e in at.error)
    assert desk.list("shot") == []


def test_reviewed_negative_frame_is_explicit(desk):
    at = app()
    at.selectbox(key="workspace").set_value("Frame labels").run()
    at.button(key="negative_frame").click().run()
    assert not at.exception
    assert desk.list("frame")[0]["negative"] is True
    assert desk.list("frame")[0]["reviewed"] is True
    assert desk.list("frame")[0]["split"] == "unassigned"
    assert desk.list("box") == []


def test_invalid_box_cannot_become_training_label(desk):
    at = app()
    at.selectbox(key="workspace").set_value("Frame labels").run()
    at.number_input(key="box_pos1:10.000_new_0").set_value(0.8)
    at.number_input(key="box_pos1:10.000_new_2").set_value(0.2)
    at.button(key="save_box").click().run()
    assert not at.exception
    assert at.error
    assert desk.list("box") == []


def test_saved_box_requires_complete_frame_review_and_prevents_negative(desk):
    at = app()
    at.selectbox(key="workspace").set_value("Frame labels").run()
    at.button(key="save_box").click().run()
    assert not at.exception
    assert len(desk.list("box")) == 1
    assert desk.list("frame") == []
    assert at.button(key="negative_frame").disabled
    at.button(key="review_frame").click().run()
    assert not at.exception
    assert desk.list("frame")[0]["reviewed"]
    assert desk.list("frame")[0]["negative"] is False


def test_cancel_and_resume_job_from_review_desk(desk):
    job = desk.enqueue("index", {"game_id": "game1"})
    at = app()
    at.selectbox(key="workspace").set_value("Jobs and export").run()
    at.button(key=f"cancel_{job}").click().run()
    assert not at.exception
    assert desk.jobs()[0]["status"] == "cancelled"
    at.button(key=f"resume_{job}").click().run()
    assert not at.exception
    assert desk.jobs()[0]["status"] == "queued"


def test_manual_foul_is_saved_without_automatic_claim(desk):
    at = app()
    at.checkbox(key="event_reviewed_pos1_1").check()
    at.button(key="save_event").click().run()
    assert not at.exception
    event = desk.list("event")[0]
    assert event["type"] == "foul" and event["reviewed"]
    assert desk.list("analysis") == []


def test_whole_game_split_moves_existing_labels_explicitly(desk):
    desk.save("event", {"id": "foul1", "game_id": "game1", "possession_id": "pos1",
                         "time": 20.0, "type": "foul", "reviewed": True, "split": "unassigned"})
    at = app()
    at.selectbox(key="workspace").set_value("Game setup").run()
    at.selectbox(key="whole_game_split").set_value("val")
    at.checkbox(key="confirm_game_split").check()
    at.button(key="assign_game_split").click().run()
    assert not at.exception
    assert desk.get("game", "game1")["split"] == "val"
    assert desk.get("possession", "pos1")["split"] == "val"
    assert desk.get("event", "foul1")["split"] == "val"


def test_pilot_blocks_guard_boundaries_and_preserve_unassigned_gaps(desk):
    desk.save("possession", {"id": "near_cut", "game_id": "game1", "start": 60.0, "end": 80.0,
                              "reviewed": False, "split": "unassigned"})
    at = app()
    at.selectbox(key="workspace").set_value("Game setup").run()
    at.selectbox(key="dataset_strategy_game1").set_value("Same-game pilot").run()
    at.number_input(key="pilot_train_end").set_value(70.0)
    at.number_input(key="pilot_val_end").set_value(100.0)
    at.button(key="save_pilot_blocks").click().run()
    assert not at.exception
    assert desk.get("game", "game1")["split_mode"] == "contiguous"
    assert desk.get("possession", "pos1")["split"] == "train"
    assert desk.get("possession", "near_cut")["split"] == "unassigned"


def test_negative_review_can_be_removed_before_positive_annotation(desk):
    at = app()
    at.selectbox(key="workspace").set_value("Frame labels").run()
    at.button(key="negative_frame").click().run()
    at.button(key="remove_frame_review").click().run()
    at.button(key="save_box").click().run()
    assert not at.exception
    assert len(desk.list("box")) == 1
    assert desk.list("frame") == []


def test_attention_requests_show_delivery_and_can_be_resolved(desk):
    desk.save("attention", {"id": "manual-review", "message": "Review 12 training frames.", "resolved": False,
                             "notification": {"delivered": False, "reason": "Phone notifications are not configured"}})
    at = app()
    at.selectbox(key="workspace").set_value("Jobs and export").run()
    assert any("Review 12 training frames" in m.value for m in at.markdown)
    assert any("Phone notifications are not configured" in c.value for c in at.caption)
    at.button(key="resolve_attention_manual-review").click().run()
    assert not at.exception
    assert desk.get("attention", "manual-review")["resolved"] is True
    assert any("No unresolved action requests" in info.value for info in at.info)


def test_alert_configuration_never_displays_secret_topic(desk, monkeypatch):
    secret_topic = "https://ntfy.example.test/secret-basketball-topic"
    monkeypatch.setenv("PLAYLAB_NTFY_URL", secret_topic)
    desk.save("attention", {"id": "accepted", "message": "Review this clip.", "resolved": False,
                             "notification": {"delivered": True}})
    at = app()
    at.selectbox(key="workspace").set_value("Jobs and export").run()
    text = " ".join(element.value for elements in [at.caption, at.markdown, at.info, at.warning]
                    for element in elements)
    assert "A phone alert endpoint is set" in text
    assert "Receipt on your phone is not verified" in text
    assert secret_topic not in text
    assert "secret-basketball-topic" not in text


def test_selected_trained_models_are_passed_to_worker(desk):
    at = app()
    at.text_input(key="model_weights").set_value("models/ball-rim.pt")
    at.text_input(key="classifier_bundle").set_value("models/shots.joblib")
    at.button(key="queue_analysis").click().run()
    assert not at.exception
    payload = desk.jobs()[0]["payload"]
    assert payload["weights"] == "models/ball-rim.pt"
    assert payload["model_bundle"] == "models/shots.joblib"


def test_automatic_release_mode_is_queued_explicitly(desk):
    at = app()
    at.selectbox(key="analysis_mode").set_value("automatic")
    at.button(key="queue_analysis").click().run()
    assert not at.exception
    assert desk.jobs()[0]["payload"]["mode"] == "automatic"
