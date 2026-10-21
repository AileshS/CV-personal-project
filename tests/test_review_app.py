"""Temporary artificial batches exercise review interaction, never model accuracy."""
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from playlab.labeling import prepare_label_batch
from playlab.media import source_fingerprint
from playlab.review import load_batch
from playlab.store import Store

APP = Path(__file__).parents[1] / "app.py"


@pytest.fixture
def batch_desk(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAYLAB_DATA", str(tmp_path / "data"))
    store = Store(tmp_path / "data")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"artificial video fixture; frame decoding mocked")
    store.save("game", {"id": "g", "name": "Artificial UI fixture", "source_path": str(source),
                        "duration": 120., "width": 80, "height": 60, "fps": 30.,
                        "split_mode": "game", "split": "train", "source_fingerprint": source_fingerprint(source)})
    store.save("possession", {"id": "p", "game_id": "g", "start": 0., "end": 120.,
                              "split": "train", "reviewed": False, "proposal": True})
    monkeypatch.setattr("playlab.labeling.frame_at", lambda *args, **kwargs: np.zeros((60, 80, 3), dtype=np.uint8))
    monkeypatch.setattr("playlab.media.frame_at", lambda *args, **kwargs: np.zeros((60, 80, 3), dtype=np.uint8))
    output = prepare_label_batch(store, tmp_path / "batch", frame_count=6, possession_count=2,
                                 allow_review_windows=True, window_seconds=30)
    return store, load_batch(output / "manifest.json")


def browser(batch):
    at = AppTest.from_file(str(APP), default_timeout=15).run()
    at.selectbox(key="workspace").set_value("Review batches").run()
    at.text_input(key="review_manifest_path").set_value(str(batch.root / "manifest.json"))
    at.button(key="load_review_batch").click().run()
    assert not at.exception
    return at


def test_navigation_notes_survive_restart_and_never_accept_suggestions(batch_desk):
    store, batch = batch_desk
    at = browser(batch)
    first = at.selectbox(key="review_frame").value
    at.button(key="review_next_frame").click().run()
    second = at.selectbox(key="review_frame").value
    assert second != first
    at.text_area[0].set_value("Uncertain control; retain offensive rebound")
    second_time = batch.frames[second]['time']
    next(n for n in at.number_input if n.label == "Corrected possession start (source seconds)").set_value(second_time - 2)
    next(n for n in at.number_input if n.label == "Corrected possession end (source seconds)").set_value(second_time + 2)
    next(c for c in at.checkbox if c.label == "Uncertain case remains unresolved").check()
    at.button(key="save_review_notes").click().run()
    assert not at.exception
    restarted = AppTest.from_file(str(APP), default_timeout=15).run()
    restarted.selectbox(key="workspace").set_value("Review batches").run()
    assert not restarted.exception
    assert restarted.selectbox(key="review_frame").value == second
    assert restarted.text_area[0].value == "Uncertain control; retain offensive rebound"
    assert next(n for n in restarted.number_input if n.label == "Corrected possession start (source seconds)").value == second_time - 2
    assert next(n for n in restarted.number_input if n.label == "Corrected possession end (source seconds)").value == second_time + 2
    restarted.button(key="review_next_window").click().run()
    assert not restarted.exception
    restarted.button(key="review_previous_window").click().run()
    assert not restarted.exception
    assert all(not f['reviewed'] for f in load_batch(batch.root / 'manifest.json').frames.values())
    assert not store.list("frame") and not store.list("box") and len(store.list("possession")) == 1


def test_thumbnail_only_batch_browsable_without_game(batch_desk):
    store, batch = batch_desk
    for frame in batch.frames.values():
        (batch.root / frame['image']).unlink()
    store.delete("game", "g")
    at = browser(batch)
    assert any("Browse only" in m.value for m in at.warning)
    assert at.button(key="confirm_review_possession").disabled
    at.button(key="review_next_frame").click().run()
    assert not at.exception
    assert store.list("attention")
    assert not store.list("frame") and not store.list("box")


def test_explicit_confirmation_exact_prefill_and_human_save(batch_desk):
    store, batch = batch_desk
    at = browser(batch)
    frame_id = at.selectbox(key="review_frame").value
    at.button(key="confirm_review_possession").click().run()
    assert any("Explicitly review" in e.value for e in at.error)
    assert len(store.list("possession")) == 1
    selected_time = batch.frames[frame_id]['time']
    next(n for n in at.number_input if n.label == "Corrected possession start (source seconds)").set_value(selected_time - 1)
    next(n for n in at.number_input if n.label == "Corrected possession end (source seconds)").set_value(selected_time + 1)
    next(c for c in at.checkbox if c.label.startswith("I reviewed the whole possession")).check()
    at.button(key="confirm_review_possession").click().run()
    assert not at.exception
    at.button(key="open_review_frame").click().run()
    assert not at.exception
    assert at.selectbox(key="workspace").value == "Frame labels"
    target_id = at.selectbox(key="possession_choice_g").value
    assert target_id != "p"
    time_input = next(n for n in at.number_input if n.label == "Frame time (game seconds)")
    assert time_input.value == batch.frames[frame_id]['time']
    assert not store.list("box") and not store.list("frame")
    at.button(key="save_box").click().run()
    assert not at.exception
    assert store.list("box")[0]['time'] == batch.frames[frame_id]['time']
    at.button(key="review_frame").click().run()
    assert not at.exception
    assert store.list("frame")[0]['reviewed']
    assert store.get("possession", target_id)['end'] - store.get("possession", target_id)['start'] == 2
    restarted = AppTest.from_file(str(APP), default_timeout=15).run()
    assert not restarted.exception
    assert Store(store.root).list("frame")[0]['reviewed']
    assert not store.jobs()


def test_stale_frame_editor_cannot_overwrite_concurrent_labels(batch_desk):
    store, _ = batch_desk
    at = AppTest.from_file(str(APP), default_timeout=15).run()
    at.selectbox(key="workspace").set_value("Frame labels").run()
    # Another reviewer adds a box after this editor was rendered.
    store.save("box", {"id": "other", "game_id": "g", "possession_id": "p", "time": 0.,
                       "label": "rim", "box": [.1, .1, .2, .2], "split": "train", "reviewed": True})
    at.button(key="save_box").click().run()
    assert not at.exception
    assert any("changed elsewhere" in e.value for e in at.error)
    assert [b['id'] for b in store.list("box")] == ['other']


def test_stale_correction_notes_require_reload(batch_desk):
    store, batch = batch_desk
    first = browser(batch)
    second = browser(batch)
    first.text_area[0].set_value("Reviewer one correction")
    first.button(key="save_review_notes").click().run()
    second.text_area[0].set_value("Stale reviewer correction")
    second.button(key="save_review_notes").click().run()
    assert not second.exception
    assert any("changed elsewhere" in e.value for e in second.error)
    assert store.list("review_frame")[0]['notes'] == "Reviewer one correction"


def test_changed_possession_blocks_stale_frame_edit(batch_desk):
    store, _ = batch_desk
    at = AppTest.from_file(str(APP), default_timeout=15).run()
    at.selectbox(key="workspace").set_value("Frame labels").run()
    possession = store.get("possession", "p")
    store.save("possession", {**possession, "end": 119.})
    at.button(key="save_box").click().run()
    assert not at.exception
    assert any("changed elsewhere" in e.value for e in at.error)
    assert not store.list("box")


def test_concurrent_uncertainty_draft_blocks_stale_confirmation(batch_desk):
    store, batch = batch_desk
    first = browser(batch)
    stale = browser(batch)
    next(c for c in first.checkbox if c.label == "Uncertain case remains unresolved").check()
    first.text_area[0].set_value("Control uncertain; do not label yet")
    first.button(key="save_review_notes").click().run()
    next(c for c in stale.checkbox if c.label.startswith("I reviewed the whole possession")).check()
    stale.button(key="confirm_review_possession").click().run()
    assert not stale.exception
    assert any("changed elsewhere" in e.value for e in stale.error)
    assert len(store.list("possession")) == 1
    assert not store.list("review_frame")[0].get("possession_id")
    assert not store.list("frame") and not store.list("box")


def test_review_clip_queue_uses_bounded_window_not_parent(batch_desk):
    store, batch = batch_desk
    at = browser(batch)
    frame = batch.frames[at.selectbox(key="review_frame").value]
    item = batch.items[frame['review_item_id']]
    at.button(key="queue_review_clip").click().run()
    assert not at.exception
    assert len(store.jobs()) == 1
    job = store.jobs()[0]
    assert job['task'] == 'review_clip'
    assert (job['payload']['start'], job['payload']['end']) == (item['start'], item['end'])
    assert job['payload']['end'] - job['payload']['start'] == 30
    assert not store.list("frame") and not store.list("box")


def test_linked_boundary_edit_route_reload_requires_reconfirmation(batch_desk):
    store, batch = batch_desk
    at = browser(batch)
    next(c for c in at.checkbox if c.label.startswith("I reviewed the whole possession")).check()
    at.button(key="confirm_review_possession").click().run()
    at.button(key="edit_review_target").click().run()
    assert not at.exception
    assert at.selectbox(key="workspace").value == "Possessions"
    target_id = at.selectbox(key="possession_choice_g").value
    target = store.get("possession", target_id)
    at.number_input(key=f"edit_end_{target_id}_{target['revision']}").set_value(target['end'] + 1)
    at.button(key="save_possession").click().run()
    assert not at.exception
    at.selectbox(key="workspace").set_value("Review batches").run()
    assert not any(b.key == "open_review_frame" for b in at.button)
    at.button(key="reload_review").click().run()
    assert next(n for n in at.number_input if n.label == "Corrected possession end (source seconds)").value == target['end'] + 1
    confirmation = next(c for c in at.checkbox if c.label.startswith("I reviewed the whole possession"))
    assert not confirmation.value
    confirmation.check()
    at.button(key="confirm_review_possession").click().run()
    assert not at.exception
    assert not at.error
    assert at.button(key="open_review_frame")
    assert not store.list("box") and not store.list("frame")
