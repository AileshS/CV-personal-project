import json

import numpy as np
import pytest

from playlab.analysis import analysis_dependencies
from playlab.dataset import canonical_hash, export_dataset, validate_splits, verify_dataset
from playlab.store import Store


def game(store, identifier="g", **extra):
    return store.save("game", {"id": identifier, "source_path": "not-bundled.mp4", "duration": 200,
                               "width": 80, "height": 60, "fps": 30, **extra})


def box(identifier="b", game_id="g", split="train", time=15, **extra):
    return {"id": identifier, "game_id": game_id, "possession_id": f"p-{game_id}", "time": time,
            "label": "ball", "box": [0.1, 0.2, 0.2, 0.3], "reviewed": True, "split": split, **extra}


def frame_review(store, time=15):
    store.save("frame", {"id": f"frame-{time}", "game_id": "g", "possession_id": "p-g", "time": time,
                         "negative": False, "reviewed": True, "split": "train"})


def test_export_only_reviewed_includes_explicit_negatives_and_verifies_hash(tmp_path, monkeypatch):
    monkeypatch.setattr("playlab.media.frame_at", lambda _path, time, max_width=None:
                        np.full((60, 80, 3), int(time), dtype=np.uint8))
    store = Store(tmp_path / "db")
    game(store)
    store.save("box", box())
    frame_review(store)
    store.save("box", box(identifier="unreviewed", time=40, reviewed=False))
    store.save("frame", {"id": "negative", "game_id": "g", "possession_id": "p-g", "time": 30,
                         "negative": True, "reviewed": True, "split": "train"})
    destination = export_dataset(store, tmp_path / "version1")
    manifest = verify_dataset(destination)
    assert manifest["record_count"] == 3
    assert len(list(destination.rglob("*.png"))) == 2
    assert any(not path.read_text() for path in (destination / "labels" / "train").glob("*.txt"))
    assert not list(destination.rglob("*.mp4"))
    with pytest.raises(ValueError, match="new version"):
        export_dataset(store, destination)
    (destination / "records.json").write_text("[]")
    with pytest.raises(ValueError, match="changed"):
        verify_dataset(destination)


def test_game_and_possession_leakage_rejected_and_block_guard_enforced():
    games = [{"id": "g", "duration": 200}]
    rows = [box(), box(identifier="v", split="val", time=160, possession_id="other")]
    with pytest.raises(ValueError, match="game crosses"):
        validate_splits(rows, games)
    games[0].update(split_mode="contiguous", split_blocks=[{"start": 0, "end": 100, "split": "train"},
                                                          {"start": 100, "end": 200, "split": "val"}])
    validate_splits(rows, games)
    rows[1]["time"] = 105
    with pytest.raises(ValueError, match="guard"):
        validate_splits(rows, games)
    rows[1].update(time=160, possession_id=rows[0]["possession_id"])
    with pytest.raises(ValueError, match="Possession"):
        validate_splits(rows, games)


def test_replay_of_cannot_cross_games_or_splits():
    games = [{"id": "g", "duration": 200}, {"id": "other", "duration": 200}]
    rows = [box(), box(identifier="replay", game_id="other", split="test", replay_of="b")]
    with pytest.raises(ValueError, match="replay"):
        validate_splits(rows, games)


def test_unreviewed_box_on_reviewed_frame_is_not_silently_omitted(tmp_path):
    store = Store(tmp_path / "db")
    game(store)
    store.save("box", box())
    frame_review(store)
    store.save("box", box(identifier="rim", label="rim", reviewed=False))
    with pytest.raises(ValueError, match="unreviewed boxes"):
        export_dataset(store, tmp_path / "v1")


def test_features_from_analysis_never_replace_human_result(tmp_path):
    store = Store(tmp_path / "db")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"explicit fixture; no decoding in shot-only export")
    game(store, source_path=str(source))
    store.save("possession", {"id": "p", "game_id": "g", "start": 0, "end": 100, "split": "train"})
    store.save("shot", {"id": "human", "game_id": "g", "possession_id": "p", "release": 20,
                        "result": "missed", "reviewed": True, "split": "train"})
    dependencies = analysis_dependencies(store, "p")
    store.save("analysis", {"id": "p", "possession_id": "p", "status": "complete", "analysis_mode": "assisted",
                         "dependencies": dependencies, "fingerprint": canonical_hash(dependencies)[:24],
                         "events": [{"id": "pred", "type": "shot", "release": 20.1,
                         "result": "made", "features": {"observation_count": 8, "below_x": None}}]})
    destination = export_dataset(store, tmp_path / "v1")
    shot = json.loads((destination / "records.json").read_text())[0]
    assert shot["result"] == "missed"
    assert shot["features"]["observation_count"] == 8
    assert shot["features_source"]["event_id"] == "pred"


def test_embedded_stale_features_and_legacy_predictions_are_removed(tmp_path):
    store = Store(tmp_path / "db")
    game(store)
    store.save("shot", {"id": "s", "game_id": "g", "possession_id": "p", "release": 20,
                        "result": "made", "reviewed": True, "split": "train",
                        "features": {"observation_count": 99}, "features_source": {"analysis_fingerprint": "old"}})
    store.save("analysis", {"id": "p", "status": "complete", "stale": True,
                            "events": [{"id": "old", "type": "shot", "release": 20,
                                        "features": {"observation_count": 99}}]})
    destination = export_dataset(store, tmp_path / "version")
    shot, = json.loads((destination / "records.json").read_text())
    assert "features" not in shot and "features_source" not in shot
    assert shot["features_status"] == "missing_or_stale_analysis"


def test_dataset_rejects_unmanifested_images(tmp_path):
    store = Store(tmp_path / "db")
    game(store)
    store.save("shot", {"id": "s", "game_id": "g", "possession_id": "p", "release": 20,
                        "result": "made", "reviewed": True, "split": "train"})
    destination = export_dataset(store, tmp_path / "version")
    image_dir = destination / "images" / "train"
    image_dir.mkdir(parents=True)
    (image_dir / "unlisted.png").write_bytes(b"unlisted")
    with pytest.raises(ValueError, match="Unmanifested"):
        verify_dataset(destination)


def test_invalid_box_coordinates_fail_before_export(tmp_path):
    store = Store(tmp_path / "db")
    game(store)
    store.save("box", box(box=[0.5, 0.1, 0.4, 0.9]))
    with pytest.raises(ValueError, match="coordinates"):
        export_dataset(store, tmp_path / "v1")


def test_positive_boxes_require_review_of_the_complete_frame(tmp_path):
    store = Store(tmp_path / "db")
    game(store)
    store.save("box", box())
    with pytest.raises(ValueError, match="complete frame reviewed"):
        export_dataset(store, tmp_path / "v1")


def test_manual_foul_and_turnover_annotations_survive_export(tmp_path):
    store = Store(tmp_path / "db")
    game(store)
    store.save("event", {"id": "called-foul", "game_id": "g", "possession_id": "p", "time": 20,
                         "type": "foul", "reviewed": True, "split": "train", "source": "human"})
    destination = export_dataset(store, tmp_path / "v1")
    event = json.loads((destination / "records.json").read_text())[0]
    assert event["kind"] == "event" and event["type"] == "foul"
    assert event["source"] == "human"
