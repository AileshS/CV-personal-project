import json

import pytest

from playlab.analysis import analysis_dependencies
from playlab.dataset import canonical_hash, export_dataset, file_hash, verify_dataset
from playlab.evaluation import detection_metrics, evaluate_locked, freeze_run, outcome_metrics
from playlab.store import Store


def test_outcomes_count_abstentions_missing_shots_false_positives_and_defender_errors():
    gold = [{"id": str(i), "possession_id": "p", "release": 10 + i * 5, "reviewed": True,
             "result": "made" if i < 2 else "missed", "nearest_defender_id": "opponent"} for i in range(3)]
    predictions = [{"id": "0", "result": "made", "nearest_defender_id": "opponent"},
                   {"id": "1", "result": "unknown", "nearest_defender_id": "wrong"},
                   {"id": "spurious", "release": 100, "possession_id": "p", "result": "made"}]
    metrics = outcome_metrics(gold, predictions)
    assert metrics["eligible"] == 3
    assert metrics["emitted"] == 1
    assert metrics["coverage"] == pytest.approx(1 / 3)
    assert metrics["confusion"]["missed"]["unknown"] == 1
    assert metrics["false_positive_events"] == 1
    assert metrics["defender"]["correct"] == 1
    assert metrics["defender"]["eligible"] == 3
    assert metrics["defender"]["accuracy_when_emitted"] == 0.5
    assert not metrics["per_class"]["made"]["supported"]


def test_event_time_matching_is_one_to_one():
    gold = [{"id": "a", "game_id": "g", "possession_id": "p", "release": 10,
             "reviewed": True, "result": "made"},
            {"id": "b", "game_id": "g", "possession_id": "p", "release": 10.3,
             "reviewed": True, "result": "missed"}]
    predictions = [{"id": "auto", "game_id": "g", "possession_id": "p", "release": 10.1, "result": "made"}]
    result = outcome_metrics(gold, predictions)
    assert result["matched"] == 1 and result["emitted"] == 1


def test_detection_duplicate_and_negative_frame_predictions_are_false_positives():
    gold = [{"image": "a", "label": "ball", "box": [0.1, 0.2, 0.2, 0.3]}]
    predictions = [dict(gold[0], confidence=0.9), dict(gold[0], confidence=0.8),
                   {"image": "negative", "label": "ball", "box": [0.4, 0.4, 0.5, 0.5], "confidence": 0.8}]
    result = detection_metrics(gold, predictions)["per_class"]["ball"]
    assert result["tp"] == 1 and result["fp"] == 2 and result["fn"] == 0
    assert result["precision"] == pytest.approx(1 / 3)


def test_locked_test_requires_unchanged_frozen_artifacts_and_can_only_run_once(tmp_path, monkeypatch):
    monkeypatch.setattr("playlab.analysis._vision_version", lambda: "unavailable")
    store = Store(tmp_path / "db")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture; no decode during evaluation")
    store.save("game", {"id": "g", "source_path": str(source), "duration": 100,
                        "width": 80, "height": 60, "fps": 30})
    store.save("possession", {"id": "p", "game_id": "g", "start": 0, "end": 100, "split": "test"})
    store.save("shot", {"id": "shot", "game_id": "g", "possession_id": "p", "release": 20,
                        "result": "made", "reviewed": True, "split": "test"})
    dataset = export_dataset(store, tmp_path / "dataset")
    weights = tmp_path / "model.txt"
    weights.write_text("frozen weights")
    dependencies = analysis_dependencies(store, "p", mode="automatic", weights=weights)
    payload = {"schema": 2, "dataset_hash": verify_dataset(dataset)["dataset_hash"], "split": "test",
               "analysis_mode": "automatic", "inference_artifacts": {str(weights): file_hash(weights)},
               "analysis_provenance": {"p": dependencies},
               "analysis_fingerprints": {"p": canonical_hash(dependencies)[:24]},
               "events": [{"id": "auto-20", "release": 20, "game_id": "g", "possession_id": "p",
                           "origin": "automatic_rim_approach", "result": "made"}]}
    predictions = tmp_path / "predictions.json"
    predictions.write_text(json.dumps(payload))
    report_dir = tmp_path / "report"
    freeze = freeze_run(dataset, [weights, predictions], report_dir)
    weights.write_text("changed")
    with pytest.raises(ValueError, match="model/configuration changed"):
        evaluate_locked(dataset, predictions, report_dir, freeze)
    weights.write_text("frozen weights")
    result = evaluate_locked(dataset, predictions, report_dir, freeze)
    assert result["correct"] == 1
    assert json.loads((report_dir / "test-report.json").read_text())["test_used_for_selection"] is False
    with pytest.raises(ValueError, match="already consumed"):
        evaluate_locked(dataset, predictions, tmp_path / "another-report", freeze)


@pytest.mark.parametrize("field,value", [("dataset_hash", "wrong"), ("split", "val")])
def test_prediction_metadata_rejected_before_consuming_locked_test(tmp_path, field, value):
    store = Store(tmp_path / "db")
    store.save("game", {"id": "g", "source_path": "unused", "duration": 100})
    store.save("shot", {"id": "s", "game_id": "g", "possession_id": "p", "release": 20,
                        "result": "made", "reviewed": True, "split": "test"})
    dataset = export_dataset(store, tmp_path / "dataset")
    payload = {"schema": 2, "dataset_hash": verify_dataset(dataset)["dataset_hash"], "split": "test",
               "analysis_mode": "automatic", "inference_artifacts": {}, "analysis_provenance": {},
               "analysis_fingerprints": {}, "events": []}
    payload[field] = value
    predictions = tmp_path / "predictions.json"
    predictions.write_text(json.dumps(payload))
    output = tmp_path / "report"
    freeze = freeze_run(dataset, [predictions], output)
    with pytest.raises(ValueError, match="hash/split"):
        evaluate_locked(dataset, predictions, output, freeze)
    assert not (output / "locked-test-used.json").exists()


def test_origin_metrics_keep_full_denominators_and_scope_matching():
    gold = [{"id": "same", "game_id": "g", "possession_id": "p", "release": 20,
             "result": "made", "reviewed": True},
            {"id": "second", "game_id": "g", "possession_id": "p", "release": 30,
             "result": "missed", "reviewed": True}]
    predictions = [{"id": "same", "game_id": "wrong", "possession_id": "p", "release": 20,
                    "result": "made", "origin": "automatic_rim_approach"},
                   {"id": "second", "game_id": "g", "possession_id": "p", "release": 30,
                    "result": "missed", "origin": "human_release_anchor"}]
    report = outcome_metrics(gold, predictions)
    assert report["correct"] == 1 and report["false_positive_events"] == 1
    assert report["by_release_origin"]["automatic"]["eligible"] == 2
    assert report["by_release_origin"]["automatic"]["correct"] == 0
    assert report["by_release_origin"]["human_anchored"]["coverage"] == .5
