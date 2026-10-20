"""Prediction/export provenance contracts, not a basketball accuracy evaluation."""
import json

import pytest

from playlab.analysis import analysis_dependencies, analysis_record_id
from playlab.benchmark import benchmark_detector, predict_dataset
from playlab.dataset import canonical_hash, export_dataset
from playlab.evaluation import evaluate_locked, freeze_run
from playlab.store import Store


def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr("playlab.analysis._vision_version", lambda: "unavailable")
    store = Store(tmp_path / "db")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"mock fixture; no video decoding or real inference")
    weights = tmp_path / "fixture-model.pt"
    weights.write_bytes(b"explicit mock weights")
    store.save("game", {"id": "g", "source_path": str(source), "duration": 100})
    store.save("possession", {"id": "p", "game_id": "g", "start": 0, "end": 100, "split": "test"})
    store.save("shot", {"id": "gold", "game_id": "g", "possession_id": "p", "release": 20,
                        "result": "made", "reviewed": True, "split": "test"})
    dataset = export_dataset(store, tmp_path / "dataset")
    dependencies = analysis_dependencies(store, "p", mode="automatic", weights=weights)
    store.save("analysis", {"id": analysis_record_id("p", "automatic"), "possession_id": "p", "game_id": "g",
                            "status": "complete", "analysis_mode": "automatic", "dependencies": dependencies,
                            "fingerprint": canonical_hash(dependencies)[:24],
                            "events": [{"id": "auto-20", "game_id": "g", "possession_id": "p", "release": 20,
                                        "type": "shot", "source": "prediction", "result": "missed",
                                        "predicted_result": "missed", "origin": "automatic_rim_approach"}]})
    return store, dataset, weights


def test_prediction_export_keeps_outcomes_separate_and_binds_locked_artifacts(tmp_path, monkeypatch):
    store, dataset, weights = prepared(tmp_path, monkeypatch)
    predictions = predict_dataset(store, dataset, tmp_path / "predictions.json")
    payload = json.loads(predictions.read_text())
    assert payload["events"][0]["result"] == "missed"
    assert payload["analysis_mode"] == "automatic"
    assert payload["missing_analysis_possessions"] == []
    freeze = freeze_run(dataset, [weights, predictions], tmp_path / "run")
    report = evaluate_locked(dataset, predictions, tmp_path / "run", freeze)
    assert report["correct"] == 0 and report["eligible"] == 1
    assert report["end_to_end_automatic"] is True


def test_prediction_export_rechecks_model_and_current_inputs(tmp_path, monkeypatch):
    store, dataset, weights = prepared(tmp_path, monkeypatch)
    weights.write_bytes(b"changed weights")
    predictions = predict_dataset(store, dataset, tmp_path / "predictions.json")
    payload = json.loads(predictions.read_text())
    assert payload["events"] == []
    assert payload["missing_analysis_possessions"] == ["p"]


def test_prediction_export_does_not_use_assisted_analysis_for_automatic_claim(tmp_path, monkeypatch):
    store, dataset, _ = prepared(tmp_path, monkeypatch)
    automatic = store.get("analysis", analysis_record_id("p", "automatic"))
    store.delete("analysis", automatic["id"])
    dependencies = analysis_dependencies(store, "p", mode="assisted")
    store.save("analysis", {**automatic, "id": "p", "analysis_mode": "assisted", "dependencies": dependencies,
                            "fingerprint": canonical_hash(dependencies)[:24]})
    predictions = predict_dataset(store, dataset, tmp_path / "predictions.json")
    assert json.loads(predictions.read_text())["missing_analysis_possessions"] == ["p"]


def test_benchmark_rejects_unlisted_images_before_loading_optional_models(tmp_path, monkeypatch):
    _, dataset, weights = prepared(tmp_path, monkeypatch)
    images = dataset / "images" / "test"
    images.mkdir(parents=True)
    (images / "unlisted.png").write_bytes(b"not in immutable dataset")
    with pytest.raises(ValueError, match="Unmanifested"):
        benchmark_detector(dataset, weights, tmp_path / "report.json")


def test_locked_evaluation_requires_actual_inference_models_in_freeze(tmp_path, monkeypatch):
    store, dataset, _ = prepared(tmp_path, monkeypatch)
    predictions = predict_dataset(store, dataset, tmp_path / "predictions.json")
    freeze = freeze_run(dataset, [predictions], tmp_path / "run")
    with pytest.raises(ValueError, match="Every inference model/bundle"):
        evaluate_locked(dataset, predictions, tmp_path / "run", freeze)
    assert not (tmp_path / "run" / "locked-test-used.json").exists()
