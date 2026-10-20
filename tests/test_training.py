import copy
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from playlab.features import FEATURE_NAMES, compute_features, geometry_prediction, predict_result
from playlab.training import train_classifier, train_detector


def feature_rows():
    rows = []
    for split in ("train", "val", "test"):
        for index in range(8):
            made = index % 2 == 0
            values = [0.1 if made else 1.8, 0.1, 1.0, 0.1 if made else 1.7, 0.8, 8.0]
            features = dict(zip(FEATURE_NAMES, values))
            features["geometry_result"] = "made" if made else "missed"
            rows.append({"id": f"{split}-{index}", "game_id": split, "possession_id": f"{split}-p-{index}",
                         "release": index * 10 + 10, "split": split, "reviewed": True,
                         "result": "made" if made else "missed", "features": features})
    return rows


def test_classifier_train_only_statistics_and_test_labels_do_not_select_model(tmp_path):
    rows = feature_rows()
    rows[0]["features"]["min_abs_y"] = None
    first = train_classifier(rows, tmp_path / "first")
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["split"] == "test":
            row["result"] = "missed"
            row["features"] = {name: 1e12 for name in FEATURE_NAMES}
    second = train_classifier(changed, tmp_path / "second")
    a, b = joblib.load(first), joblib.load(second)
    np.testing.assert_array_equal(a["model"].named_steps["logisticregression"].coef_,
                                  b["model"].named_steps["logisticregression"].coef_)
    assert a["threshold"] == b["threshold"]
    assert a["provenance"]["development_hash"] == b["provenance"]["development_hash"]
    assert a["model"].named_steps["simpleimputer"].statistics_[1] == pytest.approx(0.1)
    assert a["provenance"]["test_used_for_selection"] is False
    assert predict_result(first, rows[2]["features"])["result"] == "made"
    unknown = dict(rows[2]["features"], observation_count=1)
    assert predict_result(first, unknown)["result"] == "unknown"
    assert json.loads((first.parent / "training.json").read_text())["train_count"] == 8


def test_classifier_rejects_group_leakage_and_missing_visual_features(tmp_path):
    rows = feature_rows()
    rows[9]["game_id"] = "train"
    with pytest.raises(ValueError, match="cross splits"):
        train_classifier(rows, tmp_path / "bad")
    rows = feature_rows()
    rows[0].pop("features")
    with pytest.raises(ValueError, match="features missing"):
        train_classifier(rows, tmp_path / "missing")


def test_classifier_requires_both_results_in_development_splits(tmp_path):
    rows = [r for r in feature_rows() if not (r["split"] == "val" and r["result"] == "missed")]
    with pytest.raises(ValueError, match="validation"):
        train_classifier(rows, tmp_path / "bad")


def test_temporal_crossing_requires_consecutive_observations_and_rim_alignment():
    observations = [{"time": i / 10, "x": 100, "y": y, "rim_x": 100, "rim_y": 100, "rim_width": 20}
                    for i, y in enumerate((75, 90, 105, 120))]
    features = compute_features(observations)
    assert features["below_after_above"] == 1
    assert geometry_prediction(features) == "made"
    observations[2]["time"], observations[3]["time"] = 2, 2.1
    assert geometry_prediction(compute_features(observations)) == "unknown"
    assert geometry_prediction(compute_features([])) == "unknown"
    json.dumps(compute_features([]), allow_nan=False)
    # A wholly absent net crossing cannot by itself become a missed shot.
    assert geometry_prediction(compute_features(observations[:2])) == "unknown"


def test_detector_rejects_invalid_dataset_without_optional_vision_imports(tmp_path):
    with pytest.raises(FileNotFoundError):
        train_detector(tmp_path / "missing", tmp_path / "run")


def test_validation_keeps_geometry_when_classifier_loses(tmp_path):
    rows = feature_rows()
    for row in rows:
        # Identical visual vectors cannot teach a classifier the different labels;
        # independently provided geometry evidence wins on validation.
        row["features"].update(dict(zip(FEATURE_NAMES, [0.5, 0.1, 1, 0.5, 0.8, 8])))
    bundle = train_classifier(rows, tmp_path / "baseline-wins")
    assert joblib.load(bundle)["selected_method"] == "geometry"
    prediction = predict_result(bundle, rows[1]["features"])
    assert prediction["result"] == "missed" and prediction["method"] == "geometry_baseline"


def test_scaler_and_imputer_never_fit_validation_extremes(tmp_path):
    rows = feature_rows()
    for row in rows:
        if row["split"] == "val":
            row["features"]["min_abs_y"] = 1e9
    model = joblib.load(train_classifier(rows, tmp_path / "stats"))["model"]
    assert model.named_steps["simpleimputer"].statistics_[1] == pytest.approx(0.1)
    assert model.named_steps["standardscaler"].mean_[1] == pytest.approx(0.1)


def test_detector_pause_resume_keeps_dataset_and_rng_provenance(tmp_path, monkeypatch):
    """Exercise orchestration with stub ML libraries, not a model-accuracy test."""
    from playlab.dataset import canonical_hash, file_hash

    dataset = tmp_path / "dataset"
    files = {}
    for split in ("train", "val"):
        path = dataset / "images" / split / "fixture.png"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"stub-ML-does-not-decode-this")
        files[path.relative_to(dataset).as_posix()] = file_hash(path)
    manifest = {"files": files}
    manifest["dataset_hash"] = canonical_hash(manifest)
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    restored = []

    def save(value, destination):
        with Path(destination).open("wb") as stream:
            pickle.dump(value, stream)

    def load(source, **_kwargs):
        with Path(source).open("rb") as stream:
            return pickle.load(stream)

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), save=save, load=load,
                                 get_rng_state=lambda: b"rng-state", set_rng_state=restored.append)

    class FakeYOLO:
        def __init__(self, _weights):
            self.callbacks = {}

        def add_callback(self, name, callback):
            self.callbacks[name] = callback

        def train(self, **options):
            last = Path(options["project"]) / options["name"] / "weights" / "last.pt"
            last.parent.mkdir(parents=True, exist_ok=True)
            trainer = SimpleNamespace(last=last, epoch=int(options["resume"]))
            self.callbacks["on_train_start"](trainer)
            last.write_bytes(b"recoverable-model-and-optimizer-stub")
            self.callbacks["on_model_save"](trainer)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.setattr("playlab.training.code_version", lambda: {"revision": "unit", "source_hash": "unit"})
    monkeypatch.setenv("PLAYLAB_MODEL_DIR", str(tmp_path / "models"))
    output = tmp_path / "run"
    checkpoint = train_detector(dataset, output, epochs=4, stop_after_epochs=1)
    assert Path(str(checkpoint) + ".rng.pt").is_file()
    assert json.loads((output / "training.json").read_text())["status"] == "paused"
    train_detector(dataset, output, resume=checkpoint, epochs=4)
    assert restored == [b"rng-state"]
    assert json.loads((output / "training.json").read_text())["status"] == "completed"
    with pytest.raises(ValueError, match="Completed training"):
        train_detector(dataset, output, resume=checkpoint, epochs=4)
