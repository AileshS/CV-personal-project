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
        label = dataset / "labels" / split / "fixture.txt"
        label.parent.mkdir(parents=True)
        label.write_text("0 0.5 0.5 0.1 0.1\n1 0.5 0.2 0.2 0.1\n")
        files[label.relative_to(dataset).as_posix()] = file_hash(label)
    manifest = {"files": files}
    manifest["dataset_hash"] = canonical_hash(manifest)
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    restored = []
    calls = []
    # Valid-looking extras must not be silently included in either split.
    for split in ("train", "val", "test"):
        extra = dataset / "images" / split / "not-in-manifest.png"
        extra.parent.mkdir(parents=True, exist_ok=True)
        extra.write_bytes(b"unverified")

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
            calls.append(options)
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
    with pytest.raises(ValueError, match="Unmanifested"):
        train_detector(dataset, output, epochs=4, stop_after_epochs=1, image_size=96)
    for extra in (dataset / "images").rglob("not-in-manifest.png"):
        extra.unlink()
    checkpoint = train_detector(dataset, output, epochs=4, stop_after_epochs=1, image_size=96)
    assert Path(str(checkpoint) + ".rng.pt").is_file()
    assert json.loads((output / "training.json").read_text())["status"] == "paused"
    assert calls[0]["imgsz"] == 96
    assert "test:" not in (output / "train-data.yaml").read_text()
    for split in ("train", "val"):
        assert (output / f"{split}-images.txt").read_text().splitlines() == [
            str(dataset / "images" / split / "fixture.png")]
    with pytest.raises(ValueError, match="Epoch limit or image size"):
        train_detector(dataset, output, resume=checkpoint, epochs=5, image_size=96)
    with pytest.raises(ValueError, match="Epoch limit or image size"):
        train_detector(dataset, output, resume=checkpoint, epochs=4, image_size=128)
    with pytest.raises(ValueError, match="this run's"):
        train_detector(dataset, output, resume=tmp_path / "other-last.pt", epochs=4, image_size=96)
    sidecar = Path(str(checkpoint) + ".rng.pt")
    original = sidecar.read_bytes()
    sidecar.write_bytes(b"changed sidecar")
    with pytest.raises(ValueError, match="sidecar changed"):
        train_detector(dataset, output, resume=checkpoint, epochs=4, image_size=96)
    sidecar.write_bytes(original)
    train_detector(dataset, output, resume=checkpoint, epochs=4, image_size=96)
    assert restored == [b"rng-state"]
    assert json.loads((output / "training.json").read_text())["status"] == "completed"
    with pytest.raises(ValueError, match="Completed training"):
        train_detector(dataset, output, resume=checkpoint, epochs=4, image_size=96)


@pytest.mark.parametrize("image_size", [31, 65, 4096, True, 96.0])
def test_detector_rejects_invalid_image_size_before_optional_imports(tmp_path, image_size):
    from playlab.dataset import canonical_hash

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest = {"files": {}}
    manifest["dataset_hash"] = canonical_hash(manifest)
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="image_size"):
        train_detector(dataset, tmp_path / "run", image_size=image_size)


# Ball centers relative to the detected rim box center, in rim widths, traced at
# 30 fps from the 990 s clip (a shot that slid off the left of the rim). The box
# includes the net (41 x 44 px), so the rim line is its top edge, not its center.
MISS_990 = [(1004.63, -.39, -.89), (1004.67, -.40, -.90), (1004.70, -.41, -.91), (1004.73, -.42, -.91),
            (1004.77, -.44, -.90), (1004.80, -.45, -.87), (1004.97, -.62, -.41), (1005.03, -.73, -.13),
            (1005.07, -.79, .01), (1005.13, -.76, .55), (1005.17, -.76, .71), (1005.23, -.79, 1.32),
            (1005.33, -.53, 1.50)]


def _rows(trace, height=44 / 41):
    return [{"time": t, "x": x, "y": y, "rim_x": 0, "rim_y": 0, "rim_width": 1, "rim_height": height}
            for t, x, y in trace]


def test_ball_falling_outside_the_rim_edge_is_missed_and_through_the_middle_is_made():
    missed = compute_features(_rows(MISS_990))
    assert missed["descent_abs_x"] == pytest.approx(.76, abs=.03)
    assert geometry_prediction(missed) == "missed"
    made = compute_features(_rows([(t, x * .15, y) for t, x, y in MISS_990]))
    assert geometry_prediction(made) == "made"
    between = compute_features(_rows([(t, x * .65, y) for t, x, y in MISS_990]))
    assert geometry_prediction(between) == "unknown"  # ~0.5 rim widths: not confident either way
    # Without the approach above the rim, a ball seen only below it decides nothing.
    assert compute_features(_rows([r for r in MISS_990 if r[2] > 0]))["descent_abs_x"] is None


# Rim-out from the 34:07 miss (rim box ~44 x 44 px): reaches the front of the rim,
# then bounces back up and away to the left without dropping below it.
RIM_OUT_3407 = [(2049.27, -.62, -1.83), (2049.30, -.45, -1.57), (2049.33, -.27, -1.30), (2049.37, -.10, -1.01),
                (2049.40, .09, -.74), (2049.47, .07, -.68), (2049.50, -.13, -.87), (2049.53, -.27, -1.06),
                (2049.60, -.58, -1.38), (2049.70, -1.00, -1.75), (2049.80, -1.48, -1.95), (2049.97, -2.23, -1.95)]
# The 23:12 make: the arc above the rim was not detected, a stray sighting sits off
# to the side, then the ball drops straight through the net.
MADE_2312 = [(1392.33, .72, .83), (1393.27, -.15, .93), (1393.33, -.12, 1.33), (1393.40, -.05, 1.69),
             (1393.47, .01, 2.10), (1393.53, .03, 2.61), (1393.60, .05, 3.13)]


def test_rim_out_is_missed_and_unseen_arc_drop_through_net_is_made():
    rim_out = compute_features(_rows(RIM_OUT_3407, height=1.0))
    assert rim_out["bounced_out"] == 1 and rim_out["descent_abs_x"] is None
    assert geometry_prediction(rim_out) == "missed"
    made = compute_features(_rows(MADE_2312, height=.9))
    assert made["descent_abs_x"] == pytest.approx(.1, abs=.05)
    assert geometry_prediction(made) == "made"
    # A ball moving slowly under the rim (e.g. held by a player) is not a drop through the net.
    held = compute_features(_rows([(1393 + i * .2, .05, 1.0 + .05 * i) for i in range(6)], height=.9))
    assert held["descent_abs_x"] is None and geometry_prediction(held) == "unknown"
