"""Label Studio round trip on artificial records; no model or video is used."""
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

SPEC = importlib.util.spec_from_file_location(
    "label_studio_frames", Path(__file__).parents[1] / "scripts" / "label_studio_frames.py")
frames = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frames)


def test_windows_skip_test_blocks_and_apply_guards():
    game = {"split_blocks": [{"start": 720, "end": 2880, "split": "train"},
                             {"start": 3420, "end": 4260, "split": "val"},
                             {"start": 4380, "end": 5640, "split": "test"}]}
    assert frames.allowed_windows(game) == [(730., 2870., "train"), (3430., 4250., "val")]
    with pytest.raises(ValueError):
        frames.allowed_windows({"split": "unassigned", "duration": 10})


def test_selection_mixes_near_rim_and_spread_frames_with_spacing():
    scan = [{"time": float(t), "near_rim": .9 if t in (10, 11, 40) else 0.} for t in range(0, 100, 1)]
    chosen = frames.choose_times(scan, 10)
    assert 10. in chosen and 40. in chosen and 11. not in chosen  # 3 s spacing
    assert len(chosen) == 10 and min(b - a for a, b in zip(chosen, chosen[1:])) >= 3


def test_near_rim_uses_rim_width_reach():
    rim = {"bbox": [100, 100, 120, 110], "confidence": .9}
    assert frames.near_rim([{"bbox": [130, 100, 134, 104], "confidence": .4}], [rim]) == .4
    assert frames.near_rim([{"bbox": [400, 400, 404, 404], "confidence": .9}], [rim]) == 0


def test_prediction_percent_boxes_and_threshold():
    result = frames.prediction({"balls": [{"bbox": [96, 54, 192, 108], "confidence": .8},
                                          {"bbox": [0, 0, 5, 5], "confidence": .1}], "rims": []}, 1920, 1080)
    assert len(result) == 1
    value = result[0]["value"]
    assert (value["x"], value["y"], value["width"], value["height"]) == pytest.approx((5, 5, 5, 5))


def test_convert_writes_yolo_labels_negatives_and_skips_unlabeled(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for name in ("train-00990.00.jpg", "train-01000.00.jpg", "val-03500.00.jpg", "train-01010.00.jpg"):
        Image.new("RGB", (20, 10)).save(images / name)

    def task(name, annotations):
        return {"data": {"image": f"http://localhost:8081/images/{name}"}, "annotations": annotations}

    box = {"type": "rectanglelabels", "value": {"x": 10, "y": 20, "width": 10, "height": 20, "rotation": 0,
                                                "rectanglelabels": ["ball"]}}
    rim = {"type": "rectanglelabels", "value": {"x": 50, "y": 0, "width": 20, "height": 10, "rotation": 0,
                                                "rectanglelabels": ["rim"]}}
    export = [task("train-00990.00.jpg", [{"result": [box, rim], "updated_at": "2"},
                                          {"result": [], "updated_at": "1"}]),
              task("train-01000.00.jpg", [{"result": []}]),               # reviewed negative
              task("val-03500.00.jpg", [{"result": [rim]}]),
              task("train-01010.00.jpg", [{"result": [box], "was_cancelled": True}])]  # skipped
    path = tmp_path / "export.json"
    path.write_text(json.dumps(export))

    class Args:
        export_json, out = str(path), str(tmp_path / "dataset")
    Args.images = str(images)
    frames.convert(Args)
    out = tmp_path / "dataset"
    assert (out / "train/labels/train-00990.00.txt").read_text().splitlines() == [
        "0 0.150000 0.300000 0.100000 0.200000", "1 0.600000 0.050000 0.200000 0.100000"]
    assert (out / "train/labels/train-01000.00.txt").read_text() == ""
    assert (out / "valid/labels/val-03500.00.txt").read_text().startswith("1 ")
    assert not (out / "train/images/train-01010.00.jpg").exists()
    assert "names:\n- ball\n- rim" in (out / "data.yaml").read_text()
    assert (tmp_path / "dataset.zip").is_file()
