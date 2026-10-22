import json

import pytest
from PIL import Image

from playlab.dataset import verify_dataset
from scripts.prepare_public_detector import REVISION, convert


@pytest.fixture
def public_source(tmp_path):
    raw = tmp_path / "raw"
    paths = []
    for index, split in enumerate(("train", "valid")):
        image = raw / f"Basketball-1/{split}/images/f{index}.jpg"
        label = raw / f"Basketball-1/{split}/labels/f{index}.txt"
        image.parent.mkdir(parents=True)
        label.parent.mkdir(parents=True)
        Image.new("RGB", (32, 32), (index * 100, 20, 30)).save(image)
        label.write_text("0 .2 .2 .1 .1\n1 .7 .7 .1 .1\n")
        paths.extend([image.relative_to(raw).as_posix(), label.relative_to(raw).as_posix()])
    (raw / "Basketball-1/data.yaml").write_text("names:\n- basketball\n- rim\n- sports ball\n")
    (raw / "Basketball-1/README.dataset.txt").write_text("License: CC BY 4.0")
    (raw / "selection.json").write_text(json.dumps({"revision": REVISION, "paths": paths}))
    return raw


def test_public_conversion_keeps_external_provenance_without_local_gold(public_source, tmp_path):
    output = tmp_path / "dataset"
    manifest = convert(public_source, output)
    assert verify_dataset(output) == manifest
    assert manifest["kind"] == "external_labeled_detection" and manifest["games"] == []
    assert manifest["provenance"]["counts"]["train"] == {"images": 1, "ball": 1, "rim": 1}
    records = json.loads((output / "records.json").read_text())
    assert {r["label"] for r in records} == {"ball", "rim"}
    assert all(r["source"] == "external_publisher_labels" and not r["reviewed"] for r in records)
    assert not (output / "images/test").exists()
    with pytest.raises(ValueError, match="exists"):
        convert(public_source, output)


def test_ambiguous_sports_ball_is_not_silently_mapped_or_a_negative(public_source, tmp_path):
    label = public_source / "Basketball-1/train/labels/f0.txt"
    label.write_text(label.read_text() + "2 .4 .4 .1 .1\n")
    output = tmp_path / "dataset"
    with pytest.raises(ValueError, match="both external splits"):
        convert(public_source, output)
    assert not output.exists()


@pytest.mark.parametrize("contents", ["0 nan .2 .1 .1\n", "0 .2 .2 -1 .1\n", "0 .95 .2 .4 .1\n"])
def test_invalid_external_boxes_fail_closed(public_source, tmp_path, contents):
    (public_source / "Basketball-1/train/labels/f0.txt").write_text(contents)
    with pytest.raises(ValueError, match="annotation|exceeds"):
        convert(public_source, tmp_path / "dataset")


def test_exact_pixel_duplicates_do_not_cross_splits(public_source, tmp_path):
    train = public_source / "Basketball-1/train/images/f0.jpg"
    train.write_bytes((public_source / "Basketball-1/valid/images/f1.jpg").read_bytes())
    with pytest.raises(ValueError, match="both external splits"):
        convert(public_source, tmp_path / "dataset")
