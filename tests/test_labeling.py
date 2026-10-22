import ast
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from playlab.dataset import canonical_hash, file_hash
from playlab.labeling import prepare_label_batch
from playlab.store import Store


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    store = Store(tmp_path / "data")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source identity fixture; decoding is independently covered by media tests")
    store.save("game", {"id": "g", "source_path": str(source), "duration": 200,
                        "width": 80, "height": 60, "split_mode": "game", "split": "train"})
    store.save("possession", {"id": "p", "game_id": "g", "start": 0, "end": 200,
                              "split": "train", "reviewed": False, "proposal": True})
    monkeypatch.setattr("playlab.labeling.frame_at", lambda _path, at, max_width=None:
                        np.full((60, 80, 3), int(at), dtype=np.uint8))
    return store


def load(path):
    return json.loads((path / "manifest.json").read_text())


def test_batch_is_deterministic_portable_unreviewed_and_does_not_edit_labels(workspace, tmp_path):
    workspace.save("box", {"id": "human", "game_id": "g", "possession_id": "p", "time": 15,
                            "label": "ball", "box": [.1, .2, .2, .3], "reviewed": True})
    before = {kind: workspace.list(kind) for kind in ("game", "possession", "box", "frame", "shot")}
    first = prepare_label_batch(workspace, tmp_path / "v1", frame_count=12, possession_count=4,
                                allow_review_windows=True, window_seconds=50)
    second = prepare_label_batch(workspace, tmp_path / "v2", frame_count=12, possession_count=4,
                                 allow_review_windows=True, window_seconds=50)
    manifest = load(first)
    assert manifest == load(second)
    assert manifest["kind"] == "unreviewed_label_batch" and manifest["reviewed"] is False
    assert manifest["frame_count"] == 12 and manifest["review_item_count"] == 4
    assert all(item["item_kind"] == "review_window" for item in manifest["review_items"])
    assert all(frame["reviewed"] is False and frame["negative"] is False for frame in manifest["frames"])
    assert all(frame["width"] == 80 and frame["height"] == 60 for frame in manifest["frames"])
    assert "source_path" not in manifest["sources"]["g"]
    assert not list(first.rglob("*.mp4"))
    assert (first / "index.html").is_file()
    assert all(file_hash(first / path) == digest for path, digest in manifest["files"].items())
    assert canonical_hash({key: value for key, value in manifest.items() if key != "batch_hash"}) == manifest["batch_hash"]
    assert before == {kind: workspace.list(kind) for kind in before}
    with pytest.raises(ValueError, match="new batch"):
        prepare_label_batch(workspace, first)


def test_default_requires_real_split_assigned_proposals_and_declares_shortage(workspace, tmp_path):
    with pytest.raises(ValueError, match="Only 1.*requested 50"):
        prepare_label_batch(workspace, tmp_path / "too-many")
    batch = prepare_label_batch(workspace, tmp_path / "one", frame_count=3, possession_count=1)
    assert load(batch)["review_items"][0]["item_kind"] == "possession_proposal"
    game = workspace.get("game", "g")
    workspace.save("game", {**game, "split": "unassigned", "split_mode": "unassigned"})
    with pytest.raises(ValueError, match="before preparing"):
        prepare_label_batch(workspace, tmp_path / "unassigned", frame_count=3, possession_count=1)


def test_same_game_windows_stay_inside_guards_including_context(workspace, tmp_path):
    game = workspace.get("game", "g")
    workspace.save("game", {**game, "split_mode": "contiguous", "split": "unassigned", "split_blocks": [
        {"start": 0, "end": 100, "split": "train"}, {"start": 100, "end": 200, "split": "val"}]})
    parent = workspace.get("possession", "p")
    workspace.save("possession", {**parent, "split": "unassigned"})
    result = load(prepare_label_batch(workspace, tmp_path / "guarded", frame_count=8, possession_count=4,
                                      allow_review_windows=True, window_seconds=50))
    assert {frame["split"] for frame in result["frames"]} == {"train", "val"}
    for item in result["review_items"]:
        assert item["end"] <= 90 if item["split"] == "train" else item["start"] >= 110
        assert item["requires_parent_split_edit"]
    assert all(frame["time"] < 90 or frame["time"] > 110 for frame in result["frames"])


def test_test_partition_not_opened_or_decoded_by_default(workspace, tmp_path):
    workspace.save("game", {"id": "heldout", "source_path": "/unavailable/test.mp4", "duration": 200,
                            "split_mode": "game", "split": "test"})
    batch = prepare_label_batch(workspace, tmp_path / "development", frame_count=2, possession_count=1)
    assert list(load(batch)["sources"]) == ["g"]
    with pytest.raises(ValueError, match="indexing"):
        prepare_label_batch(workspace, tmp_path / "test", frame_count=2, possession_count=1, splits=("test",))


def test_model_proposals_are_separate_from_human_gold_and_empty_is_not_negative(workspace, tmp_path):
    class Detector:
        mode = "test_adapter"

        def reset(self):
            pass

        def __call__(self, pixels):
            return {"balls": [{"bbox": [8, 12, 16, 18], "confidence": .7}], "rims": []}

    workspace.save("box", {"id": "human", "game_id": "g", "possession_id": "p", "time": 100,
                            "box": [.5, .5, .6, .6], "reviewed": True})
    old = workspace.get("box", "human")
    batch = prepare_label_batch(workspace, tmp_path / "assisted", frame_count=2, possession_count=1,
                                model_assisted=True, detector=Detector())
    manifest = load(batch)
    assert manifest["model"]["mode"] == "test_adapter"
    for frame in manifest["frames"]:
        suggestion, = frame["proposed_boxes"]
        assert suggestion["box"] == [.1, .2, .2, .3]
        assert suggestion["reviewed"] is False and suggestion["source"] == "model_proposal"
        assert not frame["negative"]
    assert workspace.get("box", "human") == old


def test_partial_decode_failure_leaves_no_batch(workspace, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("Decoder unavailable")

    monkeypatch.setattr("playlab.labeling.frame_at", fail)
    destination = tmp_path / "failed"
    with pytest.raises(ValueError, match="Decoder"):
        prepare_label_batch(workspace, destination, frame_count=2, possession_count=1)
    assert not destination.exists()
    assert not list(tmp_path.glob(".review-*"))


def test_nonfinite_intervals_and_bad_counts_rejected(workspace, tmp_path):
    with pytest.raises(ValueError, match="positive review counts"):
        prepare_label_batch(workspace, tmp_path / "bad-count", frame_count=2, possession_count=3)
    parent = workspace.get("possession", "p")
    workspace.save("possession", {**parent, "end": 300})
    with pytest.raises(ValueError, match="outside"):
        prepare_label_batch(workspace, tmp_path / "outside", frame_count=2, possession_count=1)


def test_notebook_cells_compile_and_archive_extraction_checks_integrity(tmp_path):
    notebook = json.loads((Path(__file__).parents[1] / "notebooks/train_colab.ipynb").read_text())
    cells = [''.join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for source in cells:
        compile(source, "colab_cell", "exec")
    extraction = next(source for source in cells if "def extract_verified" in source)
    function = next(node for node in ast.parse(extraction).body if isinstance(node, ast.FunctionDef))
    namespace = {"hashlib": hashlib, "zipfile": zipfile}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "extraction", "exec"), namespace)
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("playlab/module.py", "# clean source")
    digest = file_hash(archive)
    result = namespace["extract_verified"](archive, digest, tmp_path / "clean")
    assert (result / "playlab/module.py").read_text() == "# clean source"
    with pytest.raises((AssertionError, ValueError), match="changed"):
        namespace["extract_verified"](archive, "f" * 64, tmp_path / "bad-hash")
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.py", "# invalid")
    with pytest.raises((AssertionError, ValueError), match="Unsafe zip"):
        namespace["extract_verified"](archive, file_hash(archive), tmp_path / "unsafe")
    assert not (tmp_path / "escape.py").exists()
