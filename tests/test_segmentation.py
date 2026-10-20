import numpy as np
import pytest

from playlab import segmentation
from playlab.store import Store


def test_same_team_offensive_rebounds_do_not_split():
    samples = [{"time": i / 2, "activity": .1, "team": "home"} for i in range(80)]
    boundaries = segmentation.boundaries_from_samples(samples, 40)
    assert boundaries == [(0.0, 40.0, "Video start")]


def test_control_change_requires_sustained_evidence():
    samples = [{"time": i / 2, "activity": .1, "team": "a" if i < 20 else "b"} for i in range(60)]
    boundaries = segmentation.boundaries_from_samples(samples, 30)
    assert len(boundaries) == 2
    assert boundaries[0][1] == 10
    assert "control" in boundaries[1][2]
    for i in range(20, 60):
        samples[i]["team"] = "a" if i % 2 else "b"
    assert len(segmentation.boundaries_from_samples(samples, 30)) == 1


def test_scene_cut_alone_never_creates_possession():
    samples = [{"time": i, "activity": .9 if i == 10 else .1, "team": None} for i in range(30)]
    assert len(segmentation.boundaries_from_samples(samples, 30)) == 1


def test_proposals_preserve_manual_boundary_corrections(tmp_path, monkeypatch):
    store = Store(tmp_path)
    store.save("game", {"id": "g", "source_path": "unused.mp4", "duration": 30})
    def iterator(*args, **kwargs):
        for i in range(60):
            yield i / 2, np.full((50, 100, 3), i % 255, dtype=np.uint8)
    monkeypatch.setattr(segmentation, "iter_frames", iterator)
    first = segmentation.propose_possessions(store, "g")
    assert first[0]["reviewed"] is False
    store.save("possession", {**first[0], "start": 2., "reviewed": True})
    result = segmentation.propose_possessions(store, "g")
    assert result[0]["start"] == 2.
    assert result[0]["reviewed"] is True


def test_cancelled_scan_retains_checkpoint(tmp_path, monkeypatch):
    store = Store(tmp_path)
    store.save("game", {"id": "g", "source_path": "unused.mp4", "duration": 30})
    monkeypatch.setattr(segmentation, "iter_frames", lambda *a, **kw: iter([
        (0., np.zeros((50, 100, 3), dtype=np.uint8))]))
    with pytest.raises(InterruptedError):
        segmentation.propose_possessions(store, "g", cancelled=lambda: True)
    assert store.list("segmentation_checkpoint")
    assert not store.list("possession")


def test_split_inheritance_respects_guarded_contiguous_blocks():
    game = {"duration": 100, "split": "train", "split_mode": "game"}
    assert segmentation._inherited_split(game, 0, 100) == "train"
    game.update(split_mode="contiguous", split_blocks=[
        {"start": 0, "end": 50, "split": "train"},
        {"start": 50, "end": 100, "split": "test"},
    ])
    assert segmentation._inherited_split(game, 0, 39) == "train"
    assert segmentation._inherited_split(game, 61, 100) == "test"
    assert segmentation._inherited_split(game, 39, 41) == "unassigned"
    assert segmentation._inherited_split(game, 49, 61) == "unassigned"
