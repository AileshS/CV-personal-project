"""Real video/codec checks with explicit fake model boxes, not accuracy evidence."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from playlab import tracked_video
from playlab.dataset import file_hash
from playlab.media import frame_at, make_fixture, probe_video


def fake_box(category, confidence, box, track_id=None):
    return SimpleNamespace(cls=np.array([category]), conf=np.array([confidence]),
                           xyxy=np.array([box]), id=np.array([track_id]) if track_id is not None else None)


class FakeModel:
    names = {0: "Basketball", 1: "Basketball Hoop"}

    def __init__(self):
        self.calls = []

    def track(self, frame, **kwargs):
        assert kwargs["persist"] is True
        assert Path(kwargs["tracker"]).name == "bytetrack.yaml"
        assert json.loads(Path(kwargs["tracker"]).read_text())["tracker_type"] == "bytetrack"
        assert kwargs["save"] is False
        self.calls.append((frame.shape, kwargs))
        if len(self.calls) == 1:
            boxes = [fake_box(0, .815123, [80, 60, 110, 90], 7),
                     fake_box(1, .888543, [210, 100, 290, 125], 8)]
        elif len(self.calls) == 3:
            boxes = [fake_box(0, .123456, [90, 70, 115, 95], 7)]
        else:
            boxes = []
        return [SimpleNamespace(boxes=boxes)]


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return make_fixture(tmp_path_factory.mktemp("tracked-source") / "source.mp4", duration=1.2)


@pytest.fixture
def model(tmp_path, monkeypatch):
    weights = tmp_path / "explicit-fake-model.pt"
    weights.write_bytes(b"fake model, never loaded as a real checkpoint")
    model = FakeModel()
    monkeypatch.setattr(tracked_video, "_load_model", lambda path: model)
    return model, weights


def test_real_overlay_video_retains_audio_time_mapping_and_current_raw_scores(source, model, tmp_path):
    fake_model, weights = model
    destination = tmp_path / "rendered"
    result = tracked_video.render_tracked_video(source, destination, start=.1, end=.6,
                                                weights=weights, source_offset_seconds=750)
    metadata = probe_video(result["video_path"])
    assert metadata["has_audio"] and metadata["duration"] == pytest.approx(.5, abs=.06)
    assert result["fps"] == 20 and result["frame_count"] == len(fake_model.calls) == 10
    assert result["source_start"] == pytest.approx(750.1)
    assert result["source_end"] == pytest.approx(750.6)
    assert result["model_sha256"] == file_hash(weights)
    assert result["video_sha256"] == file_hash(result["video_path"])
    assert result["samples_sha256"] == file_hash(result["samples_path"])
    assert result["frames_with_detections"] == {"ball": 2, "rim": 1}
    samples = json.loads(Path(result["samples_path"]).read_text())
    first, missing, low = samples["frames"][:3]
    assert first["source_seconds"] == pytest.approx(750.1)
    assert missing["source_seconds"] == pytest.approx(750.15)
    assert first["detections"][0]["label"] == "ball"
    assert first["detections"][1]["label"] == "rim"
    assert first["detections"][0]["confidence"] == .815123
    assert first["detections"][0]["confidence_percent"] == pytest.approx(81.5123)
    assert low["detections"][0]["confidence"] == .123456
    assert first["detections"][0]["track_id"] == 7
    assert missing["detections"] == []  # No stale box/score from the previous frame.
    assert all(detection["reviewed"] is False for sample in samples["frames"]
               for detection in sample["detections"])
    assert not list(destination.rglob("*.sqlite*"))
    assert set(path.name for path in destination.iterdir()) == {"tracked.mp4", "samples.json", "report.json"}
    pixels = frame_at(result["video_path"], 0)
    ball = pixels[57:94, 77:114]
    rim = pixels[97:129, 207:294]
    assert ((ball[:, :, 0] > 180) & (ball[:, :, 1] > 150) & (ball[:, :, 2] < 160)).sum() > 50
    assert ((rim[:, :, 0] < 120) & (rim[:, :, 1] > 150) & (rim[:, :, 2] > 150)).sum() > 50
    assert pixels.shape == (360, 640, 3)
    info = subprocess.run([tracked_video.ffmpeg_executable(), "-hide_banner", "-i", result["video_path"]],
                          capture_output=True, text=True, timeout=15)
    assert "Video: h264" in info.stderr


def test_cancellation_cleans_staging_and_only_owned_processes(source, model, tmp_path, monkeypatch):
    _, weights = model
    processed = [0]
    processes = []
    original_popen = subprocess.Popen

    def capture_process(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(tracked_video.subprocess, "Popen", capture_process)

    def progress(fraction, checkpoint):
        processed[0] = checkpoint["frames_processed"]

    destination = tmp_path / "cancelled"
    with pytest.raises(InterruptedError, match="cancelled"):
        tracked_video.render_tracked_video(source, destination, start=0, end=1,
                                           weights=weights, cancelled=lambda: processed[0] >= 3,
                                           progress=progress)
    assert not destination.exists()
    assert not list(tmp_path.glob(".cancelled-*"))
    assert processes and all(process.poll() is not None for process in processes)
    assert source.is_file() and weights.is_file()


def test_inference_failure_has_no_fallback_or_published_video(source, model, tmp_path, monkeypatch):
    fake_model, weights = model

    def fail(*args, **kwargs):
        raise RuntimeError("explicit inference failure")

    monkeypatch.setattr(fake_model, "track", fail)
    destination = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="explicit inference"):
        tracked_video.render_tracked_video(source, destination, start=0, end=.5, weights=weights)
    assert not destination.exists()
    assert not list(tmp_path.glob(".failed-*"))


@pytest.mark.parametrize("names", [{0: "person", 1: "sports ball"}, {0: "ball"},
                                   {0: "ball", 1: "basketball"}, dict(enumerate(["person"] * 80))])
def test_incompatible_model_classes_reject_before_inference(source, model, tmp_path, names):
    fake_model, weights = model
    fake_model.names = names
    with pytest.raises(ValueError, match="class|classes|model"):
        tracked_video.render_tracked_video(source, tmp_path / "invalid", start=0, end=.5, weights=weights)
    assert fake_model.calls == []
    assert not (tmp_path / "invalid").exists()


def test_public_class_names_map_ball_and_hoop_to_product_labels():
    mapping = tracked_video._class_mapping({0: "Basketball", 1: "Basketball Hoop"})
    assert mapping[0]["label"] == "ball" and mapping[1]["label"] == "rim"
    assert tracked_video._class_mapping(["Sports-ball", "Basketball_Rim"])[1]["label"] == "rim"


@pytest.mark.parametrize("start,end,offset", [(1, 0, 0), (0, 100, 0), (0, .5, -1),
                                           (float("nan"), 1, 0), (0, .5, float("inf"))])
def test_invalid_intervals_reject_without_loading_model(source, model, tmp_path, start, end, offset):
    fake_model, weights = model
    with pytest.raises(ValueError):
        tracked_video.render_tracked_video(source, tmp_path / "invalid", start=start, end=end,
                                           source_offset_seconds=offset, weights=weights)
    assert fake_model.calls == []


def test_existing_output_is_preserved(source, model, tmp_path):
    _, weights = model
    destination = tmp_path / "retained"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("existing output")
    with pytest.raises(ValueError, match="already exists"):
        tracked_video.render_tracked_video(source, destination, start=0, end=.5, weights=weights)
    assert sentinel.read_text() == "existing output"
