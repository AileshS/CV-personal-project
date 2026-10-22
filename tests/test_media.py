import subprocess

import numpy as np
import pytest

from playlab.media import (
    clip_video,
    ffmpeg_executable,
    frame_at,
    import_game,
    iter_frames,
    make_fixture,
    probe_video,
)
from playlab.store import Store


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return make_fixture(tmp_path_factory.mktemp("video") / "source.mp4", duration=3)


def test_audio_clipping_elapsed_timestamps_and_resampling(source, tmp_path):
    meta = probe_video(source)
    assert meta["duration"] == pytest.approx(3, abs=.1)
    assert meta["has_audio"]
    frames = list(iter_frames(source, start=.5, end=2, sample_fps=4, max_width=320))
    assert [t for t, _ in frames] == pytest.approx([.5,.75,1,1.25,1.5,1.75])
    assert frames[0][1].shape == (180,320,3)
    reference = frame_at(source, 1, max_width=320)
    assert np.abs(frames[2][1].astype(float)-reference).mean() < 10
    clipped = clip_video(source, tmp_path / "clip.mp4", .5, 2)
    result = probe_video(clipped)
    assert result["duration"] == pytest.approx(1.5, abs=.15)
    assert result["has_audio"]


def test_two_hour_source_seek_has_no_short_clip_limit(tmp_path):
    path = tmp_path / "long.mp4"
    subprocess.run([ffmpeg_executable(), "-v", "error", "-y", "-f", "lavfi", "-i",
                    "color=black:s=32x24:r=1", "-t", "7201", "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", str(path)], check=True, timeout=60)
    assert probe_video(path)["duration"] >= 7200
    frames = list(iter_frames(path, start=7198, end=7200, sample_fps=2))
    assert len(frames) == 4
    assert frames[-1][0] == 7199.5


def test_import_is_idempotent_without_copying_game(source, tmp_path):
    store = Store(tmp_path)
    a = import_game(store, source)
    b = import_game(store, source)
    assert a["id"] == b["id"]
    assert len(store.list("game")) == 1


def test_invalid_video_and_seek_rejected(tmp_path):
    invalid = tmp_path / "bad.mp4"
    invalid.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        probe_video(invalid)
    with pytest.raises(ValueError):
        frame_at(invalid, -1)


def test_youtube_download_uses_node_reports_size_progress_and_keeps_title(tmp_path, monkeypatch):
    import json

    from playlab import media

    assert media.is_youtube_url("https://youtu.be/abc") and not media.is_youtube_url("https://vimeo.com/1")
    monkeypatch.setattr(media.shutil, "which", lambda name: None if name == "deno" else "/bin/node")
    assert media._js_runtime_args() == ["--js-runtimes", "node:/bin/node"]
    sources, calls, seen = tmp_path / "store" / "sources", [], []

    def capture(args, timeout=180, cancelled=None, poll=None):
        calls.append(args)
        if "--skip-download" in args:
            return 0, b"abc 1000\n", b""
        (sources / "abc.f137.mp4.part").write_bytes(b"x" * 400)
        media.time.sleep(1.05)  # progress is throttled to once a second
        poll()
        video = sources / "abc.mp4"
        video.write_bytes(b"video")
        (sources / "abc.info.json").write_text(json.dumps({"title": "St. Mark's vs Dallas"}))
        return 0, f"{video}\n".encode(), b""

    monkeypatch.setattr(media, "_capture", capture)
    monkeypatch.setattr(media, "probe_video", lambda path: {"duration": 7038., "width": 1920, "height": 1080})
    monkeypatch.setattr(media, "source_fingerprint", lambda path: "fp")
    game = media.import_game(Store(tmp_path / "store"), "https://www.youtube.com/watch?v=abc",
                             progress=lambda done, total: seen.append((done, total)))
    assert all("node:/bin/node" in args for args in calls)
    assert seen[0] == (0, 1000) and (400, 1000) in seen     # partial file counted, info.json ignored
    assert game["name"] == "St. Mark's vs Dallas" and game["source_url"].endswith("v=abc")
    with pytest.raises(ValueError, match="YouTube only"):
        media.download_source("https://example.com/v.mp4", sources)
