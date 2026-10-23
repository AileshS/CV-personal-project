"""Website API on artificial data; detection is faked so background jobs finish in seconds."""
import time

import numpy as np
import pytest
from starlette.testclient import TestClient

from playlab import analysis, web
from playlab import quick_analysis as quick
from playlab.store import Store


def test_times_are_hh_mm_ss_and_parse_flexibly():
    assert web.hms(0) == "00:00:00" and web.hms(1003.4) == "00:16:43" and web.hms(7038.4) == "01:57:18"
    assert web.parse_time("00:16:30") == web.parse_time("16:30") == web.parse_time("990") == 990
    for bad in ("", "1:2:3:4", "ab:cd"):
        with pytest.raises(ValueError):
            web.parse_time(bad)


@pytest.fixture
def site(tmp_path, monkeypatch):
    store = Store(tmp_path / "data")
    source = tmp_path / "game.mp4"
    source.write_bytes(b"fixture; decoding is replaced")
    store.save("game", {"id": "g", "name": "Oakridge vs St. Marks", "source_path": str(source), "duration": 3000.,
                        "source_offset_seconds": 0., "split": "train"})

    class OneShot:
        """A ball dropping through the middle of a rim around t=12 s: one made shot."""
        mode = "fixture"

        def __init__(self, *args):
            self.t = 0

        def reset(self):
            pass

        def __call__(self, frame):
            return {"balls": [], "rims": [{"bbox": [100, 100, 110, 110], "confidence": .9}], "players": []}

    def frames(path, start, end, **kwargs):
        for at in np.arange(start, end, .2):
            yield float(at), np.zeros((80, 160, 3), dtype=np.uint8)

    def playback(source, destination, *args, **kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"mp4 fixture")
        return destination

    monkeypatch.setattr(quick, "clip_video", playback)
    monkeypatch.setattr(analysis, "BasketballDetector", OneShot)
    monkeypatch.setattr(analysis, "iter_frames", frames)
    monkeypatch.setattr(analysis, "frame_at", lambda *a, **kw: np.zeros((80, 160, 3), dtype=np.uint8))
    monkeypatch.setattr(analysis, "default_ball_rim_weights", lambda: None)
    monkeypatch.setattr(analysis, "_reached_rim", lambda features: True)
    monkeypatch.setattr(analysis, "shot_proposals", lambda observations: [990. + 5])
    return TestClient(web.create_app(tmp_path / "data")), store


def test_page_and_video_list(site):
    client, _ = site
    assert "Play Lab" in client.get("/").text
    video, = client.get("/api/videos").json()
    assert video == {"id": "g", "name": "Oakridge vs St. Marks", "start": 0.0, "end": 3000.0, "length": "00:50:00",
                     "youtube": None}
    assert client.post("/api/videos", json={"source": "C:/missing.mp4"}).status_code == 400


def test_analyze_runs_in_background_and_reports_shots_with_timestamps(site):
    client, store = site
    assert client.post("/api/analyze", json={"video_id": "g", "start": "nope"}).status_code == 400
    started = client.post("/api/analyze", json={"video_id": "g", "start": "00:16:30", "length": 20})
    assert started.status_code == 202
    clip_id = started.json()["clip_id"]
    for _ in range(100):  # the request returned at once; the job finishes on the worker thread
        job, = client.get("/api/jobs").json()
        if job["status"] not in ("queued", "running"):
            break
        time.sleep(.1)
    assert job["status"] == "done" and job["kind"] == "analysis" and job["title"] == "00:16:30 – 00:16:50"
    summary, = client.get("/api/analyses?video=g").json()
    assert summary["id"] == clip_id and summary["range"] == "00:16:30 – 00:16:50"
    shot, = summary["shots"]
    assert shot["time"] == "00:16:35" and shot["at"] == pytest.approx(5)
    assert summary["made"] + summary["missed"] + summary["unclear"] == 1
    assert client.get(f"/api/analyses/{clip_id}").json()["playback"] == f"/media/{clip_id}/playback.mp4"
    assert client.get(f"/media/{clip_id}/playback.mp4").status_code == 200
    boxes = client.get(f"/api/analyses/{clip_id}/boxes").json()
    assert boxes["step"] == pytest.approx(.2) and len(boxes["frames"]) >= 90   # 20 s sampled every 0.2 s
    first = boxes["frames"][0]
    assert first["t"] == pytest.approx(0) and first["balls"] == []
    assert first["rims"] == [[0.625, 1.25, 0.6875, 1.375, 0.9]]             # fractions of the 160x80 frame
    assert client.get("/api/analyses/unknown/boxes").status_code == 404
    assert client.get("/media/..%2F..%2Fsecret/playback.mp4").status_code == 404
