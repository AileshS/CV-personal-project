"""Website API on artificial data; detection is faked so background jobs finish in seconds."""
import time

import numpy as np
import pytest
from starlette.testclient import TestClient

from playlab import analysis, teams, web
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
    monkeypatch.setattr(teams, "find_halftime", lambda *a, **k: {"halftime": None, "readings": 0, "agreement": None})
    return TestClient(web.create_app(tmp_path / "data")), store


def test_page_and_video_list(site):
    client, _ = site
    assert "Play Lab" in client.get("/").text
    video, = client.get("/api/videos").json()
    assert video == {"id": "g", "name": "Oakridge vs St. Marks", "start": 0.0, "end": 3000.0, "length": "00:50:00",
                     "youtube": None, "teams": {"names": {"light": "Light jerseys", "dark": "Dark jerseys"},
                                                "halftime_status": "reading", "left_first": None, "votes": 0,
                                                "against": 0, "shooters": 0, "shots": 0, "halftime": None}}
    assert client.post("/api/videos", json={"source": "C:/missing.mp4"}).status_code == 400


def test_analyze_runs_in_background_and_reports_shots_with_timestamps(site):
    client, store = site
    assert client.post("/api/analyze", json={"video_id": "g", "start": "nope"}).status_code == 400
    started = client.post("/api/analyze", json={"video_id": "g", "start": "00:16:30", "length": 20})
    assert started.status_code == 202
    clip_id = started.json()["clip_id"]
    for _ in range(100):  # the request returned at once; the job finishes on the worker thread
        job = next(j for j in client.get("/api/jobs").json() if j["kind"] == "analysis")
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


def test_passcode_guards_page_api_and_video(tmp_path):
    client = TestClient(web.create_app(tmp_path / "data", passcode="hoops"))
    assert client.get("/", follow_redirects=False).headers["location"] == "/login"
    assert client.get("/api/videos").status_code == 401
    assert client.get("/media/abc/playback.mp4").status_code == 401
    assert client.post("/login", data={"passcode": "wrong"}).status_code == 401
    assert client.post("/login", data={"passcode": "hoops"}, follow_redirects=False).status_code == 303
    assert client.get("/api/videos").json() == [] and "Play Lab" in client.get("/").text


def test_other_devices_need_a_passcode(tmp_path):
    with pytest.raises(ValueError, match="PLAYLAB_PASSCODE"):
        web.serve(tmp_path / "data", host="0.0.0.0")
    assert web._is_loopback("127.0.0.1") and web._is_loopback("localhost") and not web._is_loopback("0.0.0.0")
    assert all(line.split(": ", 1)[1].endswith(":8600") for line in web.phone_addresses(8600))


def test_teams_come_from_halftime_and_shooter_jerseys(tmp_path):
    store = Store(tmp_path / "data")
    game = store.save("game", {"id": "g", "name": "A vs B", "source_path": "x.mp4", "duration": 3000.})
    crowd = [40., 45., 50., 55., 60., 170., 175., 180., 185., 190.]  # dark and light jerseys at the releases

    def view(release, basket, lightness):
        return {"result": {"events": [{"release": release, "basket": basket, "shooter_lightness": lightness,
                                       "players_lightness": crowd}]}}

    # Dark shoots right in the 1st half (two votes, one wrong), so left in the 2nd half.
    views = [view(100, "right", 50.), view(200, "right", 45.), view(300, "left", 48.), view(2000, "left", None)]
    context = web._teams(store, game, views)
    assert context["halftime_status"] == "reading" and context["left_first"] is None
    assert web.shot_team(context, "right", 100, 50.) == "Dark jerseys"   # before halftime is known: own shooter
    store.save("game_periods", {"id": "g", "halftime": 1500., "readings": 90, "agreement": 1.})
    store.save("team_setup", {"id": "g", "light": "Oakridge"})
    context = web._teams(store, game, views)
    assert (context["left_first"], context["votes"], context["against"]) == ("light", 2, 1)
    assert [web.shot_team(context, side, at) for side, at in (("left", 300), ("right", 100), ("left", 2000), (None, 5))] == [
        "Oakridge", "Dark jerseys", "Dark jerseys", None]
    assert web._video(game, context)["teams"]["halftime"] == "00:25:00"
