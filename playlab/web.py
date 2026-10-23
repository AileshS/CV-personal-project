"""Basketball Play Lab web app: pick a video, analyze a short clip in the background,
read made/missed shots on a frame-scrubbable timeline.

Run: uv run --extra vision playlab web   (then open http://localhost:8600)
Phone: see "Use it on your phone" in README.md (Tailscale, or --host 0.0.0.0 with PLAYLAB_PASSCODE).

Long work (YouTube downloads, analysis) runs on one background thread, one job at a
time, so the page stays usable; the browser polls /api/jobs for progress.
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import queue
import socket
import threading
from pathlib import Path
from urllib.parse import parse_qs

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route

from playlab import teams
from playlab.media import is_youtube_url
from playlab.quick_analysis import MAX_DURATION, get_analysis_clip, queue_analysis_clip
from playlab.quick_import import import_quick_source
from playlab.store import Store
from playlab.worker import run_job_now

PAGE = Path(__file__).with_name("web") / "index.html"
ACTIVE = {"queued", "running"}
COOKIE = "playlab_pass"
LOGIN_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Play Lab</title>
<style>body{font:16px system-ui,sans-serif;max-width:22rem;margin:20vh auto;padding:0 16px}
input,button{font:inherit;width:100%;padding:.7rem;margin-top:.6rem;box-sizing:border-box}</style></head>
<body><h1>Basketball Play Lab</h1><form method="post" action="/login">
<input type="password" name="passcode" placeholder="Passcode" autofocus autocomplete="current-password">
<button>Open</button></form>{error}</body></html>"""


def hms(seconds) -> str:
    """Whole seconds as HH:MM:SS (hours always shown)."""
    total = max(0, int(round(float(seconds))))
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def parse_time(value) -> float:
    """Accept 990, '990', '16:30' or '00:16:30'."""
    text = str(value).strip()
    if not text:
        raise ValueError("Enter a start time.")
    parts = text.split(":")
    if len(parts) > 3 or not all(p.strip().replace(".", "", 1).isdigit() for p in parts):
        raise ValueError("Use hh:mm:ss, mm:ss or seconds.")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def stage_label(job: dict) -> str:
    checkpoint = job.get("checkpoint") if isinstance(job.get("checkpoint"), dict) else {}
    stage = checkpoint.get("stage")
    if job["status"] == "queued":
        return "Waiting for the previous job"
    if stage == "downloading":
        done, total = checkpoint.get("bytes_done") or 0, checkpoint.get("bytes_total")
        return f"Downloading · {done / 1e9:.2f} of ~{total / 1e9:.2f} GB" if total else "Downloading"
    if stage == "preparing_playback":
        return "Preparing the clip"
    if stage == "detecting" and checkpoint.get("seconds_total"):
        return f"Finding the ball and rim · {checkpoint['seconds_done']:.0f}s of {checkpoint['seconds_total']:.0f}s"
    if stage == "checking_shots":
        return f"Checking shot {checkpoint.get('shot')} of {checkpoint.get('shots')}"
    return "Starting"


class Jobs:
    """One background thread runs queued jobs in order (analysis and downloads are CPU/IO heavy)."""

    def __init__(self, store: Store):
        self.store, self.pending = store, queue.Queue()
        self.labels: dict[str, dict] = {}
        self.tasks: dict[str, dict] = {}
        threading.Thread(target=self._run, daemon=True, name="playlab-jobs").start()

    def submit(self, job_id: str, **label):
        self.labels[job_id] = label
        self.pending.put(job_id)

    def submit_task(self, task_id: str, work, **label):
        """Run work(progress) on the same thread; for work that is not a store job."""
        self.tasks[task_id] = {"id": task_id, **label, "status": "queued", "progress": 0., "stage": "Waiting",
                               "error": None}
        self.pending.put((task_id, work))

    def _run(self):
        while True:
            item = self.pending.get()
            if isinstance(item, tuple):
                task_id, work = item
                task = self.tasks[task_id]
                task.update(status="running", stage=task.get("running_stage", "Working"))
                try:
                    work(lambda fraction: task.update(progress=float(fraction)))
                    task.update(status="done", progress=1.)
                except Exception as exc:  # shown in the tray; the page keeps working
                    task.update(status="failed", error=str(exc))
                continue
            try:
                run_job_now(self.store, item)
            except Exception:  # run_job_now records failures on the job itself
                pass

    def visible(self):
        jobs = {j["id"]: j for j in self.store.jobs()}
        rows = []
        for job_id, label in self.labels.items():
            job = jobs.get(job_id)
            if job:
                rows.append({"id": job_id, **label, "status": job["status"], "progress": job.get("progress") or 0,
                             "stage": stage_label(job), "error": job.get("error")})
        return rows + [{k: v for k, v in task.items() if k != "running_stage"} for task in self.tasks.values()]


SHADE_NAMES = {"light": "Light jerseys", "dark": "Dark jerseys"}


def _teams(store: Store, game: dict, views: list[dict]) -> dict:
    """Who took each shot, worked out from this video's analyzed shots (no setup needed).

    Halftime comes from the scoreboard's period digit; the shooters' jerseys vote for which
    shade attacks which basket in each half; the vote then labels every shot.
    """
    saved = store.get("team_setup", game["id"]) or {}
    names = {shade: saved.get(shade) or default for shade, default in SHADE_NAMES.items()}
    periods = store.get("game_periods", game["id"])
    halftime = periods.get("halftime") if periods else None
    shots = []
    for view in views:
        result = view.get("result") or {}
        frames = result.get("diagnostic_frames") or []
        for event in result.get("events") or []:
            shots.append({"seconds": float(event.get("release", event.get("time"))),
                          "basket": _event_basket(event, frames), "lightness": event.get("shooter_lightness"),
                          "visible": event.get("players_lightness") or []})
    split = teams.shade_split([v for shot in shots for v in shot["visible"]])
    for shot in shots:
        shot["shade"] = teams.jersey_shade(shot["lightness"], split)
    owner, votes, against = teams.side_owner(shots, halftime) if periods and halftime else (None, 0, 0)
    status = "reading" if not periods else "found" if halftime else "unreadable"
    return {"names": names, "split": split, "halftime": halftime, "halftime_status": status,
            "left_first": owner, "votes": votes, "against": against,
            "shooters": sum(shot["shade"] is not None for shot in shots), "shots": len(shots)}


def shot_team(ctx: dict, basket: str | None, seconds: float, lightness=None) -> str | None:
    """The shooting team's name: from the video's vote when there is one, else this shot's own shooter."""
    shade = teams.shot_shade(ctx.get("left_first"), basket, seconds, ctx.get("halftime"))
    if shade is None and ctx.get("halftime_status") != "found":
        shade = teams.jersey_shade(lightness, ctx.get("split"))
    return ctx["names"][shade] if shade else None


def _event_basket(event: dict, frames: list[dict]) -> str | None:
    """The saved basket side, or for older results the side of the rim seen in the 3 s after release."""
    if event.get("basket"):
        return event["basket"]
    release = float(event.get("release", event.get("time")))
    spots = sorted((f["rim"]["bbox"][0] + f["rim"]["bbox"][2]) / 2 / f["width"] for f in frames
                   if f.get("rim") and f.get("width") and release <= f["time"] <= release + 3)
    if not spots:
        return None
    return "left" if spots[len(spots) // 2] < .5 else "right"


def _video(game: dict, teams: dict | None = None) -> dict:
    offset = float(game.get("source_offset_seconds") or 0)
    row = {"id": game["id"], "name": game.get("name") or "Video", "start": offset,
           "end": offset + float(game.get("duration") or 0),
           "length": hms(game.get("duration") or 0), "youtube": game.get("source_url")}
    if teams:
        row["teams"] = {key: teams[key] for key in ("names", "halftime_status", "left_first", "votes", "against",
                                                    "shooters", "shots")}
        row["teams"]["halftime"] = hms(teams["halftime"]) if teams["halftime"] is not None else None
    return row


def _summary(view: dict, teams: dict | None = None) -> dict | None:
    clip, result = view, view.get("result") or {}
    events = result.get("events") or []
    frames = result.get("diagnostic_frames") or []
    made = sum(e.get("result") == "made" for e in events)
    missed = sum(e.get("result") == "missed" for e in events)
    shots = []
    for e in sorted(events, key=lambda e: e.get("release", e.get("time"))):
        seconds = float(e.get("release", e.get("time")))
        basket = _event_basket(e, frames)
        shots.append({"time": hms(seconds), "seconds": seconds, "at": seconds - clip["original_start"],
                      "result": e.get("result"), "basket": basket,
                      "team": shot_team(teams, basket, seconds, e.get("shooter_lightness")) if teams else None})
    by_team = {}
    for shot in shots:
        if shot["team"] and shot["result"] in ("made", "missed"):
            by_team.setdefault(shot["team"], {"made": 0, "missed": 0})[shot["result"]] += 1
    return {"id": clip["id"], "video_id": clip["game_id"], "start": clip["original_start"], "end": clip["original_end"],
            "range": f"{hms(clip['original_start'])} – {hms(clip['original_end'])}", "made": made, "missed": missed,
            "unclear": len(events) - made - missed, "shots": shots, "by_team": by_team, "created": clip.get("created_at"),
            "playback": f"/media/{clip['id']}/playback.mp4" if view.get("playback_path") else None}


def _boxes(store: Store, clip: dict) -> dict:
    """Ball/rim detections saved during the scan, as fractions of the frame, keyed by clip time.

    The scan samples a few frames per second, so the page shows the nearest sample.
    """
    def norm(box, width, height):
        x1, y1, x2, y2 = box["bbox"]
        return [round(x1 / width, 4), round(y1 / height, 4), round(x2 / width, 4), round(y2 / height, 4),
                round(float(box.get("confidence", 1)), 2)]

    frames = []
    for chunk in store.list("quick_analysis_chunk", clip_id=clip["id"]):
        for frame in chunk.get("frames", []):
            width, height = float(frame.get("width") or 0), float(frame.get("height") or 0)
            if width <= 0 or height <= 0:
                continue
            raw = frame.get("raw_detections") or {}
            frames.append({"t": round(float(frame["time"]) - float(clip["start"]), 3),
                           "balls": [norm(b, width, height) for b in raw.get("balls", [])],
                           "rims": [norm(r, width, height) for r in raw.get("rims", [])],
                           "rim": norm(frame["rim"], width, height) if frame.get("rim") else None,
                           "ball": norm(frame["ball"], width, height) if frame.get("ball") else None})
    frames.sort(key=lambda f: f["t"])
    step = min((b["t"] - a["t"] for a, b in zip(frames, frames[1:]) if b["t"] > a["t"]), default=0.2)
    return {"step": round(step, 3), "frames": frames}


def passcode_guard(passcode: str) -> Middleware:
    """Ask for the passcode once per browser (cookie lasts 30 days) before any page, API or video."""
    token = hashlib.sha256(f"playlab:{passcode}".encode()).hexdigest()

    async def login(request):
        if request.method == "POST":
            given = parse_qs((await request.body()).decode()).get("passcode", [""])[0]
            if hmac.compare_digest(given.encode(), passcode.encode()):
                response = RedirectResponse("/", status_code=303)
                response.set_cookie(COOKIE, token, max_age=30 * 86400, httponly=True, samesite="lax")
                return response
            return HTMLResponse(LOGIN_PAGE.replace("{error}", "<p>Wrong passcode.</p>"), status_code=401)
        return HTMLResponse(LOGIN_PAGE.replace("{error}", ""))

    async def dispatch(request, call_next):
        if request.url.path == "/login":
            return await login(request)
        if hmac.compare_digest(request.cookies.get(COOKIE, ""), token):
            return await call_next(request)
        if request.url.path == "/":
            return RedirectResponse("/login", status_code=303)
        return JSONResponse({"error": "Enter the passcode first."}, status_code=401)

    return Middleware(BaseHTTPMiddleware, dispatch=dispatch)


def create_app(data_dir: str | Path = "data", passcode: str | None = None) -> Starlette:
    store = Store(data_dir)
    jobs = Jobs(store)
    # Freshness checks hash model files, so cache them until a model or result changes.
    fresh_cache: dict = {}

    def model_stamp():
        root = Path(__file__).resolve().parent
        models = root.parent / "artifacts" / "models"
        code = [root / name for name in ("analysis.py", "detectors.py", "features.py")]
        return tuple(sorted((p.name, p.stat().st_mtime_ns) for p in [*models.glob("*.pt"), *code]))

    def finished(clip_id):
        key = (clip_id, model_stamp())
        if key in fresh_cache:
            return fresh_cache[key]
        view = get_analysis_clip(store, clip_id)
        if view and view.get("fresh") and view.get("result"):
            fresh_cache[key] = view
            return view
        if view and (view.get("result") or {}).get("status") == "complete":
            fresh_cache[key] = None  # made by an older model or code: stays hidden until those change
        return None  # pending results are checked again

    async def page(request):
        return FileResponse(PAGE, media_type="text/html")

    async def videos(request):
        if request.method == "POST":
            body = await request.json()
            source = str(body.get("source", "")).strip()
            if is_youtube_url(source):
                known = next((g for g in store.list("game") if g.get("source_url") == source), None)
                if known:
                    return JSONResponse({"video": video_row(known)})
                job_id = store.enqueue("import", {"source": source, "name": None})
                jobs.submit(job_id, kind="download", title="Downloading from YouTube")
                return JSONResponse({"job": job_id}, status_code=202)
            try:
                game = import_quick_source(store, source)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                return JSONResponse({"error": f"Could not open that video: {exc}"}, status_code=400)
            return JSONResponse({"video": video_row(game)})
        rows = [video_row(g) for g in store.list("game")]
        return JSONResponse(sorted(rows, key=lambda v: v["name"].lower()))

    async def analyze(request):
        body = await request.json()
        game = store.get("game", str(body.get("video_id")))
        if not game:
            return JSONResponse({"error": "Choose a video first."}, status_code=400)
        try:
            start = parse_time(body.get("start", ""))
            length = min(MAX_DURATION, max(1., float(body.get("length", 20))))
            offset = float(game.get("source_offset_seconds") or 0)
            end = min(start + length, offset + float(game["duration"]))
            clip = queue_analysis_clip(store, game["id"], start - offset, end - offset)
        except (ValueError, TypeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        if clip.get("status") in ACTIVE:
            for key in [k for k in fresh_cache if k[0] == clip["id"]]:
                del fresh_cache[key]  # a rerun replaces the old result under the same clip id
            jobs.submit(clip["job_id"], kind="analysis", video_id=game["id"], clip_id=clip["id"],
                        title=f"{hms(start)} – {hms(end)}")
        read_periods(game)
        return JSONResponse({"job": clip["job_id"], "clip_id": clip["id"]}, status_code=202)

    async def job_list(request):
        return JSONResponse(jobs.visible())

    def video_views(game_id):
        return [view for c in store.list("analysis_clip", game_id=game_id) if (view := finished(c["id"]))]

    def video_row(game):
        return _video(game, _teams(store, game, video_views(game["id"])))

    def read_periods(game):
        """Find halftime once per video, in the background (it reads a frame every 30 s)."""
        task_id = f"periods-{game['id']}"
        if store.get("game_periods", game["id"]) or task_id in jobs.tasks and jobs.tasks[task_id]["status"] != "failed":
            return

        def work(progress):
            found = teams.find_halftime(game["source_path"], float(game.get("duration") or 0), progress=progress)
            store.save("game_periods", {"id": game["id"], **found})

        jobs.submit_task(task_id, work, kind="periods", video_id=game["id"], title="Finding halftime",
                         running_stage="Reading the period on the scoreboard")

    async def analyses(request):
        video_id = request.query_params.get("video")
        clips = [c for c in store.list("analysis_clip") if not video_id or c["game_id"] == video_id]
        views = [view for c in clips if (view := finished(c["id"]))]
        contexts = {}
        rows = []
        for view in views:
            if view["game_id"] not in contexts:
                game = store.get("game", view["game_id"])
                contexts[view["game_id"]] = _teams(store, game, video_views(game["id"])) if game else None
            rows.append(_summary(view, contexts[view["game_id"]]))
        return JSONResponse(sorted(rows, key=lambda r: r["created"] or "", reverse=True))

    async def analysis(request):
        view = finished(request.path_params["clip_id"])
        if not view:
            return JSONResponse({"error": "Not ready"}, status_code=404)
        game = store.get("game", view["game_id"])
        return JSONResponse(_summary(view, _teams(store, game, video_views(game["id"])) if game else None))

    async def team_names(request):
        game = store.get("game", request.path_params["video_id"])
        if not game:
            return JSONResponse({"error": "Video not found."}, status_code=404)
        body = await request.json()
        names = {shade: str(body.get(shade) or "").strip()[:60] for shade in SHADE_NAMES}
        store.save("team_setup", {"id": game["id"], **{k: v for k, v in names.items() if v}})
        return JSONResponse({"video": video_row(game)})

    async def boxes(request):
        view = finished(request.path_params["clip_id"])
        if not view:
            return JSONResponse({"error": "Not ready"}, status_code=404)
        return JSONResponse(_boxes(store, view))

    async def media(request):
        clip_id = request.path_params["clip_id"]
        if not clip_id.isalnum():
            return Response(status_code=404)
        path = store.root / "quick_analysis" / clip_id / "playback.mp4"
        return FileResponse(path, media_type="video/mp4") if path.is_file() else Response(status_code=404)

    return Starlette(routes=[
        Route("/", page), Route("/api/videos", videos, methods=["GET", "POST"]),
        Route("/api/videos/{video_id}/teams", team_names, methods=["POST"]),
        Route("/api/analyze", analyze, methods=["POST"]), Route("/api/jobs", job_list),
        Route("/api/analyses", analyses), Route("/api/analyses/{clip_id}", analysis),
        Route("/api/analyses/{clip_id}/boxes", boxes),
        Route("/media/{clip_id}/playback.mp4", media),
    ], middleware=[passcode_guard(passcode)] if passcode else [])


def _is_loopback(host: str) -> bool:
    try:
        return host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def phone_addresses(port: int) -> list[str]:
    """This computer's IPv4 addresses a phone could open, labelled Tailscale or same Wi-Fi."""
    try:
        found = {info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}
    except OSError:
        return []
    tailnet = ipaddress.ip_network("100.64.0.0/10")
    lines = []
    for address in sorted(found):
        ip = ipaddress.ip_address(address)
        if ip.is_loopback or ip.is_link_local:
            continue
        label = "Tailscale (from anywhere)" if ip in tailnet else "Same Wi-Fi"
        lines.append(f"{label}: http://{address}:{port}")
    return lines


def serve(data_dir="data", port=8600, host="127.0.0.1", passcode=None):
    import uvicorn

    if not _is_loopback(host) and not passcode:
        raise ValueError("Set PLAYLAB_PASSCODE before opening the website to other devices "
                         "(anyone on the network could otherwise start downloads and read results).")
    print(f"Basketball Play Lab: http://localhost:{port}")
    if not _is_loopback(host):
        for line in phone_addresses(port):
            print(line)
    uvicorn.run(create_app(data_dir, passcode=passcode), host=host, port=port, log_level="warning")
