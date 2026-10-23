"""Basketball Play Lab web app: pick a video, analyze a short clip in the background,
read made/missed shots on a frame-scrubbable timeline.

Run: uv run --extra vision playlab web   (then open http://localhost:8600)

Long work (YouTube downloads, analysis) runs on one background thread, one job at a
time, so the page stays usable; the browser polls /api/jobs for progress.
"""
from __future__ import annotations

import queue
import threading
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

from playlab.media import is_youtube_url
from playlab.quick_analysis import MAX_DURATION, get_analysis_clip, queue_analysis_clip
from playlab.quick_import import import_quick_source
from playlab.store import Store
from playlab.worker import run_job_now

PAGE = Path(__file__).with_name("web") / "index.html"
ACTIVE = {"queued", "running"}


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
        threading.Thread(target=self._run, daemon=True, name="playlab-jobs").start()

    def submit(self, job_id: str, **label):
        self.labels[job_id] = label
        self.pending.put(job_id)

    def _run(self):
        while True:
            job_id = self.pending.get()
            try:
                run_job_now(self.store, job_id)
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
        return rows


def _video(game: dict) -> dict:
    offset = float(game.get("source_offset_seconds") or 0)
    return {"id": game["id"], "name": game.get("name") or "Video", "start": offset,
            "end": offset + float(game.get("duration") or 0),
            "length": hms(game.get("duration") or 0), "youtube": game.get("source_url")}


def _summary(view: dict) -> dict | None:
    clip, result = view, view.get("result") or {}
    events = result.get("events") or []
    made = sum(e.get("result") == "made" for e in events)
    missed = sum(e.get("result") == "missed" for e in events)
    shots = [{"time": hms(e.get("release", e.get("time"))), "seconds": float(e.get("release", e.get("time"))),
              "at": float(e.get("release", e.get("time"))) - clip["original_start"], "result": e.get("result")}
             for e in sorted(events, key=lambda e: e.get("release", e.get("time")))]
    return {"id": clip["id"], "video_id": clip["game_id"], "start": clip["original_start"], "end": clip["original_end"],
            "range": f"{hms(clip['original_start'])} – {hms(clip['original_end'])}", "made": made, "missed": missed,
            "unclear": len(events) - made - missed, "shots": shots, "created": clip.get("created_at"),
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
                           "rim": norm(frame["rim"], width, height) if frame.get("rim") else None})
    frames.sort(key=lambda f: f["t"])
    step = min((b["t"] - a["t"] for a, b in zip(frames, frames[1:]) if b["t"] > a["t"]), default=0.2)
    return {"step": round(step, 3), "frames": frames}


def create_app(data_dir: str | Path = "data") -> Starlette:
    store = Store(data_dir)
    jobs = Jobs(store)
    # Freshness checks hash model files, so cache them until a model or result changes.
    fresh_cache: dict = {}

    def model_stamp():
        models = Path(__file__).resolve().parents[1] / "artifacts" / "models"
        return tuple(sorted((p.name, p.stat().st_mtime_ns) for p in models.glob("*.pt")))

    def finished(clip_id):
        key = (clip_id, model_stamp())
        if key in fresh_cache:
            return fresh_cache[key]
        view = get_analysis_clip(store, clip_id)
        if view and view.get("fresh") and view.get("result"):
            fresh_cache[key] = view  # only finished results; pending ones are checked again
            return view
        return None

    async def page(request):
        return FileResponse(PAGE, media_type="text/html")

    async def videos(request):
        if request.method == "POST":
            body = await request.json()
            source = str(body.get("source", "")).strip()
            if is_youtube_url(source):
                known = next((g for g in store.list("game") if g.get("source_url") == source), None)
                if known:
                    return JSONResponse({"video": _video(known)})
                job_id = store.enqueue("import", {"source": source, "name": None})
                jobs.submit(job_id, kind="download", title="Downloading from YouTube")
                return JSONResponse({"job": job_id}, status_code=202)
            try:
                game = import_quick_source(store, source)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                return JSONResponse({"error": f"Could not open that video: {exc}"}, status_code=400)
            return JSONResponse({"video": _video(game)})
        rows = [_video(g) for g in store.list("game")]
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
            jobs.submit(clip["job_id"], kind="analysis", video_id=game["id"], clip_id=clip["id"],
                        title=f"{hms(start)} – {hms(end)}")
        return JSONResponse({"job": clip["job_id"], "clip_id": clip["id"]}, status_code=202)

    async def job_list(request):
        return JSONResponse(jobs.visible())

    async def analyses(request):
        video_id = request.query_params.get("video")
        clips = [c for c in store.list("analysis_clip") if not video_id or c["game_id"] == video_id]
        rows = [_summary(view) for c in clips if (view := finished(c["id"]))]
        return JSONResponse(sorted(rows, key=lambda r: r["created"] or "", reverse=True))

    async def analysis(request):
        view = finished(request.path_params["clip_id"])
        return JSONResponse(_summary(view)) if view else JSONResponse({"error": "Not ready"}, status_code=404)

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
        Route("/api/analyze", analyze, methods=["POST"]), Route("/api/jobs", job_list),
        Route("/api/analyses", analyses), Route("/api/analyses/{clip_id}", analysis),
        Route("/api/analyses/{clip_id}/boxes", boxes),
        Route("/media/{clip_id}/playback.mp4", media),
    ])


def serve(data_dir="data", port=8600):
    import uvicorn

    print(f"Basketball Play Lab: http://localhost:{port}")
    uvicorn.run(create_app(data_dir), host="127.0.0.1", port=port, log_level="warning")
