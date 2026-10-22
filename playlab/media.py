"""Streaming video access on an elapsed source timeline, including variable FPS.

FFmpeg seeks/decodes by timestamps; sample times never use frame_index/source_fps.
Frame sampling is intentionally approximate to the requested sampling period.
"""
from __future__ import annotations

import glob
import hashlib
import io
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image


def ffmpeg_executable() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _capture(args, timeout=180, cancelled=None, poll=None):
    """Bound output on disk while polling cancellation, including child encoders.

    `poll()` is called about every 0.2 s while the process runs (e.g. to report progress).
    """
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as errors:
        flags = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {
            "start_new_session": True}
        process = subprocess.Popen(args, stdout=out, stderr=errors, **flags)
        started = time.monotonic()
        try:
            while process.poll() is None:
                if cancelled and cancelled():
                    raise InterruptedError("Processing cancelled; partial files are retained for retry")
                if time.monotonic()-started > timeout:
                    raise ValueError("Processing exceeded its time limit")
                if poll:
                    poll()
                time.sleep(.2)
            out.seek(0)
            errors.seek(0, 2)
            errors.seek(max(0, errors.tell()-5000))
            return process.returncode, out.read(), errors.read()
        finally:
            if process.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   check=False, capture_output=True, timeout=15)
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=15)


def _run(args, timeout=180, cancelled=None):
    code, output, error = _capture([ffmpeg_executable(), "-hide_banner", "-loglevel", "error", *args],
                                  timeout=timeout, cancelled=cancelled)
    if code:
        raise ValueError(error.decode(errors="replace")[-3000:] or "Video processing failed")
    return output


def probe_video(path: str | Path) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"Video does not exist: {path}")
    probe = shutil.which("ffprobe")
    if probe:
        try:
            run = subprocess.run([probe, "-v", "error", "-show_format", "-show_streams", "-of", "json",
                                  str(path)], capture_output=True, timeout=30, check=True)
            data = json.loads(run.stdout)
        except (subprocess.SubprocessError, json.JSONDecodeError) as exc:
            raise ValueError("Could not read video metadata. Try exporting a normal MP4.") from exc
        video = next((s for s in data["streams"] if s.get("codec_type") == "video"), None)
        if video is None:
            raise ValueError("This source has no video stream")
        numerator, denominator = video.get("avg_frame_rate", "0/1").split("/")
        fps = float(numerator) / max(float(denominator), 1)
        duration = float(data["format"].get("duration", video.get("duration", 0)))
        width, height = int(video["width"]), int(video["height"])
        audio = any(s.get("codec_type") == "audio" for s in data["streams"])
        start = float(data["format"].get("start_time", 0))
    else:
        # imageio bundles FFmpeg, but not ffprobe. Metadata inspection does not decode the game.
        run = subprocess.run([ffmpeg_executable(), "-hide_banner", "-i", str(path)],
                             capture_output=True, timeout=30)
        info = run.stderr.decode(errors="replace")
        dur = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", info)
        stream = next((line for line in info.splitlines() if "Video:" in line), "")
        size = re.search(r"\b(\d{2,5})x(\d{2,5})\b", stream)
        if not dur or not size:
            raise ValueError("Could not read video dimensions/duration. Try exporting a normal MP4.")
        duration = int(dur[1]) * 3600 + int(dur[2]) * 60 + float(dur[3])
        width, height = map(int, size.groups())
        rate = re.search(r"([\d.]+) fps", stream)
        fps = float(rate[1]) if rate else 0.0
        audio = "Audio:" in info
        match = re.search(r"start:\s*(-?[\d.]+)", info)
        start = float(match[1]) if match else 0.0
    if not math.isfinite(duration) or duration <= 0 or width <= 0 or height <= 0:
        raise ValueError("Video metadata is invalid")
    return {"duration": duration, "width": width, "height": height, "fps": fps,
            "has_audio": audio, "source_start_time": start,
            "timeline": "elapsed seconds from source playback start; timestamp-based FFmpeg seeking"}


def source_fingerprint(path: str | Path) -> str:
    """Fast source identity (size + first/last MiB), not a full cryptographic video checksum."""
    path = Path(path)
    digest = hashlib.sha256(str(path.stat().st_size).encode())
    with path.open("rb") as stream:
        digest.update(stream.read(1024**2))
        stream.seek(max(0, path.stat().st_size - 1024**2))
        digest.update(stream.read(1024**2))
    return digest.hexdigest()


def frame_at(path: str | Path, seconds: float, max_width: int | None = None) -> np.ndarray:
    if seconds < 0 or not math.isfinite(seconds):
        raise ValueError("Frame time must be a finite nonnegative number")
    args = ["-ss", f"{seconds:.6f}", "-i", str(path), "-map", "0:v:0", "-frames:v", "1"]
    if max_width:
        args += ["-vf", f"scale='min({int(max_width)},iw)':-1"]
    raw = _run([*args, "-f", "image2pipe", "-vcodec", "png", "pipe:1"], timeout=60)
    if not raw:
        raise ValueError("No video frame at this time")
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")).copy()


def iter_frames(path: str | Path, start: float = 0, end: float | None = None,
                sample_fps: float = 5, max_width: int | None = 960):
    meta = probe_video(path)
    end = min(meta["duration"], meta["duration"] if end is None else end)
    if not 0 <= start < end or not 0 < sample_fps <= 120:
        raise ValueError("Invalid video sampling interval or rate")
    width = min(meta["width"], int(max_width)) if max_width else meta["width"]
    height = max(1, round(meta["height"] * width / meta["width"]))
    args = [ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-ss", f"{start:.6f}",
            "-i", str(path), "-t", f"{end-start:.6f}", "-map", "0:v:0", "-an",
            "-vf", f"fps={sample_fps},scale={width}:{height}", "-pix_fmt", "rgb24",
            "-f", "rawvideo", "pipe:1"]
    with tempfile.TemporaryFile() as error_log:
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=error_log)
        try:
            index = 0
            count = width * height * 3
            while True:
                raw = process.stdout.read(count)
                if not raw:
                    break
                if len(raw) != count:
                    raise ValueError("Truncated decoded video frame")
                time = start + index / sample_fps
                if time >= end:
                    break
                yield time, np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
                index += 1
            code = process.wait(timeout=30)
            if code:
                error_log.seek(0)
                raise ValueError(error_log.read().decode(errors="replace")[-2000:])
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)
            process.stdout.close()


def clip_video(source: str | Path, destination: str | Path, start: float, end: float,
               max_width: int = 960, cancelled=None) -> Path:
    if not 0 <= start < end:
        raise ValueError("Clip must have positive duration")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".partial.mp4")
    _run(["-y", "-ss", str(start), "-i", str(source), "-t", str(end-start),
          "-map", "0:v:0", "-map", "0:a?", "-vf",
          f"scale='min({max_width},iw)':-2,pad=ceil(iw/2)*2:ceil(ih/2)*2",
          "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-c:a", "aac",
          "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)],
         timeout=max(180, (end-start)*6), cancelled=cancelled)
    temporary.replace(destination)
    return destination


YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


def is_youtube_url(value: str) -> bool:
    from urllib.parse import urlparse
    value = str(value).strip()
    return value.startswith(("https://", "http://")) and (urlparse(value).hostname or "").lower() in YOUTUBE_HOSTS


def _js_runtime_args() -> list[str]:
    """YouTube streams need a JS challenge solver (yt-dlp-ejs) run by Deno or Node.

    yt-dlp only enables Deno by default, so point it at Node when that is what exists.
    """
    if shutil.which("deno"):
        return []
    node = shutil.which("node")
    return ["--js-runtimes", f"node:{node}"] if node else []


def download_source(url: str, directory: str | Path, *, metadata_only: bool = False,
                    cancelled=None, progress=None) -> Path | dict:
    """Download a YouTube video (<= 1080p, MP4) into `directory`.

    progress(done_bytes, total_bytes) is estimated from the partial files on disk
    against yt-dlp's approximate size; total may be None when YouTube omits it.
    """
    if not is_youtube_url(url):
        raise ValueError("URL import currently supports YouTube only; use a local path for other videos")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    args = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--socket-timeout", "25",
            "--retries", "1", "--no-warnings", *_js_runtime_args()]
    if metadata_only:
        args += ["--skip-download", "--dump-single-json", url]
        code, output, error = _capture(args, timeout=120, cancelled=cancelled)
        if code:
            raise ValueError(error.decode(errors="replace")[-2500:])
        info = json.loads(output)
        return {k: info.get(k) for k in ("id", "title", "duration", "webpage_url", "channel", "upload_date")}
    formats = "bv*[height<=1080]+ba/b[height<=1080]/best"
    poll = None
    if progress:
        probe = [*args, "--skip-download", "-f", formats, "--print", "%(id)s %(filesize_approx)s", url]
        code, output, error = _capture(probe, timeout=120, cancelled=cancelled)
        if code:
            raise ValueError("YouTube retrieval failed. A local MP4 can be imported instead.\n" +
                             error.decode(errors="replace")[-2500:])
        video_id, _, size = output.decode(errors="replace").strip().splitlines()[-1].partition(" ")
        total = int(size) if size.isdigit() else None
        last = [0.0]

        def poll():
            if time.monotonic() - last[0] < 1:
                return
            last[0] = time.monotonic()
            done = sum(p.stat().st_size for p in directory.glob(f"{glob.escape(video_id)}*")
                       if p.is_file() and not p.name.endswith(".json"))
            progress(done, total)

        progress(0, total)
    args += ["--no-progress", "-f", formats,
             "--merge-output-format", "mp4", "--ffmpeg-location", ffmpeg_executable(),
             "--write-info-json", "-o", str(directory / "%(id)s.%(ext)s"),
             "--print", "after_move:filepath", url]
    code, output, error = _capture(args, timeout=3600, cancelled=cancelled, poll=poll)
    if code:
        raise ValueError("YouTube retrieval failed. A local MP4 can be imported instead.\n" +
                         error.decode(errors="replace")[-2500:])
    lines = output.decode(errors="replace").strip().splitlines()
    path = Path(lines[-1]) if lines else None
    if path is None or not path.is_file():
        raise ValueError("The downloader did not produce a video")
    return path


def import_game(store, source: str | Path, name: str | None = None, source_url: str | None = None,
                cancelled=None, progress=None) -> dict:
    value = str(source)
    if value.startswith(("https://", "http://")):
        source_url = value
        source = download_source(value, store.root / "sources", cancelled=cancelled, progress=progress)
        if not name:
            # yt-dlp writes <id>.info.json beside the video; use its title as the display name.
            try:
                name = json.loads(Path(source).with_suffix(".info.json").read_text(encoding="utf-8")).get("title")
            except (OSError, ValueError):
                name = None
    source = Path(source).resolve()
    meta = probe_video(source)
    fingerprint = source_fingerprint(source)
    for existing in store.list("game"):
        if existing.get("source_fingerprint") == fingerprint:
            return existing
    record = store.save("game", {"name": name or source.stem, "source_path": str(source),
                                 "source_url": source_url, "source_fingerprint": fingerprint,
                                 "fingerprint_method": "size+head/tail SHA256; not full video checksum",
                                 "proxy_path": None, **meta})
    return record


def make_fixture(destination: str | Path, duration: float = 9) -> Path:
    """Own synthetic fixture, for mechanics only; never training/evaluation evidence."""
    import cv2
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = destination.with_suffix(".avi")
    width, height, fps = 640, 360, 20
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        raise ValueError("Could not create synthetic fixture")
    try:
        for index in range(round(duration*fps)):
            t = (index/fps) % 3 / 3
            rgb = np.full((height, width, 3), (32, 42, 55), np.uint8)
            cv2.rectangle(rgb, (0, 310), (639, 359), (160, 125, 85), -1)
            cv2.ellipse(rgb, (520, 170), (22, 5), 0, 0, 360, (240, 100, 45), 3)
            cv2.circle(rgb, (int(120+400*t), int(280-480*t+370*t*t)), 7, (245, 140, 30), -1)
            cv2.putText(rgb, "SYNTHETIC - mechanics only", (15, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, (240, 240, 240), 1)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    _run(["-y", "-i", str(raw), "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000",
          "-t", str(duration), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
          "-movflags", "+faststart", str(destination)])
    raw.unlink()
    return destination
