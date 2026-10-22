"""Render current ball/rim model detections on every frame of a short video.

This module does not classify shots or create human annotations. A missing
detection stays missing; no earlier box or score is carried into a later frame.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from playlab.dataset import canonical_hash, file_hash
from playlab.media import _capture, ffmpeg_executable, probe_video

TRACKING_SETTINGS = {
    "renderer_version": 1,
    "fps_cap": 30,
    "width_cap": 1280,
    "max_duration_seconds": 60,
    "tracker": "bytetrack.yaml",
    "confidence_threshold": 0.10,
    "iou_threshold": 0.50,
    "image_size": 960,
    "tracker_settings": {
        "tracker_type": "bytetrack", "track_high_thresh": 0.25,
        "track_low_thresh": 0.10, "new_track_thresh": 0.25,
        "track_buffer": 30, "match_thresh": 0.80, "fuse_score": True,
    },
    "encoder": "libx264",
    "pixel_format": "yuv420p",
    "audio_codec": "aac",
    "frame_policy": "infer_every_output_frame; no carried boxes or scores",
    "score_definition": "raw detector score; not a confidence interval or calibrated accuracy",
}


def _cancel(cancelled):
    if cancelled and cancelled():
        raise InterruptedError("Tracking video cancelled; no incomplete video was published.")


def _package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def _class_mapping(names):
    """Require a custom two-class ball/rim detector, including public labels."""
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or len(names) != 2:
        raise ValueError("Tracking video requires a two-class basketball and rim model; COCO is unsupported.")
    mapping = {}
    for category, name in names.items():
        normalized = " ".join("".join(c if c.isalnum() else " " for c in str(name).casefold()).split())
        if normalized in {"ball", "basketball", "sports ball"}:
            label = "ball"
        elif normalized in {"rim", "hoop", "basketball hoop", "basketball rim"}:
            label = "rim"
        else:
            raise ValueError(f"The tracking model has an unsupported class: {name!s}.")
        mapping[int(category)] = {"label": label, "model_label": str(name)}
    if {item["label"] for item in mapping.values()} != {"ball", "rim"}:
        raise ValueError("Tracking video needs both basketball and rim model classes.")
    return mapping


def _load_model(weights):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ball/rim tracking needs the optional vision runtime; use uv run --extra vision.") from exc
    return YOLO(str(weights))


def _current_detections(result, mapping, width, height):
    detections = []
    boxes = result.boxes
    if boxes is None:
        return detections
    for box in boxes:
        category, confidence = int(box.cls[0]), float(box.conf[0])
        if category not in mapping or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("The tracking model returned an invalid class or detector score.")
        coordinates = box.xyxy[0]
        if hasattr(coordinates, "cpu"):
            coordinates = coordinates.cpu()
        coordinates = [float(v) for v in coordinates.tolist()]
        if len(coordinates) != 4 or not all(math.isfinite(v) for v in coordinates):
            raise ValueError("The tracking model returned invalid box coordinates.")
        x1, y1, x2, y2 = coordinates
        x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
        y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        track_id = int(box.id[0]) if box.id is not None else None
        detections.append({**mapping[category], "class_id": category,
                           "confidence": confidence, "confidence_percent": confidence * 100,
                           "box": [x1, y1, x2, y2],
                           "normalized_box": [x1 / width, y1 / height, x2 / width, y2 / height],
                           "track_id": track_id, "source": "model_prediction", "reviewed": False})
    return detections


def _clock(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:d}:{minutes:02d}:{seconds:06.3f}"


def _draw_frame(frame, detections, source_seconds, width, height):
    image = Image.new("RGB", (width, height))
    image.paste(Image.fromarray(frame), (0, 0))
    draw = ImageDraw.Draw(image)
    font_size = max(14, min(24, width // 48))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default(size=font_size)
    line_width = max(2, width // 320)
    for detection in detections:
        color = "#ffd34e" if detection["label"] == "ball" else "#26e6e6"
        draw.rectangle(detection["box"], outline="black", width=line_width + 2)
        draw.rectangle(detection["box"], outline=color, width=line_width)
        label = f"{'Ball' if detection['label'] == 'ball' else 'Rim'} {detection['confidence']:.0%}"
        if detection["track_id"] is not None:
            label += f"  #{detection['track_id']}"
        bounds = draw.textbbox((0, 0), label, font=font)
        label_width, label_height = bounds[2] - bounds[0] + 10, bounds[3] - bounds[1] + 8
        left = max(0, min(width - label_width, int(detection["box"][0])))
        top = max(0, int(detection["box"][1]) - label_height)
        draw.rectangle((left, top, left + label_width, top + label_height), fill="black", outline=color, width=1)
        draw.text((left + 5, top + 3 - bounds[1]), label, font=font, fill=color)
    stamp = f"Original {_clock(source_seconds)}  |  model scores"
    bounds = draw.textbbox((0, 0), stamp, font=font)
    stamp_height = bounds[3] - bounds[1] + 10
    draw.rectangle((0, height - stamp_height, min(width, bounds[2] + 16), height), fill="black")
    draw.text((8, height - stamp_height + 4 - bounds[1]), stamp, font=font, fill="white")
    return np.asarray(image)


def _stop(process):
    if process is None:
        return
    # A decoder can be blocked writing a full stdout pipe. Close our read end
    # before waiting so FFmpeg can exit, then escalate only this owned child.
    if process.stdout and not process.stdout.closed:
        process.stdout.close()
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
    if process.stdin and not process.stdin.closed:
        process.stdin.close()


def _decode_frames(source, start, end, fps, width, height, cancelled):
    """Decode with prompt owned-child cleanup when iteration is cancelled."""
    args = [ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-nostdin",
            "-ss", str(start), "-i", str(source), "-t", str(end - start),
            "-map", "0:v:0", "-an", "-vf", f"fps={fps},scale={width}:{height}",
            "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=errors, bufsize=0)
        try:
            index, size = 0, width * height * 3
            while True:
                _cancel(cancelled)
                data, received = bytearray(size), 0
                while received < size:
                    _cancel(cancelled)
                    count = process.stdout.readinto(memoryview(data)[received:])
                    if not count:
                        break
                    received += count
                if not received:
                    break
                if received != size:
                    raise ValueError("The tracking decoder returned a truncated frame.")
                at = start + index / fps
                if at >= end:
                    break
                yield at, np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
                index += 1
            process.stdout.close()
            deadline = time.monotonic() + 30
            while process.poll() is None:
                _cancel(cancelled)
                if time.monotonic() >= deadline:
                    raise ValueError("The tracking decoder exceeded its finalization time limit.")
                time.sleep(.1)
            if process.returncode:
                errors.seek(0)
                raise ValueError("Tracking video decoding failed: " + errors.read().decode(errors="replace")[-1000:])
        finally:
            _stop(process)


def _encode_frames(frames, destination, width, height, fps, cancelled):
    """Consume annotated RGB frames and terminate only this encoder on failure."""
    args = [ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", f"{width}x{height}",
            "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(destination)]
    process = None
    with tempfile.TemporaryFile() as errors:
        try:
            process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                       stderr=errors, bufsize=0)
            for frame in frames:
                _cancel(cancelled)
                remaining = memoryview(np.ascontiguousarray(frame)).cast("B")
                while remaining:
                    _cancel(cancelled)
                    written = process.stdin.write(remaining)
                    if not written:
                        raise ValueError("The tracking video encoder stopped accepting frames.")
                    remaining = remaining[written:]
            process.stdin.close()
            deadline = time.monotonic() + 120
            while process.poll() is None:
                _cancel(cancelled)
                if time.monotonic() >= deadline:
                    raise ValueError("The tracking video encoder exceeded its finalization time limit.")
                time.sleep(.1)
            if process.returncode:
                errors.seek(0)
                raise ValueError("Tracking video encoding failed: " + errors.read().decode(errors="replace")[-1000:])
        except BrokenPipeError as exc:
            raise ValueError("The tracking video encoder failed while writing frames.") from exc
        finally:
            _stop(process)
            frames.close()


def render_tracked_video(source, destination_dir, *, start, end, weights, device="cpu",
                         source_offset_seconds=0, cancelled=None, progress=None):
    """Publish a browser-playable box-overlay MP4 and framewise model evidence.

    Inference runs on every output frame at min(source fps, 30), width <=1280.
    Audio is retained from the same source interval. Only a completed directory
    is published; failures and cancellation remove their private staging files.
    """
    _cancel(cancelled)
    source, destination, weights = Path(source).resolve(), Path(destination_dir).resolve(), Path(weights).resolve()
    start, end, offset = float(start), float(end), float(source_offset_seconds)
    if not all(math.isfinite(v) for v in (start, end, offset)) or offset < 0:
        raise ValueError("Clip times and source offset must be finite; the offset cannot be negative.")
    if not 0 <= start < end or end - start > TRACKING_SETTINGS["max_duration_seconds"]:
        raise ValueError("Choose a positive tracking interval no longer than 60 seconds.")
    if destination.exists():
        raise ValueError("Tracking output already exists. Use a new attempt directory.")
    if not weights.is_file():
        raise FileNotFoundError("The basketball/rim detector weights are unavailable.")
    metadata = probe_video(source)
    source_stat = source.stat()
    source_identity = {"size": source_stat.st_size, "mtime_ns": source_stat.st_mtime_ns}
    if end > metadata["duration"] + .001:
        raise ValueError("The tracking interval is outside the source video.")
    native_fps = float(metadata["fps"])
    if not math.isfinite(native_fps) or native_fps <= 0:
        raise ValueError("The source frame rate is unavailable; export a normal MP4 before tracking.")
    fps = min(native_fps, TRACKING_SETTINGS["fps_cap"])
    native_width, native_height = int(metadata["width"]), int(metadata["height"])
    decoded_width = min(native_width, TRACKING_SETTINGS["width_cap"])
    decoded_height = max(1, round(native_height * decoded_width / native_width))
    width, height = decoded_width + decoded_width % 2, decoded_height + decoded_height % 2
    weights_hash = file_hash(weights)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(destination.parent / ".tracking-runtime" / "ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(destination.parent / ".tracking-runtime" / "matplotlib"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    _cancel(cancelled)
    model = _load_model(weights)
    mapping = _class_mapping(model.names)
    samples = []
    counts, frames_with = {"ball": 0, "rim": 0}, {"ball": 0, "rim": 0}
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        published = staging / "completed"
        published.mkdir()
        tracker_path = staging / "bytetrack.yaml"
        # JSON is valid YAML; pin ByteTrack settings instead of inheriting a
        # changed local tracker file or mutable package defaults.
        tracker_path.write_text(json.dumps(TRACKING_SETTINGS["tracker_settings"], indent=2), encoding="utf-8")
        video_frames = _decode_frames(source, start, end, fps, decoded_width, decoded_height, cancelled)

        def annotated_frames():
            try:
                for index, (at, frame) in enumerate(video_frames):
                    _cancel(cancelled)
                    if index >= math.ceil((end - start) * fps + 1e-6):
                        raise ValueError("The decoder exceeded the requested tracking interval.")
                    results = model.track(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), persist=True,
                                          tracker=str(tracker_path), device=device,
                                          conf=TRACKING_SETTINGS["confidence_threshold"],
                                          iou=TRACKING_SETTINGS["iou_threshold"],
                                          imgsz=TRACKING_SETTINGS["image_size"], verbose=False, save=False)
                    if len(results) != 1:
                        raise ValueError("The tracking model did not return one result for this video frame.")
                    detections = _current_detections(results[0], mapping, frame.shape[1], frame.shape[0])
                    for label in counts:
                        count = sum(item["label"] == label for item in detections)
                        counts[label] += count
                        frames_with[label] += bool(count)
                    samples.append({"frame_index": index, "file_seconds": float(at),
                                    "relative_seconds": index / fps, "source_seconds": float(at) + offset,
                                    "image_width": frame.shape[1], "image_height": frame.shape[0],
                                    "detections": detections})
                    if progress:
                        progress(min(.9, (index + 1) / max(1, (end - start) * fps) * .9),
                                 {"frames_processed": index + 1})
                    yield _draw_frame(frame, detections, float(at) + offset, width, height)
            finally:
                video_frames.close()

        silent = staging / "overlay-silent.mp4"
        _encode_frames(annotated_frames(), silent, width, height, fps, cancelled)
        if not samples:
            raise ValueError("No frames were decoded from the selected tracking interval.")
        _cancel(cancelled)
        video_path = published / "tracked.mp4"
        code, _, error = _capture([
            ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(silent),
            "-ss", str(start), "-i", str(source), "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "copy", "-c:a", "aac", "-t", str(end - start),
            "-movflags", "+faststart", str(video_path),
        ], timeout=max(60, (end - start) * 4), cancelled=cancelled)
        if code:
            raise ValueError("Could not preserve tracking video audio: " + error.decode(errors="replace")[-1000:])
        _cancel(cancelled)
        output_metadata = probe_video(video_path)
        if metadata["has_audio"] and not output_metadata["has_audio"]:
            raise ValueError("The completed tracking video lost its source audio.")
        if abs(output_metadata["duration"] - (end - start)) > 1 / fps + .05:
            raise ValueError("The completed tracking video duration does not match the requested interval.")
        if file_hash(weights) != weights_hash:
            raise ValueError("The detector weights changed during tracking. Start a fresh request.")
        current_source_stat = source.stat()
        if source_identity != {"size": current_source_stat.st_size, "mtime_ns": current_source_stat.st_mtime_ns}:
            raise ValueError("The source video changed during tracking. Start a fresh request.")
        settings = json.loads(json.dumps(TRACKING_SETTINGS))
        evidence = {"schema": 1, "kind": "ball_rim_tracking_samples", "reviewed": False,
                    "source_offset_seconds": offset, "model_sha256": weights_hash,
                    "settings": settings, "frames": samples}
        samples_path = published / "samples.json"
        samples_path.write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8")
        report = {
            "schema": 1, "kind": "ball_rim_tracking_video", "status": "complete", "reviewed": False,
            "video_path": str(destination / "tracked.mp4"), "samples_path": str(destination / "samples.json"),
            "report_path": str(destination / "report.json"), "video_sha256": file_hash(video_path),
            "samples_sha256": file_hash(samples_path), "frame_count": len(samples), "fps": fps,
            "duration": output_metadata["duration"], "requested_duration": end - start,
            "width": output_metadata["width"], "height": output_metadata["height"],
            "has_audio": output_metadata["has_audio"], "file_start": start, "file_end": end,
            "source_start": start + offset, "source_end": end + offset,
            "source_offset_seconds": offset, "model_path": str(weights), "model_sha256": weights_hash,
            "source_path": str(source), "source_identity": source_identity,
            "source_metadata": metadata, "renderer_sha256": file_hash(Path(__file__)),
            "model_classes": {str(key): item for key, item in mapping.items()},
            "settings": settings, "settings_hash": canonical_hash(settings),
            "tracker_settings_hash": canonical_hash(settings["tracker_settings"]),
            "runtime": {"ultralytics": _package_version("ultralytics"), "torch": _package_version("torch")},
            "detection_count": counts, "frames_with_detections": frames_with,
            "limitations": ["Raw model scores are not confidence intervals or calibrated accuracy.",
                            "Missing detections remain absent; no previous box or score is carried forward.",
                            "Track IDs apply only to this short rendering, not player identities.",
                            "This output does not classify shot outcomes or create verified labels."],
        }
        from playlab.model_assets import DEFAULT_MODEL_SHA256, model_provenance
        report["model_provenance"] = (model_provenance() if weights_hash == DEFAULT_MODEL_SHA256 else {
            "kind": "user_supplied_model", "sha256": weights_hash, "path": str(weights),
            "limitations": "User-supplied detector; no basketball accuracy estimate is established."})
        (published / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        _cancel(cancelled)
        published.rename(destination)
    if progress:
        progress(1.0, {"frames_processed": len(samples), "video_path": report["video_path"]})
    return report
