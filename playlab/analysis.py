"""Resumable selected-possession analysis; human results never enter predictions."""
from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from playlab.dataset import canonical_hash, file_hash
from playlab.detectors import BasketballDetector, ColorCandidateDetector, nearest_defender, team_for_player
from playlab.features import compute_features, geometry_prediction, predict_result
from playlab.media import frame_at, iter_frames

ANALYSIS_MODES = {"automatic", "assisted"}


def analysis_record_id(possession_id, mode="assisted"):
    """Keep independent automatic inference separate from human-assisted runs."""
    if mode not in ANALYSIS_MODES:
        raise ValueError("Analysis mode must be automatic or assisted.")
    return possession_id if mode == "assisted" else f"{possession_id}:automatic"


def _default_model_path():
    model_dir = Path(os.environ.get("PLAYLAB_MODEL_DIR", Path(__file__).parents[1] / "artifacts" / "models"))
    return (model_dir / "yolo11n.pt").resolve()


def _artifact_versions(weights, model_bundle):
    paths = [_default_model_path()] if _vision_version() != "unavailable" else []
    paths.extend(Path(p) for p in (weights, model_bundle) if p)
    return {str(p.resolve()): file_hash(p) if p.is_file() else "missing" for p in paths}


def analysis_dependencies(store, possession_id, *, mode="assisted", weights=None, model_bundle=None, device="cpu"):
    """Snapshot inference inputs, excluding every human outcome label.

    Model files are content hashed; large source videos use path/stat identity.
    The latter detects ordinary edits, not adversarial timestamp-preserving edits.
    """
    analysis_record_id(possession_id, mode)
    possession = store.get("possession", possession_id)
    if possession is None:
        raise ValueError("Possession not found.")
    game = store.get("game", possession["game_id"])
    if game is None:
        raise ValueError("Source game not found.")
    anchors = ([{k: p.get(k) for k in ("id", "release", "shooter_id", "shot_type")}
                for p in store.list("shot", possession_id=possession_id)
                if p.get("reviewed") is True] if mode == "assisted" else [])
    source = Path(__file__).parent
    artifact_state = _artifact_versions(weights, model_bundle)
    return {"version": 6, "mode": mode, "device": str(device),
            "source": _file_version(game["source_path"]), "game_id": game["id"],
            "source_offset_seconds": game.get("source_offset_seconds", 0),
            "source_provenance": game.get("source_provenance", game.get("provenance")),
            "interval_context": game.get("quick_analysis_context"),
            "start": float(possession["start"]), "end": float(possession["end"]),
            "possession_revision": possession.get("revision"),
            "setup": store.get("setup", game["id"]) or {}, "anchors": anchors,
            "weights": str(Path(weights).resolve()) if weights else None,
            "model_bundle": str(Path(model_bundle).resolve()) if model_bundle else None,
            # Missing optional weights are availability state, not files used
            # by diagnostic inference. They cannot be hashed into a freeze.
            "artifacts": {path: digest for path, digest in artifact_state.items() if digest != "missing"},
            "unavailable_artifacts": sorted(path for path, digest in artifact_state.items() if digest == "missing"),
            "vision_version": _vision_version(),
            "code_hash": canonical_hash({p.name: file_hash(p) for p in
                                         (source / "analysis.py", source / "detectors.py", source / "features.py")})}


def analysis_is_fresh(store, result):
    """Legacy results without input provenance must be recomputed before export."""
    if not result or result.get("stale") or result.get("status") != "complete":
        return False
    saved = result.get("dependencies")
    if not isinstance(saved, dict) or len(saved.get("source") or []) != 3:
        return False
    try:
        current = analysis_dependencies(store, result["possession_id"], mode=result["analysis_mode"],
                                        weights=saved.get("weights"), model_bundle=saved.get("model_bundle"),
                                        device=saved["device"])
    except (KeyError, OSError, ValueError):
        return False
    return current == saved and canonical_hash(saved)[:24] == result.get("fingerprint")


def _file_version(path):
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        return [str(path), "missing"]
    if path.is_dir():
        return [[p.name, p.stat().st_size, p.stat().st_mtime_ns]
                for p in sorted(path.glob("*")) if p.is_file()]
    return [str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns]


def _vision_version():
    try:
        return version("ultralytics")
    except PackageNotFoundError:
        return "unavailable"


def _center(box):
    x1, y1, x2, y2 = box["bbox"]
    return (x1 + x2) / 2, (y1 + y2) / 2


def _rim(detections, setup, width, height):
    rims = detections.get("rims", [])
    if len(rims) == 1:
        return rims[0]
    if len(rims) > 1:
        # Detection confidence cannot distinguish which basket a shot targets.
        # Associate only a uniquely nearby ball, otherwise withhold rim evidence.
        balls = detections.get("balls", [])
        if len(balls) == 1:
            bx, by = _center(balls[0])
            ranked = sorted((float(np.hypot(_center(r)[0] - bx, _center(r)[1] - by)), i, r)
                            for i, r in enumerate(rims))
            if ranked[0][0] < .65 * ranked[1][0]:
                return ranked[0][2]
        return None
    box = setup.get("rim_box") or setup.get("rim")
    if box and len(box) == 4:
        x1, y1, x2, y2 = [float(v) for v in box]
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("Rim box must be normalized [left, top, right, bottom].")
        return {"bbox": [x1 * width, y1 * height, x2 * width, y2 * height],
                "confidence": 1.0, "manual": True}
    return None


def shot_proposals(observations, minimum_spacing=3.0):
    """Discover isolated rim approaches; timestamps are release candidates only."""
    proposals = []
    for i, point in enumerate(observations):
        if point.get("rim_width", 0) <= 0:
            continue
        distance = np.hypot(point["x"] - point["rim_x"], point["y"] - point["rim_y"]) / point["rim_width"]
        if distance > 2.5:
            continue
        neighbors = [p for p in observations[max(0, i - 8):i + 9]
                     if abs(p["time"] - point["time"]) <= .8]
        # A single orange spot near a hoop is insufficient evidence of a shot.
        if len(neighbors) < 4 or max(p["y"] for p in neighbors) - min(p["y"] for p in neighbors) < point["rim_width"] * .5:
            continue
        time = max(observations[0]["time"], point["time"] - 1.5)
        if proposals and time - proposals[-1] < minimum_spacing:
            continue
        proposals.append(float(time))
    return proposals


def _shooter(players, observations, release, explicit=None):
    if explicit is not None:
        # IDs are scoped to a chunk and represent tracks, not jersey numbers.
        found = next((p for p in players if str(p.get("track_id")) == str(explicit)), None)
        return found.get("track_id") if found else None
    points = [p for p in observations if abs(p["time"] - release) <= .4]
    if not points or not players:
        return None
    ball = min(points, key=lambda p: abs(p["time"] - release))
    ranked = []
    for player in players:
        x1, y1, x2, y2 = player["bbox"]
        # Ball must be close to the upper body at release, not merely to feet.
        dx = max(x1 - ball["x"], 0, ball["x"] - x2)
        dy = max(y1 - .2 * (y2 - y1) - ball["y"], 0, ball["y"] - (y1 + .7 * (y2 - y1)))
        ranked.append((float(np.hypot(dx, dy)), str(player.get("track_id")), player))
    ranked.sort(key=lambda row: row[:2])
    if not ranked or ranked[0][0] > 35 or (len(ranked) > 1 and ranked[1][0] - ranked[0][0] < 12):
        return None
    return ranked[0][2].get("track_id")


def _evidence(source, at, players, shooter, defender, destination, source_offset_seconds=0):
    frame = frame_at(source, at, max_width=1280)
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    for player in players:
        track = player.get("track_id")
        if track == shooter or (defender.get("status") == "approximate" and track == defender.get("track_id")):
            color = "#39dd87" if track == shooter else "#ffb547"
            draw.rectangle(player["bbox"], outline=color, width=3)
            draw.text((player["bbox"][0], max(0, player["bbox"][1] - 15)),
                      "Shooter" if track == shooter else "Closest visible opponent", fill=color)
    draw.text((12, 12), f"Source {at + source_offset_seconds:.2f}s | approximate image-space evidence", fill="white")
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return str(destination.resolve())


def analyze_possession(store, possession_id, device="cpu", weights=None, model_bundle=None,
                       progress=None, cancelled=None, mode="assisted"):
    """Process complete possessions in restartable 20-second, 10-fps chunks.

    Callback: progress(fraction: float, checkpoint: dict). Cancellation raises
    InterruptedError after retaining every completed chunk. Human shot.result is
    deliberately excluded both from inference and its cache key. Automatic mode
    never reads human shot anchors; assisted mode may use their release times.
    """
    possession = store.get("possession", possession_id)
    if possession is None:
        raise ValueError("Possession not found.")
    game = store.get("game", possession["game_id"])
    if game is None:
        raise ValueError("Source game not found.")
    start, end = float(possession["start"]), float(possession["end"])
    if start < 0 or end <= start or end > game["duration"] + .1:
        raise ValueError("Possession times must lie within the source game.")
    setup = store.get("setup", game["id"]) or {}
    record_id = analysis_record_id(possession_id, mode)
    annotations = ([p for p in store.list("shot", possession_id=possession_id) if p.get("reviewed") is True]
                   if mode == "assisted" else [])
    anchors = [{k: p.get(k) for k in ("id", "release", "shooter_id", "shot_type")} for p in annotations]
    dependencies = analysis_dependencies(store, possession_id, mode=mode, weights=weights,
                                         model_bundle=model_bundle, device=device)
    fingerprint = canonical_hash(dependencies)[:24]
    existing = store.get("analysis", record_id)
    if (existing and not existing.get("stale") and existing.get("fingerprint") == fingerprint
            and existing.get("status") == "complete"):
        if progress:
            progress(1.0, {"cached": True})
        return existing
    detector = None
    warnings = []
    observations, frames, modes = [], [], set()
    previous_view = None
    manual_view_invalid = False
    chunks = int(np.ceil((end - start) / 20))
    for index in range(chunks):
        if cancelled and cancelled():
            raise InterruptedError("Analysis cancelled; completed chunks can be resumed.")
        chunk_id = f"{fingerprint}:{index}"
        saved = store.get("analysis_chunk", chunk_id)
        if saved is None:
            if detector is None:
                try:
                    detector = BasketballDetector(weights, device)
                except Exception as exc:
                    if weights or str(device) != "cpu":
                        raise RuntimeError(f"Requested detector/device could not start ({device}): {exc}") from exc
                    detector = ColorCandidateDetector()
                    warnings.append(f"Learned detector unavailable: {exc}. Using color candidates; no player tracks.")
                # A first neural startup may download only its default model.
                # Preserve all existing/custom snapshots so a concurrent model
                # or classifier edit makes this result stale instead of silently
                # attributing in-memory inference to the replacement file.
                default_model = _default_model_path()
                requested = {str(Path(p).resolve()) for p in (weights, model_bundle) if p}
                if (str(default_model) not in requested
                        and str(default_model) in dependencies["unavailable_artifacts"]
                        and default_model.is_file()):
                    dependencies["artifacts"][str(default_model)] = file_hash(default_model)
                    dependencies["unavailable_artifacts"].remove(str(default_model))
            detector.reset()
            chunk_start, chunk_end = start + index * 20, min(end, start + (index + 1) * 20)
            chunk_rows, chunk_frames = [], []
            last_ball = None
            view_segment = 0
            for at, frame in iter_frames(game["source_path"], start=max(start, chunk_start - .5),
                                         end=chunk_end, sample_fps=10, max_width=1280):
                if cancelled and cancelled():
                    raise InterruptedError("Analysis cancelled; completed chunks can be resumed.")
                view = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (96, 54))
                view_changed = previous_view is not None and np.mean(cv2.absdiff(previous_view, view)) > 65
                if view_changed:
                    detector.reset()
                    view_segment += 1
                    last_ball = None
                    manual_view_invalid = True
                    warnings.append("Strong view change reset player tracks; a fixed manual rim is withheld after it.")
                previous_view = view
                detections = detector(frame)
                height, width = frame.shape[:2]
                rim = _rim(detections, {} if manual_view_invalid else setup, width, height)
                players = []
                for p in detections.get("players", []):
                    player = dict(p)
                    raw_id = player.get("track_id")
                    player["track_id"] = f"{index}:{view_segment}:{raw_id}" if raw_id is not None else None
                    player["team"] = team_for_player(frame, player["bbox"], setup.get("team_colors"))
                    players.append(player)
                if at < chunk_start:
                    continue
                chunk_frames.append({"time": float(at), "players": players, "rim": rim,
                                     "view_changed": bool(view_changed)})
                balls = detections.get("balls", [])
                if balls:
                    # Reject ambiguous detections and implausible jumps instead
                    # of allowing a distractor to fabricate a rim crossing.
                    balls = sorted(balls, key=lambda b: b["confidence"], reverse=True)
                    ball = balls[0] if len(balls) == 1 or balls[0]["confidence"] - balls[1]["confidence"] > .2 else None
                    if ball is not None:
                        x, y = _center(ball)
                        if last_ball and at - last_ball[0] < .35 and np.hypot(x - last_ball[1], y - last_ball[2]) > width * .18:
                            ball = None
                        if ball is not None:
                            last_ball = (at, x, y)
                            if rim:
                                rx, ry = _center(rim)
                                chunk_rows.append({"time": float(at), "x": float(x), "y": float(y),
                                                   "rim_x": rx, "rim_y": ry,
                                                   "rim_width": rim["bbox"][2] - rim["bbox"][0],
                                                   "confidence": float(ball["confidence"])})
            saved = store.save("analysis_chunk", {"id": chunk_id, "observations": chunk_rows,
                               "frames": chunk_frames, "mode": detector.mode, "warnings": list(warnings),
                               "manual_view_invalid": manual_view_invalid,
                               "last_view": previous_view.tolist() if previous_view is not None else None})
        manual_view_invalid = bool(saved.get("manual_view_invalid"))
        if saved.get("last_view") is not None:
            previous_view = np.asarray(saved["last_view"], dtype=np.uint8)
        observations.extend(saved["observations"])
        frames.extend(saved["frames"])
        modes.add(saved["mode"])
        warnings.extend(saved.get("warnings", []))
        if progress:
            progress(.85 * (index + 1) / chunks, {"completed_chunks": index + 1, "chunks": chunks})
    if not frames:
        raise ValueError("No video frames could be read within this possession.")
    observations.sort(key=lambda p: p["time"])
    frames.sort(key=lambda p: p["time"])
    candidates = [{"id": p["id"], "release": float(p["release"]), "shooter_id": p.get("shooter_id"),
                   "shot_type": p.get("shot_type", "field_goal"), "origin": "human_release_anchor"}
                  for p in annotations if p.get("release") is not None and start <= float(p["release"]) <= end]
    for release in shot_proposals(observations):
        if not any(abs(release - p["release"]) < 3 for p in candidates):
            candidates.append({"id": f"auto-{release:.2f}", "release": release,
                               "origin": "automatic_rim_approach", "shot_type": "field_goal"})
    events = []
    candidates.sort(key=lambda p: p["release"])
    for candidate_index, candidate in enumerate(candidates):
        release = candidate["release"]
        # An offensive rebound followed by a make must not turn the preceding
        # missed attempt into a make. Bound evidence at the next shot release.
        evidence_end = min(end, release + 5,
                           candidates[candidate_index + 1]["release"] - .05
                           if candidate_index + 1 < len(candidates) else end)
        selected = [p for p in observations if release - .25 <= p["time"] <= evidence_end]
        if any(f.get("view_changed") and release - .25 <= f["time"] <= evidence_end for f in frames):
            # Separate camera views cannot establish one continuous ball path.
            selected = []
        features = compute_features(selected)
        features = {key: (None if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value)
                    for key, value in features.items()}
        baseline = geometry_prediction(features)
        prediction = predict_result(model_bundle, features) if model_bundle else {
            "result": baseline, "score": None, "method": "provisional_geometry"}
        # Color segmentation is not sufficient evidence for an automatic made/
        # missed claim, even when it produces a visually plausible trajectory.
        if "color_candidates_manual_rim" in modes:
            prediction = {"result": "unknown", "score": None, "method": "color_diagnostic_only"}
        release_frame = min(frames, key=lambda f: abs(f["time"] - release))
        shooter = _shooter(release_frame["players"], selected, release, candidate.get("shooter_id"))
        if candidate["shot_type"] == "free_throw":
            defender = {"status": "not_applicable", "reason": "Free throw"}
        else:
            defender = nearest_defender(release_frame["players"], shooter)
            if candidate["origin"] != "human_release_anchor" and defender.get("status") == "approximate":
                defender["release_unconfirmed"] = True
                defender["reason"] = "Suggested at an estimated release; confirm the release and highlighted shooter."
                if game.get("quick_analysis_context"):
                    defender = {"status": "unknown", "reason": "Release and shooter require review."}
        image_path = store.root / "evidence" / fingerprint / f"{len(events):04d}.png"
        evidence = _evidence(game["source_path"], release_frame["time"], release_frame["players"],
                             shooter, defender, image_path,
                             **({"source_offset_seconds": game["source_offset_seconds"]}
                                if game.get("source_offset_seconds") else {}))
        events.append({**candidate, "type": "shot", "time": release, "release": release,
                       "game_id": game["id"], "possession_id": possession_id,
                       "release_confirmed": candidate["origin"] == "human_release_anchor",
                       "result": prediction["result"], "predicted_result": prediction["result"],
                       "score": prediction.get("score"), "method": prediction.get("method"),
                       "baseline_result": baseline, "features": features,
                       "shooter_id": shooter, "defender": defender, "evidence_path": evidence,
                       "reviewed": False, "source": "prediction", "status": "needs_review"})
    if not setup.get("team_colors"):
        warnings.append("Set both uniform colors to identify opposing teams.")
    warnings.append("Uniform-color tracking does not reliably exclude benches or officials; review the highlighted players.")
    if setup.get("rim_box") or setup.get("rim"):
        warnings.append("A manual rim box is valid only for its corrected camera view; review it after pans or cuts.")
    if not events:
        warnings.append("No dependable shot proposals. Add a shot release anchor, correct the rim view, and reanalyze.")
    current_possession = store.get("possession", possession_id)
    current_setup = store.get("setup", game["id"]) or {}
    current_anchors = ([{k: p.get(k) for k in ("id", "release", "shooter_id", "shot_type")}
                        for p in store.list("shot", possession_id=possession_id)
                        if p.get("reviewed") is True] if mode == "assisted" else [])
    try:
        current_dependencies = analysis_dependencies(store, possession_id, mode=mode, weights=weights,
                                                     model_bundle=model_bundle, device=device)
    except ValueError:
        current_dependencies = None
    stale = (current_possession is None or current_possession.get("revision") != possession.get("revision")
             or current_setup != setup or current_anchors != anchors or current_dependencies != dependencies)
    fingerprint = canonical_hash(dependencies)[:24]
    result = store.save("analysis", {"id": record_id, "game_id": game["id"],
        "possession_id": possession_id, "fingerprint": fingerprint, "status": "complete",
        "analysis_mode": mode, "dependencies": dependencies,
        "possession_revision": possession.get("revision"), "stale": stale,
        "events": events, "shots": events, "observation_count": len(observations),
        "frames_processed": len(frames), "chunks": chunks, "warnings": sorted(set(warnings)),
        "foul": "manual_only", "turnover": "manual_only",
        "accuracy_claim": "Unverified predictions; inspect evidence and review results."})
    if progress:
        progress(1.0, {"events": len(events), "fingerprint": fingerprint})
    return result
