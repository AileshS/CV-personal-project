"""Editable possession proposals from activity and approximate color control.

This is a coarse baseline, not a possession model. A pan or scene cut alone is
never called a possession change. Long active spans may remain unsplit.
"""
from __future__ import annotations

import hashlib

import cv2
import numpy as np

from playlab.detectors import hex_rgb
from playlab.domain import inherited_split as _inherited_split
from playlab.media import iter_frames


def _control(frame, colors):
    """A tiny orange circular candidate near a distinctive jersey-color region."""
    if isinstance(colors, list):
        colors = {str(i): c for i, c in enumerate(colors)}
    if not colors or len(colors) != 2:
        return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 120, 90), (27, 255, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    balls = []
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if 4 <= area <= 150 and perimeter and 4 * np.pi * area / perimeter ** 2 > .65:
            (x, y), radius = cv2.minEnclosingCircle(c)
            if radius < 9:
                balls.append((x, y))
    # Multiple candidates commonly include court markings. Abstain rather than
    # allowing the court's orange paint to invent control changes.
    if len(balls) != 1:
        return None
    ball = np.array(balls[0])
    scores = []
    for team, color in colors.items():
        difference = np.linalg.norm(frame.astype(float) - hex_rgb(color), axis=2)
        ys, xs = np.where(difference < 55)
        if len(xs) < 15:
            continue
        distance = np.linalg.norm(np.column_stack([xs, ys]) - ball, axis=1)
        scores.append((float(np.percentile(distance, 5)), team))
    scores.sort()
    if len(scores) != 2 or scores[0][0] > 65 or scores[1][0] - scores[0][0] < 15:
        return None
    return scores[0][1]


def boundaries_from_samples(samples, duration):
    """Require sustained changed control or inactivity before proposing a split."""
    changes = []
    owner = None
    pending = None
    count = 0
    last_active = None
    inactive_since = None
    for row in samples:
        t, activity, team = row["time"], row["activity"], row.get("team")
        # Near-total frame changes can be cuts/pans; do not use them as stops.
        if .003 < activity < .65:
            if inactive_since is not None and t - inactive_since >= 10 and last_active is not None:
                changes.append((t, "Activity resumed after a long quiet interval"))
            inactive_since = None
            last_active = t
        elif activity <= .003:
            inactive_since = t if inactive_since is None else inactive_since
        if team is None:
            pending, count = None, 0
            continue
        if team == pending:
            count += 1
        else:
            pending, count = team, 1
        if count >= 4 and team != owner:
            if owner is not None:
                changes.append((max(0, t - 1.5), "Sustained apparent team-control change"))
            owner = team
    retained = [(0.0, "Video start")]
    for t, reason in sorted(changes):
        if t - retained[-1][0] >= 5 and duration - t >= 3:
            retained.append((float(t), reason))
    retained.append((float(duration), "Video end"))
    return [(retained[i][0], retained[i + 1][0], retained[i][1])
            for i in range(len(retained) - 1) if retained[i + 1][0] > retained[i][0]]


def propose_possessions(store, game_id, progress=None, cancelled=None):
    """Scan at 2 fps; persist checkpoints and preserve every existing correction."""
    existing = store.list("possession", game_id=game_id)
    if existing:
        return sorted(existing, key=lambda p: p["start"])
    game = store.get("game", game_id)
    if game is None:
        raise ValueError("Game not found.")
    setup = store.get("setup", game_id) or {}
    checkpoint_id = f"{game_id}:setup{setup.get('revision', 0)}"
    checkpoint = store.get("segmentation_checkpoint", checkpoint_id) or {}
    samples = checkpoint.get("samples", [])
    start = max(0, samples[-1]["time"] - .5) if samples else 0
    source = game.get("proxy_path") or game["source_path"]
    previous = None
    last_saved = start
    for t, frame in iter_frames(source, start=start, end=game["duration"], sample_fps=2, max_width=480):
        if cancelled and cancelled():
            store.save("segmentation_checkpoint", {"id": checkpoint_id, "samples": samples})
            raise InterruptedError("Possession scan cancelled; its checkpoint is saved.")
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)
        activity = float(np.mean(cv2.absdiff(previous, gray) > 18)) if previous is not None else 0.0
        previous = gray
        if samples and t <= samples[-1]["time"]:
            continue
        samples.append({"time": float(t), "activity": activity,
                        "team": _control(frame, setup.get("team_colors"))})
        if t - last_saved >= 30:
            store.save("segmentation_checkpoint", {"id": checkpoint_id, "samples": samples})
            last_saved = t
            if progress:
                progress(min(.98, t / max(1, game["duration"])), {"scan_time": t})
    proposals = []
    for start, end, reason in boundaries_from_samples(samples, game["duration"]):
        identifier = hashlib.sha256(f"{game_id}:{start:.3f}:{end:.3f}".encode()).hexdigest()[:20]
        proposals.append(store.save("possession", {
            "id": identifier, "game_id": game_id, "start": start, "end": end,
            "split": _inherited_split(game, start, end), "reviewed": False, "proposal": True,
            "method": "activity_and_approximate_control", "reason": reason,
            "confidence": "needs_review",
            "warning": "Editable proposal; offensive rebounds and camera movement can confuse this baseline.",
        }))
    store.save("segmentation_checkpoint", {"id": checkpoint_id, "samples": samples, "done": True})
    if progress:
        progress(1.0, {"proposals": len(proposals)})
    return proposals
