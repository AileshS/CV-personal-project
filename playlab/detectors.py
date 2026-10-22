"""Optional neural detectors and conservative, inspectable fallback helpers.

All inputs are RGB and all boxes refer to the supplied frame. Importing this
module never imports Torch, downloads weights, or starts inference.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


def _boxes(result, *, custom=False):
    output = {"balls": [], "rims": [], "players": []}
    if result.boxes is None:
        return output
    names = result.names
    for box in result.boxes:
        category = int(box.cls[0])
        name = str(names[category]).lower()
        key = None
        if name in {"ball", "basketball", "sports ball"}:
            key = "balls"
        elif custom and name in {"rim", "hoop", "basketball rim"}:
            key = "rims"
        elif not custom and category == 0:
            key = "players"
        if key:
            item = {"bbox": [float(v) for v in box.xyxy[0].cpu().tolist()],
                    "confidence": float(box.conf[0])}
            if box.id is not None:
                item["track_id"] = int(box.id[0])
            output[key].append(item)
    return output


class BasketballDetector:
    """Separate COCO person tracking and optionally fine-tuned ball/rim weights."""

    def __init__(self, weights=None, device="cpu"):
        model_dir = Path(os.environ.get("PLAYLAB_MODEL_DIR",
                                       Path(__file__).resolve().parents[1] / "artifacts" / "models"))
        model_dir.mkdir(parents=True, exist_ok=True)
        # Keep optional-library configuration and downloads in the workspace.
        os.environ.setdefault("YOLO_CONFIG_DIR", str(model_dir.parent / "ultralytics"))
        os.environ.setdefault("MPLCONFIGDIR", str(model_dir.parent / "matplotlib"))
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Learned detection needs the optional vision dependencies. "
                               "Install with `uv sync --extra vision`.") from exc
        if weights and not Path(weights).is_file():
            raise FileNotFoundError(f"Ball/rim weights were not found: {weights}")
        self.people = YOLO(str(model_dir / "yolo11n.pt"))
        self.objects = YOLO(str(weights)) if weights else None
        self.device = device
        self.mode = "fine_tuned_ball_rim" if weights else "pretrained_coco_ball"

    def reset(self):
        # Tracking state belongs to consecutive frames of one chunk, never to
        # separate clips. Track IDs are explicitly chunk-local.
        if hasattr(self.people, "predictor") and self.people.predictor is not None:
            for tracker in getattr(self.people.predictor, "trackers", []):
                tracker.reset()

    def __call__(self, frame):
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        result = self.people.track(bgr, persist=True, tracker="botsort.yaml",
                                   classes=[0] if self.objects else [0, 32],
                                   conf=0.15, imgsz=960, device=self.device, verbose=False)[0]
        output = _boxes(result)
        if self.objects is not None:
            result = self.objects.predict(bgr, conf=0.12, imgsz=960,
                                          device=self.device, verbose=False)[0]
            objects = _boxes(result, custom=True)
            output["balls"], output["rims"] = objects["balls"], objects["rims"]
        return output

    def ball_rim(self, frame, imgsz=960):
        """Ball and rim boxes only (no people): the per-frame cost of shot analysis."""
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if self.objects is None:
            result = self.people.predict(bgr, classes=[32], conf=0.15, imgsz=imgsz,
                                         device=self.device, verbose=False)[0]
            return {**_boxes(result), "players": []}
        result = self.objects.predict(bgr, conf=0.12, imgsz=imgsz, device=self.device, verbose=False)[0]
        return {**_boxes(result, custom=True), "players": []}

    def players(self, frame):
        """People in one frame, untracked; callers assign their own frame-local IDs."""
        result = self.people.predict(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), classes=[0], conf=0.15,
                                     imgsz=640, device=self.device, verbose=False)[0]
        return _boxes(result)["players"]

    def diagnostic(self, frame):
        """Best weak candidate per supported class, independent of tracking/outcomes.

        Class filtering happens before NMS so people cannot suppress weak balls.
        This reporting floor is deliberately tiny; these are guesses, not evidence
        sufficient to assert a shot outcome.
        """
        model = self.objects if self.objects is not None else self.people
        names = model.names
        names = names.items() if isinstance(names, dict) else enumerate(names)
        classes = [int(i) for i, name in names if str(name).lower() in
                   {"ball", "basketball", "sports ball", "rim", "hoop", "basketball rim"}]
        output = {"balls": [], "rims": [], "players": []}
        if not classes:
            return output
        result = model.predict(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), classes=classes,
                               conf=0.000001, imgsz=960, max_det=100,
                               device=self.device, verbose=False)[0]
        candidates = _boxes(result, custom=self.objects is not None)
        for key in ("balls", "rims"):
            if candidates[key]:
                output[key] = [max(candidates[key], key=lambda item: item["confidence"])]
        return output


class ColorCandidateDetector:
    """Dependency-free diagnostic fallback, never a trained ball detector."""

    mode = "color_candidates_manual_rim"

    def reset(self):
        pass

    def __call__(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (0, 125, 85), (27, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if 8 <= area <= 1600 and perimeter and 4 * np.pi * area / perimeter ** 2 > .7:
                (x, y), r = cv2.minEnclosingCircle(contour)
                candidates.append({"bbox": [x - r, y - r, x + r, y + r], "confidence": .25})
        # Do not arbitrarily choose among the floor, jerseys and multiple balls.
        return {"balls": candidates if len(candidates) == 1 else [], "rims": [], "players": []}

    def ball_rim(self, frame, imgsz=None):
        return self(frame)

    def players(self, frame):
        return []


def hex_rgb(value):
    value = str(value).lstrip("#")
    if len(value) != 6:
        raise ValueError("Uniform colors must be six-digit RGB hex colors.")
    return np.array([int(value[i:i + 2], 16) for i in (0, 2, 4)], dtype=float)


def _torso(frame, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    left, right = max(0, int(x1 + .2 * (x2 - x1))), min(w, int(x2 - .2 * (x2 - x1)))
    top, bottom = max(0, int(y1 + .2 * (y2 - y1))), min(h, int(y1 + .55 * (y2 - y1)))
    return frame[top:bottom, left:right]


def torso_color(frame, bbox):
    """Median torso color in CIELAB (8-bit OpenCV scale), or None when not visible."""
    region = _torso(frame, bbox)
    if not region.size:
        return None
    lab = cv2.cvtColor(np.ascontiguousarray(region), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return [float(v) for v in np.median(lab, axis=0)]


def assign_auto_teams(frames, minimum_players=12, minimum_separation=25.0):
    """Cluster torso colors into two teams plus an 'other' group (officials, bench).

    Players far from both team centers stay unresolved. Returns True when teams
    were assigned. Labels are clip-local ("auto-0"/"auto-1"), not real team names.
    """
    rows = [p for f in frames for p in f.get("players", [])
            if p.get("team") is None and p.get("torso") is not None]
    if len(rows) < minimum_players:
        return False
    from sklearn.cluster import KMeans

    colors = np.asarray([p["torso"] for p in rows], dtype=float)
    if len(np.unique(colors, axis=0)) < 3:
        return False
    model = KMeans(n_clusters=3, n_init=5, random_state=0).fit(colors)
    teams = [int(i) for i in np.argsort(np.bincount(model.labels_, minlength=3))[::-1][:2]]
    separation = float(np.linalg.norm(model.cluster_centers_[teams[0]] - model.cluster_centers_[teams[1]]))
    if separation < minimum_separation:
        return False
    for player, distances, label in zip(rows, model.transform(colors), model.labels_):
        if label in teams and distances[label] <= .5 * separation:
            player["team"] = f"auto-{teams.index(int(label))}"
    return True


def team_for_player(frame, bbox, colors, threshold=105.0):
    """Classify the torso only; unresolved color evidence remains unknown."""
    if isinstance(colors, list):
        colors = {str(i): color for i, color in enumerate(colors)}
    if not colors or len(colors) != 2:
        return None
    region = _torso(frame, bbox)
    if not region.size:
        return None
    pixels = region.reshape(-1, 3).astype(float)
    distances = []
    for team, color in colors.items():
        distances.append((float(np.percentile(np.linalg.norm(pixels - hex_rgb(color), axis=1), 35)), team))
    distances.sort()
    if distances[0][0] > threshold or distances[1][0] - distances[0][0] < 20:
        return None
    return distances[0][1]


def bottom_center(player):
    x1, _, x2, y2 = player["bbox"]
    return np.array([(x1 + x2) / 2, y2], dtype=float)


def nearest_defender(players, shooter_id):
    """Return a visible opponent by image-space foot position, never meters."""
    if shooter_id is None:
        return {"status": "unknown", "reason": "The shooter has no resolved track."}
    shooter = next((p for p in players if p.get("track_id") == shooter_id), None)
    if not shooter or shooter.get("team") is None:
        return {"status": "unknown", "reason": "Shooter or shooting team is unresolved."}
    candidates = [p for p in players if p.get("team") is not None and
                  p["team"] != shooter["team"] and p.get("track_id") is not None]
    if not candidates:
        return {"status": "unknown", "reason": "No opposing player has a usable visible track."}
    ranked = sorted((float(np.linalg.norm(bottom_center(p) - bottom_center(shooter))),
                     i, p) for i, p in enumerate(candidates))
    if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < 5:
        return {"status": "unknown", "reason": "Two opposing players are indistinguishably close in this view."}
    distance, _, opponent = ranked[0]
    return {"status": "approximate", "track_id": opponent["track_id"],
            "distance_px": distance, "bbox": opponent["bbox"],
            "label": "Closest visible opponent in this view"}
