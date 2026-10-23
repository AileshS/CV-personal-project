"""Which team took each shot, without asking anyone.

Two facts about a basketball game do the work:
- Teams switch baskets at halftime, so within one half each basket belongs to one team.
- The two teams wear clearly different shades (by rule one light, one dark).

The period digit on the scoreboard gives halftime (only the period is read, never the
score). Each shot's shooter jersey, where it can be seen, votes for which shade shoots at
which basket; the vote then labels every shot, including ones whose shooter was not seen.
These are image heuristics; they are not verified against labeled games.
"""
from __future__ import annotations

import cv2
import numpy as np

from playlab.media import frame_at

SCOREBOARD = (.7, .3)  # the scoreboard inset sits in the bottom-left: bottom 30% of height, left 30% of width
PERIOD_STEP = 30.0     # seconds between scoreboard readings
ENLARGE = 4            # the period digit is read from an enlarged crop
MINIMUM_AGREEMENT = .85  # share of period readings that must fall on the right side of halftime
MINIMUM_SHADE_GAP = 30.0  # lightness (0-255) between the dark and light groups' averages


def _lit(region, cores=False):
    """LED pixels: bright and strongly colored (red, amber or green), unlike white print or the floor.

    With cores, also an LED's overexposed, nearly white middle (only safe in a small area
    known to hold a digit; elsewhere white print and jerseys would count).
    """
    hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
    lit = (hsv[..., 2] > 150) & (hsv[..., 1] > 110)
    if cores:
        lit |= (hsv[..., 2] > 235) & (hsv[..., 1] > 25)
    return lit.astype(np.uint8)


def _components(mask, minimum_height):
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [tuple(int(v) for v in stats[i][:4]) for i in range(1, count) if stats[i][3] >= minimum_height]


def _clock(boxes):
    """The topmost row of at least three similar-height digits: the game clock."""
    for x, y, w, h in sorted(boxes, key=lambda b: b[1]):
        row = [b for b in boxes if abs((b[1] + b[3] / 2) - (y + h / 2)) < .35 * h and abs(b[3] - h) < .35 * h]
        if len(row) >= 3:
            left, right = min(b[0] for b in row), max(b[0] + b[2] for b in row)
            return left, y, right, max(b[1] + b[3] for b in row), h
    return None


def read_period(frame, cores=False):
    """The period (1-4) from the digit centered under the game clock, or None when unreadable.

    The digit is read from its strokes, which holds across LED fonts: a "1" is narrow, a "4"
    has an upper-left stroke but no lower-left one, a "2" the reverse, a "3" neither.
    """
    height, width = frame.shape[:2]
    region = np.ascontiguousarray(frame[int(height * SCOREBOARD[0]):, :int(width * SCOREBOARD[1])])
    boxes = _components(_lit(region), minimum_height=max(6, region.shape[0] // 25))
    clock = _clock(boxes)
    if not clock:
        return None
    left, _, right, bottom, digit_height = clock
    center = (left + right) / 2
    # Look again just under the clock's middle, enlarged: some period digits are ~12 px tall.
    zx1, zx2 = int(max(0, center - .45 * (right - left))), int(center + .45 * (right - left))
    zy1, zy2 = bottom + 1, int(bottom + 3 * digit_height)
    zone = cv2.resize(region[zy1:zy2, zx1:zx2], None, fx=ENLARGE, fy=ENLARGE, interpolation=cv2.INTER_CUBIC)
    mask = _lit(zone, cores=cores)
    middle = (center - zx1) * ENLARGE
    below = [b for b in _components(mask, minimum_height=max(6, region.shape[0] // 25) * ENLARGE)
             if abs(b[0] + b[2] / 2 - middle) < .25 * (right - left) * ENLARGE]
    if not below:
        return None
    top = min(b[1] for b in below)
    digit = [b for b in below if b[1] < top + digit_height * ENLARGE * .5]
    x1, y1 = min(b[0] for b in digit), min(b[1] for b in digit)
    x2, y2 = max(b[0] + b[2] for b in digit), max(b[1] + b[3] for b in digit)
    if (x2 - x1) / (y2 - y1) < .45:
        return 1
    # Bright LEDs wash out to near-white in the middle, leaving only a colored outline: fill the strokes.
    patch = cv2.morphologyEx(mask[y1:y2, x1:x2], cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    shape = cv2.resize(patch.astype(np.float32), (12, 20), interpolation=cv2.INTER_AREA)
    upper_left, lower_left = shape[2:9, 0:4].mean() > .5, shape[11:18, 0:4].mean() > .5
    return {(True, False): 4, (False, True): 2, (False, False): 3}.get((upper_left, lower_left))


def find_halftime(source, duration, step=PERIOD_STEP, progress=None):
    """Seconds into the video where the 2nd half starts, from the scoreboard's period digit.

    Readings are noisy, so halftime is the split with the fewest readings on the wrong side
    (periods 1-2 before it, 3-4 after). Each frame is read with and without washed-out LED
    cores, and the reading mode that agrees best with one split is used. Returns
    {"halftime", "readings", "agreement"}; halftime is None when the video does not show
    both halves, the scoreboard cannot be read, or the readings disagree too much.
    """
    readings = {False: [], True: []}
    times = np.arange(step / 2, duration, step)
    for i, at in enumerate(times):
        try:
            frame = frame_at(source, float(at), max_width=1280)
        except (OSError, ValueError, RuntimeError):
            continue
        for cores in readings:
            period = read_period(frame, cores=cores)
            if period:
                readings[cores].append((float(at), period))
        if progress and i % 10 == 0:
            progress(i / len(times))
    best = max((_split(rows) for rows in readings.values()), key=lambda r: (r["agreement"] or 0, r["readings"]))
    if best["agreement"] is None or best["agreement"] < MINIMUM_AGREEMENT:
        best["halftime"] = None
    return best


def _split(readings):
    result = {"halftime": None, "readings": len(readings), "agreement": None}
    if sum(p <= 2 for _, p in readings) < 3 or sum(p >= 3 for _, p in readings) < 3:
        return result
    wrong, index = min((sum(p >= 3 for _, p in readings[:i]) + sum(p <= 2 for _, p in readings[i:]), i)
                       for i in range(1, len(readings)))
    result["halftime"] = (readings[index - 1][0] + readings[index][0]) / 2
    result["agreement"] = 1 - wrong / len(readings)
    return result


def shade_split(lightness):
    """The lightness between the game's dark and light jerseys, or None without two clear groups.

    Lighting differs by gym, so no fixed value works: everyone visible at the shooters'
    releases is split into the two groups that are tightest (Otsu's method on a 1-D list).
    Officials, benches and fans add noise but rarely form a group of their own.
    """
    values = np.sort(np.asarray([v for v in lightness if v is not None], dtype=float))
    if len(values) < 10:
        return None
    best = None
    for i in range(3, len(values) - 3):
        low, high = values[:i], values[i:]
        spread = low.var() * len(low) + high.var() * len(high)
        if best is None or spread < best[0]:
            best = (spread, (low.mean() + high.mean()) / 2, high.mean() - low.mean())
    _, split, gap = best
    return split if gap >= MINIMUM_SHADE_GAP else None


def jersey_shade(lightness, split):
    """Light or dark, relative to this game's split, or None."""
    if lightness is None or split is None:
        return None
    return "light" if float(lightness) >= split else "dark"


def side_owner(shots, halftime):
    """Which shade shoots at the left basket in the 1st half, by vote of the shots whose shooter was seen.

    shots: dicts with "seconds", "basket" ("left"/"right") and "shade" ("light"/"dark"/None).
    Returns ("light"|"dark"|None, votes for, votes against).
    """
    votes = {"light": 0, "dark": 0}
    for shot in shots:
        if shot.get("shade") not in votes or shot.get("basket") not in ("left", "right"):
            continue
        first_half = halftime is None or shot["seconds"] < halftime
        # Normalize every vote to "shade at the left basket in the 1st half".
        flip = (shot["basket"] == "right") != (not first_half)
        shade = shot["shade"] if not flip else ("dark" if shot["shade"] == "light" else "light")
        votes[shade] += 1
    if votes["light"] == votes["dark"]:
        return None, votes["light"], votes["dark"]
    winner = max(votes, key=votes.get)
    return winner, votes[winner], votes["dark" if winner == "light" else "light"]


def shot_shade(owner, basket, seconds, halftime):
    """The shade shooting at this basket at this time, given who owns the left basket in the 1st half."""
    if owner not in ("light", "dark") or basket not in ("left", "right"):
        return None
    first_half = halftime is None or seconds < halftime
    left_owner = owner if first_half else ("dark" if owner == "light" else "light")
    return left_owner if basket == "left" else ("dark" if left_owner == "light" else "light")
