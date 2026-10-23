"""Halftime from the scoreboard's period digit and the light/dark vote, on drawn scoreboards and toy shots."""
import cv2
import numpy as np

from playlab import analysis, teams

SEGMENTS = {  # a, b, c, d, e, f, g
    "0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "8": "abcdefg",
}


def _digit(image, left, top, char, size, color):
    w, h, t = size, 2 * size, max(2, size // 5)
    bars = {"a": (0, 0, w, t), "b": (w - t, 0, w, h // 2), "c": (w - t, h // 2, w, h), "d": (0, h - t, w, h),
            "e": (0, h // 2, t, h), "f": (0, 0, t, h // 2), "g": (0, h // 2 - t // 2, w, h // 2 + t // 2)}
    if char == "1":  # a "1" is a single narrow stroke on LED boards
        cv2.rectangle(image, (left + w // 2, top), (left + w // 2 + t, top + h), color, -1)
        return
    for name in SEGMENTS[char]:
        x1, y1, x2, y2 = bars[name]
        cv2.rectangle(image, (left + x1, top + y1), (left + x2, top + y2), color, -1)


def scoreboard(period):
    """A 1280x720 frame with an LED board in the bottom-left: clock 8:00 and the period below it."""
    frame = np.full((720, 1280, 3), (200, 170, 120), np.uint8)   # wooden floor
    cv2.rectangle(frame, (20, 520), (360, 715), (20, 25, 40), -1)
    for i, char in enumerate("800"):
        _digit(frame, 130 + i * 32, 535, char, 22, (255, 170, 0))
    _digit(frame, 174, 600, str(period), 14, (255, 40, 30))
    return frame


def test_period_digit_is_read_from_its_strokes():
    assert [teams.read_period(scoreboard(p)) for p in (1, 2, 3, 4)] == [1, 2, 3, 4]
    assert teams.read_period(np.full((720, 1280, 3), 90, np.uint8)) is None   # no scoreboard


def test_halftime_is_the_split_with_fewest_misread_periods():
    readings = [(t * 30., 1) for t in range(10)] + [(t * 30., 2) for t in range(10, 20)]
    readings += [(t * 30., 3) for t in range(20, 30)] + [(t * 30., 4) for t in range(30, 40)]
    readings[25] = (750., 2)  # one misread
    split = teams._split(readings)
    assert split["halftime"] == 585. and split["agreement"] == 39 / 40
    assert teams._split([(t, 1) for t in range(20)])["halftime"] is None   # one half only


def test_shades_split_per_game_and_vote_for_baskets():
    dim_gym = [30., 35., 40., 45., 50., 95., 100., 105., 110., 120.]   # white jerseys are only ~100 here
    split = teams.shade_split(dim_gym)
    assert 50 < split < 95 and teams.jersey_shade(98., split) == "light" and teams.jersey_shade(40., split) == "dark"
    assert teams.shade_split([100.] * 12) is None
    shots = [{"seconds": 10, "basket": "right", "shade": "dark"}, {"seconds": 20, "basket": "left", "shade": "light"},
             {"seconds": 90, "basket": "left", "shade": "dark"}, {"seconds": 95, "basket": "left", "shade": "light"}]
    assert teams.side_owner(shots, halftime=50) == ("light", 3, 1)
    assert [teams.shot_shade("light", b, t, 50) for b, t in (("left", 10), ("right", 10), ("left", 90))] == [
        "light", "dark", "dark"]


def test_release_is_where_the_flight_to_the_rim_began():
    rim = {"rim_x": 0., "rim_y": 0., "rim_width": 1.}
    path = [(0., 9., 3.), (.2, 8., 3.5), (.4, 6., 2.), (.6, 4., 1.), (.62, 4.3, 1.2), (.8, 2., .5), (1., .3, .1)]
    points = [{"time": t, "x": x, "y": y, **rim} for t, x, y in path]
    assert analysis.release_point(points, 1.)["time"] == 0.   # the 0.62 wobble does not stop the trace
    # A dribble before the shot: the ball was closer to the rim, then moved away, then flew in.
    assert analysis.release_point([{"time": -.2, "x": 3., "y": 0., **rim}] + points, 1.)["time"] == 0.
    assert analysis.release_point(points[-2:], 1.) is None   # under 0.3 s of flight
