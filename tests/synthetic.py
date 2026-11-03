"""Synthetic hands for tests: 21 landmarks in MediaPipe order, fingers pointing up.

Coordinates are in arbitrary units with the same scale on both axes (like
pixels). ``hand()`` builds an open hand or a pinch, then scales and moves it.
"""

import math
from types import SimpleNamespace


def _p(x, y):
    return SimpleNamespace(x=x, y=y)


# Open hand around a wrist at (0, 0); palm about 1 unit tall. y grows downward.
_OPEN = {
    0: (0.00, 0.00),                                            # wrist
    1: (-0.30, -0.20), 2: (-0.50, -0.40), 3: (-0.65, -0.55), 4: (-0.80, -0.65),  # thumb
    5: (-0.25, -1.00), 6: (-0.28, -1.40), 7: (-0.30, -1.65), 8: (-0.32, -1.85),  # index
    9: (0.00, -1.05), 10: (0.00, -1.50), 11: (0.00, -1.78), 12: (0.00, -2.00),   # middle
    13: (0.22, -1.00), 14: (0.25, -1.40), 15: (0.27, -1.62), 16: (0.29, -1.80),  # ring
    17: (0.42, -0.90), 18: (0.48, -1.22), 19: (0.52, -1.42), 20: (0.55, -1.58),  # pinky
}


def hand(pinch: bool = False, scale: float = 100.0, at=(320.0, 240.0), angle: float = 0.0):
    """An open hand, or one pinching thumb and index together.

    ``scale`` is the palm height in pixels (bigger = closer to the camera) and
    ``at`` is where the wrist is. ``angle`` (radians) turns the hand around
    the wrist, clockwise on screen.
    """
    pts = dict(_OPEN)
    if pinch:
        # Index bends toward the thumb and the tips touch.
        pts[6], pts[7], pts[8] = (-0.40, -1.30), (-0.55, -1.15), (-0.62, -1.00)
        pts[3], pts[4] = (-0.60, -0.75), (-0.64, -0.98)
    ox, oy = at
    c, s = math.cos(angle), math.sin(angle)
    return [_p(ox + (x * c - y * s) * scale, oy + (x * s + y * c) * scale) for _, (x, y) in sorted(pts.items())]
