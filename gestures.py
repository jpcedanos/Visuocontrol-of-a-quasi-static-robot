"""Hand-gesture logic and arm state, independent of the camera and MediaPipe.

Landmarks are any sequence of 21 objects with normalized ``x`` and ``y``
attributes (0..1, image coordinates, y grows downward), in MediaPipe's
hand-landmark order. Keeping this module free of camera and model code lets
the gesture rules be unit-tested with synthetic hands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# MediaPipe hand landmark indices
WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_PIP, PINKY_TIP = 18, 20

FINGERS = [  # (pip, tip) for the four non-thumb fingers
    (INDEX_PIP, INDEX_TIP),
    (MIDDLE_PIP, MIDDLE_TIP),
    (RING_PIP, RING_TIP),
    (PINKY_PIP, PINKY_TIP),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def _dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def hand_size(lms) -> float:
    """Wrist to middle-finger knuckle distance, used to scale thresholds."""
    return _dist(lms[WRIST], lms[MIDDLE_MCP]) or 1e-6


def is_fist(lms) -> bool:
    """True when all four fingers are curled.

    A finger counts as curled when its tip is closer to the wrist than its
    middle joint (PIP). Using distances instead of raw y values makes the
    check work whatever way the hand is rotated.
    """
    wrist = lms[WRIST]
    return all(_dist(lms[tip], wrist) < _dist(lms[pip], wrist) for pip, tip in FINGERS)


def thumb_gesture(lms, margin: float = 0.35) -> str:
    """Classify a thumbs-up / thumbs-down gesture.

    Returns ``"up"``, ``"down"`` or ``"neutral"``. A gesture needs the four
    fingers curled and the thumb tip clearly above (up) or below (down) its
    knuckle, by at least ``margin`` hand sizes. An open hand is always
    neutral, so resting the hand does not move the arm.
    """
    if not is_fist(lms):
        return "neutral"
    dy = (lms[THUMB_TIP].y - lms[THUMB_MCP].y) / hand_size(lms)
    if dy < -margin:
        return "up"
    if dy > margin:
        return "down"
    return "neutral"


@dataclass
class ArmState:
    """Target sent to the simulation: position, height and grip."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    grab: bool = False

    z_speed: float = 30.0   # units per second while a thumb gesture is held
    z_limit: float = 100.0  # |z| is clamped to this
    smoothing: float = 0.5  # 0 = no smoothing, closer to 1 = smoother but laggier
    pixel_scale: float = 2.5  # camera pixels per simulation unit

    _has_position: bool = False

    def update_position(self, wrist_x: float, wrist_y: float, width: int, height: int) -> None:
        """Map a normalized wrist position to arm X/Y, with smoothing.

        Uses the same mapping as the original version: offset from the image
        centre in pixels, divided by ``pixel_scale``, with both axes inverted
        to match the simulation's frame.
        """
        x = -(wrist_x * width - width / 2) / self.pixel_scale
        y = (wrist_y * height - height / 2) / self.pixel_scale
        if self._has_position:
            k = self.smoothing
            x = k * self.x + (1 - k) * x
            y = k * self.y + (1 - k) * y
        self.x, self.y = x, y
        self._has_position = True

    def update_height(self, gesture: str, dt: float) -> None:
        """Move Z at a constant speed while a thumb gesture is held.

        Thumb up raises the arm, which is negative Z in the simulation.
        Speed is per second, so it no longer depends on the camera's frame rate.
        """
        step = self.z_speed * dt
        if gesture == "up":
            self.z -= step
        elif gesture == "down":
            self.z += step
        self.z = max(-self.z_limit, min(self.z_limit, self.z))

    def message(self) -> str:
        """One newline-terminated line in the wire format the sketch parses."""
        return f"{self.x:.2f},{self.y:.2f},{self.z:.2f},{self.grab}\n"
