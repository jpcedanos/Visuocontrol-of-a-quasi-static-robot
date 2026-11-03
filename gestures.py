"""One-hand control logic, independent of the camera and MediaPipe.

Landmarks are any sequence of 21 objects with ``x`` and ``y`` attributes, in
MediaPipe's hand-landmark order. Functions here expect them in pixels (or any
units with the same scale on both axes), so distances are not distorted by the
image's aspect ratio. Keeping this module free of camera and model code lets
the rules be unit-tested with synthetic hands.

Control scheme (one hand, mirrored camera view):
- palm left/right  -> arm left/right
- palm up/down     -> arm up/down
- hand closer to / farther from the camera (apparent size) -> arm deeper / nearer
- pinch thumb and index together -> grab
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# MediaPipe hand landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_TIP = 5, 8
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17
PALM = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def _dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def palm_center(lms) -> tuple[float, float]:
    """Mean of the wrist and the four knuckles.

    These points barely move when the fingers do, so pinching does not drag
    the arm around.
    """
    return (sum(lms[i].x for i in PALM) / len(PALM), sum(lms[i].y for i in PALM) / len(PALM))


def hand_size(lms) -> float:
    """Palm size: average of palm length (wrist to middle knuckle) and width (index to pinky knuckle)."""
    return (_dist(lms[WRIST], lms[MIDDLE_MCP]) + _dist(lms[INDEX_MCP], lms[PINKY_MCP])) / 2 or 1e-6


def pinch_ratio(lms) -> float:
    """Thumb-tip to index-tip distance, in palm sizes. Small means pinching."""
    return _dist(lms[THUMB_TIP], lms[INDEX_TIP]) / hand_size(lms)


def _lerp(lo: float, hi: float, u: float) -> float:
    return lo + (hi - lo) * u


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class Workspace:
    """How hand motion maps to the arm's target, in the sketch's arm coordinates.

    With the sketch's default view, coordX points toward the viewer, coordY to
    the right and coordZ down. The defaults cover both tables and the floor
    between them, within the arm's 120-unit reach.
    """

    side: tuple[float, float] = (-85.0, 85.0)    # coordY at the left / right edge of the frame
    height: tuple[float, float] = (-50.0, 60.0)  # coordZ at the top / bottom of the frame
    depth: tuple[float, float] = (-50.0, 50.0)   # coordX limits
    frame_margin: float = 0.15  # the outer 15% of the frame maps to the workspace edge
    depth_gain: float = 110.0   # depth units for an e-fold change in apparent hand size
    invert_depth: bool = False  # True: moving the hand toward the camera brings the arm nearer


@dataclass
class ArmState:
    """Target sent to the simulation, driven by one hand."""

    x: float = 0.0  # coordX: depth
    y: float = 0.0  # coordY: sideways
    z: float = 0.0  # coordZ: height (negative is up)
    grab: bool = False

    workspace: Workspace = field(default_factory=Workspace)
    smoothing: float = 0.5        # 0 = none; closer to 1 = smoother but laggier
    depth_smoothing: float = 0.7  # apparent size is noisier than position
    pinch_close: float = 0.25     # pinch ratio below this starts a grab (real pinches are ~0.1)
    pinch_open: float = 0.40      # ...and above this releases it (hysteresis)

    ref_size: float | None = None  # hand size that means "middle depth"
    _has_position: bool = False

    def recenter_depth(self) -> None:
        """Make the hand's next apparent size the middle depth."""
        self.ref_size = None

    def update(self, palm_x: float, palm_y: float, size: float, pinch: float) -> None:
        """Update from one frame.

        palm_x and palm_y are normalized (0..1) in the mirrored frame; size is
        the hand size in pixels; pinch is ``pinch_ratio``.
        """
        ws = self.workspace
        span = 1 - 2 * ws.frame_margin
        u = _clamp((palm_x - ws.frame_margin) / span, 0, 1)
        v = _clamp((palm_y - ws.frame_margin) / span, 0, 1)
        y = _lerp(*ws.side, u)
        z = _lerp(*ws.height, v)

        if self.ref_size is None:
            self.ref_size = size
        sign = 1 if ws.invert_depth else -1  # default: closer to the camera = deeper
        x = _clamp(sign * ws.depth_gain * math.log(size / self.ref_size), *ws.depth)

        if self._has_position:
            k, kd = self.smoothing, self.depth_smoothing
            y = k * self.y + (1 - k) * y
            z = k * self.z + (1 - k) * z
            x = kd * self.x + (1 - kd) * x
        self.x, self.y, self.z = x, y, z
        self._has_position = True

        if self.grab and pinch > self.pinch_open:
            self.grab = False
        elif not self.grab and pinch < self.pinch_close:
            self.grab = True

    def hand_lost(self) -> None:
        """No hand in view: keep the arm where it is and let go of the cube."""
        self.grab = False

    def message(self) -> str:
        """One newline-terminated line in the wire format the sketch parses."""
        return f"{self.x:.2f},{self.y:.2f},{self.z:.2f},{self.grab}\n"


def choose_hand(hands, previous: tuple[float, float] | None):
    """Pick which detected hand drives the arm.

    ``hands`` holds pixel-space landmark lists. With a previous palm position,
    keep following the nearest hand so the arm does not jump between hands;
    otherwise take the biggest (closest) one. Returns an index or None.
    """
    if not hands:
        return None
    if previous is None:
        return max(range(len(hands)), key=lambda i: hand_size(hands[i]))
    px, py = previous
    return min(range(len(hands)), key=lambda i: math.hypot(palm_center(hands[i])[0] - px,
                                                           palm_center(hands[i])[1] - py))
