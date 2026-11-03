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
import time
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


def hand_roll(lms) -> float:
    """How far the hand is turned in the camera's plane, in radians.

    0 with the fingers pointing up; positive when the hand turns clockwise on
    screen (fingers toward the right). Uses the wrist and the middle knuckle,
    so it is the same for either hand and pinching does not change it.
    """
    dx = lms[MIDDLE_MCP].x - lms[WRIST].x
    dy = lms[MIDDLE_MCP].y - lms[WRIST].y
    return math.atan2(dx, -dy)


def wrap_angle(a: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def pinch_ratio(lms) -> float:
    """Thumb-tip to index-tip distance, in palm sizes. Small means pinching."""
    return _dist(lms[THUMB_TIP], lms[INDEX_TIP]) / hand_size(lms)


def _lerp(lo: float, hi: float, u: float) -> float:
    return lo + (hi - lo) * u


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class OneEuroFilter:
    """Low-pass filter whose cutoff rises with speed (Casiez, Roussel & Vogel, CHI 2012).

    When the hand is still, the cutoff stays at ``min_cutoff`` and jitter is
    smoothed away. When it moves fast, the cutoff rises by ``beta`` per unit of
    speed, so the output keeps up instead of lagging behind.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: float | None = None
        self._dx = 0.0
        self._t: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self._x is None or self._t is None or t <= self._t:
            self._x, self._t = x, t
            return x
        dt = t - self._t
        self._t = t
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        return self._x


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
    # Palm position (normalized, mirrored frame) that maps to the middle of the
    # workspace. Calibration sets it to where your hand rests comfortably.
    center_x: float = 0.5
    center_y: float = 0.5
    # Depth units per e-fold change in apparent hand size, when the hand is
    # closer than the middle (near) or farther (far). Calibration sets these.
    depth_gain_near: float = 110.0
    depth_gain_far: float = 110.0
    invert_depth: bool = False  # True: moving the hand toward the camera brings the arm nearer
    # Wrist rotation follows the hand's roll beyond a small dead zone, up to a limit
    roll_deadzone: float = math.radians(10)
    roll_limit: float = math.radians(80)


@dataclass
class ArmState:
    """Target sent to the simulation, driven by one hand."""

    x: float = 0.0  # coordX: depth
    y: float = 0.0  # coordY: sideways
    z: float = 0.0  # coordZ: height (negative is up)
    grab: bool = False
    roll: float = 0.0  # wrist rotation in radians (positive: clockwise as seen on screen)

    mode: str = "run"  # what the tracker is doing: run, calib (calibrating) or rec (recording)

    workspace: Workspace = field(default_factory=Workspace)
    # One Euro filter settings. min_cutoff (Hz): lower = steadier when still.
    # beta: higher = less lag when moving fast. Depth (apparent size) is noisier.
    filtering: bool = True
    min_cutoff: float = 1.0
    beta: float = 0.02
    depth_min_cutoff: float = 0.5
    depth_beta: float = 0.02
    pinch_close: float = 0.25     # pinch ratio below this starts a grab (real pinches are ~0.1)
    pinch_open: float = 0.40      # ...and above this releases it (hysteresis)

    ref_size: float | None = None  # hand size that means "middle depth"
    roll_ref: float = 0.0          # hand roll that means "wrist straight" (calibration sets it)

    def __post_init__(self):
        self._filters = [OneEuroFilter(self.depth_min_cutoff, self.depth_beta),  # x (depth)
                         OneEuroFilter(self.min_cutoff, self.beta),              # y (side)
                         OneEuroFilter(self.min_cutoff, self.beta),              # z (height)
                         OneEuroFilter(self.min_cutoff, self.beta)]              # roll

    def recenter_depth(self) -> None:
        """Make the hand's next apparent size the middle depth."""
        self.ref_size = None

    def depth_for(self, size: float) -> float:
        """Arm depth (coordX) for a hand of apparent ``size``, relative to ``ref_size``."""
        ws = self.workspace
        r = math.log(size / self.ref_size)
        gain = ws.depth_gain_near if r > 0 else ws.depth_gain_far
        sign = 1 if ws.invert_depth else -1  # default: closer to the camera = deeper
        return _clamp(sign * gain * r, *ws.depth)

    def roll_for(self, hand_angle: float) -> float:
        """Wrist rotation for a hand turned by ``hand_angle`` (see hand_roll)."""
        ws = self.workspace
        r = wrap_angle(hand_angle - self.roll_ref)
        r = math.copysign(max(0.0, abs(r) - ws.roll_deadzone), r)  # small turns do nothing
        return _clamp(r, -ws.roll_limit, ws.roll_limit)

    def update(self, palm_x: float, palm_y: float, size: float, pinch: float, t: float | None = None,
               hand_angle: float | None = None) -> None:
        """Update from one frame.

        palm_x and palm_y are normalized (0..1) in the mirrored frame; size is
        the hand size in pixels; pinch is ``pinch_ratio``; t is the frame time
        in seconds (defaults to now); hand_angle is ``hand_roll`` (None keeps
        the wrist where it is).
        """
        if t is None:
            t = time.monotonic()
        ws = self.workspace
        span = 1 - 2 * ws.frame_margin
        # The calibrated rest position maps to the middle of the workspace
        u = _clamp((palm_x - ws.center_x) / span + 0.5, 0, 1)
        v = _clamp((palm_y - ws.center_y) / span + 0.5, 0, 1)
        y = _lerp(*ws.side, u)
        z = _lerp(*ws.height, v)

        if self.ref_size is None:
            self.ref_size = size
        x = self.depth_for(size)

        roll = self.roll if hand_angle is None else self.roll_for(hand_angle)

        if self.filtering:
            x, y, z, roll = (f(v, t) for f, v in zip(self._filters, (x, y, z, roll)))
        self.x, self.y, self.z, self.roll = x, y, z, roll

        if self.grab and pinch > self.pinch_open:
            self.grab = False
        elif not self.grab and pinch < self.pinch_close:
            self.grab = True

    def hand_lost(self) -> None:
        """No hand in view: keep the arm where it is and let go of the cube."""
        self.grab = False

    def message(self) -> str:
        """One newline-terminated line in the wire format the sketch parses: x,y,z,grab,mode,roll."""
        return f"{self.x:.2f},{self.y:.2f},{self.z:.2f},{self.grab},{self.mode},{self.roll:.3f}\n"


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
