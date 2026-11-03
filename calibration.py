"""Depth calibration: learn how much your hand's apparent size changes with distance.

Cameras, lenses and how far you sit all change how much bigger your hand looks
when you move it toward the camera. A short guided routine measures that:

1. Hold your hand still at a comfortable distance (this becomes the middle).
2. Move it closer to the camera and hold.
3. Move it farther away and hold.

The result is two depth gains, chosen so your closest and farthest positions
reach 90% of the workspace depth. They are saved to calibration.json and
reused next time; press k in the sketch window to calibrate again.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_FILE = Path(__file__).parent / "calibration.json"


@dataclass
class DepthCalibration:
    gain_near: float
    gain_far: float

    def save(self, path: Path = DEFAULT_FILE) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path = DEFAULT_FILE) -> "DepthCalibration | None":
        try:
            data = json.loads(path.read_text())
            return cls(float(data["gain_near"]), float(data["gain_far"]))
        except (OSError, ValueError, KeyError, TypeError):
            return None


class Calibrator:
    """Step-by-step routine, fed one hand size per frame (None when no hand is seen)."""

    STEPS = ("center", "near", "far")
    PROMPTS = {
        "center": "Hold your hand still at a comfortable distance",
        "near": "Now move it closer to the camera and hold",
        "far": "Now move it farther away than at the start and hold",
    }

    def __init__(self, depth_limit: float = 50.0, hold_time: float = 0.8, min_change: float = 1.12,
                 steadiness: float = 0.06, gain_range: tuple[float, float] = (40.0, 400.0)):
        self.depth_limit = depth_limit
        self.hold_time = hold_time        # seconds the hand must stay steady at each step
        self.min_change = min_change      # near/far must be at least 12% bigger/smaller than the middle
        self.steadiness = steadiness      # max size variation while holding, as a fraction
        self.gain_range = gain_range      # keeps a sloppy calibration from producing absurd gains
        self.step = 0
        self.sizes: dict[str, float] = {}
        self.result: DepthCalibration | None = None
        self._window: deque[tuple[float, float]] = deque()
        self.progress = 0.0               # 0..1 of the current hold, for the on-screen bar

    @property
    def done(self) -> bool:
        return self.result is not None

    @property
    def current(self) -> str:
        return self.STEPS[min(self.step, len(self.STEPS) - 1)]

    @property
    def prompt(self) -> str:
        return "Calibrated" if self.done else self.PROMPTS[self.current]

    def _in_position(self, size: float) -> bool:
        if self.current == "center":
            return True
        ref = self.sizes["center"]
        if self.current == "near":
            return size >= ref * self.min_change
        return size <= ref / self.min_change

    def update(self, size: float | None, t: float) -> None:
        if self.done:
            return
        if size is None or not self._in_position(size):
            self._window.clear()
            self.progress = 0.0
            return
        # The window holds every sample since this hold started; the hold
        # completes once it spans hold_time.
        self._window.append((t, size))
        sizes = [s for _, s in self._window]
        if max(sizes) - min(sizes) > self.steadiness * (sum(sizes) / len(sizes)):
            # moving too much: start the hold again from this frame
            self._window.clear()
            self._window.append((t, size))
            self.progress = 0.0
            return
        held = t - self._window[0][0]
        self.progress = min(1.0, held / self.hold_time)
        if held >= self.hold_time:
            self.sizes[self.current] = sorted(sizes)[len(sizes) // 2]  # median
            self._window.clear()
            self.progress = 0.0
            self.step += 1
            if self.step == len(self.STEPS):
                self.result = self._gains()

    def _gains(self) -> DepthCalibration:
        ref, near, far = self.sizes["center"], self.sizes["near"], self.sizes["far"]
        target = 0.9 * self.depth_limit
        lo, hi = self.gain_range
        gain_near = min(hi, max(lo, target / math.log(near / ref)))
        gain_far = min(hi, max(lo, target / math.log(ref / far)))
        return DepthCalibration(gain_near, gain_far)

    @property
    def center_size(self) -> float | None:
        return self.sizes.get("center")
