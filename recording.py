"""Session recordings: one JSON object per line (JSON Lines).

The first line is a header; every other line is one camera frame:

    {"type": "meta", "version": 1, "frame": [640, 480], "started": "...",
     "workspace": {...}, "settings": {...}}
    {"type": "frame", "t": 0.033, "hands": [[[x, y], ... 21 points], ...],
     "target": [x, y, z], "grab": false, "roll": 0.0}

Hand points are normalized (0..1) in the mirrored camera frame, exactly as
MediaPipe returned them, so a recording can be replayed as-is or run again
through the current control code (see replay.py). The same structure is a
starting point for teleoperation datasets later on.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

VERSION = 1
DEFAULT_DIR = Path(__file__).parent / "recordings"


def new_path(directory: Path = DEFAULT_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory / time.strftime("session_%Y%m%d_%H%M%S.jsonl")


class Recorder:
    def __init__(self, path: Path, frame_size: tuple[int, int], state=None):
        self.path = path
        self._file = open(path, "w", encoding="utf-8")
        self._t0: float | None = None
        self.frames = 0
        meta = {"type": "meta", "version": VERSION, "frame": list(frame_size),
                "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if state is not None:
            meta["workspace"] = asdict(state.workspace)
            meta["settings"] = {k: getattr(state, k) for k in
                                ("filtering", "min_cutoff", "beta", "depth_min_cutoff", "depth_beta",
                                 "pinch_close", "pinch_open")}
            meta["ref_size"] = state.ref_size
        self._write(meta)

    def _write(self, obj: dict) -> None:
        self._file.write(json.dumps(obj, separators=(",", ":")) + "\n")

    def write(self, t: float, hands, state) -> None:
        """Record one frame. ``hands`` are MediaPipe landmark lists (normalized x, y)."""
        if self._t0 is None:
            self._t0 = t
        self._write({
            "type": "frame",
            "t": round(t - self._t0, 4),
            "hands": [[[round(p.x, 5), round(p.y, 5)] for p in lms] for lms in hands],
            "target": [round(state.x, 3), round(state.y, 3), round(state.z, 3)],
            "grab": state.grab,
            "roll": round(state.roll, 4),
        })
        self.frames += 1

    @property
    def seconds(self) -> float:
        return 0.0 if self._t0 is None else time.monotonic() - self._t0

    def close(self) -> None:
        self._file.close()


def read_recording(path: Path) -> tuple[dict, list[dict]]:
    """Return (header, frames). Raises ValueError if the file is not a recording."""
    meta, frames = None, []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                if n > 1:  # a session cut short can end with a partial line
                    break
                raise ValueError(f"{path} is not a recording")
            if obj.get("type") == "meta":
                meta = obj
            elif obj.get("type") == "frame":
                frames.append(obj)
    if meta is None:
        raise ValueError(f"{path} has no header line; is it a recording?")
    return meta, frames
