"""Replay a recorded session into the Processing sketch, without a camera.

It serves the same two connections as hand_detector.py, so the sketch (and
launcher.py --replay) work unchanged. The camera panel shows the recorded hand
drawn on a dark background.

Two modes:
- default: send the arm targets exactly as recorded, to watch a session again
  or capture a demo video;
- --recompute: run the recorded hand landmarks through the current control
  code (filter, pinch thresholds...) with the recorded calibration, to see how
  a change behaves with real movements.

Usage:
    python replay.py recordings/session_20261103_101500.jsonl [--speed 1.0] [--loop] [--recompute]
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from gestures import ArmState, Workspace
from hand_detector import FrameServer, LineServer, OneHandControl, draw_overlay
from recording import read_recording


def frame_result(frame: dict) -> SimpleNamespace:
    """A recorded frame in the shape of a HandLandmarker result."""
    return SimpleNamespace(hand_landmarks=[[SimpleNamespace(x=x, y=y) for x, y in hand] for hand in frame["hands"]])


def make_control(meta: dict) -> OneHandControl:
    """Control set up with the calibration (workspace) the session was recorded with."""
    known = {f.name for f in fields(Workspace)}
    ws = Workspace(**{k: tuple(v) if isinstance(v, list) else v
                      for k, v in meta.get("workspace", {}).items() if k in known})
    state = ArmState(workspace=ws)
    state.ref_size = meta.get("ref_size")
    return OneHandControl(state)


def render(frame: dict, result, control: OneHandControl, size: tuple[int, int], label: str):
    w, h = size
    image = np.full((h, w, 3), 38, np.uint8)
    draw_overlay(image, result, control, True, hints=False)
    cv2.putText(image, label, (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
    return image


class Player:
    """Steps through a recording in real time (scaled by speed)."""

    def __init__(self, meta: dict, frames: list[dict], recompute: bool = False, speed: float = 1.0):
        if not frames:
            raise ValueError("the recording has no frames")
        self.meta, self.frames, self.recompute, self.speed = meta, frames, recompute, speed
        self.size = tuple(meta["frame"])
        self.duration = frames[-1]["t"]
        self.restart()

    def restart(self) -> None:
        self.control = make_control(self.meta)
        self.index = 0

    @property
    def finished(self) -> bool:
        return self.index >= len(self.frames)

    def step(self) -> tuple[dict, object]:
        """Advance one frame; return the frame and its result. The arm state is in self.control.state."""
        frame = self.frames[self.index]
        self.index += 1
        result = frame_result(frame)
        self.control.apply(result, *self.size, frame["t"])
        state = self.control.state
        if not self.recompute:
            state.x, state.y, state.z = frame["target"]
            state.grab = frame["grab"]
            state.roll = frame.get("roll", 0.0)  # recordings made before wrist rotation have none
        state.mode = "replay"
        return frame, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed (2 = twice as fast)")
    parser.add_argument("--loop", action="store_true", help="start over at the end")
    parser.add_argument("--recompute", action="store_true",
                        help="run the recorded hand through the current control code instead of the recorded targets")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=65432)
    args = parser.parse_args()

    meta, frames = read_recording(args.recording)
    player = Player(meta, frames, args.recompute, args.speed)
    mode = "recomputed" if args.recompute else "as recorded"
    print(f"Replaying {args.recording.name}: {len(frames)} frames, {player.duration:.1f} s ({mode})")

    server = LineServer(args.host, args.port)
    video = FrameServer(args.host, args.port + 1)
    ready_file = os.environ.get("VISUOCONTROL_READY_FILE")
    if ready_file:
        Path(ready_file).touch()

    # Start once the sketch is connected, so the beginning isn't missed
    while not server.connected:
        server.send(player.control.state.message())
        time.sleep(0.05)

    start = time.monotonic()
    frame = result = None
    try:
        while True:
            elapsed = (time.monotonic() - start) * args.speed
            while not player.finished and player.frames[player.index]["t"] <= elapsed:
                frame, result = player.step()
            if player.finished and args.loop:
                player.restart()
                start = time.monotonic()
                continue
            if frame is not None:
                done = " (finished)" if player.finished else ""
                label = f"REPLAY {min(elapsed, player.duration):5.1f} / {player.duration:.1f} s{done}"
                server.send(player.control.state.message())
                video.send_frame(render(frame, result, player.control, player.size, label))
            server.commands()  # notices if the sketch closed
            if not server.connected:
                print("Sketch closed; stopping the replay")
                break
            time.sleep(1 / 60)
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        video.close()


if __name__ == "__main__":
    main()
