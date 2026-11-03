"""Track one hand with a webcam and stream an arm target to the Processing sketch.

- Move your palm left/right and up/down: the arm follows.
- Move your hand toward the camera: the arm reaches deeper into the scene.
  The distance where your hand first appears is the middle; press c to re-center.
- Pinch thumb and index together to grab the cube; open them to drop it.

The script runs a TCP server and sends one line per frame:
``x,y,z,grab\\n``. The Processing sketch connects as a client, and can be
started or restarted at any time.

Usage:
    python hand_detector.py [--camera 0] [--invert-depth] [--host 127.0.0.1] [--port 65432]
"""

from __future__ import annotations

import argparse
import os
import socket
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from gestures import (HAND_CONNECTIONS, INDEX_TIP, THUMB_TIP, ArmState, Workspace, choose_hand, hand_size,
                      palm_center, pinch_ratio)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
DEFAULT_MODEL = Path(__file__).parent / "models" / "hand_landmarker.task"
WINDOW = "Hand tracking (q to quit)"


def ensure_model(path: Path) -> Path:
    """Download the MediaPipe hand model the first time the script runs."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading hand model to {path} ...")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def create_landmarker(model_path: Path, running_mode=vision.RunningMode.VIDEO) -> vision.HandLandmarker:
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=running_mode,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


class LineServer:
    """TCP server that accepts one client at a time without blocking the camera loop."""

    def __init__(self, host: str, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(1)
        self.sock.setblocking(False)
        self.client: socket.socket | None = None
        print(f"Waiting for the Processing sketch on {host}:{port}")

    def _accept(self) -> None:
        try:
            conn, addr = self.sock.accept()
        except BlockingIOError:
            return
        conn.setblocking(True)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.client = conn
        print(f"Sketch connected from {addr[0]}:{addr[1]}")

    def send(self, line: str) -> None:
        if self.client is None:
            self._accept()
        if self.client is None:
            return
        try:
            self.client.sendall(line.encode())
        except OSError:
            print("Sketch disconnected; waiting for it to reconnect")
            self.client.close()
            self.client = None

    @property
    def connected(self) -> bool:
        return self.client is not None

    def close(self) -> None:
        if self.client:
            self.client.close()
        self.sock.close()


class Pixel:
    """A landmark scaled to pixels, so distances keep the same scale on both axes."""

    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float):
        self.x, self.y = x, y


def to_pixels(landmarks, width: int, height: int) -> list[Pixel]:
    return [Pixel(p.x * width, p.y * height) for p in landmarks]


class OneHandControl:
    """Turns HandLandmarker results into arm targets, following one hand."""

    def __init__(self, state: ArmState):
        self.state = state
        self.palm: tuple[float, float] | None = None  # last palm position, in pixels
        self.active: int | None = None                # index of the hand in control
        self.pinch: float | None = None

    def apply(self, result, width: int, height: int) -> None:
        hands = [to_pixels(lms, width, height) for lms in result.hand_landmarks]
        self.active = choose_hand(hands, self.palm)
        if self.active is None:
            self.state.hand_lost()
            self.palm = self.pinch = None
            return
        hand = hands[self.active]
        self.palm = palm_center(hand)
        self.pinch = pinch_ratio(hand)
        self.state.update(self.palm[0] / width, self.palm[1] / height, hand_size(hand), self.pinch)


def draw_overlay(image, result, control: OneHandControl, connected: bool) -> None:
    h, w = image.shape[:2]
    state = control.state
    for i, landmarks in enumerate(result.hand_landmarks):
        pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
        active = i == control.active
        color = (0, 200, 0) if active else (110, 110, 110)
        for a, b in HAND_CONNECTIONS:
            cv2.line(image, pts[a], pts[b], color, 2)
        for p in pts:
            cv2.circle(image, p, 3, (0, 0, 255) if active else color, -1)
        if active:
            # Thumb-index line: orange while open, green and thick while grabbing
            pinch_color, thickness = ((0, 255, 0), 5) if state.grab else ((0, 165, 255), 2)
            cv2.line(image, pts[THUMB_TIP], pts[INDEX_TIP], pinch_color, thickness)
            label = "GRAB" if state.grab else f"pinch {control.pinch:.2f}"
            cv2.putText(image, label, (pts[INDEX_TIP][0] + 10, pts[INDEX_TIP][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, pinch_color, 2, cv2.LINE_AA)

    status = "sketch connected" if connected else "waiting for sketch"
    lines = [
        f"side {state.y:6.1f}  height {-state.z:6.1f}  depth {-state.x:6.1f}  grab={state.grab}  |  {status}",
        "pinch to grab   c: re-center depth   q: quit" if control.active is not None else "show one hand to the camera",
    ]
    for i, text in enumerate(lines):
        cv2.putText(image, text, (10, 25 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, text, (10, 25 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=65432)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="path to hand_landmarker.task")
    parser.add_argument("--invert-depth", action="store_true",
                        help="moving the hand toward the camera brings the arm nearer instead of deeper")
    args = parser.parse_args()

    landmarker = create_landmarker(ensure_model(args.model))

    # Open the camera before listening, so the launcher only starts the sketch
    # once the tracker can actually run.
    webcam = cv2.VideoCapture(args.camera)
    if not webcam.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    server = LineServer(args.host, args.port)
    control = OneHandControl(ArmState(workspace=Workspace(invert_depth=args.invert_depth)))

    # Tell launcher.py we are listening (it sets this variable).
    ready_file = os.environ.get("VISUOCONTROL_READY_FILE")
    if ready_file:
        Path(ready_file).touch()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    start = time.monotonic()
    last_ts = -1
    try:
        while True:
            ok, frame = webcam.read()
            if not ok:
                print("Could not read a frame from the camera")
                break

            # Mirror the image so moving your hand right moves it right on screen
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            now = time.monotonic()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # VIDEO mode needs strictly increasing timestamps
            last_ts = max(int((now - start) * 1000), last_ts + 1)
            result = landmarker.detect_for_video(mp_image, last_ts)

            control.apply(result, w, h)
            server.send(control.state.message())

            draw_overlay(frame, result, control, server.connected)
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                control.state.recenter_depth()
    finally:
        webcam.release()
        cv2.destroyAllWindows()
        server.close()
        landmarker.close()


if __name__ == "__main__":
    main()
