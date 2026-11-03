"""Track both hands with a webcam and stream an arm target to the Processing sketch.

- Left hand: wrist position moves the arm in X/Y; a closed fist grabs the cube.
- Right hand: thumbs-up raises the arm, thumbs-down lowers it.

The script runs a TCP server and sends one line per frame:
``x,y,z,grab\\n``. The Processing sketch connects as a client, and can be
started or restarted at any time.

Usage:
    python hand_detector.py [--camera 0] [--host 127.0.0.1] [--port 65432]
"""

from __future__ import annotations

import argparse
import socket
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from gestures import HAND_CONNECTIONS, WRIST, ArmState, is_fist, thumb_gesture

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


def user_side(category_name: str) -> str:
    """Which of the user's hands MediaPipe saw, given a mirrored frame.

    The HandLandmarker task labels hands correctly on an unmirrored camera
    image. We run it on the mirrored frame (so the overlay and the X axis
    behave like a mirror), which flips its labels, so swap them back.
    The legacy ``mp.solutions.hands`` API behaved the other way round, which
    is why the original script did not need this.
    """
    return {"Left": "Right", "Right": "Left"}.get(category_name, category_name)


def apply_result(result, state: ArmState, width: int, height: int, dt: float) -> dict[str, str]:
    """Update the arm state from one HandLandmarker result on a mirrored frame.

    Returns the gesture shown for each detected hand, for the on-screen overlay.
    If the left hand is not visible, the arm keeps its last position and
    releases the cube.
    """
    labels: dict[str, str] = {}
    left_seen = False
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        side = user_side(handedness[0].category_name)
        if side == "Left":
            left_seen = True
            state.update_position(landmarks[WRIST].x, landmarks[WRIST].y, width, height)
            state.grab = is_fist(landmarks)
            labels[side] = "fist (grab)" if state.grab else "open"
        elif side == "Right":
            gesture = thumb_gesture(landmarks)
            state.update_height(gesture, dt)
            labels[side] = f"thumb {gesture}"
    if not left_seen:
        state.grab = False
    return labels


def draw_overlay(image, result, labels: dict[str, str], state: ArmState, connected: bool) -> None:
    h, w = image.shape[:2]
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(image, pts[a], pts[b], (0, 200, 0), 2)
        for p in pts:
            cv2.circle(image, p, 3, (0, 0, 255), -1)
        side = user_side(handedness[0].category_name)
        cv2.putText(image, f"{side}: {labels.get(side, '')}", (pts[WRIST][0] - 40, pts[WRIST][1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    status = "sketch connected" if connected else "waiting for sketch"
    hud = f"x={state.x:6.1f}  y={state.y:6.1f}  z={state.z:6.1f}  grab={state.grab}  |  {status}"
    cv2.putText(image, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=65432)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="path to hand_landmarker.task")
    args = parser.parse_args()

    landmarker = create_landmarker(ensure_model(args.model))
    server = LineServer(args.host, args.port)
    state = ArmState()

    webcam = cv2.VideoCapture(args.camera)
    if not webcam.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    start = last = time.monotonic()
    last_ts = -1
    try:
        while True:
            ok, frame = webcam.read()
            if not ok:
                print("Could not read a frame from the camera")
                break

            # Mirror the image so the preview and the arm's X axis behave like
            # a mirror (see user_side() for what this does to hand labels).
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            now = time.monotonic()
            dt, last = now - last, now
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # VIDEO mode needs strictly increasing timestamps
            last_ts = max(int((now - start) * 1000), last_ts + 1)
            result = landmarker.detect_for_video(mp_image, last_ts)

            labels = apply_result(result, state, w, h, dt)
            server.send(state.message())

            draw_overlay(frame, result, labels, state, server.connected)
            cv2.imshow(WINDOW, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        webcam.release()
        cv2.destroyAllWindows()
        server.close()
        landmarker.close()


if __name__ == "__main__":
    main()
