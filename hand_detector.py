"""Track one hand with a webcam and stream an arm target to the Processing sketch.

- Move your palm left/right and up/down: the arm follows.
- Move your hand toward the camera: the arm reaches deeper into the scene.
  The distance where your hand first appears is the middle; press c to re-center.
- Pinch thumb and index together to grab the cube; open them to drop it.

On the first run (or with --calibrate, or k in the sketch) a short routine
measures how your hand's size changes with distance; see calibration.py.
v in the sketch starts/stops recording the session to recordings/ (replay.py).

The script runs two TCP servers the Processing sketch connects to (it can be
started or restarted at any time):
- port 65432: one ``x,y,z,grab,mode\\n`` line per frame (mode is run, calib or
  rec); the sketch sends back ``recenter``, ``calibrate`` or ``record`` lines
  when c, k or v is pressed in its window.
- port 65433: the camera preview as JPEG frames, each prefixed with its
  4-byte big-endian length, so the sketch can show it in the same window.

Usage:
    python hand_detector.py [--camera 0] [--invert-depth] [--no-window] [--calibrate] [--record]
                            [--host 127.0.0.1] [--port 65432]
"""

from __future__ import annotations

import argparse
import os
import select
import socket
import struct
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from calibration import DEFAULT_FILE as CALIBRATION_FILE
from calibration import Calibrator, DepthCalibration
from gestures import (HAND_CONNECTIONS, INDEX_TIP, THUMB_TIP, ArmState, Workspace, choose_hand, hand_size,
                      palm_center, pinch_ratio)
from recording import Recorder, new_path

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


class StreamServer:
    """TCP server that accepts one client at a time without blocking the camera loop."""

    def __init__(self, host: str, port: int, name: str):
        self.name = name
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(1)
        self.sock.setblocking(False)
        self.client: socket.socket | None = None

    def _accept(self) -> None:
        try:
            conn, addr = self.sock.accept()
        except BlockingIOError:
            return
        conn.settimeout(0.5)  # a stalled client is dropped instead of freezing the camera loop
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.client = conn
        print(f"Sketch connected ({self.name}) from {addr[0]}:{addr[1]}")

    def send_bytes(self, data: bytes) -> None:
        if self.client is None:
            self._accept()
        if self.client is None:
            return
        try:
            self.client.sendall(data)
        except OSError:
            self._drop()

    def _drop(self) -> None:
        print(f"Sketch disconnected ({self.name}); waiting for it to reconnect")
        if self.client:
            self.client.close()
        self.client = None

    @property
    def connected(self) -> bool:
        return self.client is not None

    def close(self) -> None:
        if self.client:
            self.client.close()
        self.sock.close()


class LineServer(StreamServer):
    """Sends one text line per frame, and reads command lines sent back by the sketch."""

    def __init__(self, host: str, port: int):
        super().__init__(host, port, "control")
        self._inbox = b""
        print(f"Waiting for the Processing sketch on {host}:{port}")

    def send(self, line: str) -> None:
        self.send_bytes(line.encode())

    def commands(self) -> list[str]:
        """Command lines received since the last call (e.g. "recenter")."""
        if self.client is None:
            return []
        try:
            readable, _, _ = select.select([self.client], [], [], 0)
            if not readable:
                return []
            data = self.client.recv(4096)
        except OSError:
            self._drop()
            return []
        if not data:  # the sketch closed the connection
            self._drop()
            return []
        self._inbox += data
        *lines, self._inbox = self._inbox.split(b"\n")
        return [ln.decode(errors="ignore").strip() for ln in lines if ln.strip()]


class FrameServer(StreamServer):
    """Streams the camera preview to the sketch as JPEG frames.

    Each frame is a 4-byte big-endian length followed by the JPEG bytes.
    """

    def __init__(self, host: str, port: int, width: int = 480, quality: int = 70):
        super().__init__(host, port, "video")
        self.width = width
        self.quality = quality

    def send_frame(self, image) -> None:
        if self.client is None:
            self._accept()
            if self.client is None:
                return
        self.send_bytes(encode_frame(image, self.width, self.quality))


def encode_frame(image, width: int = 480, quality: int = 70) -> bytes:
    """Resize to ``width`` pixels wide, JPEG-encode, and prefix with the length."""
    h, w = image.shape[:2]
    if w != width:
        image = cv2.resize(image, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return b""
    data = jpg.tobytes()
    return struct.pack(">I", len(data)) + data


class Pixel:
    """A landmark scaled to pixels, so distances keep the same scale on both axes."""

    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float):
        self.x, self.y = x, y


def to_pixels(landmarks, width: int, height: int) -> list[Pixel]:
    return [Pixel(p.x * width, p.y * height) for p in landmarks]


class OneHandControl:
    """Turns HandLandmarker results into arm targets, following one hand.

    While a calibration is running, the arm holds still and the hand's size
    feeds the calibration routine instead.
    """

    def __init__(self, state: ArmState, calibration: DepthCalibration | None = None):
        self.state = state
        self.palm: tuple[float, float] | None = None  # last palm position, in pixels
        self.active: int | None = None                # index of the hand in control
        self.pinch: float | None = None
        self.calibrator: Calibrator | None = None
        self.just_calibrated: DepthCalibration | None = None  # set for one frame when a calibration ends
        if calibration:
            self.apply_calibration(calibration)

    @property
    def calibrating(self) -> bool:
        return self.calibrator is not None

    def start_calibration(self) -> None:
        self.calibrator = Calibrator(depth_limit=max(map(abs, self.state.workspace.depth)))
        self.state.hand_lost()  # let go of the cube

    def apply_calibration(self, cal: DepthCalibration) -> None:
        self.state.workspace.depth_gain_near = cal.gain_near
        self.state.workspace.depth_gain_far = cal.gain_far

    def apply(self, result, width: int, height: int, t: float | None = None) -> None:
        self.just_calibrated = None
        hands = [to_pixels(lms, width, height) for lms in result.hand_landmarks]
        self.active = choose_hand(hands, self.palm)
        hand = hands[self.active] if self.active is not None else None
        if hand is None:
            self.state.hand_lost()
            self.palm = self.pinch = None
        else:
            self.palm = palm_center(hand)
            self.pinch = pinch_ratio(hand)

        if self.calibrator is not None:
            self.calibrator.update(hand_size(hand) if hand else None, time.monotonic() if t is None else t)
            if self.calibrator.done:
                self.apply_calibration(self.calibrator.result)
                self.state.ref_size = self.calibrator.center_size  # the comfortable distance is the middle
                self.just_calibrated = self.calibrator.result
                self.calibrator = None
            return  # the arm holds still while calibrating

        if hand is not None:
            self.state.update(self.palm[0] / width, self.palm[1] / height, hand_size(hand), self.pinch, t)


def draw_overlay(image, result, control: OneHandControl, connected: bool, hints: bool = True,
                 recording: Recorder | None = None) -> None:
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

    if hints:
        status = "sketch connected" if connected else "waiting for sketch"
        lines = [
            f"side {state.y:6.1f}  height {-state.z:6.1f}  depth {-state.x:6.1f}  grab={state.grab}  |  {status}",
            "pinch to grab   c: re-center   k: calibrate   v: record   q: quit" if control.active is not None
            else "show one hand to the camera",
        ]
    else:  # shown inside the sketch, which has its own status bar
        lines = [] if control.active is not None else ["show one hand to the camera"]
    for i, text in enumerate(lines):
        cv2.putText(image, text, (10, 25 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, text, (10, 25 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    if control.calibrator is not None:
        draw_calibration(image, control.calibrator)
    if recording is not None:
        # red dot and elapsed time, top right
        cv2.circle(image, (w - 110, 24), 8, (0, 0, 255), -1)
        cv2.putText(image, f"REC {int(recording.seconds) // 60:02d}:{int(recording.seconds) % 60:02d}",
                    (w - 95, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)


def draw_calibration(image, cal: Calibrator) -> None:
    """Instruction banner with step number and a hold-progress bar."""
    h, w = image.shape[:2]
    top = h - 110
    band = image[top:h].copy()
    cv2.rectangle(band, (0, 0), (w, 110), (0, 0, 0), -1)
    image[top:h] = cv2.addWeighted(band, 0.65, image[top:h], 0.35, 0)
    step = min(cal.step + 1, len(cal.STEPS))
    cv2.putText(image, f"Calibrating depth  {step}/{len(cal.STEPS)}", (16, top + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)
    scale = 0.7  # shrink the instruction until it fits the frame
    while scale > 0.4 and cv2.getTextSize(cal.prompt, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0] > w - 32:
        scale -= 0.05
    cv2.putText(image, cal.prompt, (16, top + 62), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    bar_w = w - 32
    cv2.rectangle(image, (16, top + 80), (16 + bar_w, top + 94), (90, 90, 90), 1)
    cv2.rectangle(image, (16, top + 80), (16 + int(bar_w * cal.progress), top + 94), (0, 200, 255), -1)


class Session:
    """What the main loop keeps track of besides the arm: calibration file and recording."""

    def __init__(self, control: OneHandControl, frame_size: tuple[int, int], calibration_file: Path):
        self.control = control
        self.frame_size = frame_size
        self.calibration_file = calibration_file
        self.recorder: Recorder | None = None

    def command(self, cmd: str) -> None:
        """Commands from the sketch window (or keys in the camera window)."""
        if cmd == "recenter":
            self.control.state.recenter_depth()
        elif cmd == "calibrate":
            self.control.start_calibration()
            print("Calibrating depth: follow the instructions on the camera image")
        elif cmd == "record":
            self.toggle_recording()

    def toggle_recording(self) -> None:
        if self.recorder is None:
            self.recorder = Recorder(new_path(), self.frame_size, self.control.state)
            print(f"Recording to {self.recorder.path}")
        else:
            self.recorder.close()
            print(f"Saved {self.recorder.frames} frames to {self.recorder.path}")
            self.recorder = None

    def after_frame(self, result, t: float) -> None:
        control = self.control
        if control.just_calibrated:
            control.just_calibrated.save(self.calibration_file)
            print(f"Calibration saved to {self.calibration_file}: "
                  f"near gain {control.just_calibrated.gain_near:.0f}, far gain {control.just_calibrated.gain_far:.0f}")
        if self.recorder is not None:
            self.recorder.write(t, result.hand_landmarks, control.state)
        control.state.mode = "calib" if control.calibrating else "rec" if self.recorder else "run"

    def close(self) -> None:
        if self.recorder is not None:
            self.toggle_recording()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=65432)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="path to hand_landmarker.task")
    parser.add_argument("--invert-depth", action="store_true",
                        help="moving the hand toward the camera brings the arm nearer instead of deeper")
    parser.add_argument("--no-window", action="store_true",
                        help="don't open the camera window; the sketch shows the camera instead (used by the launcher)")
    parser.add_argument("--calibrate", action="store_true", help="run the depth calibration even if one is saved")
    parser.add_argument("--calibration-file", type=Path, default=CALIBRATION_FILE)
    parser.add_argument("--record", action="store_true", help="start recording the session right away")
    args = parser.parse_args()

    landmarker = create_landmarker(ensure_model(args.model))

    # Open the camera before listening, so the launcher only starts the sketch
    # once the tracker can actually run.
    webcam = cv2.VideoCapture(args.camera)
    if not webcam.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    frame_size = (int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    server = LineServer(args.host, args.port)
    video = FrameServer(args.host, args.port + 1)  # camera preview for the sketch's window

    saved = DepthCalibration.load(args.calibration_file)
    control = OneHandControl(ArmState(workspace=Workspace(invert_depth=args.invert_depth)), saved)
    if saved is None or args.calibrate:
        control.start_calibration()
        print("Calibrating depth: follow the instructions on the camera image")
    else:
        print(f"Using saved calibration ({args.calibration_file.name}); press k in the sketch to redo it")
    session = Session(control, frame_size, args.calibration_file)
    if args.record:
        session.toggle_recording()

    # Tell launcher.py we are listening (it sets this variable).
    ready_file = os.environ.get("VISUOCONTROL_READY_FILE")
    if ready_file:
        Path(ready_file).touch()

    if not args.no_window:
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

            session.frame_size = (w, h)
            for command in server.commands():  # keys pressed in the sketch window
                session.command(command)
            control.apply(result, w, h, now)
            session.after_frame(result, now)
            server.send(control.state.message())

            draw_overlay(frame, result, control, server.connected, hints=not args.no_window,
                         recording=session.recorder)
            video.send_frame(frame)
            if args.no_window:
                continue
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            session.command({ord("c"): "recenter", ord("k"): "calibrate", ord("v"): "record"}.get(key, ""))
    finally:
        session.close()
        webcam.release()
        cv2.destroyAllWindows()
        server.close()
        video.close()
        landmarker.close()


if __name__ == "__main__":
    main()
