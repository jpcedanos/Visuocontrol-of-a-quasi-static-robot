from types import SimpleNamespace

import pytest

pytest.importorskip("mediapipe")
pytest.importorskip("cv2")

from gestures import ArmState  # noqa: E402
from hand_detector import OneHandControl  # noqa: E402
from synthetic import hand  # noqa: E402

W, H = 640, 480


def _result(*hands_px):
    """Fake HandLandmarker result from pixel-space hands (MediaPipe gives normalized coords)."""
    return SimpleNamespace(hand_landmarks=[[SimpleNamespace(x=p.x / W, y=p.y / H) for p in h] for h in hands_px])


def test_pinch_grabs_and_opening_releases():
    c = OneHandControl(ArmState())
    c.apply(_result(hand(pinch=False)), W, H)
    assert c.state.grab is False
    c.apply(_result(hand(pinch=True)), W, H)
    assert c.state.grab is True
    c.apply(_result(hand(pinch=False)), W, H)
    assert c.state.grab is False


def test_hand_leaving_the_frame_drops_the_cube():
    c = OneHandControl(ArmState())
    c.apply(_result(hand(pinch=True)), W, H)
    assert c.state.grab
    c.apply(_result(), W, H)
    assert c.state.grab is False
    assert c.active is None


def test_follows_the_same_hand_when_a_second_one_appears():
    c = OneHandControl(ArmState(filtering=False))
    first = hand(scale=80, at=(200, 400))
    c.apply(_result(first), W, H)
    side = c.state.y
    intruder = hand(scale=140, at=(500, 400))  # bigger and on the other side
    c.apply(_result(intruder, first), W, H)
    assert c.active == 1
    assert c.state.y == pytest.approx(side)


# --- streams to the sketch ------------------------------------------------------

import socket  # noqa: E402
import struct  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from hand_detector import FrameServer, LineServer, encode_frame  # noqa: E402


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _connect(server, port):
    client = socket.create_connection(("127.0.0.1", port))
    for _ in range(50):  # the server accepts on its next send
        server.send_bytes(b"")
        if server.connected:
            return client
        time.sleep(0.01)
    raise AssertionError("server did not accept")


def test_encode_frame_is_length_prefixed_jpeg_at_the_requested_width():
    import cv2
    data = encode_frame(np.zeros((480, 640, 3), np.uint8), width=320)
    (n,) = struct.unpack(">I", data[:4])
    assert n == len(data) - 4
    img = cv2.imdecode(np.frombuffer(data[4:], np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (240, 320, 3)


def test_frame_server_streams_frames_back_to_back():
    port = _free_port()
    server = FrameServer("127.0.0.1", port, width=160)
    client = _connect(server, port)
    for _ in range(3):
        server.send_frame(np.full((120, 160, 3), 128, np.uint8))
    client.settimeout(2)
    buf = b""
    frames = 0
    while frames < 3:
        buf += client.recv(65536)
        while len(buf) >= 4 and len(buf) >= 4 + struct.unpack(">I", buf[:4])[0]:
            n = struct.unpack(">I", buf[:4])[0]
            assert buf[4:6] == b"\xff\xd8"  # JPEG start marker
            buf = buf[4 + n:]
            frames += 1
    client.close()
    server.close()


def test_line_server_reads_commands_split_across_packets():
    port = _free_port()
    server = LineServer("127.0.0.1", port)
    client = _connect(server, port)
    client.sendall(b"recen")
    time.sleep(0.05)
    assert server.commands() == []
    client.sendall(b"ter\nrecenter\n")
    time.sleep(0.05)
    assert server.commands() == ["recenter", "recenter"]
    client.close()
    time.sleep(0.05)
    server.commands()  # notices the disconnect
    assert not server.connected
    server.close()


# --- calibration and recording in the tracker ------------------------------------

from calibration import DepthCalibration  # noqa: E402
from gestures import hand_size  # noqa: E402
from hand_detector import Session  # noqa: E402
from recording import read_recording  # noqa: E402


def _run(control, scale, seconds, t0, pinch=False, fps=30):
    t = t0
    for i in range(int(seconds * fps)):
        t = t0 + i / fps
        control.apply(_result(hand(pinch=pinch, scale=scale)), W, H, t)
    return t + 1 / fps


def test_calibration_holds_the_arm_then_applies_gains():
    c = OneHandControl(ArmState(filtering=False))
    c.start_calibration()
    t = _run(c, 100, 1.0, 0, pinch=True)
    assert c.state.grab is False and c.state.x == 0  # arm frozen, no grabbing while calibrating
    t = _run(c, 130, 1.0, t)
    t = _run(c, 80, 1.0, t)
    assert not c.calibrating
    assert c.state.ref_size == pytest.approx(hand_size(hand(scale=100)))  # the comfortable distance is the middle
    # closest position now reaches 90% of the depth range (deeper is negative)
    t = _run(c, 130, 0.2, t)
    assert c.state.x == pytest.approx(-45, abs=1.5)


def test_saved_calibration_is_applied_without_running_the_routine():
    c = OneHandControl(ArmState(), DepthCalibration(200, 60))
    assert not c.calibrating
    assert (c.state.workspace.depth_gain_near, c.state.workspace.depth_gain_far) == (200, 60)


def test_session_saves_calibration_and_reports_mode(tmp_path):
    c = OneHandControl(ArmState(filtering=False))
    session = Session(c, (W, H), tmp_path / "calibration.json")
    session.command("calibrate")
    t = 0.0
    for scale in (100, 130, 80):
        for i in range(30):
            r = _result(hand(scale=scale))
            c.apply(r, W, H, t)
            session.after_frame(r, t)
            t += 1 / 30
            if c.calibrating:
                assert c.state.mode == "calib"
    assert c.state.mode == "run"
    assert DepthCalibration.load(tmp_path / "calibration.json") is not None


def test_recording_round_trip(tmp_path, monkeypatch):
    import recording
    monkeypatch.setattr(recording, "DEFAULT_DIR", tmp_path)
    monkeypatch.setattr("hand_detector.new_path", lambda: recording.new_path(tmp_path))
    c = OneHandControl(ArmState(filtering=False), DepthCalibration(110, 110))
    session = Session(c, (W, H), tmp_path / "cal.json")
    session.command("record")
    for i in range(10):
        r = _result(hand(pinch=i >= 5, at=(200 + 10 * i, 300)))
        c.apply(r, W, H, i / 30)
        session.after_frame(r, i / 30)
        assert c.state.mode == "rec"
    path = session.recorder.path
    session.command("record")  # stop
    meta, frames = read_recording(path)
    assert meta["frame"] == [W, H] and meta["version"] == 1
    assert len(frames) == 10
    assert frames[0]["t"] == 0 and frames[-1]["t"] == pytest.approx(9 / 30, abs=1e-3)
    assert [f["grab"] for f in frames] == [False] * 5 + [True] * 5
    assert len(frames[0]["hands"]) == 1 and len(frames[0]["hands"][0]) == 21


def test_calibration_sets_centre_and_rest_tilt_then_turning_rotates_the_wrist():
    import math
    from calibration import DepthCalibration
    c = OneHandControl(ArmState(filtering=False), DepthCalibration(110, 110, center_x=0.3, center_y=0.5,
                                                                    roll_ref=math.radians(15)))
    # a hand resting where it was calibrated: arm in the middle, wrist straight
    rest = hand(scale=100, at=(0.3 * W, 0.5 * H), angle=math.radians(15))
    from gestures import palm_center
    cx, cy = palm_center(rest)
    rest = hand(scale=100, at=(0.3 * W - (cx - 0.3 * W), 0.5 * H - (cy - 0.5 * H)), angle=math.radians(15))
    c.apply(_result(rest), W, H, t=0.0)
    assert c.state.y == pytest.approx(0, abs=1e-6)
    assert c.state.roll == 0
    turned = hand(scale=100, at=(0.3 * W, 0.5 * H), angle=math.radians(65))
    c.apply(_result(turned), W, H, t=0.1)
    assert math.degrees(c.state.roll) == pytest.approx(40, abs=0.5)
