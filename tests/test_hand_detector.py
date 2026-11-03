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
    c = OneHandControl(ArmState(smoothing=0.0))
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
