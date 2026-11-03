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
