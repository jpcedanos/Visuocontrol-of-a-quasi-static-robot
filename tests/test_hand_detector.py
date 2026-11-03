from types import SimpleNamespace

import pytest

pytest.importorskip("mediapipe")
pytest.importorskip("cv2")

from gestures import ArmState  # noqa: E402
from hand_detector import apply_result, user_side  # noqa: E402
from synthetic import FIST, OPEN, THUMB_UP  # noqa: E402


def _result(*hands):
    """Fake HandLandmarker result: hands are (mediapipe_label, landmarks)."""
    return SimpleNamespace(
        hand_landmarks=[lms for _, lms in hands],
        handedness=[[SimpleNamespace(category_name=label)] for label, _ in hands],
    )


def test_labels_are_swapped_for_the_mirrored_frame():
    assert user_side("Left") == "Right"
    assert user_side("Right") == "Left"


def test_users_left_hand_moves_and_grabs_right_hand_sets_height():
    # On the mirrored frame, MediaPipe calls the user's left hand "Right".
    state = ArmState(smoothing=0.0, z_speed=10)
    labels = apply_result(_result(("Right", FIST), ("Left", THUMB_UP)), state, 640, 480, dt=1.0)
    assert labels == {"Left": "fist (grab)", "Right": "thumb up"}
    assert state.grab is True
    assert state.z == pytest.approx(-10.0)


def test_grab_is_released_when_left_hand_leaves_the_frame():
    state = ArmState(smoothing=0.0)
    apply_result(_result(("Right", FIST)), state, 640, 480, dt=0.03)
    assert state.grab
    x_before = state.x
    apply_result(_result(("Left", OPEN)), state, 640, 480, dt=0.03)  # only the right hand
    assert state.grab is False
    assert state.x == x_before  # position is kept
