import pytest

from gestures import ArmState, is_fist, thumb_gesture
from synthetic import FIST, OPEN, THUMB_DOWN, THUMB_UP, _hand, _rotate


# --- gestures ---------------------------------------------------------------


def test_open_hand_is_not_a_fist():
    assert not is_fist(OPEN)


def test_fist_is_detected():
    assert is_fist(FIST)


@pytest.mark.parametrize("angle", [30, 90, 180, -60])
def test_fist_detection_works_when_hand_is_rotated(angle):
    assert is_fist(_rotate(FIST, angle))
    assert not is_fist(_rotate(OPEN, angle))


def test_thumbs_up_and_down():
    assert thumb_gesture(THUMB_UP) == "up"
    assert thumb_gesture(THUMB_DOWN) == "down"


def test_open_hand_never_moves_the_arm():
    # The original rule (thumb above the wrist) fired for any upright open hand.
    open_thumb_up = _hand(fingers_curled=False, thumb="up")
    assert thumb_gesture(open_thumb_up) == "neutral"
    assert thumb_gesture(OPEN) == "neutral"


def test_fist_with_thumb_to_the_side_is_neutral():
    assert thumb_gesture(FIST) == "neutral"


# --- arm state --------------------------------------------------------------


def _original_mapping(wx, wy, w, h):
    """Mapping from the original hand_detector.py, for comparison."""
    x_px, y_px = wx * w - w / 2, h / 2 - wy * h
    return (x_px / 2.5) * -1, (y_px / 2.5) * -1


@pytest.mark.parametrize("wx,wy", [(0.5, 0.5), (0.1, 0.9), (0.8, 0.2)])
def test_position_mapping_matches_original(wx, wy):
    state = ArmState(smoothing=0.0)
    state.update_position(wx, wy, 640, 480)
    assert (state.x, state.y) == pytest.approx(_original_mapping(wx, wy, 640, 480))


def test_smoothing_moves_part_of_the_way():
    state = ArmState(smoothing=0.5)
    state.update_position(0.5, 0.5, 640, 480)  # first sample: no smoothing
    assert state.x == pytest.approx(0.0)
    state.update_position(0.0, 0.5, 640, 480)  # target x = 128
    assert state.x == pytest.approx(64.0)


def test_height_speed_is_per_second_not_per_frame():
    fast, slow = ArmState(z_speed=30), ArmState(z_speed=30)
    for _ in range(60):  # 1 s at 60 fps
        fast.update_height("down", 1 / 60)
    for _ in range(15):  # 1 s at 15 fps
        slow.update_height("down", 1 / 15)
    assert fast.z == pytest.approx(30.0)
    assert slow.z == pytest.approx(30.0)


def test_thumb_up_raises_arm_which_is_negative_z():
    state = ArmState(z_speed=10)
    state.update_height("up", 1.0)
    assert state.z == pytest.approx(-10.0)


def test_height_is_clamped():
    state = ArmState(z_speed=1000, z_limit=100)
    state.update_height("down", 1.0)
    assert state.z == 100
    state.update_height("up", 5.0)
    assert state.z == -100


def test_message_is_one_line_in_the_sketch_format():
    state = ArmState(x=1.234, y=-5.0, z=7.5, grab=True)
    assert state.message() == "1.23,-5.00,7.50,True\n"
