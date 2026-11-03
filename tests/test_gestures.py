import math

import pytest

from gestures import ArmState, Workspace, choose_hand, hand_size, palm_center, pinch_ratio
from synthetic import hand

# --- hand measurements --------------------------------------------------------


def test_pinch_ratio_is_small_only_when_pinching():
    assert pinch_ratio(hand(pinch=True)) < 0.25
    assert pinch_ratio(hand(pinch=False)) > 0.40


def test_measurements_do_not_depend_on_distance_to_camera():
    near, far = hand(scale=160), hand(scale=60)
    assert pinch_ratio(near) == pytest.approx(pinch_ratio(far))
    assert hand_size(near) / hand_size(far) == pytest.approx(160 / 60)


def test_palm_center_does_not_move_when_pinching():
    assert palm_center(hand(pinch=True)) == pytest.approx(palm_center(hand(pinch=False)))


# --- arm state ----------------------------------------------------------------


def fresh(**kw):
    return ArmState(smoothing=0.0, depth_smoothing=0.0, **kw)


def test_palm_position_maps_to_side_and_height():
    s = fresh()
    s.update(0.15, 0.15, 100, 1.0)  # top-left corner of the usable frame
    assert (s.y, s.z) == pytest.approx((-85, -50))
    s.update(0.85, 0.85, 100, 1.0)  # bottom-right corner
    assert (s.y, s.z) == pytest.approx((85, 60))


def test_moving_hand_right_moves_arm_right_and_up_moves_it_up():
    s = fresh()
    s.update(0.5, 0.5, 100, 1.0)
    y0, z0 = s.y, s.z
    s.update(0.6, 0.4, 100, 1.0)
    assert s.y > y0  # coordY + is to the right in the sketch
    assert s.z < z0  # coordZ - is up


def test_edges_of_the_frame_are_clamped():
    s = fresh()
    s.update(0.0, 1.0, 100, 1.0)
    assert (s.y, s.z) == pytest.approx((-85, 60))


def test_first_hand_size_is_the_middle_depth():
    s = fresh()
    s.update(0.5, 0.5, 120, 1.0)
    assert s.x == 0
    assert s.ref_size == 120


def test_hand_closer_to_camera_reaches_deeper():
    s = fresh()
    s.update(0.5, 0.5, 100, 1.0)
    s.update(0.5, 0.5, 100 * math.exp(0.25), 1.0)
    # deeper into the scene is negative coordX (away from the viewer)
    assert s.x == pytest.approx(-110 * 0.25)


def test_invert_depth():
    s = fresh(workspace=Workspace(invert_depth=True))
    s.update(0.5, 0.5, 100, 1.0)
    s.update(0.5, 0.5, 100 * math.exp(0.25), 1.0)
    assert s.x == pytest.approx(110 * 0.25)


def test_depth_is_clamped():
    s = fresh()
    s.update(0.5, 0.5, 100, 1.0)
    s.update(0.5, 0.5, 1000, 1.0)
    assert s.x == -50


def test_recenter_depth():
    s = fresh()
    s.update(0.5, 0.5, 100, 1.0)
    s.update(0.5, 0.5, 150, 1.0)
    assert s.x != 0
    s.recenter_depth()
    s.update(0.5, 0.5, 150, 1.0)
    assert s.x == 0


def test_pinch_has_hysteresis():
    s = fresh()
    seq = [(0.6, False), (0.2, True), (0.33, True), (0.45, False), (0.33, False), (0.15, True)]
    for ratio, expected in seq:
        s.update(0.5, 0.5, 100, ratio)
        assert s.grab is expected, ratio


def test_losing_the_hand_releases_but_keeps_position():
    s = fresh()
    s.update(0.3, 0.3, 100, 0.1)
    before = (s.x, s.y, s.z)
    s.hand_lost()
    assert s.grab is False
    assert (s.x, s.y, s.z) == before


def test_smoothing_moves_part_of_the_way():
    s = ArmState(smoothing=0.5)
    s.update(0.5, 0.5, 100, 1.0)
    s.update(0.85, 0.5, 100, 1.0)  # target side = 85
    assert s.y == pytest.approx(42.5)


def test_default_workspace_stays_within_arm_reach():
    ws = Workspace()
    corner = math.dist((0, 0, 0), (max(map(abs, ws.depth)), max(map(abs, ws.side)), max(map(abs, ws.height))))
    assert corner < 150  # beyond 120 the arm just stretches toward the target
    assert max(map(abs, ws.side)) >= 80  # far edge of the tables
    cube_on_floor_center = 62.5 - 13.8 / 2  # floor depth minus half the cube, in coordZ
    assert ws.height[1] >= cube_on_floor_center  # can reach a cube lying on the floor


def test_message_is_one_line_in_the_sketch_format():
    s = ArmState(x=1.234, y=-5.0, z=7.5, grab=True)
    assert s.message() == "1.23,-5.00,7.50,True\n"


# --- choosing a hand ----------------------------------------------------------


def test_choose_biggest_hand_first_then_follow_it():
    small = hand(scale=60, at=(100, 300))
    big = hand(scale=120, at=(500, 300))
    assert choose_hand([small, big], None) == 1
    # later, keep following the hand near the last palm position even if it is smaller
    assert choose_hand([small, big], palm_center(small)) == 0


def test_no_hands():
    assert choose_hand([], None) is None
