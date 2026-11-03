import math

import pytest

from gestures import ArmState, OneEuroFilter, Workspace, choose_hand, hand_size, palm_center, pinch_ratio
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
    return ArmState(filtering=False, **kw)


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


def _noisy_still(n=300, fps=30, noise=2.0, seed=1):
    import random
    rnd = random.Random(seed)
    return [(i / fps, 50 + rnd.gauss(0, noise)) for i in range(n)]


def _std(values):
    m = sum(values) / len(values)
    return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5


def test_one_euro_removes_most_jitter_when_still():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.02)
    samples = _noisy_still()
    out = [f(v, t) for t, v in samples][60:]  # skip warm-up
    assert _std(out) < 0.4 * _std([v for _, v in samples][60:])


def test_one_euro_keeps_up_with_fast_motion():
    # hand sweeping across the workspace at 200 units/s, 30 fps
    f = OneEuroFilter(min_cutoff=1.0, beta=0.02)
    out = [(f(200 * t, t), 200 * t) for t in (i / 30 for i in range(30))]
    lag = out[-1][1] - out[-1][0]
    assert 0 <= lag < 8  # a plain low-pass at 1 Hz would trail by ~30 units here


def test_one_euro_beats_a_plain_average_at_both():
    """Steadier than the old 50% moving average when still, with no more lag when moving."""
    def ema(samples, k=0.5):
        y, out = None, []
        for _, v in samples:
            y = v if y is None else k * y + (1 - k) * v
            out.append(y)
        return out
    still = _noisy_still()
    f = OneEuroFilter(min_cutoff=1.0, beta=0.02)
    assert _std([f(v, t) for t, v in still][60:]) < _std(ema(still)[60:])
    ramp = [(i / 30, 200 * i / 30) for i in range(30)]
    f = OneEuroFilter(min_cutoff=1.0, beta=0.02)
    euro_lag = ramp[-1][1] - [f(v, t) for t, v in ramp][-1]
    ema_lag = ramp[-1][1] - ema(ramp)[-1]
    assert euro_lag <= ema_lag * 1.1


def test_arm_state_filters_with_frame_times():
    s = ArmState()
    for i, (t, v) in enumerate(_noisy_still(noise=0.01)):
        s.update(0.5 + (v - 50) / 100, 0.5, 100, 1.0, t=t)
    assert abs(s.y) < 1.0


def test_depth_uses_separate_gains_for_near_and_far():
    s = fresh(workspace=Workspace(depth_gain_near=200, depth_gain_far=50))
    s.update(0.5, 0.5, 100, 1.0)
    assert s.depth_for(100 * math.exp(0.1)) == pytest.approx(-20)  # closer: deeper (negative)
    assert s.depth_for(100 * math.exp(-0.1)) == pytest.approx(5)    # farther: nearer


def test_default_workspace_stays_within_arm_reach():
    ws = Workspace()
    corner = math.dist((0, 0, 0), (max(map(abs, ws.depth)), max(map(abs, ws.side)), max(map(abs, ws.height))))
    assert corner < 150  # beyond 120 the arm just stretches toward the target
    assert max(map(abs, ws.side)) >= 80  # far edge of the tables
    cube_on_floor_center = 62.5 - 13.8 / 2  # floor depth minus half the cube, in coordZ
    assert ws.height[1] >= cube_on_floor_center  # can reach a cube lying on the floor


def test_message_is_one_line_in_the_sketch_format():
    s = ArmState(x=1.234, y=-5.0, z=7.5, grab=True)
    assert s.message() == "1.23,-5.00,7.50,True,run\n"


# --- choosing a hand ----------------------------------------------------------


def test_choose_biggest_hand_first_then_follow_it():
    small = hand(scale=60, at=(100, 300))
    big = hand(scale=120, at=(500, 300))
    assert choose_hand([small, big], None) == 1
    # later, keep following the hand near the last palm position even if it is smaller
    assert choose_hand([small, big], palm_center(small)) == 0


def test_no_hands():
    assert choose_hand([], None) is None
