import math

import pytest

from calibration import Calibrator, DepthCalibration


def feed(cal, size, seconds, t0, fps=30, wobble=0.0):
    """Feed a steady (or slightly wobbling) hand size for some seconds."""
    t = t0
    for i in range(int(seconds * fps)):
        t = t0 + i / fps
        cal.update(None if size is None else size * (1 + wobble * math.sin(i)), t)
    return t + 1 / fps


def test_full_routine_gives_gains_that_reach_90_percent_of_depth():
    cal = Calibrator(depth_limit=50)
    t = feed(cal, 100, 1.0, 0)
    assert cal.current == "near"
    t = feed(cal, 130, 1.0, t)
    assert cal.current == "far"
    feed(cal, 80, 1.0, t)
    assert cal.done
    assert cal.result.gain_near * math.log(130 / 100) == pytest.approx(45)
    assert cal.result.gain_far * math.log(100 / 80) == pytest.approx(45)
    assert cal.center_size == pytest.approx(100)


def test_waits_while_the_hand_is_moving_or_missing():
    cal = Calibrator()
    feed(cal, 100, 2.0, 0, wobble=0.2)  # shaking too much to count as holding
    assert cal.current == "center"
    feed(cal, None, 2.0, 2)  # no hand
    assert cal.current == "center"


def test_near_step_needs_the_hand_clearly_closer():
    cal = Calibrator()
    t = feed(cal, 100, 1.0, 0)
    t = feed(cal, 105, 2.0, t)  # only 5% bigger: not enough
    assert cal.current == "near"
    feed(cal, 120, 1.0, t)
    assert cal.current == "far"


def test_completes_with_accumulated_float_timestamps():
    # real frame times drift (0.1 + 0.2 != 0.3); the hold must still complete
    cal = Calibrator(hold_time=0.8)
    t = 0.0
    for size in [100] * 30 + [130] * 30 + [80] * 30:
        cal.update(size, t)
        t += 1 / 30
    assert cal.done


def test_progress_rises_while_holding():
    cal = Calibrator(hold_time=1.0)
    feed(cal, 100, 0.5, 0)
    assert 0.4 < cal.progress < 0.6


def test_gains_are_kept_in_a_sane_range():
    cal = Calibrator(gain_range=(40, 400))
    t = feed(cal, 100, 1.0, 0)
    t = feed(cal, 400, 1.0, t)  # absurdly close
    feed(cal, 89, 1.0, t)        # barely farther
    assert cal.result.gain_near == 40
    assert cal.result.gain_far == pytest.approx(min(400, 45 / math.log(100 / 89)))


def test_save_and_load(tmp_path):
    path = tmp_path / "calibration.json"
    DepthCalibration(123.4, 56.7).save(path)
    assert DepthCalibration.load(path) == DepthCalibration(123.4, 56.7)
    assert DepthCalibration.load(tmp_path / "missing.json") is None
    (tmp_path / "bad.json").write_text("{not json")
    assert DepthCalibration.load(tmp_path / "bad.json") is None
