import pytest

pytest.importorskip("mediapipe")
pytest.importorskip("cv2")

from calibration import DepthCalibration  # noqa: E402
from gestures import ArmState  # noqa: E402
from hand_detector import OneHandControl, Session  # noqa: E402
from recording import read_recording  # noqa: E402
from replay import Player, render  # noqa: E402
from synthetic import hand  # noqa: E402
from test_hand_detector import H, W, _result  # noqa: E402


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    """Record a short session: hand moves right, pinches, moves back."""
    import recording
    monkeypatch.setattr("hand_detector.new_path", lambda: recording.new_path(tmp_path))
    c = OneHandControl(ArmState(), DepthCalibration(150, 90))
    session = Session(c, (W, H), tmp_path / "cal.json")
    session.command("record")
    targets = []
    for i in range(60):
        x = 200 + 4 * i if i < 30 else 320 - 4 * (i - 30)
        r = _result(hand(pinch=20 <= i < 45, scale=100 + i, at=(x, 300)))
        t = 1000 + i / 30  # absolute clock, like time.monotonic()
        c.apply(r, W, H, t)
        session.after_frame(r, t)
        targets.append((c.state.x, c.state.y, c.state.z, c.state.grab))
    path = session.recorder.path
    session.command("record")
    return path, targets


def test_replay_as_recorded_sends_the_same_targets(recorded):
    path, targets = recorded
    p = Player(*read_recording(path))
    sent = []
    while not p.finished:
        p.step()
        s = p.control.state
        sent.append((s.x, s.y, s.z, s.grab))
        assert s.mode == "replay"
    assert sent == [(pytest.approx(x, abs=1e-3), pytest.approx(y, abs=1e-3), pytest.approx(z, abs=1e-3), g)
                    for x, y, z, g in targets]


def test_recompute_reproduces_the_session_with_the_same_settings(recorded):
    path, targets = recorded
    p = Player(*read_recording(path), recompute=True)
    assert p.control.state.workspace.depth_gain_near == 150  # recorded calibration
    i = 0
    while not p.finished:
        p.step()
        s = p.control.state
        x, y, z, g = targets[i]
        assert (s.x, s.y, s.z) == (pytest.approx(x, abs=0.05), pytest.approx(y, abs=0.05), pytest.approx(z, abs=0.05))
        assert s.grab == g
        i += 1


def test_recompute_uses_current_settings(recorded):
    path, targets = recorded
    p = Player(*read_recording(path), recompute=True)
    p.control.state.pinch_close = 0.0  # pretend the code changed: pinching never grabs
    while not p.finished:
        p.step()
        assert p.control.state.grab is False


def test_render_draws_the_hand(recorded):
    path, _ = recorded
    p = Player(*read_recording(path))
    frame, result = p.step()
    image = render(frame, result, p.control, p.size, "REPLAY")
    assert image.shape == (H, W, 3)
    assert image.max() > 100  # landmarks and label drawn on the dark background
