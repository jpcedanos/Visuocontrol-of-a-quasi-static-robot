import math
from types import SimpleNamespace

# --- synthetic hands (normalized image coords, y grows downward) -------------
# Built around a wrist at (0.5, 0.8), palm about 0.2 tall, fingers pointing up.


def _p(x, y):
    return SimpleNamespace(x=x, y=y)


def _hand(fingers_curled: bool, thumb: str):
    """21 landmarks in MediaPipe order. thumb: 'side', 'up' or 'down'."""
    lms = [None] * 21
    lms[0] = _p(0.50, 0.80)  # wrist
    # thumb: CMC, MCP, IP, TIP
    thumb_pts = {
        "side": [(0.44, 0.76), (0.40, 0.72), (0.36, 0.70), (0.33, 0.69)],
        "up": [(0.44, 0.74), (0.42, 0.66), (0.42, 0.58), (0.42, 0.51)],
        "down": [(0.44, 0.76), (0.42, 0.72), (0.42, 0.80), (0.42, 0.87)],
    }[thumb]
    for i, (x, y) in enumerate(thumb_pts, start=1):
        lms[i] = _p(x, y)
    # four fingers: MCP, PIP, DIP, TIP at x offsets
    for base, x in zip((5, 9, 13, 17), (0.46, 0.50, 0.54, 0.58)):
        lms[base] = _p(x, 0.60)  # knuckle
        lms[base + 1] = _p(x, 0.52)  # PIP
        if fingers_curled:
            lms[base + 2] = _p(x, 0.56)
            lms[base + 3] = _p(x, 0.63)  # tip folded back toward the palm
        else:
            lms[base + 2] = _p(x, 0.46)
            lms[base + 3] = _p(x, 0.41)
    return lms


def _rotate(lms, degrees):
    """Rotate a hand around its wrist, to check the rules are rotation-proof."""
    cx, cy = lms[0].x, lms[0].y
    a = math.radians(degrees)
    out = []
    for p in lms:
        dx, dy = p.x - cx, p.y - cy
        out.append(_p(cx + dx * math.cos(a) - dy * math.sin(a), cy + dx * math.sin(a) + dy * math.cos(a)))
    return out


OPEN = _hand(fingers_curled=False, thumb="side")
FIST = _hand(fingers_curled=True, thumb="side")
THUMB_UP = _hand(fingers_curled=True, thumb="up")
THUMB_DOWN = _hand(fingers_curled=True, thumb="down")
