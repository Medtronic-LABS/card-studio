"""Straighten a photographed card before layout detection. OpenCV only, local.

detect.py finds the ruled table with horizontal/vertical morphology, so it needs level lines (it fails beyond ~2 degrees of tilt).
- estimate_tilt: the dominant ruling-line angle (mod 90 degrees), from Hough segments weighted by length.
- straighten: rotate by that angle. A quarter turn (90/180/270) cannot be told from the lines alone, so that is a user choice
  (rotate_quarter); estimate_tilt only ever returns -45..45 degrees.
"""
import cv2, numpy as np

MIN_CONFIDENCE = 0.30      # share of line length that agrees with the peak; below this the tilt is not trusted
MIN_LINES = 0.02           # a level table has at least this share of its ink on long straight runs; noise and photos of things have ~0
MAX_LOSS = 0.95            # a rotation is refused if it makes the ruled lines weaker than this fraction of before
PEAK_WINDOW = 1.5          # degrees


def _gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr


def estimate_tilt(bgr, max_side=1400):
    """Returns (angle_deg, confidence). angle is how far the ruling lines are rotated clockwise on screen (-45..45]."""
    g = _gray(bgr)
    s = min(1.0, max_side / max(g.shape))
    if s < 1:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    H, W = g.shape
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    seg = cv2.HoughLinesP(bw, 1, np.pi / 720, threshold=int(0.12 * min(H, W)), minLineLength=int(0.15 * min(H, W)), maxLineGap=12)
    if seg is None or len(seg) < 4:
        return 0.0, 0.0
    seg = seg.reshape(-1, 4).astype(np.float64)
    dx, dy = seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]
    ang = np.degrees(np.arctan2(dy, dx))
    ang = (ang + 45.0) % 90.0 - 45.0            # fold into [-45, 45): a vertical rule counts as a horizontal one
    w = np.hypot(dx, dy)
    bins = np.arange(-45.0, 45.25, 0.25)
    hist, _ = np.histogram(ang, bins=bins, weights=w)
    k = np.ones(7) / 7.0
    peak = bins[int(np.argmax(np.convolve(hist, k, mode="same")))] + 0.125
    near = np.abs(ang - peak) <= PEAK_WINDOW
    if not near.any():
        return 0.0, 0.0
    refined = float(np.average(ang[near], weights=w[near]))
    return refined, float(w[near].sum() / w.sum())


def line_strength(bgr, max_side=1400):
    """Share of ink that sits on long horizontal/vertical runs: high for a level ruled table, low for tilted or non-tables."""
    g = _gray(bgr)
    s = min(1.0, max_side / max(g.shape))
    if s < 1:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    H, W = g.shape
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, W // 25), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, H // 25)))
    lines = cv2.bitwise_or(cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk), cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk))
    return float(np.count_nonzero(lines)) / (np.count_nonzero(bw) + 1.0)


def _fill(bgr):
    border = np.concatenate([bgr[0], bgr[-1], bgr[:, 0], bgr[:, -1]])
    return tuple(int(v) for v in np.median(border, axis=0))


def rotate_by(bgr, angle):
    """Rotate counter-clockwise on screen by `angle` degrees, growing the canvas so nothing is cut off."""
    if abs(angle) < 1e-6:
        return bgr
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(bgr, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=_fill(bgr))


def rotate_quarter(bgr, deg):
    deg %= 360
    if deg not in (0, 90, 180, 270):
        raise ValueError("rotation must be 0, 90, 180 or 270")
    return bgr if deg == 0 else cv2.rotate(bgr, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[deg])


def straighten(bgr, quarter=0, enabled=True):
    """quarter turn (user's choice), then level the ruling lines. Returns (image, info)."""
    im = rotate_quarter(bgr, quarter)
    info = {"quarter": quarter, "tilt_deg": 0.0, "confidence": 0.0, "straightened": False}
    if not enabled:
        return im, info
    a, conf = estimate_tilt(im)
    info.update(tilt_deg=round(a, 2), confidence=round(conf, 3))
    if conf >= MIN_CONFIDENCE and abs(a) >= 0.3:
        cand = rotate_by(im, a)
        before, after = line_strength(im), line_strength(cand)
        info["line_gain"] = round(after / max(before, 1e-6), 2)
        if after > MIN_LINES and after >= MAX_LOSS * before:
            im = cand
            info["straightened"] = True
    return im, info
