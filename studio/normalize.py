"""Photo normalization options for the ANC reader, in Python. The same maths as leapwell_ocr_demo/lib/bangla_digit/normalization.dart
(tests/test_cleanup.py checks it against the reference values the Dart code is checked against). numpy only."""
import math
import numpy as np

METHODS = ("none", "flatten", "contrast_match", "cell_stretch", "flatten_cell_stretch")
LABELS = {"none": "No clean-up", "flatten": "Even out shadows", "contrast_match": "Match contrast to training scans",
          "cell_stretch": "Stretch each cell", "flatten_cell_stretch": "Even out shadows, then stretch each cell"}
MIN_COVER = 0.05


def hist256(a, mask=None):
    a = np.asarray(a).reshape(-1)
    if mask is not None:
        a = a[np.asarray(mask).reshape(-1) != 0]
    return np.bincount(a, minlength=256).astype(int).tolist()


def hist_stats(h):
    n = sum(h)
    if n == 0:
        return 0.0, 0.0
    s = 0.0
    for v in range(256):
        s += v * h[v]
    mean = s / n
    q = 0.0
    for v in range(256):
        d = v - mean
        q += d * d * h[v]
    return mean, math.sqrt(q / n)


def _clamp(v):
    return 0 if v < 0 else 255 if v > 255 else math.floor(v + 0.5)


def contrast_lut(h, ref_mean, ref_std):
    mean, std = hist_stats(h)
    if std < 1.0:
        return np.arange(256, dtype=np.uint8)
    return np.array([_clamp((v - mean) / (std + 1e-6) * ref_std + ref_mean) for v in range(256)], np.uint8)


def _order_stat(h, idx):
    seen = 0
    for v in range(256):
        seen += h[v]
        if seen > idx:
            return v
    return 255


def percentile(h, q):
    n = sum(h)
    if n == 0:
        return 0.0
    pos = q / 100.0 * (n - 1)
    k = math.floor(pos)
    frac = pos - k
    a = _order_stat(h, k)
    b = _order_stat(h, k + 1) if k + 1 < n else a
    return a + (b - a) * frac


def stretch_lut(h, lo=2.0, hi=98.0, min_range=20.0):
    a, b = percentile(h, lo), percentile(h, hi)
    if b - a < min_range:
        return np.arange(256, dtype=np.uint8)
    return np.array([_clamp((v - a) * 255.0 / (b - a)) for v in range(256)], np.uint8)


def flatten(gray, kernel):
    """The app's _flatten: divide by the blurred, dilated background."""
    import cv2
    bg = cv2.GaussianBlur(cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))), (kernel, kernel), 0)
    return cv2.divide(gray, bg, scale=255 / 0.8)


def normalize_page(method, warped, covered, kernel, reference):
    """Page-level step. covered: nonzero where the photo really covered the card. reference: (mean, std) or None.
    Returns (page, used_method): contrast_match without a reference falls back to flatten, as the app does."""
    if method in ("flatten", "flatten_cell_stretch"):
        return flatten(warped, kernel), method
    if method in ("none", "cell_stretch"):
        return warped, method
    if method == "contrast_match":
        if reference is None:
            return flatten(warped, kernel), "flatten"
        h = hist256(warped, covered)
        if sum(h) < MIN_COVER * warped.size:
            return warped, method
        return contrast_lut(h, *reference)[warped], method
    raise ValueError(f"unknown method {method}")


def stretch_cell(crop):
    return stretch_lut(hist256(crop))[crop]
