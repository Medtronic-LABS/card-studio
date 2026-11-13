"""Read a cell with a CTC text recogniser that was trained somewhere else. numpy and onnxruntime only; no frappe, no torch.

An outside model is an ONNX file plus three facts the file does not carry: which characters its output columns mean, which column is the blank, and how a
picture must be prepared (height, channels, scaling). Those facts are the model's `spec`. Given a spec, `ExternalReader.read(gray)` prepares a cell, runs the
model, collapses the CTC output and returns (text, confidence), the same shape as our own readers, so everything that takes a suggestion (labeling, the
clean-up lab, benchmarks) can take an outside model's read.

    spec = make_spec("paddle-rec", charset_list)           # PP-OCR style recognisers: 3 channels, height 48, blank first, a trailing space
    ExternalReader(onnx_path_or_bytes, spec).read(gray)    -> ("22/3/26", 0.93)

Nothing here trains. An outside model can only be used as it is (a suggester, a benchmark); starting a training run from it needs weights in our own network's
shape (studio/pretrained.py). This file is kept identical in Card Studio (studio/extreader.py) and the Frappe app (card_ocr/engine/extreader.py).
"""
import json, math, re

import numpy as np

BENGALI_DIGITS = "০১২৩৪৫৬৭৮৯"
PROFILES = {
    "paddle-rec": {"label": "PaddleOCR recogniser (PP-OCR family)", "height": 48, "channels": 3, "mean": 0.5, "std": 0.5, "max_width": 640, "blank": 0, "trailing_space": True},
    "gray-ctc": {"label": "Single-channel CTC recogniser, blank first", "height": 32, "channels": 1, "mean": 0.5, "std": 0.5, "max_width": 640, "blank": 0, "trailing_space": False},
}
NAME = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def parse_charset(text):
    """A character list from a file's text: a JSON list, a PaddleOCR inference.yml (PostProcess.character_dict), or one character per line."""
    t = text.lstrip("﻿")
    if t.lstrip().startswith("["):
        v = json.loads(t)
        if isinstance(v, list) and all(isinstance(c, str) for c in v): return v
        raise ValueError("a JSON charset must be a list of strings")
    if "character_dict:" in t:
        import yaml
        v = (yaml.safe_load(t).get("PostProcess") or {}).get("character_dict")
        if not isinstance(v, list): raise ValueError("this file has no PostProcess.character_dict")
        return [str(c) for c in v]
    return [ln for ln in t.split("\n") if ln != ""] if "\n" in t else list(t)


def make_spec(profile, charset, **override):
    """A model's spec. profile: a key of PROFILES. charset: the characters its output columns mean, in order, WITHOUT the blank and without a trailing space
    (the profile says where those go). override: height, channels, mean, std, max_width, blank, trailing_space, digit_map (map Bengali digits to 0-9 in the read)."""
    if profile not in PROFILES: raise ValueError(f"unknown profile {profile!r}: {', '.join(PROFILES)}")
    p = {**PROFILES[profile], **override}; chars = list(charset)
    if not chars: raise ValueError("the charset is empty")
    classes = chars + ([" "] if p["trailing_space"] else [])
    classes = classes[:p["blank"]] + ["<blank>"] + classes[p["blank"]:]
    return {"profile": profile, "height": int(p["height"]), "channels": int(p["channels"]), "mean": float(p["mean"]), "std": float(p["std"]), "max_width": int(p["max_width"]),
            "blank": int(p["blank"]), "classes": classes, "digit_map": bool(p.get("digit_map", False))}


def prepare(gray, spec):
    """A gray cell (uint8, any size) -> float32 [1, channels, height, width]: scaled to the model's height keeping its shape, then normalised."""
    import cv2
    g = np.asarray(gray)
    if g.ndim == 3: g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    h = spec["height"]; w = int(min(spec["max_width"], max(h // 2, math.ceil(h * g.shape[1] / max(1, g.shape[0])))))
    x = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA if g.shape[0] > h else cv2.INTER_LINEAR).astype(np.float32) / 255.0
    x = (x - spec["mean"]) / spec["std"]
    return np.repeat(x[None, None], spec["channels"], axis=1) if spec["channels"] > 1 else x[None, None]


def decode(out, spec):
    """Model output (any layout holding [time, classes]) -> (text, confidence). Greedy CTC; confidence = the lowest character probability read, 0 when nothing."""
    a = np.asarray(out, np.float32); k = len(spec["classes"])
    a = np.squeeze(a)
    if a.ndim != 2: raise ValueError(f"expected a [time, classes] output, got shape {np.asarray(out).shape}")
    if a.shape[1] != k:
        if a.shape[0] == k: a = a.T
        else: raise ValueError(f"the model has {a.shape[1]} output columns but the spec has {k} classes: the charset or blank position is wrong")
    if not np.allclose(a.sum(1), 1.0, atol=1e-2):          # raw scores, not probabilities
        z = a - a.max(1, keepdims=True); a = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    best = a.argmax(1); text, conf, prev = [], [], -1
    for t, b in enumerate(best):
        if b != spec["blank"] and b != prev: text.append(spec["classes"][b]); conf.append(float(a[t, b]))
        prev = b
    s = "".join(text)
    if spec.get("digit_map"): s = s.translate(str.maketrans(BENGALI_DIGITS, "0123456789"))
    return s, (min(conf) if conf else 0.0)


class ExternalReader:
    def __init__(self, onnx, spec, threads=2):
        import onnxruntime as ort
        o = ort.SessionOptions(); o.intra_op_num_threads = threads
        self.session = ort.InferenceSession(onnx if isinstance(onnx, (bytes, bytearray)) else str(onnx), o, providers=["CPUExecutionProvider"])
        self.spec = spec; self.input = self.session.get_inputs()[0].name

    def read(self, gray):
        out = self.session.run(None, {self.input: prepare(gray, self.spec)})[0]
        return decode(out, self.spec)

    def check(self):
        """Run a blank cell through the model: raises ValueError saying what is wrong with the spec (wrong class count, wrong channels). -> None"""
        try: self.read(np.full((40, 120), 255, np.uint8))
        except ValueError: raise
        except Exception as e: raise ValueError(f"the model did not run with this spec: {str(e)[:200]}")


# ---------------------------------------------------------------- scoring against labels
def _wilson(k, n, z=1.96):
    if n == 0: return 0.0, 1.0
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _edit(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1): cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm(text):
    """Compare a read with a label whatever the punctuation or spaces: '22 . 3 . 26' == '22/3/26'."""
    return re.sub(r"\s+", "", (text or "")).replace(".", "/") if re.search(r"\d\s*[./]\s*\d.*[./]", text or "") else re.sub(r"\s+", "", text or "")


def score(rows):
    """rows: [(field, label, read)] -> {overall: {...}, by_field: {field: {...}}}. exact = the read equals the label; cer = edit distance / label length."""
    def agg(rs):
        n = len(rs); k = sum(norm(p) == norm(l) for _, l, p in rs); lo, hi = _wilson(k, n)
        chars = sum(len(norm(l)) for _, l, _ in rs)
        return {"n": n, "correct": k, "exact": round(k / max(1, n), 3), "lo": round(lo, 3), "hi": round(hi, 3), "cer": round(sum(_edit(norm(p), norm(l)) for _, l, p in rs) / max(1, chars), 3)}
    fields = sorted({f for f, _, _ in rows})
    return {"overall": agg(rows), "by_field": {f: agg([r for r in rows if r[0] == f]) for f in fields}}
