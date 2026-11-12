"""Synthetic labeled cells for pretraining a reader, made to look like the cells it will read. Local only; numpy and OpenCV, no torch.

The first generator stitched handwritten digit images into a string on a white square. This one differs in what the reader actually sees:
  grammars     values are drawn the way the fields are written: weights with an optional decimal, dates in every punctuation seen
               (22/2/26, 22.2.26, 2/2/26), blood pressure as systolic/diastolic
  layouts      single-line cells, a date stacked over a fraction bar, and a blood pressure either side of a printed diagonal, drawn
               into the cell the way people write them (slanted, off the line, bar not quite level)
  backgrounds  the empty cells of the real cards (a blank card estimated from the scans, cut with the same grid), so ruled lines, shading
               and the printed diagonal are the real ones; handwriting is laid on top with the darkest pixel winning
  damage       blur, noise, contrast and brightness change, JPEG, a slight turn: what a photocopy or a phone does
A sample is {image, kind, layout, label, background} with the image the size of the background cell. `pieces` turns a sample into what is
trained on: the cell itself, or for a two-line layout the same two pieces `subcells.parts` cuts from a real cell.

Handwriting comes from digit glyph pools. Only the Bangla pool (NumtaDB, data/bangla_digits) is wired in; `load_pool` is where another
script's digits would be added. Digits are drawn at their glyph size, never recolored, so stroke weight varies the way real data does.
"""
import random
from pathlib import Path

import cv2, numpy as np

import subcells

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NUMTA = ROOT / "data" / "bangla_digits"
KINDS = ("number", "date", "bp")
SEPARATORS = "/.|"          # how a date's parts are divided on the page: a slash, a dot, or (very common on the ANC card) a tall vertical bar; a bar is labeled "/"
DATE_STYLE = (("|", 0.4), ("/", 0.3), (".", 0.3))
GLYPH = 32


class Pool:
    """Handwritten digit images of one script: `by_digit[d]` is a list of uint8 images with dark strokes on a white ground."""
    def __init__(self, name, by_digit):
        self.name, self.by_digit = name, by_digit


def load_pool(name="bangla"):
    if name != "bangla": raise ValueError(f"no digit pool for {name!r} wired in yet")
    X = np.load(NUMTA / "numtaX_train.npy", allow_pickle=False); Y = np.load(NUMTA / "numtaY_train.npy", allow_pickle=False).argmax(1)
    imgs = (255 - np.clip(X.astype(np.int32) * 13, 0, 255)).astype(np.uint8); by = {d: [] for d in range(10)}
    for im, y in zip(imgs, Y):
        ys, xs = np.where(im < 200)
        if len(ys) < 8: continue
        by[int(y)].append(np.ascontiguousarray(im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]))      # tight to the digit: its own width, no square of white around it
    return Pool(name, by)


# ---------------------------------------------------------------- values
def value(kind, rng):
    """A (label, parts) pair: the label a person would give the whole cell, and its text pieces for drawing."""
    d = lambda lo, hi: rng.randint(lo, hi)
    if kind == "number":
        s = str(d(1, 199)) if rng.random() < 0.7 else f"{d(1, 99)}.{d(0, 9)}"
        return s, (s,)
    if kind == "date":
        sep = rng.choices([s for s, _ in DATE_STYLE], [w for _, w in DATE_STYLE])[0]; dd, mm = d(1, 31), d(1, 12); pad = rng.random() < 0.4
        day, mon = (f"{dd:02d}", f"{mm:02d}") if pad else (str(dd), str(mm)); yr = str(d(20, 27)); lab = "/" if sep == "|" else sep
        return f"{day}{lab}{mon}{lab}{yr}", (f"{day}{sep}{mon}", yr)          # the parts are drawn with the written separator, the label has the plain one
    if kind == "bp":
        sy, di = d(90, 220), d(50, 130); return f"{sy}/{di}", (str(sy), str(di))
    raise ValueError(kind)


# ---------------------------------------------------------------- drawing
def _symbol(ch, size):
    g = np.full((size, size), 255, np.uint8)
    if ch == ".": cv2.circle(g, (size // 2, int(size * .86)), max(2, size // 9), 0, -1)
    elif ch == "/": cv2.line(g, (int(size * .15), size - 1), (int(size * .85), 0), 0, max(2, size // 12))
    elif ch == "-": cv2.line(g, (int(size * .1), size // 2), (int(size * .9), size // 2), 0, max(2, size // 10))
    elif ch == "|": cv2.line(g, (size // 2, 0), (size // 2, size - 1), 0, max(2, size // 11))
    return g


def string_image(text, pool, rng, height):
    """The text as one line of handwriting `height` pixels tall (glyphs vary +-15%), white ground, dark strokes."""
    gl = []
    for ch in text:
        if ch.isdigit():
            g = pool.by_digit[int(ch)][rng.randrange(len(pool.by_digit[int(ch)]))]; h = max(8, int(height * rng.uniform(.85, 1.15)))
            gl.append(cv2.resize(g, (max(3, int(round(g.shape[1] * h / g.shape[0]))), h), interpolation=cv2.INTER_AREA))
        elif ch == "|": gl.append(_symbol("|", max(12, int(height * 1.0 * rng.uniform(.85, 1.15)))))          # a tall bar the height of the digits, as written
        else: gl.append(_symbol(ch, max(10, int(height * .6))))
    gap = max(2, height // 8); hm = max(g.shape[0] for g in gl); row = None
    for g in gl:
        t = (hm - g.shape[0]) // 2; g = np.pad(g, ((t, hm - g.shape[0] - t), (0, 0)), constant_values=255)
        row = g if row is None else np.hstack([row, np.full((hm, rng.randint(gap, 2 * gap)), 255, np.uint8), g])
    return row


def _shear_turn(im, rng, shear=0.25, turn=4.0):
    h, w = im.shape; s = rng.uniform(-shear, shear); a = rng.uniform(-turn, turn)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0); M[0, 1] += s
    pad = int(abs(s) * h) + 4; return cv2.warpAffine(cv2.copyMakeBorder(im, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=255), M, (w + 2 * pad, h), borderValue=255)


def _stroke(im, rng):
    r = rng.random()
    if r < 0.25: return cv2.erode(im, np.ones((2, 2), np.uint8))       # darker, thicker strokes
    if r < 0.40: return cv2.dilate(im, np.ones((2, 2), np.uint8))      # thinner strokes
    return im


def lay(cell, ink, cx, cy):
    """Put `ink` (dark strokes on white) on `cell` with its centre at (cx, cy); the darker pixel wins. Clipped to the cell."""
    h, w = cell.shape; ih, iw = ink.shape; x0, y0 = int(round(cx - iw / 2)), int(round(cy - ih / 2))
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(w, x0 + iw), min(h, y0 + ih)
    if xb > xa and yb > ya: cell[ya:yb, xa:xb] = np.minimum(cell[ya:yb, xa:xb], ink[ya - y0:yb - y0, xa - x0:xb - x0])


def _fit(ink, max_w, max_h):
    s = min(1.0, max_w / ink.shape[1], max_h / ink.shape[0]); return ink if s >= 1 else cv2.resize(ink, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def _single_text(kind, parts):
    """What is written when the whole value is on one line: a date's parts joined with the separator its first part was written with."""
    if len(parts) == 1: return parts[0]
    if kind == "date": sep = next(c for c in parts[0] if not c.isdigit()); return parts[0] + sep + parts[1]
    return parts[0] + "/" + parts[1]


def draw(bg, kind, layout, parts, pool, rng):
    """Handwriting for one cell drawn onto a copy of the background. -> uint8 image the size of `bg`."""
    cell = bg.copy(); h, w = cell.shape
    if layout == "single":
        ink = _stroke(_shear_turn(string_image(_single_text(kind, parts), pool, rng, int(h * rng.uniform(.38, .62))), rng), rng)
        ink = _fit(ink, w * rng.uniform(.6, .86), h * .85); lay(cell, ink, w * rng.uniform(.42, .6), h * rng.uniform(.42, .58))
    elif layout == "stacked":
        hl = int(h * rng.uniform(.22, .34)); top = _fit(_stroke(_shear_turn(string_image(parts[0], pool, rng, hl), rng, .3, 5), rng), w * .82, h * .42)
        bot = _fit(_stroke(_shear_turn(string_image(parts[1], pool, rng, hl), rng, .3, 5), rng), w * .6, h * .4)
        yb = h * rng.uniform(.46, .54); tilt = rng.uniform(-.12, .12); x0, x1 = w * rng.uniform(.04, .14), w * rng.uniform(.86, .96)       # the bar, not quite level
        cv2.line(cell, (int(x0), int(yb - tilt * (x1 - x0) / 2)), (int(x1), int(yb + tilt * (x1 - x0) / 2)), int(rng.uniform(10, 70)), rng.choice([2, 2, 3]), cv2.LINE_AA)
        lay(cell, top, w * rng.uniform(.4, .55), yb - top.shape[0] / 2 - h * rng.uniform(.01, .05)); lay(cell, bot, w * rng.uniform(.45, .6), yb + bot.shape[0] / 2 + h * rng.uniform(.01, .05))
    else:                       # diagonal: the real printed diagonal is in the background when there is one; otherwise draw it
        if (bg < 110).sum() < 0.02 * bg.size: cv2.line(cell, (0, h - 1), (w - 1, 0), 40, 2, cv2.LINE_AA)
        hl = int(h * rng.uniform(.26, .36)); sy = _fit(_stroke(_shear_turn(string_image(parts[0], pool, rng, hl), rng, .25, 5), rng), w * .42, h * .36)
        di = _fit(_stroke(_shear_turn(string_image(parts[1], pool, rng, hl), rng, .25, 5), rng), w * .38, h * .36)
        lay(cell, sy, w * rng.uniform(.28, .42), h * rng.uniform(.22, .36)); lay(cell, di, w * rng.uniform(.62, .74), h * rng.uniform(.64, .78))
    return cell


def damage(im, rng):
    """What a photocopy or a phone does to a cell."""
    out = im.astype(np.float32)
    if rng.random() < .6: s = rng.uniform(.3, 1.2); out = cv2.GaussianBlur(out, (0, 0), s)
    out = out * rng.uniform(.82, 1.12) + rng.uniform(-14, 10)
    if rng.random() < .7: out += np.random.default_rng(rng.randrange(2 ** 31)).normal(0, rng.uniform(1, 7), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < .5: out = cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(55, 95)])[1], cv2.IMREAD_GRAYSCALE)
    if rng.random() < .5:
        h, w = out.shape; M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-1.5, 1.5), 1.0); out = cv2.warpAffine(out, M, (w, h), borderValue=255)
    return out


# ---------------------------------------------------------------- backgrounds
def _flatten(img):
    """A shaded cell brought up to the white of a cell people write in (the reader's line removal reads a grey ground as ink)."""
    med = float(np.median(img))
    return img if med >= 235 else np.clip(img.astype(np.float32) * (245.0 / max(med, 1.0)), 0, 255).astype(np.uint8)


def _clean_cells(cells, ink_rate=None):
    """Keep the cells whose blank estimate has no ghost of handwriting. A blank card estimated as the median of scans keeps a faint trace of what
    nearly every scan has in a cell, so a cell is kept when it is rarely written on (ink_rate: {(field, visit): share of scans with ink},
    when known) and is not noticeably darker than the lightest cell of its field."""
    least = {}
    for c in cells: least[c["field"]] = min(least.get(c["field"], 1e18), float((255 - c["image"].astype(np.int32)).sum() / c["image"].size))
    out = []
    for c in cells:
        dark = float((255 - c["image"].astype(np.int32)).sum() / c["image"].size)
        rare = ink_rate is None or ink_rate.get((c["field"], c["visit"]), 0.0) <= 0.05
        if rare and dark <= least[c["field"]] * 1.6 + 4: out.append({**c, "image": _flatten(c["image"])})
    return out


def backgrounds(card):
    """Clean empty cells of one card: [{field, visit, layout, kind, image}] cut from its blank card with its grid. [] when the card has no blank card yet."""
    import cards, cleanup, scans
    if card == cards.LEGACY:
        import json, cardpack as cp
        blank = cv2.imread(str(ROOT / "work" / "median.png"), cv2.IMREAD_GRAYSCALE)
        packs = sorted(cards.PACKS.glob("anc_v*.cardpack"), key=lambda p: int(p.stem.split("_v")[1]))
        if blank is None or not packs: return []
        m = cp.read(packs[-1]); grid = json.loads(m["grid.json"]); margin = json.loads(m["manifest.json"])["grid"]["cell_margin"]
        meta = {f["key"]: f for f in cards.fields(card)}
        t = type("T", (), {"grid": grid, "margin": margin})()
        if list(blank.shape[::-1]) != grid["image_size"]: return []
        return _clean_cells([{"field": f, "visit": f"c{c + 1}", "layout": "single", "kind": meta[f]["kind"], "image": crop} for ri, c, f, crop in cleanup.cut_cells(blank, t, "none") if f in meta])
    try:
        t = scans.target(card); blank = scans._blank(t)
    except (KeyError, ValueError, OSError): return []
    if blank is None: return []
    meta = {f["key"]: f for f in cards.fields(card)}; by_col = t.grid.get("axis") == "columns"; rate = {}
    import csv
    p = cards.CARDS / card / "cells.csv"
    if p.exists():
        with open(p, newline="") as fh:
            tot, ink = {}, {}
            for r in csv.DictReader(fh): k = (r["row"], r["col"]); tot[k] = tot.get(k, 0) + 1; ink[k] = ink.get(k, 0) + (r["ink"] == "1")
        rate = {k: ink[k] / tot[k] for k in tot}
    return _clean_cells([{"field": f, "visit": f"c{(ri if by_col else c) + 1}", "layout": meta[f]["layout"], "kind": meta[f]["kind"], "image": crop}
                         for ri, c, f, crop in cleanup.cut_cells(blank, t, "none") if f in meta], rate or None)


def sample(rng, pool, bgs, mix=None):
    """One synthetic cell. mix: {(kind, layout): weight}. Backgrounds are chosen to suit the layout: a two-line layout takes a cell of a field that
    has it when any exists, otherwise a plain cell; a plain layout takes a plain cell."""
    mix = mix or DEFAULT_MIX; kind, layout = rng.choices(list(mix), list(mix.values()))[0]
    fit = [b for b in bgs if b["layout"] == layout] or [b for b in bgs if b["layout"] == "single"] or bgs
    if not fit: raise ValueError("no background cells: a card needs a blank card (add 5 or more scans) before synthetic cells can be drawn on it")
    b = fit[rng.randrange(len(fit))]; label, parts = value(kind, rng)
    return {"image": damage(draw(b["image"], kind, layout, parts, pool, rng), rng), "kind": kind, "layout": layout, "label": label, "background": b}


DEFAULT_MIX = {("number", "single"): 4, ("date", "single"): 2, ("bp", "single"): 1.5, ("date", "stacked"): 1.5, ("bp", "diagonal"): 1.5}


def pieces(s):
    """[(image, label, piece suffix)] a sample is trained on: the cell itself with suffix "", or its two pieces ("#1", "#2")."""
    if s["layout"] == "single": return [(s["image"], s["label"], "")]
    pcs = subcells.parts(s["image"], s["layout"], s["background"]["image"] if s["layout"] == "diagonal" else None)
    labs = subcells.split_label(s["layout"], s["label"])
    return [(img, lab, f"#{i + 1}") for i, (img, lab) in enumerate(zip(pcs, labs))]
