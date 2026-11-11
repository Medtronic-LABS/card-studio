"""Cells written on two lines: split them into two single-line pieces the reader can read. Local only.

The reader reads one line of writing left to right. Two kinds of cell on the cards are not that:
  stacked   a date written as day.month over a bar over the year (the EPI vaccination card):        top "22.2"   bottom "26"
  diagonal  a blood pressure written either side of a printed diagonal (the NCD card):              top-left "120"   bottom-right "80"
`layout` on a field in fields.json says which (default `single`: nothing changes). The label stays one string for the whole cell
("22/2/26", "120/80"): a person labels what the cell says. For training, each labeled cell becomes two pieces whose labels are the
parts of that string; for reading, the two pieces are read separately and joined. A piece is read with its own tighter character mask.

Geometry is plain image processing and makes no use of the model:
  stacked   the fraction bar is the longest straight stroke near the middle of the cell, level or slanted as people draw it; everything
            above it is the top piece and everything below it the bottom piece, with the bar itself whited out. With no bar found the cell
            is cut level through the middle
  diagonal  the cell is cut along the line from its bottom-left to its top-right corner, each side keeping a margin so digits that
            cross the line are not cut in half; when a blank card is known, what is printed there (the diagonal itself) is whited out

Every piece goes through the same line removal as a whole cell before it is stored or read, so training sees what reading sees.
"""
import re
from pathlib import Path

import cv2, numpy as np

LAYOUTS = ("single", "stacked", "diagonal")
KIND_FOR = {"stacked": "date", "diagonal": "bp"}      # the field kind each layout belongs to
STACKED_LABEL = re.compile(r"(\d{1,2}[./]\d{1,2})[./](\d{2}(?:\d{2})?)")
DIAGONAL_LABEL = re.compile(r"(\d{2,3})/(\d{2,3})")
SOFT = 0.10             # how far across the diagonal each piece keeps pixels (fraction of the cell), so strokes that cross it are not cut
PRINT_BELOW = 110       # blank-card pixels darker than this are print (a blank estimated from scans keeps faint ghosts of handwriting; those are lighter)
BAR_SLOPE = 0.55        # a fraction bar may rise or fall by this much of its length (about 29 degrees)
MIN_SIDE = 4


def check_layout(kind, layout):
    """Raise ValueError when the layout does not fit the field's kind."""
    if layout not in LAYOUTS: raise ValueError(f"layout must be one of {', '.join(LAYOUTS)}")
    if layout != "single" and KIND_FOR[layout] != kind: raise ValueError(f"layout {layout} is for a {KIND_FOR[layout]} field, not {kind}")


def piece_names(key):
    return [f"{key}#1", f"{key}#2"]


def base_key(name):
    """`bp#2` -> (`bp`, 2); a name without a piece suffix -> (name, None)."""
    m = re.fullmatch(r"(.+)#([12])", name)
    return (m[1], int(m[2])) if m else (name, None)


def split_label(layout, label):
    """The two labels a whole-cell label breaks into, or None when it does not have the shape of this layout."""
    m = (STACKED_LABEL if layout == "stacked" else DIAGONAL_LABEL).fullmatch(label.strip()) if layout in KIND_FOR else None
    return (m[1], m[2]) if m else None


def canonical(layout, label):
    """A label in the form `join` produces, so a read can be compared with a label however the date was punctuated."""
    return label.strip().replace(".", "/") if layout == "stacked" else label.strip()


def join(layout, a, b):
    """The whole-cell string from the two read pieces. A piece that read nothing leaves the whole read as the other piece alone."""
    if not a or not b: return ""
    return (a.replace(".", "/") + "/" + b) if layout == "stacked" else f"{a}/{b}"


def piece_masks(layout, classes, blank):
    """{`#1`: allowed class indices, `#2`: ...} for the two pieces: digits (and the day-month separator for the top of a date)."""
    dig = sorted({blank} | {classes.index(c) for c in "0123456789"})
    top = sorted(set(dig) | {classes.index(c) for c in "./-"}) if layout == "stacked" else dig
    return {"#1": top, "#2": dig}


def _clean(piece):
    import cleanup
    return cleanup.remove_lines(piece)


def _find_bar(g):
    """(y at the left edge, y at the right edge) of the fraction bar, or None. The strongest straight segment of ink that is long, close to
    level, and crosses the middle third of the cell's height."""
    h, w = g.shape; ink = ((g < 150) * 255).astype(np.uint8)
    segs = cv2.HoughLinesP(ink, 1, np.pi / 180, threshold=max(20, int(w * 0.25)), minLineLength=int(w * 0.4), maxLineGap=max(4, int(w * 0.06)))
    best = None
    for x1, y1, x2, y2 in ([] if segs is None else np.asarray(segs).reshape(-1, 4)):        # (n, 1, 4) in some OpenCV versions, (n, 4) in others
        if x2 == x1: continue
        m = (y2 - y1) / (x2 - x1)
        if abs(m) > BAR_SLOPE: continue
        yl, yr = y1 - m * x1, y1 + m * (w - 1 - x1); mid = (yl + yr) / 2; length = float(np.hypot(x2 - x1, y2 - y1))
        if not h * 0.3 <= mid <= h * 0.7: continue
        if best is None or length > best[0]: best = (length, yl, yr)
    return None if best is None else (best[1], best[2])


def _split_stacked(g):
    h, w = g.shape; bar = _find_bar(g) or (h / 2, h / 2)
    yy, xx = np.mgrid[0:h, 0:w]; line = bar[0] + (bar[1] - bar[0]) * xx / max(1, w - 1); gap = max(2.0, h * 0.025)
    return np.where(yy < line - gap, g, 255).astype(np.uint8), np.where(yy > line + gap, g, 255).astype(np.uint8)


def _split_diagonal(g, blank=None):
    h, w = g.shape
    if blank is not None and blank.shape == g.shape:
        g = np.where(cv2.dilate((blank < PRINT_BELOW).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0, 255, g).astype(np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]; s = xx / w + yy / h          # 1 on the diagonal, below 1 above-left of it, above 1 below-right of it
    return np.where(s < 1 + SOFT, g, 255).astype(np.uint8), np.where(s > 1 - SOFT, g, 255).astype(np.uint8)


def parts(crop, layout, blank=None):
    """The two pieces of one cell crop (gray), each already line-cleaned, in reading order (stacked: top, bottom; diagonal: above, below)."""
    if layout not in KIND_FOR: raise ValueError(f"{layout!r} has no pieces")
    g = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if min(g.shape) < 2 * MIN_SIDE: raise ValueError("cell is too small to split")
    a, b = _split_stacked(g) if layout == "stacked" else _split_diagonal(g, blank)
    return _clean(a), _clean(b)


def expand(card, rows, dropped=None):
    """Replace every labeled cell of a layout field by its two pieces for training. rows: [{page,row,col,label,crop}] (cards.labels).
    Pieces are written to studio/cards/<card>/pieces/<field>#k/<visit>/<page>.png (derived: rebuilt every run). A cell whose label does not
    have the layout's shape, or whose crop cannot be read, is left out and counted in `dropped`. -> (rows for training, {field: layout})."""
    import cards, scans
    layouts = {f["key"]: f["layout"] for f in cards.fields(card) if f.get("layout", "single") != "single"} if card != cards.LEGACY else {}
    if not layouts: return rows, {}
    blanks = {}
    try:
        t = scans.target(card); bl = scans._blank(t); blanks = scans._blank_cells(t, bl)
    except (KeyError, ValueError, OSError): pass
    out = []; root = cards.CARDS / card / "pieces"
    for r in rows:
        lay = layouts.get(r["row"])
        if lay is None: out.append(r); continue
        sp = split_label(lay, r["label"]); crop = cv2.imread(r["crop"], cv2.IMREAD_GRAYSCALE) if r.get("crop") else None
        if sp is None or crop is None:
            if dropped is not None: dropped["label_does_not_split" if sp is None else "missing_crop"] = dropped.get("label_does_not_split" if sp is None else "missing_crop", 0) + 1
            continue
        try: pcs = parts(crop, lay, blanks.get((r["row"], r["col"])))
        except ValueError:
            if dropped is not None: dropped["cell_too_small"] = dropped.get("cell_too_small", 0) + 1
            continue
        for name, img, lab in zip(piece_names(r["row"]), pcs, sp):
            p = root / name / r["col"] / f"{r['page']}.png"; p.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(p), img)
            out.append({**r, "row": name, "label": lab, "crop": str(p)})
    return out, layouts


def compose(preds, layouts, whole_labels):
    """Predictions on pieces -> predictions on whole cells. preds: [{page,row,col,label,pred,conf}] with piece rows for layout fields;
    whole_labels: {(page, field, col): whole label}. Cells missing a piece are left out. The joined read is compared with the whole label
    in the same punctuation, so `22.2.26` and `22/2/26` are the same date."""
    out, held = [], {}
    for p in preds:
        key, k = base_key(p["row"])
        if k is None: out.append(p); continue
        held.setdefault((p["page"], key, p["col"]), {})[k] = p
    for (page, key, col), d in held.items():
        if len(d) != 2: continue
        lay = layouts[key]; whole = whole_labels.get((page, key, col))
        if whole is None: continue
        out.append({"page": page, "row": key, "col": col, "label": canonical(lay, whole), "pred": join(lay, d[1]["pred"], d[2]["pred"]), "conf": min(d[1]["conf"], d[2]["conf"])})
    return out
