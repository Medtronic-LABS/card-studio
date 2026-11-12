"""Clean-up lab: take one photo, align it to a card pack, apply every photo clean-up option, read the cells with the pack's reader.

A port of the app's ANC path (anc_reader.dart: align, normalize, cut cells, remove lines, ink gate, masked CTC decode), so what
Studio shows is what the phone would do, up to OpenCV/onnxruntime differences nobody has compared. Local only; nothing is saved
except small preview PNGs in uploads/ (deleted after a day with the other uploads).
"""
import json, math, sys, time, uuid
from pathlib import Path
import cv2, numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cardpack"))
import cardpack as cp
import cardlogic, cards, grammar
import normalize as N
from idtest import _sift, _homography

PACKS = HERE.parent / "cardpack" / "out"
_cache = {}


class Pack:
    """Everything the lab needs from a .cardpack, loaded once."""
    def __init__(self, path):
        import onnxruntime as ort
        self.path = Path(path); m = cp.read(self.path)
        self.man = json.loads(m["manifest.json"]); self.fields = json.loads(m["fields.json"]); self.grid = json.loads(m["grid.json"])
        self.tpl = cv2.imdecode(np.frombuffer(m["template.png"], np.uint8), cv2.IMREAD_GRAYSCALE)
        self.tpl_feats = _sift(self.tpl)
        self.session = ort.InferenceSession(m["reader.onnx"], providers=["CPUExecutionProvider"])
        r = self.man["reader"]; self.in_name, self.out_name = r["input"]["name"], r["output"]["name"]
        pp = self.man["preprocess"]; self.flatten_k = pp["flatten_kernel"]; self.remove_lines = pp["remove_lines"]; self.ink = pp["ink_gate"]
        nz = pp.get("normalize") or {"default": "flatten"}; self.default = nz["default"]
        ref = nz.get("reference"); self.reference = (ref["mean"], ref["std"]) if ref else None
        self.margin = self.man["grid"]["cell_margin"]; self.scale = self.man["template"]["grid_scale"]
        self.classes = self.fields["classes"]; self.blank = self.fields["blank_idx"]
        self.spec = {f["key"]: f for f in self.fields["fields"]}
        self.id, self.version = self.man["id"], self.man["version"]

    @property
    def name(self):
        return f"{self.id} v{self.version}"


def load_pack(path):
    p = Path(path); key = (str(p), p.stat().st_mtime)
    if key not in _cache:
        _cache[key] = Pack(p)
    return _cache[key]


def installed():
    return [load_pack(p) for p in sorted(PACKS.glob("*.cardpack"))]


def align(bgr, pack):
    """-> (H photo->template, inliers, outline problem or None). H maps original photo pixels to template pixels."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, inl = _homography(_sift(gray), pack.tpl_feats)
    if H is None:
        return None, 0, "no homography"
    qh, qw = gray.shape; th, tw = pack.tpl.shape
    quad = cv2.perspectiveTransform(np.float32([[0, 0], [qw, 0], [qw, qh], [0, qh]]).reshape(-1, 1, 2), H).reshape(4, 2)
    e = [quad[(i + 1) % 4] - quad[i] for i in range(4)]; cr = [float(e[i][0] * e[(i + 1) % 4][1] - e[i][1] * e[(i + 1) % 4][0]) for i in range(4)]   # z of the 2-D cross product (np.cross no longer takes 2-D vectors)
    area = 0.5 * abs(np.dot(quad[:, 0], np.roll(quad[:, 1], -1)) - np.dot(quad[:, 1], np.roll(quad[:, 0], -1))) / (tw * th)
    if not (all(c > 0 for c in cr) or all(c < 0 for c in cr)):
        return H, inl, "not convex"
    if cr[0] < 0:
        return H, inl, "mirrored"
    if not 0.2 <= area <= 10.0:
        return H, inl, f"implausible size ({area:.2f}x the template)"
    return H, inl, None


def best_pack(bgr, packs):
    """(pack, H, inliers, tried): the pack with the most inliers that clears its floor and the outline check (ties go to the newest
    version), or pack None. `tried` lists every pack's inliers and why it was rejected, for the error message."""
    best, tried = None, []
    for pk in packs:
        H, inl, bad = align(bgr, pk); floor = pk.man["match"]["min_inliers"]
        why = bad or (f"only {inl} inliers, needs {floor}" if inl < floor else None)
        tried.append({"pack": pk.name, "inliers": inl, "floor": floor, "why": why})
        if why is None and (best is None or (inl, pk.version) > (best[2], best[0].version)):
            best = (pk, H, inl)
    return (*best, tried) if best else (None, None, 0, tried)


def warp(bgr, H, pack):
    """Photo -> upright gray card at the grid's resolution, plus a mask of where the photo really covered it."""
    W, Hh = pack.grid["image_size"]; S = np.diag([pack.scale, pack.scale, 1.0]); M = S @ H
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    warped = cv2.warpPerspective(gray, M, (W, Hh), borderValue=255)
    covered = cv2.warpPerspective(np.ones_like(gray), M, (W, Hh), flags=cv2.INTER_NEAREST, borderValue=0)
    return warped, covered


def _xat(top, bot, y, g):
    return top + (bot - top) * (y - g["y_ref_top"]) / (g["y_ref_bot"] - g["y_ref_top"])


def _rnd(v):
    return int(math.floor(v + 0.5))


def remove_lines(gray):
    _, ink = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kh, kv = max(9, _rnd(gray.shape[1] * 0.6)), max(9, _rnd(gray.shape[0] * 0.6))
    h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kh, 1)))
    v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kv)))
    lines = cv2.dilate(cv2.bitwise_or(h, v), cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    return cv2.bitwise_or(gray, lines)


def prepare(gray, pack):
    """Ink gate, then fit to 48 high keeping aspect and pad white. None = blank cell (never reaches the model)."""
    h, w = gray.shape; ix0 = _rnd(w * 0.12); ix1 = w - ix0; iy0 = _rnd(h * 0.2); iy1 = h - iy0
    dark = int((gray[iy0:iy1, ix0:ix1] < pack.ink["dark_below"]).sum()); inner = max(1, (ix1 - ix0) * (iy1 - iy0))
    if dark < pack.ink["min_dark_pixels"] or dark / inner < pack.ink["min_dark_fraction"]:
        return None
    nw = min(192, max(1, _rnd(w * 48 / h)))
    out = np.full((48, 192), 255, np.uint8); out[:, :nw] = cv2.resize(gray, (nw, 48), interpolation=cv2.INTER_AREA)
    return out


def cut_cells(card, pack, method):
    """Yield (row_index, col_index, field, crop) like AncAligner.cells. The field is the row's name, or the column's when the grid's
    fields run down the columns (axis "columns": each row is then a visit)."""
    g = pack.grid; m = pack.margin; H, W = card.shape; by_col = g.get("axis") == "columns"
    for c in range(len(g["col_lines_top"]) - 1):
        rl = g["row_lines_by_col"][c]
        for ri, row in enumerate(g["rows"]):
            field = g["cols"][c] if by_col else row
            ymid = (rl[ri] + rl[ri + 1]) / 2.0
            xa = _xat(g["col_lines_top"][c], g["col_lines_bot"][c], ymid, g) - m
            xb = _xat(g["col_lines_top"][c + 1], g["col_lines_bot"][c + 1], ymid, g) + m
            my = min(m, max(2.0, (rl[ri + 1] - rl[ri]) / 6))
            x0 = min(max(_rnd(xa), 0), W - 1); x1 = min(max(_rnd(xb), x0 + 1), W)
            y0 = min(max(_rnd(rl[ri] - my), 0), H - 1); y1 = min(max(_rnd(rl[ri + 1] + my), y0 + 1), H)
            crop = card[y0:y1, x0:x1].copy()
            yield ri, c, field, (N.stretch_cell(crop) if method in ("cell_stretch", "flatten_cell_stretch") else crop)


UNREAL_DATE_CONFIDENCE = 0.1


def read_cell(pack, px, field, piece_grammar=None, piece=False):
    """Masked greedy CTC like AncReader.read. Returns (text, confidence). piece_grammar: read this piece against a grammar kind instead of the field's
    own (the top line of a two-line date is a "daymonth"), and falls back to greedy when nothing fits. piece: this is one line of a two-line cell, so the field's
    whole-cell grammar (a complete date or pressure) does not apply to it."""
    x = ((px.astype(np.float32) / 255.0 - 0.5) / 0.5).reshape(1, 1, 48, 192)
    lg = pack.session.run([pack.out_name], {pack.in_name: x})[0].reshape(-1, len(pack.classes))
    spec = pack.spec[field]
    if piece_grammar:
        idx = sorted(spec["allowed_idx"]); text, margin = grammar.decode(lg[:, idx], [pack.classes[i] for i in idx], idx.index(pack.blank), piece_grammar)
        if text: return text, float(min(1.0, math.exp(margin)))
    if spec.get("decode") == "grammar" and not piece:          # the field's own grammar instead of the best class at every step (constrained.py, CARDPACK.md)
        try:
            import constrained
            text, margin = constrained.decode(lg, spec["allowed_idx"], spec["kind"], spec.get("range"), (spec.get("date") or {}).get("years"))
            return text, float(min(1.0, math.exp(margin)))
        except ImportError: pass                 # grammar scoring needs torch; without it this field is read greedily
    ok = np.zeros(len(pack.classes), bool); ok[spec["allowed_idx"]] = True
    z = np.where(ok[None, :], lg, -np.inf); best = z.argmax(1)
    out, confs, prev = [], [], -1
    for t, b in enumerate(best):
        if b != pack.blank and b != prev:
            out.append(pack.classes[int(b)]); confs.append(float(1.0 / np.exp(z[t][ok] - z[t, b]).sum()))
        prev = b
    return "".join(out), (min(confs) if confs else 0.0)


def read_two_line(pack, crop, field, layout):
    """A cell written on two lines (subcells.py): read its two pieces separately and join them. -> (text, confidence). A piece the ink gate
    calls empty, or that cannot be cut, makes the whole read empty: half a date or half a pressure is not a value."""
    import subcells
    try: pieces = subcells.parts(crop, layout)
    except ValueError: return "", 0.0
    reads = []
    for piece in pieces:
        px = prepare(piece, pack)
        if px is None: return "", 0.0
        reads.append(read_cell(pack, px, field, "daymonth" if layout == "stacked" and not reads and pack.spec[field].get("decode") == "grammar" else None, piece=True))
    text, conf = subcells.join(layout, reads[0][0], reads[1][0]), min(reads[0][1], reads[1][1])
    if text and layout == "stacked" and pack.spec[field].get("kind") == "date" and not grammar.date_ok(reads[0][0], reads[1][0]):
        conf = min(conf, UNREAL_DATE_CONFIDENCE)         # the two lines do not name a real day (a 29 February without a leap year, a year line that is not a year): a person should look
    return text, conf


def card_checks(pack, reads):
    """The card-level checks the card asks for (cards.checks, cardlogic.py) over one method's reads. A cell the pack flags for low confidence does not vote.
    -> [{field, visit, rule, suggestion, note}]; [] for a card with no checks or no card folder (a pack Studio did not make)."""
    try: asked = cards.checks(pack.id)
    except (KeyError, ValueError, OSError): return []
    if not asked: return []
    g = pack.grid; by_col = g.get("axis") == "columns"; texts, weak = {}, set()
    for (ri, c), (text, conf, flagged) in reads.items():
        key = ((g["cols"][c] if by_col else g["rows"][ri]), (ri if by_col else c) + 1)
        if text: texts[key] = text
        if flagged: weak.add(key)
    return cardlogic.check(texts, asked, weak)


def run_method(pack, warped, covered, method):
    t0 = time.time()
    page, used = N.normalize_page(method, warped, covered, pack.flatten_k, pack.reference)
    reads, ink = {}, 0
    for ri, c, field, crop in cut_cells(page, pack, method):
        clean = remove_lines(crop) if pack.remove_lines else crop
        px = prepare(clean, pack)
        if px is None:
            reads[(ri, c)] = ("", 0.0, False); continue
        layout = pack.spec[field].get("layout", "single")
        ink += 1; text, conf = read_two_line(pack, crop, field, layout) if layout != "single" else read_cell(pack, px, field)
        reads[(ri, c)] = (text, conf, conf < (pack.spec[field].get("min_confidence") or 0.0))
    filled = [v for v in reads.values() if v[0]]
    return page, used, reads, {"seconds": round(time.time() - t0, 1), "cells": len(reads), "with_ink": ink, "read": len(filled),
                               "flagged": sum(v[2] for v in filled), "mean_confidence": round(float(np.mean([v[1] for v in filled])), 3) if filled else None}


def analyze(photo, out_dir, pack=None, packs=None, methods=N.METHODS, preview_width=1000):
    """photo: BGR image. pack: a Pack, or None to pick the best installed one. Returns a JSON-able report; previews go to out_dir."""
    packs = packs if packs is not None else installed()
    if pack is None:
        pack, H, inl, tried = best_pack(photo, packs)
        if pack is None:
            top = max(tried, key=lambda t: t["inliers"]) if tried else None
            raise ValueError("Unknown card type: this photo does not match any installed pack. "
                             + (f"Closest: {top['pack']} with {top['inliers']} matching features ({top['why']}); a real card photo gives hundreds. " if top else "")
                             + "The photo needs the printed headings and the row-label column in view, not just handwriting or a few columns, "
                               "and it must be this card type: EPI and NCD cards have no pack here.")
    else:
        H, inl, bad = align(photo, pack)
        if bad or inl < pack.man["match"]["min_inliers"]:
            raise ValueError(f"{pack.name}: this photo does not match ({inl} inliers, floor {pack.man['match']['min_inliers']}{'; ' + bad if bad else ''}).")
    warped, covered = warp(photo, H, pack)
    uid = uuid.uuid4().hex; out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    g = pack.grid; rows, ncol = g["rows"], len(g["col_lines_top"]) - 1
    results, all_reads = [], {}
    for method in methods:
        note = None
        if method == "contrast_match" and pack.reference is None:
            results.append({"method": method, "label": N.LABELS[method], "available": False, "note": "this pack carries no reference (mean/std of its training scans)"}); continue
        page, used, reads, stats = run_method(pack, warped, covered, method)
        s = preview_width / page.shape[1]
        cv2.imwrite(str(out_dir / f"{uid}_{method}.png"), cv2.resize(page, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
        all_reads[method] = reads
        results.append({"method": method, "label": N.LABELS[method], "available": True, "image": f"/api/cleanup/{uid}/{method}.png", "is_pack_default": method == pack.default, "checks": card_checks(pack, reads), **stats})
    done = [r["method"] for r in results if r["available"]]
    diff = []
    for ri in range(len(rows)):
        for c in range(ncol):
            vals = {m: all_reads[m][(ri, c)] for m in done}
            texts = {v[0] for v in vals.values()}
            if len(texts) > 1:
                diff.append({"field": g["cols"][c] if g.get("axis") == "columns" else rows[ri], "visit": (ri if g.get("axis") == "columns" else c) + 1, "reads": {m: {"text": v[0], "confidence": round(v[1], 2), "flagged": v[2]} for m, v in vals.items()}})
    return {"id": uid, "pack": pack.name, "pack_default": pack.default, "inliers": inl, "covered_percent": round(100.0 * float((covered != 0).mean()), 1),
            "methods": results, "disagreements": diff, "cells_total": len(rows) * ncol,
            "notes": ["No ground truth for an uploaded photo: compare the reads with the card by eye.",
                      "Reads use the chosen pack's reader; the app runs the same steps but this port has not been compared with it pixel for pixel."]}
