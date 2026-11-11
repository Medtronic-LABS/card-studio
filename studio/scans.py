"""Filled scans for a card type: say which card a scan is, warp it onto that card's template, cut every grid cell. Local only.

The steps are the ones the phone path uses (cleanup.py: SIFT + RANSAC align with an outline check, warp to the grid's size, cut cells with
the pack margin), so a crop the reader is trained on is cut the way a crop it will read is cut. Crops are stored as cut: gray, no line
removal, no page clean-up (the same as the ANC training crops). Results go to studio/cards/<card>/ (patient data, gitignored):
    aligned/<page>.png             the scan on the template, at the grid's resolution
    crops/<field>/<visit>/<page>.png   one image per grid cell (visit = c1..cN: a column, or a row on a card whose fields run down the columns)
    scans.csv                      one row per scan: source file name, content hash, inliers, status, template version, grid hash
    cells.csv                      one row per cut cell: does it hold ink (the pack's ink gate, after line removal)
    blank.png                      the card with nothing written on it: the pixel median of up to BLANK_SCANS aligned scans

What is printed inside a cell (a diagonal that splits a blood-pressure cell, a dotted line, shading) is not handwriting. Once a card has
BLANK_MIN scans, the blank card is estimated from them and anything it already shows is ignored when deciding whether a cell holds ink.
With fewer scans there is no estimate, and a cell with print in it counts as having ink.

A scan is `ok`, or `check` when it matched with fewer than CHECK_BELOW inliers (look at it before trusting its cells). A scan that does not
match is not stored. Adding the same file again replaces its earlier result. If the grid is edited, `recut` cuts every stored scan again
from its aligned image; if the template image itself is replaced, stored scans are `stale` and have to be added again.

CLI: python scans.py add <card|auto> FILE_OR_FOLDER... [--writer CODE]   |   python scans.py recut <card>   |   python scans.py summary [card]
"""
import csv, hashlib, json, re, sys, threading, time
from pathlib import Path
import cv2, numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cards, cleanup
from detect import _decode
from idtest import _sift

AUTO = "auto"
MIN_INLIERS, CHECK_BELOW, MARGIN = 40, 100, 2.0     # floor (the packs' match.min_inliers); below CHECK_BELOW a person should look; auto: best must beat the runner-up by MARGIN
MAX_SIDE, MAX_PDF_PAGES, PDF_DPI = 3000, 200, 200
BLANK_MIN, BLANK_SCANS, PRINT_BELOW = 5, 25, 170      # scans needed for / used in the blank-card estimate; blank-card pixels darker than this are print
EXT = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".heic", ".heif", ".pdf"}
SCAN_COLS = ["page", "source", "sha256", "inliers", "status", "template_version", "grid_sha", "cells", "with_ink", "added"]
CELL_COLS = ["page", "row", "col", "ink"]
_lock = threading.Lock()
_targets = {}


class Target:
    """A card's template and grid in the shape cleanup.py's align / warp / cut_cells / prepare expect from a pack."""
    def __init__(self, card):
        d = cards.template_dir(card)
        self.card, self.version = card, cards.template_version(card)
        self.tpl = cv2.imread(str(d / "template.png"), cv2.IMREAD_GRAYSCALE)
        if self.tpl is None: raise ValueError(f"{card}: template image is missing or unreadable")
        self.tpl_feats = _sift(self.tpl)
        self.grid, self.scale = cards.pack_grid(card); self.margin, self.ink = cards.CELL_MARGIN, cards.INK_GATE
        self.grid_sha = hashlib.sha256(json.dumps(self.grid, sort_keys=True).encode()).hexdigest()[:16]


def target(card):
    if card == cards.LEGACY or not cards.exists(card): raise KeyError("no such card")
    d = cards.template_dir(card); key = (card, str(d), (d / "grid.json").stat().st_mtime, (d / "template.png").stat().st_mtime)
    if _targets.get(card, (None,))[0] != key: _targets[card] = (key, Target(card))
    return _targets[card][1]


def targets():
    """Every card that has a template with a grid (the cards a scan can be routed to)."""
    return [target(c["id"]) for c in cards.list_cards() if c["id"] != cards.LEGACY and c["template_version"] is not None]


def pages_from(data: bytes):
    """One BGR image per page: an image file is one page, a PDF is every page (rendered at PDF_DPI)."""
    if data[:4] == b"%PDF":
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(data)
        if len(doc) > MAX_PDF_PAGES: raise ValueError(f"PDF has {len(doc)} pages (limit {MAX_PDF_PAGES}): split it first")
        for i in range(len(doc)):
            yield cv2.cvtColor(np.array(doc[i].render(scale=PDF_DPI / 72).to_pil().convert("RGB")), cv2.COLOR_RGB2BGR)
        return
    try:
        im = _decode(data)
    except Exception as e:
        raise ValueError(f"could not read this file ({type(e).__name__}: {e})")
    if im is None: raise ValueError("unsupported file: use JPEG, PNG, WebP, TIFF, HEIC or PDF")
    yield im


def identify(bgr, among):
    """Score a scan against each target. -> (best target or None, H, tried). `tried` = [{card, inliers, why}] best first; the winner needs
    MIN_INLIERS, a sane outline, and (when there is a runner-up) MARGIN times the runner-up's inliers."""
    res = []
    for t in among:
        H, inl, bad = cleanup.align(bgr, t)
        res.append((t, H, inl, bad or (f"only {inl} matching features, needs {MIN_INLIERS}" if inl < MIN_INLIERS else None)))
    res.sort(key=lambda r: -r[2])
    tried = [{"card": t.card, "inliers": inl, "why": why} for t, _, inl, why in res]
    if not res or res[0][3] is not None: return None, None, tried
    if len(res) > 1 and res[0][2] < MARGIN * res[1][2]:
        tried[0]["why"] = f"too close to {res[1][0].card} ({res[0][2]} against {res[1][2]} matching features)"; return None, None, tried
    return res[0][0], res[0][1], tried


def _read(card, name, cols):
    p = cards.CARDS / card / name
    if not p.exists(): return []
    with open(p, newline="") as fh: return [{c: r.get(c, "") for c in cols} for r in csv.DictReader(fh)]


def _write(card, name, cols, rows):
    d = cards.CARDS / card; d.mkdir(parents=True, exist_ok=True)
    with open(d / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols); w.writeheader(); w.writerows(rows)


def _visit(t, ri, c):
    return f"c{(ri if t.grid.get('axis') == 'columns' else c) + 1}"        # a visit is a column, or a row when the fields run down the columns


def _blank(t):
    """The estimated blank card (see the module text), or None when there is none yet or it is for another image size."""
    p = cards.CARDS / t.card / "blank.png"; im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.exists() else None
    return im if im is not None and [im.shape[1], im.shape[0]] == t.grid["image_size"] else None


def _blank_cells(t, blank):
    return {} if blank is None else {(field, _visit(t, ri, c)): crop for ri, c, field, crop in cleanup.cut_cells(blank, t, "none")}


def _has_ink(t, crop, blank_crop=None):
    if blank_crop is not None and blank_crop.shape == crop.shape:
        printed = cv2.dilate((blank_crop < PRINT_BELOW).astype(np.uint8), np.ones((7, 7), np.uint8))      # widened a little: scans never align to the pixel
        crop = np.where(printed > 0, 255, crop).astype(np.uint8)
    return cleanup.prepare(cleanup.remove_lines(crop), t) is not None


def _cut(t, page, aligned):
    """Write every cell crop of one aligned scan. -> cells.csv rows."""
    root = cards.crops_dir(t.card); cols = len(t.grid["col_lines_top"]) - 1; out = []; bl = _blank_cells(t, _blank(t))
    for ri, c, field, crop in cleanup.cut_cells(aligned, t, "none"):
        visit = _visit(t, ri, c)
        f = root / field / visit / f"{page}.png"; f.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(f), crop)
        out.append({"page": page, "row": field, "col": visit, "ink": int(_has_ink(t, crop, bl.get((field, visit))))})
    assert len(out) == len(t.grid["rows"]) * cols
    return out


def _drop_crops(card, page):
    for f in cards.crops_dir(card).glob(f"*/*/{page}.png"): f.unlink()


def _page_id(card, source, sha, existing):
    """The file's name when it can be a page id and is not taken by different content, else the name plus a bit of the hash."""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(source).stem).strip("-")[:30] or "scan"
    same = next((r["page"] for r in existing if r["sha256"] == sha), None)
    if same: return same
    return stem if all(r["page"] != stem for r in existing) else f"{stem}-{sha[:6]}"


def add(card, data: bytes, source="scan", writer=None):
    """Add one uploaded file (an image, or a PDF of several cards) to `card`, or to whichever card each page matches when card is AUTO.
    writer: who wrote the cards in this file, as a code (cards.set_writers); every page added gets it. Not a name.
    -> one result per page: {source, page, card, status: ok | check | rejected, inliers, reason, cells, with_ink, tried}."""
    if writer and not cards.WRITER.match(writer): raise ValueError("a writer is a code of letters, digits, _ or - (at most 24 characters): use W07 or a clinic id, never a name")
    among = targets() if card == AUTO else [target(card)]
    if not among: raise ValueError("no card has a template with a saved grid yet")
    out = []
    for i, bgr in enumerate(pages_from(data)):
        src = source if data[:4] != b"%PDF" else f"{Path(source).stem}-p{i + 1:03d}"
        s = MAX_SIDE / max(bgr.shape[:2])
        if s < 1: bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        sha = hashlib.sha256(cv2.imencode(".png", bgr)[1].tobytes()).hexdigest()
        t, H, tried = identify(bgr, among)
        if t is None:
            top = tried[0] if tried else None
            out.append({"source": src, "page": None, "card": None, "status": "rejected", "inliers": top["inliers"] if top else 0,
                        "reason": (f"closest is {top['card']}: {top['why']}" if top else "no card to match against"), "cells": 0, "with_ink": 0, "tried": tried})
            continue
        aligned, covered = cleanup.warp(bgr, H, t); inl = tried[0]["inliers"]
        cover = round(100.0 * float((covered != 0).mean()), 1)
        status = "ok" if inl >= CHECK_BELOW and cover >= 90 else "check"
        why = "" if status == "ok" else (f"only {inl} matching features" if inl < CHECK_BELOW else f"the scan covers {cover}% of the card")
        with _lock:
            rows = _read(t.card, "scans.csv", SCAN_COLS); page = _page_id(t.card, src, sha, rows)
            d = cards.aligned_dir(t.card); d.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(d / f"{page}.png"), aligned)
            _drop_crops(t.card, page); cells = _cut(t, page, aligned)
            rows = [r for r in rows if r["page"] != page] + [{"page": page, "source": Path(src).name[:80], "sha256": sha, "inliers": inl, "status": status,
                    "template_version": t.version, "grid_sha": t.grid_sha, "cells": len(cells), "with_ink": sum(c["ink"] for c in cells), "added": time.strftime("%Y-%m-%d %H:%M")}]
            _write(t.card, "scans.csv", SCAN_COLS, sorted(rows, key=lambda r: r["page"]))
            _write(t.card, "cells.csv", CELL_COLS, [r for r in _read(t.card, "cells.csv", CELL_COLS) if r["page"] != page] + cells)
            if writer: cards.set_writers(t.card, writer, [page])
        out.append({"source": src, "page": page, "card": t.card, "status": status, "inliers": inl, "reason": why, "cells": len(cells),
                    "with_ink": sum(c["ink"] for c in cells), "tried": tried})
    return out


def refresh_ink(card):
    """Rebuild the blank-card estimate when the scans it should come from have changed, then decide again which cells hold ink.
    -> True when something was rebuilt. Cheap when nothing changed, so summary() calls it."""
    t = target(card); d = cards.CARDS / card
    with _lock:
        rows = _read(card, "scans.csv", SCAN_COLS)
        use = [r for r in rows if r["status"] == "ok" and str(r["template_version"]) == str(t.version)][:BLANK_SCANS]
        key = hashlib.sha256("|".join(r["sha256"] for r in use).encode()).hexdigest()[:16] if len(use) >= BLANK_MIN else None
        try: have = json.loads((d / "blank.json").read_text())["key"]
        except (OSError, ValueError, KeyError): have = None
        if key == have: return False
        if key is None:
            (d / "blank.png").unlink(missing_ok=True); (d / "blank.json").unlink(missing_ok=True)
        else:
            stack = [im for im in (cv2.imread(str(cards.aligned_dir(card) / f"{r['page']}.png"), cv2.IMREAD_GRAYSCALE) for r in use)
                     if im is not None and [im.shape[1], im.shape[0]] == t.grid["image_size"]]
            if len(stack) < BLANK_MIN: return False
            cv2.imwrite(str(d / "blank.png"), np.median(np.stack(stack), axis=0).astype(np.uint8))
            (d / "blank.json").write_text(json.dumps({"key": key, "scans": len(stack)}))
        bl = _blank_cells(t, _blank(t)); cells = _read(card, "cells.csv", CELL_COLS); per = {}
        for c in cells:
            crop = cv2.imread(str(cards.crops_dir(card) / c["row"] / c["col"] / f"{c['page']}.png"), cv2.IMREAD_GRAYSCALE)
            if crop is not None: c["ink"] = int(_has_ink(t, crop, bl.get((c["row"], c["col"]))))
            per[c["page"]] = per.get(c["page"], 0) + int(c["ink"])
        for r in rows:
            if r["page"] in per: r["with_ink"] = per[r["page"]]
        _write(card, "scans.csv", SCAN_COLS, rows); _write(card, "cells.csv", CELL_COLS, cells)
    return True


def recut(card):
    """Cut every stored scan again with the grid as it is now. Scans aligned to an older template image are marked stale and their crops removed."""
    t = target(card)
    with _lock:
        rows = _read(card, "scans.csv", SCAN_COLS); cells = []; done = stale = 0
        for r in rows:
            f = cards.aligned_dir(card) / f"{r['page']}.png"; _drop_crops(card, r["page"])
            im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) if f.exists() else None
            if str(r["template_version"]) != str(t.version) or im is None or [im.shape[1], im.shape[0]] != t.grid["image_size"]:
                r["status"] = "stale"; r["cells"] = r["with_ink"] = 0; stale += 1; continue
            c = _cut(t, r["page"], im); cells += c; done += 1
            r["grid_sha"] = t.grid_sha; r["cells"] = len(c); r["with_ink"] = sum(x["ink"] for x in c)
        _write(card, "scans.csv", SCAN_COLS, rows); _write(card, "cells.csv", CELL_COLS, cells)
    refresh_ink(card)
    return {"card": card, "recut": done, "stale": stale}


def remove(card, page):
    target(card)
    if not cards.PAGE.match(page): raise ValueError("bad page id")
    with _lock:
        rows = _read(card, "scans.csv", SCAN_COLS)
        if all(r["page"] != page for r in rows): raise KeyError("no such scan")
        (cards.aligned_dir(card) / f"{page}.png").unlink(missing_ok=True); _drop_crops(card, page); cards.set_writers(card, "", [page])
        _write(card, "scans.csv", SCAN_COLS, [r for r in rows if r["page"] != page])
        _write(card, "cells.csv", CELL_COLS, [r for r in _read(card, "cells.csv", CELL_COLS) if r["page"] != page])
    return {"card": card, "removed": page}


def summary(card):
    refresh_ink(card)
    t = target(card); rows = _read(card, "scans.csv", SCAN_COLS)
    for r in rows:
        if r["status"] != "stale" and str(r["template_version"]) != str(t.version): r["status"] = "stale"
    live = [r for r in rows if r["status"] != "stale"]
    labeled = len(cards.labels(card)[0])
    return {"card": card, "template_version": t.version, "grid": {**cards.grid_shape(t.grid), "size": t.grid["image_size"]},
            "scans": [{**{k: (int(r[k]) if k in ("inliers", "cells", "with_ink") and str(r[k]).isdigit() else r[k]) for k in SCAN_COLS if k != "sha256"}, "writer": cards.writers(card).get(r["page"], "")} for r in rows],
            "totals": {"scans": len(rows), "ok": sum(r["status"] == "ok" for r in rows), "check": sum(r["status"] == "check" for r in rows),
                       "stale": len(rows) - len(live), "cells": sum(int(r["cells"] or 0) for r in live), "with_ink": sum(int(r["with_ink"] or 0) for r in live), "labeled": labeled},
            "blank_card": (cards.CARDS / card / "blank.json").exists(), "writers": cards.writer_status(card),
            "needs_recut": sum(r["grid_sha"] != t.grid_sha for r in live)}


def summaries():
    out = []
    for c in cards.list_cards():
        if c["id"] == cards.LEGACY or c["template_version"] is None: continue
        try: out.append(summary(c["id"]))
        except (ValueError, OSError, KeyError) as e: out.append({"card": c["id"], "error": f"{type(e).__name__}: {e}"})
    return out


def overlay(card, page, width=1100):
    """PNG of one aligned scan with the grid drawn on it, to check by eye that the cells sit on the printed lines."""
    t = target(card)
    if not cards.PAGE.match(page): raise KeyError("no such scan")
    f = cards.aligned_dir(card) / f"{page}.png"
    im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) if f.exists() else None
    if im is None: raise KeyError("no such scan")
    g = t.grid; vis = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR); n = len(g["col_lines_top"]) - 1
    if [im.shape[1], im.shape[0]] == g["image_size"]:
        for c in range(n):
            rl = g["row_lines_by_col"][c]
            for y in rl:
                x0, x1 = (cleanup._xat(g["col_lines_top"][k], g["col_lines_bot"][k], y, g) for k in (c, c + 1))
                cv2.line(vis, (int(x0), int(y)), (int(x1), int(y)), (0, 160, 0), 2)
        for k in range(n + 1):
            rl = g["row_lines_by_col"][min(k, n - 1)]
            p = [(int(cleanup._xat(g["col_lines_top"][k], g["col_lines_bot"][k], y, g)), int(y)) for y in (rl[0], rl[-1])]
            cv2.line(vis, p[0], p[1], (0, 0, 220), 2)
    s = width / vis.shape[1]
    return cv2.imencode(".png", cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else vis)[1].tobytes()


def add_path(card, path, writer=None):
    """CLI helper: a file, or every supported file in a folder (sorted)."""
    p = Path(path); files = sorted(f for f in p.rglob("*") if f.suffix.lower() in EXT) if p.is_dir() else [p]
    out = []
    for f in files:
        try: res = add(card, f.read_bytes(), f.name, writer)
        except ValueError as e: res = [{"source": f.name, "page": None, "card": None, "status": "rejected", "inliers": 0, "reason": str(e), "cells": 0, "with_ink": 0}]
        for r in res: print(f"{r['source']:28s} {r['status']:8s} {r['card'] or '-':16s} {r['inliers']:5d} inliers  {r['with_ink']}/{r['cells']} cells with ink  {r['reason']}", flush=True)
        out += res
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "summary"
    if cmd == "add" and len(sys.argv) >= 4:
        args = sys.argv[3:]; writer = None
        if "--writer" in args: i = args.index("--writer"); writer = args[i + 1]; args = args[:i] + args[i + 2:]
        res = [r for p in args for r in add_path(sys.argv[2], p, writer)]
        print({s: sum(r["status"] == s for r in res) for s in ("ok", "check", "rejected")})
    elif cmd == "recut" and len(sys.argv) == 3: print(recut(sys.argv[2]))
    elif cmd == "summary":
        for s in ([summary(sys.argv[2])] if len(sys.argv) > 2 else summaries()): print(json.dumps({k: v for k, v in s.items() if k != "scans"}))
    else: sys.exit(__doc__.split("CLI:")[1].strip())
