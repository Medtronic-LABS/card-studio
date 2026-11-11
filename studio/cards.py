"""Card types Studio can train: where each card's labels, cell crops, aligned scans and pack come from. Local only.

`anc` is the first card and keeps its original places (work/labels.csv + labels_new.csv, work/crops, work/aligned; fields from
model_meta.json; pack derived from the latest anc pack). Every other card is a saved Studio template that has a grid. Its data is in
studio/cards/<name>/ (patient data: gitignored):
    labels.csv                     one row per labeled cell, columns: page, row, col, label (row = field key, col = visit c1..cN)
    crops/<row>/<col>/<page>.png   the cell image, same layout as work/crops
    aligned/<page>.png             scans warped onto the template at the grid's resolution (scans.py writes these and the crops)
    scans.csv, cells.csv           what scans.py recorded per scan and per cut cell (which cells have ink)
    fields.json                    optional, {"fields": {"<row>": {"kind": "number", "range": [30, 150]}}}; a row not listed is `text`

A cell is always keyed (field, visit). On most cards a field is a grid row and a visit is a grid column; on a card whose grid says
axis "columns" (one row per visit, like the NCD card) the field is the column and visit cN is the N-th row. Files keep the names row / col.

All cards share one class list (model_meta.json) and one pretrained reader; a card only chooses which classes each field may emit."""
import csv, json, re, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEMPLATES, CARDS, PACKS, WORK = HERE / "templates", HERE / "cards", ROOT / "cardpack" / "out", ROOT / "work"
META = ROOT / "model_meta.json"
LEGACY = "anc"
NAME = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
PACK_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
COL = re.compile(r"^c[1-9][0-9]?$")
PAGE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
KINDS = ("number", "date", "bp", "mark", "text")
KIND_CHARS = {"number": "-.0123456789", "date": "-./0123456789", "bp": "/0123456789", "mark": "-xy"}   # text = every class
META_KIND = {"numeric": "number", "date": "date", "bp": "bp", "mark": "mark", "text": "text"}
MIN_PAGES = 10          # 5 folds by page, and each fold still needs validation and training pages
CELL_MARGIN = 12
INK_GATE = {"dark_below": 110, "min_dark_pixels": 40, "min_dark_fraction": 0.006}
GRID_LONG_SIDE = 2000   # a card is read at about this many pixels on its long side (the ANC grid is 2338), whatever size the template photo is


def meta():
    return json.loads(META.read_text())


def allowed_idx(kind, classes, blank):
    """Class indices a field of this kind may emit (always includes the blank)."""
    if kind not in KINDS:
        raise ValueError(f"unknown field kind {kind!r} (use one of {', '.join(KINDS)})")
    if kind == "text":
        return list(range(len(classes)))
    return sorted({blank} | {classes.index(c) for c in KIND_CHARS[kind]})


def pack_id(card):
    return card.lower()


def template_version(card):
    """Newest version of the template that has a grid, or None."""
    d = TEMPLATES / card
    vs = sorted((int(p.name[1:]) for p in d.glob("v*") if p.name[1:].isdigit() and (p / "grid.json").exists()), reverse=True) if d.is_dir() else []
    return vs[0] if vs else None


def template_dir(card):
    v = template_version(card)
    if v is None:
        raise ValueError(f"card {card!r} has no template with a saved grid")
    return TEMPLATES / card / f"v{v}"


def pack_grid(card):
    """-> (grid, scale): the template's grid at reading resolution. A template photo is often small (Studio caps it for the editor), so the
    grid is scaled by a whole number and the pack says so in template.grid_scale; scans are warped to this size before cells are cut."""
    g = json.loads((template_dir(card) / "grid.json").read_text())
    k = min(3, max(1, round(GRID_LONG_SIDE / max(g["image_size"]))))
    f = lambda v: round(v * k, 1)
    out = {"image_size": [g["image_size"][0] * k, g["image_size"][1] * k], "rows": g["rows"], "y_ref_top": f(g["y_ref_top"]), "y_ref_bot": f(g["y_ref_bot"]),
           "col_lines_top": [f(v) for v in g["col_lines_top"]], "col_lines_bot": [f(v) for v in g["col_lines_bot"]],
           "row_lines_by_col": [[f(v) for v in col] for col in g["row_lines_by_col"]]}
    if g.get("axis") == "columns": out["axis"], out["cols"] = "columns", g["cols"]
    return out, float(k)


def grid_shape(g):
    """{rows, cols, axis, fields, visits} of a grid, for display."""
    r, c = len(g["rows"]), len(g["col_lines_top"]) - 1; by_col = g.get("axis") == "columns"
    return {"rows": r, "cols": c, "axis": "columns" if by_col else "rows", "fields": c if by_col else r, "visits": r if by_col else c}


def pack_versions(card):
    pid = pack_id(card)
    return sorted(int(p.stem[len(pid) + 2:]) for p in PACKS.glob(f"{pid}_v*.cardpack") if p.stem[len(pid) + 2:].isdigit())


def fields(card):
    """[{key, kind, range}] in grid order (rows, or columns when the grid's fields run down the columns). `anc`: every field in model_meta.json."""
    if card == LEGACY:
        return [{"key": k, "kind": META_KIND[v["type"]], "range": None} for k, v in meta()["fields"].items()]
    import gridconv
    rows = gridconv.field_names(json.loads((template_dir(card) / "grid.json").read_text()))
    fp = CARDS / card / "fields.json"
    spec = json.loads(fp.read_text()).get("fields", {}) if fp.exists() else {}
    unknown = sorted(set(spec) - set(rows))
    if unknown:
        raise ValueError(f"fields.json names fields that are not in the grid: {', '.join(unknown)}")
    out = []
    for r in rows:
        s = spec.get(r, {}); kind = s.get("kind", "text"); rng = s.get("range")
        if kind not in KINDS:
            raise ValueError(f"{r}: unknown field kind {kind!r}")
        if rng is not None and (kind != "number" or len(rng) != 2 or not rng[0] < rng[1]):
            raise ValueError(f"{r}: range needs kind number and [low, high]")
        out.append({"key": r, "kind": kind, "range": rng})
    return out


def masks(card):
    """{field key: allowed class indices}: what the trainer and the decoder restrict each field to."""
    m = meta()
    if card == LEGACY:
        return {k: sorted(v["allowed_idx"]) for k, v in m["fields"].items()}
    return {f["key"]: allowed_idx(f["kind"], m["classes"], m["blank_idx"]) for f in fields(card)}


def crops_dir(card):
    return WORK / "crops" if card == LEGACY else CARDS / card / "crops"


def aligned_dir(card):
    return WORK / "aligned" if card == LEGACY else CARDS / card / "aligned"


def _imported(card):
    """Labels imported from phones / Label Studio that belong to this card (a phone correction names its pack; Label Studio rows are ANC)."""
    import corrections
    pid = pack_id(card)
    return [(r, str(corrections.IMPORTS / r["crop"]) if r["crop"] else "") for r in corrections.read_imported()
            if (r.get("pack") or LEGACY) == pid]


def labels(card):
    """-> (rows, problems). rows: [{page,row,col,label,crop}], sorted, one per cell, later sources win (labels.csv, labels_new.csv, imports).
    `anc` rows keep crop "" (the trainer reads work/crops) unless the crop was imported. Other cards carry the crop path and are checked:
    a row whose field is not in the grid, whose crop is missing or whose label has a character the field cannot emit is left out and counted."""
    cells = {}
    if card == LEGACY:
        for f in ("labels.csv", "labels_new.csv"):
            p = WORK / f
            if p.exists():
                for r in csv.DictReader(open(p, newline="")):
                    cells[(r["page"], r["row"], r["col"])] = (r["label"], "")
        for r, crop in _imported(card):
            cells[(r["page"], r["row"], r["col"])] = (r["label"], crop)
        return [{"page": k[0], "row": k[1], "col": k[2], "label": v[0], "crop": v[1]} for k, v in sorted(cells.items()) if v[0].strip() != ""], {}
    p = CARDS / card / "labels.csv"
    if p.exists():
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                cells[(r["page"], r["row"], r["col"])] = (r["label"], "")
    for r, crop in _imported(card):
        cells[(r["page"], r["row"], r["col"])] = (r["label"], crop)
    m = meta(); allowed = {k: {m["classes"][i] for i in v} for k, v in masks(card).items()}
    bn = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
    rows, bad = [], {"unknown_field": 0, "bad_cell": 0, "missing_crop": 0, "bad_characters": 0}
    for (page, row, col), (label, crop) in sorted(cells.items()):
        if label.strip() == "":
            continue
        if row not in allowed: bad["unknown_field"] += 1; continue
        if not PAGE.match(page) or not COL.match(col): bad["bad_cell"] += 1; continue
        crop = crop or str(crops_dir(card) / row / col / f"{page}.png")
        if not Path(crop).exists(): bad["missing_crop"] += 1; continue
        if any(c not in allowed[row] for c in label.strip().translate(bn)): bad["bad_characters"] += 1; continue
        rows.append({"page": page, "row": row, "col": col, "label": label, "crop": crop})
    return rows, {k: v for k, v in bad.items() if v}


def pack_field(key, kind, rng, classes, blank, reason="not trained yet"):
    e = {"key": key, "kind": kind, "allowed_idx": allowed_idx(kind, classes, blank), "range": list(rng) if rng and kind == "number" else None,
         "min_confidence": 0.5 if kind == "mark" else 0.7, "status": "needs-review", "status_reason": reason}
    if kind == "bp": e["bp"] = {"sys": [70, 220], "dia": [40, 130], "sys_above_dia": True}
    if kind == "mark": e["valid_values"] = ["y", "x", "-"]
    return e


def base_members(card):
    """The pack members a training run starts from. `anc`: the latest anc pack. Other cards: built from the template as it is now (template
    image, grid, field kinds), numbered after the card's latest pack, so a grid or field edit reaches the next pack. The reader member is
    empty until the trainer fills it. Match and preprocessing settings are the ANC pack's; they have not been tuned on other cards."""
    import cardpack as cp
    if card == LEGACY:
        return cp.read(max(PACKS.glob("anc_v*.cardpack"), key=lambda p: int(p.stem.split("_v")[1])))
    import cv2
    pid = pack_id(card)
    if not PACK_ID.match(pid):
        raise ValueError(f"card name {card!r} cannot be a pack id (lower-case letters, digits, _ or -, starting with a letter or digit, max 32)")
    d = template_dir(card); m = meta()
    gray = cv2.imread(str(d / "template.png"), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("template image is missing or unreadable")
    th, tw = gray.shape
    if json.loads((d / "grid.json").read_text())["image_size"] != [tw, th]:
        raise ValueError(f"grid is for a different image size than the template ({[tw, th]}): save the grid again")
    grid, scale = pack_grid(card)
    fld = [pack_field(f["key"], f["kind"], f["range"], m["classes"], m["blank_idx"]) for f in fields(card)]
    mem = {"template.png": cv2.imencode(".png", gray)[1].tobytes(), "grid.json": json.dumps(grid, indent=1).encode(),
           "fields.json": json.dumps({"schema": "cardpack-fields/1", "classes": m["classes"], "blank_idx": m["blank_idx"], "fields": fld}, indent=1, ensure_ascii=False).encode(),
           "reader.onnx": b"",
           "fallback.json": json.dumps({"engine": "qwen3-vl", "when": "flagged", "image_px": 256, "prompt": "short", "apply_field_checks": True}, indent=1).encode()}
    man = {"format": "cardpack/1", "id": pid, "version": (pack_versions(card) or [0])[-1], "name": card, "created": time.strftime("%Y-%m-%d"),
           "template": {"file": "template.png", "width": tw, "height": th, "grid_scale": scale},
           "match": {"method": "sift-ransac", "detect_long_side": 1600, "ratio": 0.75, "ransac_px": 4.0, "min_matches": 10, "min_inliers": 40},
           "grid": {"file": "grid.json", "type": "ruled-rows-v1" if grid.get("axis") == "columns" else "ruled-columns-v1", "cell_margin": CELL_MARGIN},
           "preprocess": {"flatten_kernel": 61, "remove_lines": True, "ink_gate": dict(INK_GATE)},
           "reader": {"file": "reader.onnx", "input": {"name": "image", "height": m["img_h"], "width": m["img_w"], "mean": 0.5, "std": 0.5},
                      "output": {"name": "logits", "layout": "TBV"}, "decode": "ctc-greedy-masked", "confidence": "min-char-softmax"},
           "fields": "fields.json", "fallback": "fallback.json", "metrics": {"heldout": None}, "files": {}}
    mem["manifest.json"] = json.dumps(man, indent=1).encode()
    return mem


def describe(card):
    """One row for the card list: what exists for this card and, if it cannot be trained yet, why."""
    out = {"id": card, "pack_id": pack_id(card), "source": "legacy" if card == LEGACY else "template", "template_version": None, "grid": None,
           "labels": {"cells": 0, "pages": 0}, "label_problems": {}, "pack_versions": [], "trainable": False, "blocker": ""}
    try:
        out["pack_versions"] = pack_versions(card)
        if card != LEGACY:
            v = template_version(card)
            if v is None:
                out["blocker"] = "no grid saved for this template yet (Cards tab, Grid)"; return out
            g = json.loads((TEMPLATES / card / f"v{v}" / "grid.json").read_text())
            out["template_version"] = v; out["grid"] = grid_shape(g)
            if not PACK_ID.match(pack_id(card)):
                out["blocker"] = "the template name cannot be used as a pack id"; return out
        elif not out["pack_versions"]:
            out["blocker"] = "no anc pack in cardpack/out to start from"; return out
        rows, problems = labels(card)
        pages = len({r["page"] for r in rows})
        out["labels"] = {"cells": len(rows), "pages": pages}; out["label_problems"] = problems
        if pages < MIN_PAGES:
            out["blocker"] = f"{pages} labeled page(s); training needs at least {MIN_PAGES}"; return out
        out["trainable"] = True
    except (ValueError, OSError, KeyError) as e:
        out["blocker"] = f"{type(e).__name__}: {e}"
    return out


def list_cards():
    names = sorted(d.name for d in TEMPLATES.glob("*") if d.is_dir() and NAME.match(d.name) and d.name != LEGACY) if TEMPLATES.exists() else []
    return [describe(LEGACY)] + [describe(n) for n in names]


def exists(card):
    return card == LEGACY or (bool(NAME.match(card)) and (TEMPLATES / card).is_dir())
