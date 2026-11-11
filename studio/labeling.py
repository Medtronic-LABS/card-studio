"""Labeling for a card's cut cells, and the field kinds a label is checked against. Local only.

scans.py cuts every cell of every filled scan and says which hold ink. A person then types what each inked cell says. Labels go to
studio/cards/<card>/labels.csv (patient data, gitignored): one row per cell with the columns page, row (field), col (visit), label, note.
    label      what is written, with Bangla digits stored as 0-9 (the reader's classes); empty when there is nothing to train on
    note       "" for a normal label, "blank" (nothing is written: the ink gate was wrong) or "unreadable" (written, but nobody can say what)
    how        where the label came from: "typed" (no AI suggestion existed), "accepted" (the person kept the AI suggestion), "corrected"
               (the person changed it) or "labelstudio"
    suggested  what the AI had suggested for the cell at that moment (suggest.py), kept so its accuracy on this card can be counted
A cell that has a row here is done and leaves the queue; only rows with a label are trained on (cards.labels).

A label may only use characters its field can emit (cards.KIND_CHARS), because the reader could never produce the others; that is refused.
A label that is allowed but looks wrong for the kind (a date without a month, a weight outside the range) is saved with a warning.
Field kinds are in studio/cards/<card>/fields.json; changing a kind does not rewrite labels that were saved under the old one, but
`overview` counts the ones the new kind no longer allows.
"""
import csv, json, re, threading

import cards

COLS = ["page", "row", "col", "label", "note", "how", "suggested"]
SURE = 0.8          # a suggestion at least this confident, from a reader trained on this card, is put in the box to confirm; otherwise it is only shown
NOTES = ("", "blank", "unreadable")
MAX_LABEL = 40
BN2EN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
SHAPE = {"number": (re.compile(r"^\d+(\.\d+)?$"), "a number like 57 or 9.6"),
         "date": (re.compile(r"^\d{1,2}[./]\d{1,2}[./]\d{2}(\d{2})?$"), "day, month, year like 22/2/26"),
         "bp": (re.compile(r"^\d{2,3}/\d{2,3}$"), "systolic/diastolic like 120/80"),
         "mark": (re.compile(r"^[yx-]$"), "y (tick), x (cross) or - (dash)")}
_lock = threading.Lock()


def _card(card):
    if card == cards.LEGACY or not cards.exists(card) or cards.template_version(card) is None: raise KeyError("no such card")
    return card


def _read(card):
    p = cards.CARDS / card / "labels.csv"
    if not p.exists(): return {}
    with open(p, newline="") as fh:
        return {(r["page"], r["row"], r["col"]): {"label": r.get("label", ""), "note": r.get("note") or "", "how": r.get("how") or "", "suggested": r.get("suggested") or ""}
                for r in csv.DictReader(fh)}


def _write(card, rows):
    d = cards.CARDS / card; d.mkdir(parents=True, exist_ok=True); tmp = d / "labels.csv.tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, COLS); w.writeheader()
        w.writerows({"page": k[0], "row": k[1], "col": k[2], **v} for k, v in sorted(rows.items()))
    tmp.replace(d / "labels.csv")


def _cells(card):
    """Cells scans.py cut for this card: {(page, field, visit): has ink}."""
    p = cards.CARDS / card / "cells.csv"
    if not p.exists(): return {}
    with open(p, newline="") as fh:
        return {(r["page"], r["row"], r["col"]): r["ink"] == "1" for r in csv.DictReader(fh)}


def normalize(label):
    return " ".join(str(label).translate(BN2EN).split())


def check(field, label):
    """-> (error or None, warning or None) for a label of this field ({key, kind, range})."""
    m = cards.meta(); allowed = {m["classes"][i] for i in cards.allowed_idx(field["kind"], m["classes"], m["blank_idx"])} - {m["classes"][m["blank_idx"]]}
    bad = sorted({c for c in label if c not in allowed})
    if bad:
        return f"{field['key']} is a {field['kind']} field and cannot hold {' '.join(repr(c) for c in bad)}: fix the label, or change the field's kind", None
    shape = SHAPE.get(field["kind"])
    if shape and not shape[0].match(label) and not (field["kind"] != "mark" and label == "-"):
        return None, f"does not look like {shape[1]}"
    if field["kind"] == "number" and field.get("range") and shape[0].match(label) and not field["range"][0] <= float(label) <= field["range"][1]:
        return None, f"outside the expected range {field['range'][0]:g} to {field['range'][1]:g}"
    if field["kind"] == "date" and shape[0].match(label):
        d, mo = (int(x) for x in re.split(r"[./]", label)[:2])
        if not (1 <= d <= 31 and 1 <= mo <= 12): return None, "not a real day and month"
    return None, None


def overview(card):
    """Per field: its kind, how many inked cells there are and how far labeling has got; plus what training still needs."""
    import suggest
    _card(card); cells = _cells(card); done = _read(card); fl = cards.fields(card); out = []; sug = suggest.read(card)
    for f in fl:
        mine = [k for k, ink in cells.items() if k[1] == f["key"]]
        rows = [done[k] for k in mine if k in done]; labeled = [r for r in rows if r["label"]]
        out.append({**f, "cells": len(mine), "inked": sum(cells[k] for k in mine), "labeled": len(labeled),
                    "ai_right": sum(r["how"] == "accepted" for r in rows), "ai_wrong": sum(r["how"] == "corrected" for r in rows),
                    "ai_waiting": sum(1 for k in mine if cells[k] and k not in done and k in sug),
                    "blank": sum(r["note"] == "blank" for r in rows), "unreadable": sum(r["note"] == "unreadable" for r in rows),
                    "todo": sum(1 for k in mine if cells[k] and k not in done), "not_allowed_now": sum(check(f, r["label"])[0] is not None for r in labeled)})
    c = cards.describe(card)
    return {"card": card, "fields": out, "kinds": list(cards.KINDS), "hints": {k: v[1] for k, v in SHAPE.items()}, "labeled_cells": c["labels"]["cells"], "labeled_pages": c["labels"]["pages"],
            "min_pages": cards.MIN_PAGES, "trainable": c["trainable"], "blocker": c["blocker"], "todo": sum(f["todo"] for f in out),
            "ai": {**suggest.status(card), "sure": SURE, "right": sum(f["ai_right"] for f in out), "wrong": sum(f["ai_wrong"] for f in out)}}


def queue(card, field=None, show="todo", limit=60, after=None, order_by="reading"):
    """Cells to label. show: `todo` = inked and not done, `done` = already has a row (to review), `uninked` = cells the ink gate called
    empty (to catch what it missed), `disagree` = done cells whose label differs from what the AI suggests now (to catch typos, and AI errors
    that were accepted). order_by: `reading` (field, page, visit), `unsure` (the AI's least confident first: the labels that teach the reader
    most), `sure` (most confident first: the quickest to confirm). `after` = "page|field|visit" of the last cell already shown (reading order)."""
    import suggest
    _card(card); cells = _cells(card); done = _read(card); order = {f["key"]: i for i, f in enumerate(cards.fields(card))}; sug = suggest.read(card)
    own = cards.pack_id(card) + " v"       # a suggestion from this card's own reader; one borrowed from another card's reader is never pre-filled
    if field is not None and field not in order: raise KeyError("no such field")
    if show not in ("todo", "done", "uninked", "disagree"): raise ValueError("show must be todo, done, uninked or disagree")
    if order_by not in ("reading", "unsure", "sure"): raise ValueError("order_by must be reading, unsure or sure")
    differs = lambda k: k in done and k in sug and done[k]["label"] != "" and sug[k]["text"] != done[k]["label"]
    keys = [k for k, ink in cells.items() if k[1] in order and (field is None or k[1] == field)
            and ((show == "todo" and ink and k not in done) or (show == "done" and k in done) or (show == "uninked" and not ink and k not in done) or (show == "disagree" and differs(k)))]
    sk = lambda k: (order[k[1]], k[0], int(k[2][1:]))
    keys.sort(key=sk)
    if order_by != "reading":       # cells the AI said nothing about go last either way
        conf = lambda k: sug[k]["confidence"] if k in sug else None
        keys.sort(key=lambda k: (conf(k) is None, (conf(k) or 0) * (1 if order_by == "unsure" else -1)))
    elif after:
        a = tuple(after.split("|"))
        if len(a) == 3 and a[1] in order and cards.COL.match(a[2]): keys = [k for k in keys if sk(k) > (order[a[1]], a[0], int(a[2][1:]))]
    return {"card": card, "total": len(keys), "cells": [{"page": k[0], "field": k[1], "visit": k[2], **done.get(k, {"label": "", "note": "", "how": "", "suggested": ""}), "done": k in done,
                                                         "suggestion": ({"text": sug[k]["text"], "confidence": sug[k]["confidence"], "source": sug[k]["source"],
                                                                         "sure": sug[k]["confidence"] >= SURE and sug[k]["source"].startswith(own)} if k in sug else None)}
                                                        for k in keys[:max(1, min(int(limit), 500))]]}


def save(card, page, field, visit, label="", note="", confirm=False, how=None):
    """Record what one cell says. note "blank" / "unreadable" stores no label. -> {saved, label, note, warning}.
    A label that draws a warning is not saved until it is sent again with confirm (a typo is the usual reason for a warning)."""
    _card(card); fl = {f["key"]: f for f in cards.fields(card)}
    if field not in fl: raise KeyError("no such field")
    if (page, field, visit) not in _cells(card): raise KeyError("no such cell")
    if note not in NOTES: raise ValueError("note must be empty, blank or unreadable")
    label = "" if note else normalize(label)
    if not note and not label: raise ValueError("type what the cell says, or mark it blank or unreadable")
    if len(label) > MAX_LABEL: raise ValueError(f"label is longer than {MAX_LABEL} characters")
    err, warn = check(fl[field], label) if label else (None, None)
    if err: raise ValueError(err)
    if warn and not confirm: return {"saved": False, "label": label, "note": note, "warning": warn}
    import suggest
    s = suggest.read(card).get((page, field, visit))        # what the AI said about this cell, if anything: kept beside the label
    if how is None: how = "typed" if s is None else ("accepted" if (s["text"] == label and not note) or (s["text"] == "" and note == "blank") else "corrected")
    with _lock:
        rows = _read(card); rows[(page, field, visit)] = {"label": label, "note": note, "how": how, "suggested": s["text"] if s else ""}; _write(card, rows)
    return {"saved": True, "label": label, "note": note, "warning": warn, "how": how}


def clear(card, page, field, visit):
    """Forget what was recorded for a cell, so it comes back into the queue."""
    _card(card)
    with _lock:
        rows = _read(card)
        if rows.pop((page, field, visit), None) is None: raise KeyError("that cell has no label")
        _write(card, rows)
    return {"cleared": True}


def set_field(card, key, kind, rng=None):
    """Set one field's kind (and, for a number, its expected range) in fields.json."""
    _card(card)
    if key not in [f["key"] for f in cards.fields(card)]: raise KeyError("no such field")
    if kind not in cards.KINDS: raise ValueError(f"kind must be one of {', '.join(cards.KINDS)}")
    if rng is not None:
        if kind != "number" or not isinstance(rng, list) or len(rng) != 2 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in rng) or not rng[0] < rng[1]:
            raise ValueError("a range is [low, high] with low below high, and only for a number field")
    with _lock:
        p = cards.CARDS / card / "fields.json"; p.parent.mkdir(parents=True, exist_ok=True)
        spec = json.loads(p.read_text()).get("fields", {}) if p.exists() else {}
        spec[key] = {"kind": kind, **({"range": rng} if rng is not None else {})}
        p.write_text(json.dumps({"fields": spec}, indent=1))
    return {"key": key, "kind": kind, "range": rng}


def crop_path(card, field, visit, page):
    _card(card)
    if not (cards.PAGE.match(page) and cards.COL.match(visit) and re.fullmatch(r"[A-Za-z0-9_]{1,40}", field)): raise KeyError("no such cell")
    p = cards.crops_dir(card) / field / visit / f"{page}.png"
    if not p.exists(): raise KeyError("no such cell")
    return p
