"""AI suggestions for labeling: a trained reader reads each inked cell first, a person confirms or corrects. Local only.

The suggester is a reader this Studio already has (a pack's reader.onnx): the card's own newest pack once the card has been trained, and
until then the newest pack of another card. All cards share one class list, so another card's reader can be asked for "a number" or "a
date" on this card; it has never seen this card's writers or print, so expect it to be wrong often, and it is not asked about free-text
fields at all. Suggestions are written to studio/cards/<card>/suggestions.csv (page, field, visit, text, confidence, source).

A suggestion is never a label. labeling.save records, for every label, whether a suggestion existed and whether the person kept it
(`accepted`) or changed it (`corrected`), so the suggester's real accuracy on this card is measured as labeling goes on, and labels a
person merely agreed with can be told apart later. Cells are read exactly as the trainer reads them: the stored gray crop, scaled to the
reader's height, padded white, no line removal.
"""
import csv, time
from pathlib import Path
import cv2, numpy as np

import cards, cleanup

COLS = ["page", "row", "col", "text", "confidence", "source"]
BATCH = 64


def source_pack(card):
    """(pack path, own) for the reader that suggests on this card: its own newest pack, else the newest pack of any other card. None if there is none."""
    own = cards.pack_versions(card)
    if own: return cards.PACKS / f"{cards.pack_id(card)}_v{own[-1]}.cardpack", True
    others = [p for p in cards.PACKS.glob("*.cardpack") if "_v" in p.stem and p.stem.rsplit("_v", 1)[1].isdigit()]
    if not others: return None, False
    return max(others, key=lambda p: (p.stat().st_mtime, p.name)), False


def _input(path, h, w):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None: return None
    nw = min(w, max(1, int(im.shape[1] * h / im.shape[0]))); canvas = np.full((h, w), 255, np.uint8)
    canvas[:, :nw] = cv2.resize(im, (nw, h), interpolation=cv2.INTER_AREA)
    return (canvas.astype(np.float32) / 255.0 - 0.5) / 0.5


def _decode(logits, allowed, classes, blank):
    """Masked greedy CTC; confidence = the lowest probability among the characters read (0 when nothing was read). logits: [T, classes]."""
    ok = np.zeros(len(classes), bool); ok[allowed] = True
    z = np.where(ok[None, :], logits, -np.inf); best = z.argmax(1); out, confs, prev = [], [], -1
    for t, b in enumerate(best):
        if b != blank and b != prev:
            out.append(classes[int(b)]); confs.append(float(1.0 / np.exp(z[t][ok] - z[t, b]).sum()))
        prev = b
    return "".join(out), (min(confs) if confs else 0.0)


def read(card):
    p = cards.CARDS / card / "suggestions.csv"
    if not p.exists(): return {}
    with open(p, newline="") as fh:
        return {(r["page"], r["row"], r["col"]): {"text": r["text"], "confidence": float(r["confidence"] or 0), "source": r["source"]} for r in csv.DictReader(fh)}


def status(card):
    """Which reader would suggest, what is stored, and whether the stored suggestions are out of date (a newer reader, or new cells)."""
    pack, own = source_pack(card); d = cards.CARDS / card; f = d / "suggestions.csv"; have = read(card)
    name = pack.stem.replace("_v", " v") if pack else None
    stored = next(iter(have.values()))["source"] if have else None
    newer_cells = f.exists() and (d / "cells.csv").exists() and (d / "cells.csv").stat().st_mtime > f.stat().st_mtime
    return {"available": pack is not None, "reader": name, "own_reader": own, "count": len(have), "stored_reader": stored,
            "stale": pack is not None and (not f.exists() or stored != name or newer_cells)}


def make(card):
    """Read every inked cell of the card with the suggesting reader and store what it says. -> {reader, own_reader, cells, skipped_text_fields, seconds}."""
    import labeling
    labeling._card(card); pack_path, own = source_pack(card)
    if pack_path is None: raise ValueError("no trained reader in cardpack/out yet: label some cells by hand and train once, then suggestions can start")
    pk = cleanup.load_pack(pack_path); r = pk.man["reader"]["input"]; h, w = r["height"], r["width"]; t0 = time.time()
    masks = cards.masks(card); kinds = {f["key"]: f["kind"] for f in cards.fields(card)}
    layouts = {f["key"]: f["layout"] for f in cards.fields(card)}
    todo = [k for k, ink in labeling._cells(card).items() if ink and k[1] in masks and (own or kinds[k[1]] != "text") and layouts[k[1]] == "single"]       # two-line cells are not suggested: the whole-cell reader cannot read them
    rows = []
    for i in range(0, len(todo), BATCH):
        chunk = [(k, _input(cards.crops_dir(card) / k[1] / k[2] / f"{k[0]}.png", h, w)) for k in todo[i:i + BATCH]]
        chunk = [(k, x) for k, x in chunk if x is not None]
        if not chunk: continue
        lg = pk.session.run([pk.out_name], {pk.in_name: np.stack([x for _, x in chunk])[:, None]})[0]        # [T, B, classes]
        for j, (k, _) in enumerate(chunk):
            text, conf = _decode(lg[:, j], masks[k[1]], pk.classes, pk.blank)
            rows.append({"page": k[0], "row": k[1], "col": k[2], "text": text, "confidence": round(conf, 4), "source": pk.name})
    d = cards.CARDS / card; tmp = d / "suggestions.csv.tmp"
    with open(tmp, "w", newline="") as fh:
        wr = csv.DictWriter(fh, COLS); wr.writeheader(); wr.writerows(rows)
    tmp.replace(d / "suggestions.csv")
    return {"card": card, "reader": pk.name, "own_reader": own, "cells": len(rows),
            "skipped_text_fields": sorted(k for k, v in kinds.items() if v == "text") if not own else [], "skipped_two_line_fields": sorted(k for k, v in layouts.items() if v != "single"), "seconds": round(time.time() - t0, 1)}
