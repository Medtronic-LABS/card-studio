"""Ablation: the app erases printed ruled lines from each cell crop before reading. What does that cost and what does it buy?
Reader: the anc v3 pack's (trained on ALL labeled pages), so accuracy on labeled cells here is TRAINING-WRITER accuracy, not held-out.
It still isolates the effect of the preprocessing step, because only the step changes. Blank-cell false reads use unlabeled cells with almost no ink.
Writes studio/experiments/lineremoval-ablation.json. Reads work/, writes nothing there."""
import csv, collections, sys
from pathlib import Path
import cv2, numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "studio"))
import cleanup as L, experiments as E
from common import normalize_label

pk = next(p for p in L.installed() if p.name == "anc v3"); m = pk.margin
labels = collections.defaultdict(dict); labeled = set()
for r in csv.DictReader(open(HERE.parent / "work" / "labels.csv")):
    if r["label"].strip(): labeled.add((r["page"], r["row"], r["col"])); labels[r["page"]][(r["row"], r["col"])] = normalize_label(r["label"])
ink = {(r["page"], r["row"], r["col"]): float(r["ink_fraction"]) for r in csv.DictReader(open(HERE.parent / "work" / "manifest.csv"))}
blank = collections.defaultdict(set)
for k, v in ink.items():
    if k not in labeled and v < 0.004: blank[k[0]].add((k[1], k[2]))
blank_pages = sorted(blank)[:25]
mc = lambda f: pk.spec[f].get("min_confidence") or 0
def band(crop, ex, ey):
    c = crop.copy(); h, w = c.shape; ex = min(ex, w // 3); ey = min(ey, h // 3); c[:, :ex] = 255; c[:, w - ex:] = 255; c[:ey, :] = 255; c[h - ey:, :] = 255; return c
V = {"app: erase long thin lines (today)": L.remove_lines, "no line removal": lambda c: c, "whiten only the outer border band": lambda c: band(c, m + 4, 12)}
R = {k: collections.Counter() for k in V}; nb = 0
for page in sorted(labels):
    bgr = cv2.imread(str(HERE.parent / "work" / "raw" / f"{page}.png")); H, _, _ = L.align(bgr, pk); warped, cov = L.warp(bgr, H, pk)
    for ri, c, field, crop in L.cut_cells(warped, pk, "none"):
        k = (field, f"c{c + 1}"); is_blank = page in blank_pages and k in blank[page]
        for name, fn in V.items():
            if k in labels[page]:
                px = L.prepare(fn(crop), pk); t, conf = ("", 0.0) if px is None else L.read_cell(pk, px, field)
                R[name]["labeled_n"] += 1; R[name]["labeled_correct"] += t == labels[page][k]
                if field in ("pulse", "weight_kg", "glucose", "hemoglobin", "bp", "fundal_height_cm"): R[name]["num_n"] += 1; R[name]["num_correct"] += t == labels[page][k]
            elif is_blank:
                px = L.prepare(fn(crop), pk); R[name]["blank_n"] += 1
                if px is not None:
                    t, conf = L.read_cell(pk, px, field)
                    if t: R[name]["junk"] += 1; R[name]["silent_junk"] += conf >= mc(field)
rows = [{"label": n, "cells": [{"v": 100 * R[n]["labeled_correct"] / R[n]["labeled_n"]}, {"v": 100 * R[n]["num_correct"] / R[n]["num_n"]},
        {"v": 100 * R[n]["junk"] / R[n]["blank_n"]}, {"v": 100 * R[n]["silent_junk"] / R[n]["blank_n"]}]} for n in V]
E.save({"id": "lineremoval-ablation", "kind": "ablation", "title": "What the app's printed-line erasing costs and buys",
        "question": "The app erases long thin runs from every cell before reading so blank cells stay blank. Does it also erase digit strokes?",
        "data": {"eval": f"{R[next(iter(V))]['labeled_n']} labeled cells on 49 scan pages (TRAINING pages of this reader: training-writer numbers, not held-out) and {R[next(iter(V))]['blank_n']} near-blank cells on 25 pages",
                 "reader": "anc v3 pack reader"},
        "recipe": "same warped scan pages, same reader and ink gate for every variant; only the crop-cleaning step differs",
        "table": {"title": "Reading a scan through each variant", "columns": ["labeled correct %", "number cells correct %", "blank cells read as text %", "of which unflagged %"], "rows": rows},
        "baselines": [], "treatments": [], "test_confirmations": [],
        "notes": ["Labeled-cell accuracy is training-writer accuracy: it shows the preprocessing's effect, not generalisation.",
                  "Number cells are pulse, weight, glucose, hemoglobin, BP, fundal height; they have tall digit strokes. The step erases vertical runs longer than 60% of the cell height.",
                  "Near-blank = unlabeled and under 0.4% ink in work/manifest.csv; some may hold faint writing, so 'junk' is an upper bound.",
                  "Result: erasing lines costs most number reads; not erasing lets junk into blank cells. Retraining on the app's own inputs (experiment apppath-*) is the way out."],
        "source_files": ["work_phaseA/lineremoval_ablation.py", "studio/cleanup.py"]})
for n in V: print(n, dict(R[n]))
