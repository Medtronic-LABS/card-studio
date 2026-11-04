"""Retrain the reader on what the app actually feeds it, and compare on held-out pages.

The app does: align photo to the template, page clean-up, cut cells, erase printed lines, ink gate, fit to 48 high. The Phase A reader
was trained on stored gray crops that skip the line erasing, so the app's input is a shift away from its training data (the lab found
the same reader scoring 60% through the app path against 84% on stored crops, on training pages).

Here: build the app-path input for every labeled cell (studio/cleanup.py is a port of the app's steps), fine-tune with the SAME recipe
(same pretrained weights, light_lr3e-4, same early stopping on the val pages) on train pages only, and read the held-out TEST pages
through the app path. Settings were fixed before this run; nothing is tuned on TEST. Reads TEST once per model.
Run: python apppath_train.py [seeds]          (writes apppath_results.json; never writes to work/)
"""
import csv, json, sys, time
from pathlib import Path
import cv2, numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "studio"))
import common as C, train as T
import cleanup as L, normalize as N

S = json.load(open(HERE / "split.json"))
LABELS = C.read_labels(HERE / "merged_labels.csv")
VARIANTS = ("flatten", "contrast_match")          # the app's current default (v3) and the v5 default; both with line removal
SEEDS = [int(x) for x in sys.argv[1:]] or [0, 1, 2]


def build_inputs(pack, method, with_lines_removed=True):
    """{(page,row,col): 48x192 uint8 or None} exactly as the app would feed the model."""
    out, need = {}, {(r["page"], r["row"], r["col"]) for r in LABELS}
    pages = sorted({k[0] for k in need})
    for page in pages:
        bgr = cv2.imread(str(C.WORK / "raw" / f"{page}.png"))
        H, inl, bad = L.align(bgr, pack)
        assert bad is None and inl >= 40, (page, inl, bad)
        warped, cov = L.warp(bgr, H, pack)
        page_img, _ = N.normalize_page(method, warped, cov, pack.flatten_k, pack.reference)
        for ri, c, field, crop in L.cut_cells(page_img, pack, method):
            key = (page, field, f"c{c + 1}")
            if key in need:
                out[key] = L.prepare(L.remove_lines(crop) if with_lines_removed else crop, pack)
    return out


def exact_cer(model, rows):
    out = C.predict_torch(model, rows)
    hit = [p == r["label"] for (p, _), r in zip(out, rows)]
    err = sum(C.edit_distance(p, r["label"]) for (p, _), r in zip(out, rows)); tot = sum(len(r["label"]) for r in rows)
    k = int(sum(hit)); lo, hi = C.wilson(k, len(rows))
    return {"k": k, "n": len(rows), "exact": k / len(rows), "lo": lo, "hi": hi, "cer": err / tot}


def with_cache(inputs, fn):
    C._cache.clear()
    for k, v in inputs.items():
        if v is not None: C._cache[k] = v
    try:
        return fn()
    finally:
        C._cache.clear()


def main():
    torch.set_num_threads(4)
    pack = next(p for p in L.installed() if p.name == "anc v5")
    pre = torch.load(HERE / "pretrained.pt")
    base = C.CRNN(len(C.CLASSES)); base.load_state_dict(torch.load(HERE / "final_model.pt")); base.eval()
    rows = {k: [r for r in LABELS if r["page"] in set(S[k])] for k in ("train", "val", "test")}
    results = {"cfg": "light_lr3e-4", "seeds": SEEDS, "n": {k: len(v) for k, v in rows.items()}, "variants": {}}
    # reference: the Phase A model on stored crops (the number already in the report) and through the app path
    ref_stored = exact_cer(base, rows["test"])
    print(f"Phase A model, STORED crops, TEST: {ref_stored['k']}/{ref_stored['n']} = {100 * ref_stored['exact']:.1f}% [{100 * ref_stored['lo']:.0f}-{100 * ref_stored['hi']:.0f}] CER {ref_stored['cer']:.3f}", flush=True)
    results["phaseA_stored_crops"] = ref_stored
    for v in VARIANTS:
        t0 = time.time(); inputs = build_inputs(pack, v)
        gated = {k for k, x in inputs.items() if x is None}
        keep = lambda rs: [r for r in rs if (r["page"], r["row"], r["col"]) not in gated]
        tr, va, te = keep(rows["train"]), keep(rows["val"]), rows["test"]
        print(f"\n[{v}] app-path inputs built in {time.time() - t0:.0f}s; blanked by the ink gate: {len(gated)} labeled cells "
              f"(train/val dropped {len(rows['train']) - len(tr)}/{len(rows['val']) - len(va)}; test cells gated count as misses)", flush=True)
        te_k = [r for r in te if (r["page"], r["row"], r["col"]) not in gated]
        res = {"gated": len(gated), "baseline_app_path": None, "retrained": []}
        # the old reader fed through the app path
        res["baseline_app_path"] = with_cache(inputs, lambda: exact_cer(base, te_k)); b = res["baseline_app_path"]
        extra = len(te) - len(te_k)
        print(f"  old reader, app path, TEST: {b['k']}/{len(te)} = {100 * b['k'] / len(te):.1f}%  (gated test cells: {extra})", flush=True)
        for sd in SEEDS:
            def run():
                m, info = T.finetune(pre, tr, va, T.CONFIGS["light_lr3e-4"], seed=sd, log=lambda *_: None)
                return m, info
            m, info = with_cache(inputs, run)
            r = with_cache(inputs, lambda: exact_cer(m, te_k)); r["seed"] = sd; r["best_epoch"] = info["best_epoch"]; r["test_total"] = len(te)
            res["retrained"].append(r)
            torch.save(m.state_dict(), HERE / f"apppath_{v}_seed{sd}.pt")
            print(f"  retrained seed {sd} (best epoch {info['best_epoch']}): TEST {r['k']}/{len(te)} = {100 * r['k'] / len(te):.1f}%  CER {r['cer']:.3f}", flush=True)
        results["variants"][v] = res
        json.dump(results, open(HERE / "apppath_results.json", "w"), indent=1)
    print("\ndone")


if __name__ == "__main__":
    main()
