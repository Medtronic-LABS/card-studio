"""A small, fully synthetic card pack for tests and demos: a drawn template, a made-up grid, the real field schema from model_meta.json, and a
placeholder reader. No patient data, no trained weights. The placeholder is NOT a valid ONNX model, so validate it with check_reader=False."""
import json, time
from pathlib import Path
import cv2, numpy as np
import cardpack as cp

META = cp.HERE.parent / "model_meta.json"
ROWS = ["pulse", "weight_kg", "edema", "bp"]
KIND = {"numeric": "number", "date": "date", "bp": "bp", "mark": "mark", "text": "text"}
PLACEHOLDER_READER = b"synthetic-placeholder-reader: not an ONNX model"


def build_members():
    meta = json.loads(META.read_text())
    tw, th, scale = 400, 300, 2.0
    img = np.full((th, tw), 245, np.uint8)
    for x in (50, 150, 250, 350): cv2.line(img, (x, 50), (x, 250), 40, 2)
    for y in (50, 100, 150, 200, 250): cv2.line(img, (50, y), (350, y), 40, 2)
    tmpl = cv2.imencode(".png", img)[1].tobytes()
    W, H = int(tw * scale), int(th * scale)
    rl = [100, 200, 300, 400, 500]
    grid = {"image_size": [W, H], "rows": ROWS, "y_ref_top": 100.0, "y_ref_bot": 500.0, "col_lines_top": [100, 300, 500, 700],
            "col_lines_bot": [102, 302, 502, 702], "row_lines_by_col": [list(rl), [v + 1 for v in rl], [v + 2 for v in rl]]}
    fields = []
    for key in ROWS:
        f = meta["fields"][key]; kind = KIND[f["type"]]
        e = {"key": key, "kind": kind, "allowed_idx": f["allowed_idx"], "range": [30, 150] if kind == "number" else None,
             "min_confidence": 0.5 if kind == "mark" else 0.7, "status": "needs-review", "status_reason": "synthetic pack: no held-out evidence"}
        if kind == "bp": e["bp"] = {"sys": [70, 220], "dia": [40, 130], "sys_above_dia": True}
        if kind == "mark": e["valid_values"] = ["y", "x", "-"]
        fields.append(e)
    members = {"template.png": tmpl, "grid.json": json.dumps(grid, indent=1).encode(),
               "fields.json": json.dumps({"schema": "cardpack-fields/1", "classes": meta["classes"], "blank_idx": meta["blank_idx"], "fields": fields}, indent=1, ensure_ascii=False).encode(),
               "reader.onnx": PLACEHOLDER_READER,
               "fallback.json": json.dumps({"engine": "qwen3-vl", "when": "flagged", "image_px": 256, "prompt": "short", "apply_field_checks": True}, indent=1).encode()}
    manifest = {"format": "cardpack/1", "id": "synthetic", "version": 1, "name": "Synthetic demo card", "created": time.strftime("%Y-%m-%d"),
                "template": {"file": "template.png", "width": tw, "height": th, "grid_scale": scale},
                "match": {"method": "sift-ransac", "detect_long_side": 1600, "ratio": 0.75, "ransac_px": 4.0, "min_matches": 10, "min_inliers": 40},
                "grid": {"file": "grid.json", "type": "ruled-columns-v1", "cell_margin": 12},
                "preprocess": {"flatten_kernel": 61, "remove_lines": True, "ink_gate": {"dark_below": 110, "min_dark_pixels": 40, "min_dark_fraction": 0.006}},
                "reader": {"file": "reader.onnx", "input": {"name": "image", "height": meta["img_h"], "width": meta["img_w"], "mean": 0.5, "std": 0.5},
                           "output": {"name": "logits", "layout": "TBV"}, "decode": "ctc-greedy-masked", "confidence": "min-char-softmax"},
                "fields": "fields.json", "fallback": "fallback.json",
                "metrics": {"heldout": None, "note": "Synthetic pack for tests."}, "files": {n: cp.sha256(members[n]) for n in cp.HASHED}}
    members["manifest.json"] = json.dumps(manifest, indent=1).encode()
    return members
