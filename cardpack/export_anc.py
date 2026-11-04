"""Export the CURRENT ANC reader (as shipped in the Flutter app's assets) as anc_v3.cardpack.
Nothing is retrained and nothing is read from work/: the app assets are byte-identical to work/ (checked by sha256 in the test)."""
import json, sys, time
from pathlib import Path
import cardpack as cp

APP = cp.HERE.parent.parent / "leapwell_ocr_demo" / "assets" / "model"
RANGES = {"weight_kg": (30, 150), "pulse": (40, 180), "fundal_height_cm": (5, 45), "hemoglobin": (4, 18), "glucose": (2, 30),
          "gestational_month": (1, 10), "height_cm": (120, 200), "temp_f": (94, 106), "fetal_movement": (0, 100)}
KIND = {"numeric": "number", "date": "date", "bp": "bp", "mark": "mark", "text": "text"}


def build_members():
    meta = json.loads((APP / "model_meta.json").read_text())
    grid_b = (APP / "grid.json").read_bytes()
    tmpl_b = (APP / "anc_template_half.png").read_bytes()
    onnx_b = (APP / "crnn.onnx").read_bytes()
    import cv2, numpy as np
    th, tw = cv2.imdecode(np.frombuffer(tmpl_b, np.uint8), 0).shape
    grid = json.loads(grid_b)
    fields = []
    for key in grid["rows"]:
        f = meta["fields"][key]; kind = KIND[f["type"]]
        e = {"key": key, "kind": kind, "allowed_idx": f["allowed_idx"], "range": list(RANGES[key]) if key in RANGES else None,
             "min_confidence": 0.5 if kind == "mark" else 0.7,
             "status": "needs-review", "status_reason": "no held-out evaluation exists for this exact reader (Phase A evaluated a retrained model, not this one)"}
        if kind == "bp":
            e["bp"] = {"sys": [70, 220], "dia": [40, 130], "sys_above_dia": True}
        if kind == "mark":
            e["valid_values"] = ["y", "x", "-"]
        if kind == "text":
            e["status_reason"] = "free text: reader has no training data for this field"
        fields.append(e)
    fields_b = json.dumps({"schema": "cardpack-fields/1", "classes": meta["classes"], "blank_idx": meta["blank_idx"], "fields": fields}, indent=1, ensure_ascii=False).encode()
    fallback_b = json.dumps({"engine": "qwen3-vl", "when": "flagged", "image_px": 256, "prompt": "short", "apply_field_checks": True}, indent=1).encode()
    members = {"template.png": tmpl_b, "grid.json": grid_b, "fields.json": fields_b, "reader.onnx": onnx_b, "fallback.json": fallback_b}
    manifest = {
        "format": "cardpack/1", "id": "anc", "version": 3, "name": "ANC card (Bangla handwriting)", "created": time.strftime("%Y-%m-%d"),
        "template": {"file": "template.png", "width": tw, "height": th, "grid_scale": grid["image_size"][0] / tw},
        "match": {"method": "sift-ransac", "detect_long_side": 1600, "ratio": 0.75, "ransac_px": 4.0, "min_matches": 10, "min_inliers": 40},
        "grid": {"file": "grid.json", "type": "ruled-columns-v1", "cell_margin": 12},
        "preprocess": {"flatten_kernel": 61, "remove_lines": True, "ink_gate": {"dark_below": 110, "min_dark_pixels": 40, "min_dark_fraction": 0.006}},
        "reader": {"file": "reader.onnx", "input": {"name": "image", "height": meta["img_h"], "width": meta["img_w"], "mean": 0.5, "std": 0.5},
                   "output": {"name": "logits", "layout": "TBV"}, "decode": "ctc-greedy-masked", "confidence": "min-char-softmax"},
        "fields": "fields.json", "fallback": "fallback.json",
        "metrics": {"heldout": None, "note": "Current shipped reader. Trained on all labeled pages, so no honest held-out number exists for it. Match values mirror the app's current hardcoded constants."},
        "files": {n: cp.sha256(members[n]) for n in cp.HASHED},
    }
    members["manifest.json"] = json.dumps(manifest, indent=1).encode()
    return members


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else cp.HERE / "out" / "anc_v3.cardpack"
    out.parent.mkdir(parents=True, exist_ok=True)
    cp.build(build_members(), out)
    print(f"wrote {out} ({out.stat().st_size} bytes)")
