"""Local layout detection (OpenCV only) and template matching. No network, no OCR engine."""
import cv2, numpy as np

MAX_SIDE = 2000


def _decode(data: bytes):
    """JPEG/PNG/WebP/TIFF via OpenCV; else PDF (page 1), HEIC/HEIF (macOS sips), then Pillow. All local."""
    im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if im is not None:
        return im
    if data[:4] == b"%PDF":
        import pypdfium2 as pdfium
        pil = pdfium.PdfDocument(data)[0].render(scale=2.5).to_pil().convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    if data[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1", b"ftypheif", b"ftyphevc", b"ftypmsf1"):
        import subprocess, tempfile, shutil
        if not shutil.which("sips"):
            raise ValueError("HEIC photo: convert to JPEG first (macOS sips not found)")
        with tempfile.TemporaryDirectory() as d:
            src, dst = f"{d}/in.heic", f"{d}/out.png"
            open(src, "wb").write(data)
            subprocess.run(["sips", "-s", "format", "png", src, "--out", dst], capture_output=True, check=True)
            return cv2.imread(dst, cv2.IMREAD_COLOR)
    try:
        import io
        from PIL import Image
        return cv2.cvtColor(np.array(Image.open(io.BytesIO(data)).convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def load_normalized(data: bytes):
    try:
        im = _decode(data)
    except Exception as e:
        raise ValueError(f"could not convert this file ({type(e).__name__}: {e})")
    if im is None:
        raise ValueError(f"unsupported file ({len(data)} bytes, starts {data[:12].hex()}). Use JPEG, PNG, WebP, TIFF, HEIC or PDF.")
    s = MAX_SIDE / max(im.shape[:2])
    if s < 1:
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return im


def detect_layout(bgr):
    """Find ruled table lines -> cells -> rows/cols; small square cells -> 'mark' (checkbox).
    Returns {"fields": [...], "stats": {...}}. Field = {id,name,kind,section,x,y,w,h,range,charset}."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, W // 25), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, H // 25)))
    lines = cv2.bitwise_or(cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk), cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk))
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
    free = cv2.bitwise_not(lines)
    n, _, st, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    cells = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w >= W * 0.97 or h >= H * 0.97:  # page background
            continue
        if w < 10 or h < 10 or a < 0.5 * w * h:
            continue
        cells.append((int(x), int(y), int(w), int(h)))
    cells.sort(key=lambda c: (c[1], c[0]))
    # group into rows by y centre, then number columns by x within the row
    rows, tol = [], max(8, int(np.median([c[3] for c in cells]) * 0.5)) if cells else 8
    for c in cells:
        cy = c[1] + c[3] / 2
        for r in rows:
            if abs(r["cy"] - cy) <= tol:
                r["cells"].append(c); r["cy"] = (r["cy"] * (len(r["cells"]) - 1) + cy) / len(r["cells"]); break
        else:
            rows.append({"cy": cy, "cells": [c]})
    rows.sort(key=lambda r: r["cy"])
    fields = []
    for ri, r in enumerate(rows, 1):
        for ci, (x, y, w, h) in enumerate(sorted(r["cells"]), 1):
            box = 0.7 <= w / h <= 1.4 and 12 <= w <= 70
            fields.append({"id": f"f{len(fields) + 1}", "name": f"r{ri}_c{ci}", "kind": "mark" if box else "text",
                           "section": "", "x": x, "y": y, "w": w, "h": h, "range": "", "charset": ""})
    # hollow square outlines inside cells (printed checkboxes): ink with ruled lines removed
    ink = cv2.bitwise_and(bw, free)
    cnts, hier = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cell_boxes = [(f["x"], f["y"], f["w"], f["h"], f["name"]) for f in fields]
    for i, c in enumerate(cnts):
        if hier[0][i][3] != -1 or hier[0][i][2] == -1:  # need an outer contour that has a hole
            continue
        x, y, w, h = cv2.boundingRect(c)
        if not (0.75 <= w / h <= 1.33 and 14 <= w <= 70) or len(cv2.approxPolyDP(c, 0.06 * cv2.arcLength(c, True), True)) != 4:
            continue
        owner = next((n for cx, cy, cw, ch, n in cell_boxes if cx <= x and cy <= y and x + w <= cx + cw and y + h <= cy + ch and w < cw * 0.8), None)
        if owner:
            fields.append({"id": f"f{len(fields) + 1}", "name": f"{owner}_box", "kind": "mark", "section": "",
                           "x": int(x), "y": int(y), "w": int(w), "h": int(h), "range": "", "charset": "y x -"})
    unique_names(fields)
    return {"fields": fields, "stats": {"cells": len(fields), "rows": len(rows),
                                         "marks": sum(f["kind"] == "mark" for f in fields), "image": [W, H]}}


def unique_names(fields):
    """Give every field a unique name (a cell with two checkboxes would otherwise have two `<cell>_box`). Returns how many were renamed."""
    seen, renamed = set(), 0
    for f in fields:
        n, k = f["name"], 2
        while f["name"] in seen:
            f["name"] = f"{n}_{k}"; k += 1
        if f["name"] != n: renamed += 1
        seen.add(f["name"])
    return renamed


def _sift(gray):
    s = 0.5
    k, d = cv2.SIFT_create(nfeatures=3000).detectAndCompute(cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), None)
    return k, d


def match_score(bgr, tmpl_bgr):
    """0..1 = RANSAC inliers / 150 (the app needs >=100 inliers to align, so >=0.67 is a usable match)."""
    kq, dq = _sift(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    kt, dt = _sift(cv2.cvtColor(tmpl_bgr, cv2.COLOR_BGR2GRAY))
    if dq is None or dt is None or len(kq) < 10 or len(kt) < 10:
        return 0.0
    good = [m[0] for m in cv2.BFMatcher(cv2.NORM_L2).knnMatch(dq, dt, k=2) if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < 10:
        return 0.0
    src = np.float32([kq[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
    dst = np.float32([kt[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
    Hm, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    return 0.0 if Hm is None else min(1.0, int(mask.sum()) / 150)
