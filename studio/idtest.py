"""Template-identification test set + matcher comparison. Local only; reads work/ (never writes there).

  python idtest.py build    # make the labeled photo set in runs/idtest/images (patient pages: gitignored)
  python idtest.py score    # score every image with every matcher -> runs/idtest/scores.json
  python idtest.py report   # runs/idtest/report.html (self-contained)

Matchers
  baseline : SIFT + ratio 0.75 + RANSAC 4px, accept at >= 40 inliers (what the app / pack manifest does today)
  hardened : baseline + homography sanity (convex, not mirrored, plausible area) + inlier coverage of the template
  lines    : the same SIFT/RANSAC, but on the ruled-line layout only (handwriting removed)
  two-template decision : accept a type only if its score beats the runner-up by a margin, else 'ambiguous'
Rules for 'hardened' were fixed before looking at scores (see HARD_* below); only the baseline floor sweep is a sweep.
"""
import base64, json, random, sys, time, zipfile
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "idtest"
IMG = RUN / "images"
WORK = ROOT.parent / "work"
PACK = ROOT.parent / "cardpack" / "out" / "anc_v4.cardpack"
SEED = 20261104

FLOOR = 40            # pack manifest match.min_inliers
LINE_FLOOR = 25       # fixed a priori: lines carry fewer features than the full card
HARD_AREA = (0.2, 10.0)   # mapped query quad area / template area
HARD_COVER = 0.30         # convex hull of inliers / template area
MARGIN = 1.5              # winner must beat runner-up by this factor


# ------------------------------------------------------------------ build
def phone(im, rng, level):
    """Fake a hand-held photo: perspective + rotation + uneven light + blur + resize + JPEG."""
    h, w = im.shape[:2]
    s = rng.uniform(.72, .92)
    cx, cy = w / 2, h / 2
    base = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = (base - [cx, cy]) * s
    ang = np.deg2rad(rng.uniform(-1, 1) * 6 * level)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
    dst = dst @ R.T + [cx, cy]
    j = level * 0.035 * min(h, w)
    dst = dst + np.float32([[rng.uniform(-j, j), rng.uniform(-j, j)] for _ in range(4)])
    bg = [int(rng.uniform(70, 170))] * 3
    out = cv2.warpPerspective(im, cv2.getPerspectiveTransform(base, dst.astype(np.float32)), (w, h),
                              flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=bg)
    gx, gy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    th = rng.uniform(0, 2 * np.pi)
    light = 1 - level * 0.22 * rng.uniform(.4, 1) * (np.cos(th) * gx + np.sin(th) * gy + 1) / 2
    out = np.clip(out.astype(np.float32) * light[..., None], 0, 255)
    out = (255 * (out / 255) ** rng.uniform(.8, 1.0 + .35 * level)).astype(np.uint8)
    if level > 0.5:
        k = int(rng.uniform(1, 2.2 + 2.2 * level)) * 2 + 1
        out = cv2.GaussianBlur(out, (k, k), 0)
    out = out + np.random.RandomState(rng.randint(0, 99999)).normal(0, 3 + 4 * level, out.shape).astype(np.int16).clip(-255, 255).astype(np.int16)
    out = np.clip(out, 0, 255).astype(np.uint8)
    ls = rng.uniform(800, 1500)
    f = ls / max(h, w)
    out = cv2.resize(out, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(28, 80))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def synth_table(rng):
    W, H = 1654, 1169
    im = np.full((H, W, 3), 250, np.uint8)
    rows, cols = rng.randint(8, 30), rng.randint(3, 10)
    xs = np.sort(np.array([rng.uniform(60, W - 60) for _ in range(cols - 1)] + [60, W - 60]))
    ys = np.sort(np.array([rng.uniform(60, H - 60) for _ in range(rows - 1)] + [60, H - 60]))
    t = rng.randint(1, 3)
    for x in xs: cv2.line(im, (int(x), 60), (int(x), H - 60), (40, 40, 40), t)
    for y in ys: cv2.line(im, (60, int(y)), (W - 60, int(y)), (40, 40, 40), t)
    for yi in range(len(ys) - 1):
        for xi in range(len(xs) - 1):
            if rng.random() < .6:
                cv2.putText(im, str(rng.randint(1, 999)), (int(xs[xi] + 8), int(ys[yi + 1] - 10)),
                            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, rng.uniform(.6, 1.1), (60, 40, 30), 2, cv2.LINE_AA)
    return im


def non_card(rng, kind):
    W, H = 1654, 1169
    rs = np.random.RandomState(rng.randint(0, 99999))
    if kind == "noise":
        return cv2.GaussianBlur(rs.randint(0, 255, (H, W, 3)).astype(np.uint8), (0, 0), rng.uniform(.5, 3))
    if kind == "gradient":
        gx, gy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
        return (np.stack([gx, gy, (gx + gy) / 2], -1) * 200 + 30).astype(np.uint8)
    if kind == "blank":
        return np.full((H, W, 3), int(rng.uniform(150, 250)), np.uint8)
    if kind == "text":
        im = np.full((H, W, 3), 245, np.uint8)
        for y in range(80, H - 40, 46):
            cv2.putText(im, " ".join("".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(2, 9))) for _ in range(rng.randint(5, 12))),
                        (60, y), cv2.FONT_HERSHEY_SIMPLEX, rng.uniform(.8, 1.1), (30, 30, 30), 2, cv2.LINE_AA)
        return im
    im = np.full((H, W, 3), 230, np.uint8)  # shapes
    for _ in range(rng.randint(6, 25)):
        c = tuple(int(rng.uniform(0, 255)) for _ in range(3))
        p1, p2 = (rng.randint(0, W), rng.randint(0, H)), (rng.randint(0, W), rng.randint(0, H))
        (cv2.rectangle if rng.random() < .5 else cv2.line)(im, p1, p2, c, rng.randint(1, 6))
    return im


def build():
    rng = random.Random(SEED)
    IMG.mkdir(parents=True, exist_ok=True)
    items = []

    def save(name, im, cls, truth, note=""):
        cv2.imwrite(str(IMG / name), im, [cv2.IMWRITE_JPEG_QUALITY, 92] if name.endswith(".jpg") else [])
        items.append({"file": name, "cls": cls, "truth": truth, "note": note})

    anc = sorted((WORK / "raw").glob("p-*.png"))
    epi = sorted((WORK / "epi" / "aligned").glob("p-*.png"))
    epi_template = epi[0].name  # used as the second template; excluded from the test images
    rng.shuffle(anc); rng.shuffle(epi_pool := [p for p in epi if p.name != epi_template])
    epi = epi_pool

    def load(p, long_side=1600):
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        f = long_side / max(im.shape[:2])
        return cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else im

    for p in anc[:20]: save(f"anc_clean_{p.stem}.png", load(p), "anc_clean", "anc", p.name)
    pool = anc[:]
    for i in range(40):
        p = pool[i % len(pool)]
        save(f"anc_phone_{i:02d}_{p.stem}.jpg", phone(load(p), rng, rng.uniform(.3, .8)), "anc_phone", "anc", p.name)
    for i in range(20):
        p = pool[(i * 3) % len(pool)]
        save(f"anc_hard_{i:02d}_{p.stem}.jpg", phone(load(p), rng, rng.uniform(1.0, 1.6)), "anc_hard", "anc", p.name)
    for i in range(20):  # partial card in frame: informational (rejecting is acceptable, accepting is fine)
        p = pool[(i * 7) % len(pool)]
        im = load(p); h, w = im.shape[:2]
        fh, fw = rng.uniform(.5, .7), rng.uniform(.5, .7)
        y, x = rng.randint(0, int(h * (1 - fh))), rng.randint(0, int(w * (1 - fw)))
        save(f"anc_partial_{i:02d}_{p.stem}.jpg", phone(im[y:y + int(h * fh), x:x + int(w * fw)], rng, .5), "anc_partial", "anc", p.name)
    for p in epi: save(f"epi_clean_{p.stem}.png", load(p), "epi_clean", "epi", p.name)
    for i in range(20):
        p = epi[i % len(epi)]
        save(f"epi_phone_{i:02d}_{p.stem}.jpg", phone(load(p), rng, rng.uniform(.4, 1.2)), "epi_phone", "epi", p.name)
    for i in range(30):
        im = synth_table(rng)
        save(f"table_{i:02d}.jpg", phone(im, rng, rng.uniform(0, .8)) if i % 2 else im, "synth_table", "none")
    kinds = ["noise", "gradient", "blank", "text", "shapes"]
    for i in range(30):
        save(f"noncard_{i:02d}.jpg", non_card(rng, kinds[i % 5]), "non_card", "none", kinds[i % 5])
    cv2.imwrite(str(RUN / "template_epi.png"), load(WORK / "epi" / "aligned" / epi_template))
    with zipfile.ZipFile(PACK) as z:
        (RUN / "template_anc.png").write_bytes(z.read("template.png"))
    (RUN / "manifest.json").write_text(json.dumps({"seed": SEED, "epi_template_source": epi_template, "items": items}, indent=1))
    from collections import Counter
    print("built", len(items), dict(Counter(i["cls"] for i in items)))


# ------------------------------------------------------------------ matchers
def _sift(gray, long_side=1600):
    s = min(1.0, long_side / max(gray.shape))
    g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    k, d = cv2.SIFT_create(nfeatures=4000).detectAndCompute(g, None)
    return (np.float32([p.pt for p in k]) / s if k else np.zeros((0, 2), np.float32)), d, gray.shape[:2]


def line_image(gray):
    """Ruled-line layout only: printed rules survive, handwriting mostly does not."""
    s = min(1.0, 1600 / max(gray.shape))
    g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    H, W = g.shape
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, W // 25), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, H // 25)))
    lines = cv2.bitwise_or(cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk), cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk))
    out = cv2.GaussianBlur(cv2.bitwise_not(cv2.dilate(lines, np.ones((3, 3), np.uint8))), (0, 0), 1.2)
    return cv2.resize(out, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA) if s < 1 else out


def feats(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return {"full": _sift(gray), "lines": _sift(line_image(gray))}


def _poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def match(q, t, ratio=.75, px=4.0):
    """q, t = (pts, desc, (h, w)). Returns inliers + geometry sanity of the query->template homography."""
    (pq, dq, (qh, qw)), (pt, dt, (th, tw)) = q, t
    res = {"inliers": 0, "convex": False, "area_ratio": 0.0, "mirrored": None, "cover": 0.0}
    if dq is None or dt is None or len(pq) < 10 or len(pt) < 10:
        return res
    good = [m[0] for m in cv2.BFMatcher(cv2.NORM_L2).knnMatch(dq, dt, k=2) if len(m) == 2 and m[0].distance < ratio * m[1].distance]
    if len(good) < 10:
        return res
    src = pq[[g.queryIdx for g in good]].reshape(-1, 1, 2)
    dst = pt[[g.trainIdx for g in good]].reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, px)
    if H is None:
        return res
    inl = mask.ravel().astype(bool)
    res["inliers"] = int(inl.sum())
    quad = cv2.perspectiveTransform(np.float32([[0, 0], [qw, 0], [qw, qh], [0, qh]]).reshape(-1, 1, 2), H).reshape(4, 2)
    e = [quad[(i + 1) % 4] - quad[i] for i in range(4)]; cr = [float(e[i][0] * e[(i + 1) % 4][1] - e[i][1] * e[(i + 1) % 4][0]) for i in range(4)]   # z of the 2-D cross product (np.cross no longer takes 2-D vectors)
    res["convex"] = bool(all(c > 0 for c in cr) or all(c < 0 for c in cr))
    res["mirrored"] = bool(cr[0] < 0)
    res["area_ratio"] = float(_poly_area(quad) / (tw * th))
    if inl.sum() >= 3:
        hull = cv2.convexHull(dst[inl].reshape(-1, 2).astype(np.float32))
        res["cover"] = float(cv2.contourArea(hull) / (tw * th))
    return res


def hardened_ok(r, floor=FLOOR, use_cover=True):
    return (r["inliers"] >= floor and r["convex"] and not r["mirrored"]
            and HARD_AREA[0] <= r["area_ratio"] <= HARD_AREA[1] and (not use_cover or r["cover"] >= HARD_COVER))


def decide(sa, sb, ok_a, ok_b):
    """Two-template decision. ok_* = that template passes its own accept rule."""
    if not ok_a and not ok_b: return "none"
    if ok_a and not ok_b: return "anc"
    if ok_b and not ok_a: return "epi"
    if sa >= MARGIN * sb: return "anc"
    if sb >= MARGIN * sa: return "epi"
    return "ambiguous"


def score():
    man = json.loads((RUN / "manifest.json").read_text())
    T = {n: feats(cv2.imread(str(RUN / f"template_{n}.png"), cv2.IMREAD_COLOR)) for n in ("anc", "epi")}
    out, t0 = [], time.time()
    for i, it in enumerate(man["items"], 1):
        q = feats(cv2.imread(str(IMG / it["file"]), cv2.IMREAD_COLOR))
        row = dict(it)
        for n in ("anc", "epi"):
            row[n] = {"full": match(q["full"], T[n]["full"]), "lines": match(q["lines"], T[n]["lines"])}
        out.append(row)
        if i % 10 == 0 or i == len(man["items"]):
            print(f"{i}/{len(man['items'])} {time.time() - t0:.0f}s", flush=True)
    (RUN / "scores.json").write_text(json.dumps(out))


# ------------------------------------------------------------------ analysis + report
def methods(row):
    """Decision of each matcher variant for one image: 'anc' | 'epi' | 'none' | 'ambiguous'."""
    a, b = row["anc"], row["epi"]
    f = lambda r: r["full"]["inliers"]
    base_a, base_b = f(a) >= FLOOR, f(b) >= FLOOR
    ap = "none" if not (base_a or base_b) else ("anc" if f(a) >= f(b) else "epi")   # app today: best of the installed packs
    hard = decide(f(a), f(b), hardened_ok(a["full"]), hardened_ok(b["full"]))
    lin_ok = lambda r: r["lines"]["inliers"] >= LINE_FLOOR and r["lines"]["convex"] and not r["lines"]["mirrored"]
    lines = decide(a["lines"]["inliers"], b["lines"]["inliers"], lin_ok(a), lin_ok(b))
    both_a, both_b = hardened_ok(a["full"]) and lin_ok(a), hardened_ok(b["full"]) and lin_ok(b)
    comb = decide(f(a) + a["lines"]["inliers"], f(b) + b["lines"]["inliers"], both_a, both_b)
    geo = decide(f(a), f(b), hardened_ok(a["full"], use_cover=False), hardened_ok(b["full"], use_cover=False))
    return {"baseline": ap, "hardened": hard, "lines": lines, "hardened+lines": comb, "geometry-only (post-hoc)": geo}


METHOD_ORDER = ["baseline", "hardened", "geometry-only (post-hoc)", "lines", "hardened+lines"]
CLASS_ORDER = ["anc_clean", "anc_phone", "anc_hard", "anc_partial", "epi_clean", "epi_phone", "synth_table", "non_card"]


def analyse(rows):
    res = {m: {} for m in METHOD_ORDER}
    for r in rows:
        r["dec"] = methods(r)
    for m in METHOD_ORDER:
        for c in CLASS_ORDER:
            rs = [r for r in rows if r["cls"] == c]
            n = len(rs)
            cnt = {k: sum(r["dec"][m] == k for r in rs) for k in ("anc", "epi", "none", "ambiguous")}
            res[m][c] = {"n": n, **cnt, "truth": rs[0]["truth"]}
    return res


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def floor_sweep(rows):
    out = []
    for fl in (20, 30, 40, 60, 100, 150, 250):
        d = {}
        for grp, cl in (("anc (clean+phone+hard)", ("anc_clean", "anc_phone", "anc_hard")), ("not anc (epi+table+non-card)", ("epi_clean", "epi_phone", "synth_table", "non_card"))):
            rs = [r for r in rows if r["cls"] in cl]
            d[grp] = sum(r["anc"]["full"]["inliers"] >= fl for r in rs) / len(rs)
        out.append({"floor": fl, "accept_anc": d["anc (clean+phone+hard)"], "accept_other": d["not anc (epi+table+non-card)"]})
    return out


def thumb(path, w=220):
    im = cv2.imread(str(path)); f = w / im.shape[1]
    ok, buf = cv2.imencode(".jpg", cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 70])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def strip_svg(rows, key, floor, title):
    """Dot strip: x = inliers vs the ANC template (sqrt scale), one row per class."""
    W, L, Rm, rowh = 760, 120, 20, 24
    mx = max(max(r["anc"][key]["inliers"] for r in rows), floor * 2)
    X = lambda v: L + (W - L - Rm) * (np.sqrt(v) / np.sqrt(mx))
    pos = {"anc_clean", "anc_phone", "anc_hard", "anc_partial"}
    h = rowh * len(CLASS_ORDER) + 50
    s = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="{title}" style="width:100%;max-width:{W}px">',
         f'<text x="{L}" y="14" class="t">{title}</text>']
    for ci, c in enumerate(CLASS_ORDER):
        y = 34 + ci * rowh
        s.append(f'<text x="{L - 8}" y="{y + 4}" text-anchor="end" class="t">{c}</text>')
        s.append(f'<line x1="{L}" x2="{W - Rm}" y1="{y}" y2="{y}" class="g"/>')
        rng = random.Random(ci)
        for r in rows:
            if r["cls"] != c: continue
            v = r["anc"][key]["inliers"]
            s.append(f'<circle cx="{X(v):.1f}" cy="{y + rng.uniform(-6, 6):.1f}" r="3.2" class="{"p" if c in pos else "n"}"><title>{r["file"]}: {v}</title></circle>')
    s.append(f'<line x1="{X(floor):.1f}" x2="{X(floor):.1f}" y1="22" y2="{h - 16}" class="f"/><text x="{X(floor) + 4:.1f}" y="{h - 4}" class="t">floor {floor}</text>')
    for v in (0, 10, 50, 150, 500, 1000):
        if v <= mx: s.append(f'<text x="{X(v):.1f}" y="{h - 18}" text-anchor="middle" class="m">{v}</text>')
    return "".join(s) + "</svg>"


def report():
    rows = json.loads((RUN / "scores.json").read_text())
    res = analyse(rows)
    sweep = floor_sweep(rows)
    pct = lambda k, n: f"{100 * k / n:.0f}%" if n else "-"
    ci = lambda k, n: f"{100 * wilson(k, n)[0]:.0f}-{100 * wilson(k, n)[1]:.0f}"
    # headline: per method, correct-type rate on each truth group
    def grp(m, cl, want):
        k = sum(res[m][c][want] for c in cl); n = sum(res[m][c]["n"] for c in cl)
        return k, n
    head = []
    for m in METHOD_ORDER:
        a = grp(m, ("anc_clean", "anc_phone", "anc_hard"), "anc")
        e = grp(m, ("epi_clean", "epi_phone"), "epi")
        # false accepts of ANC among things that are not ANC
        fa = sum(res[m][c]["anc"] for c in ("epi_clean", "epi_phone", "synth_table", "non_card"))
        fn = sum(res[m][c]["n"] for c in ("epi_clean", "epi_phone", "synth_table", "non_card"))
        wrong_epi = sum(res[m][c]["anc"] for c in ("epi_clean", "epi_phone"))
        wn = sum(res[m][c]["n"] for c in ("epi_clean", "epi_phone"))
        amb = sum(res[m][c]["ambiguous"] for c in CLASS_ORDER)
        head.append((m, a, e, (fa, fn), (wrong_epi, wn), amb))
    h_rows = "".join(
        f"<tr><td><b>{m}</b></td><td>{a[0]}/{a[1]} ({pct(*a)}) [{ci(*a)}]</td><td>{fa[0]}/{fa[1]} ({pct(*fa)}) [{ci(*fa)}]</td>"
        f"<td>{we[0]}/{we[1]} ({pct(*we)})</td><td>{e[0]}/{e[1]} ({pct(*e)})</td><td>{amb}</td></tr>"
        for m, a, e, fa, we, amb in head)
    c_rows = ""
    for m in METHOD_ORDER:
        for c in CLASS_ORDER:
            d = res[m][c]
            c_rows += (f"<tr><td>{m if c == CLASS_ORDER[0] else ''}</td><td>{c}</td><td>{d['truth']}</td><td>{d['n']}</td>"
                       f"<td>{d['anc']}</td><td>{d['epi']}</td><td>{d['ambiguous']}</td><td>{d['none']}</td></tr>")
    sw = "".join(f"<tr><td>{s['floor']}</td><td>{100 * s['accept_anc']:.0f}%</td><td>{100 * s['accept_other']:.1f}%</td></tr>" for s in sweep)
    # failures of the baseline and of hardened+lines
    def fails(m, limit=6):
        bad = []
        for r in rows:
            d = r["dec"][m]
            if r["cls"] == "anc_partial": continue
            if (r["truth"] == "anc" and d != "anc") or (r["truth"] != "anc" and d == "anc"):
                bad.append((r, d))
        return bad[:limit], len(bad)
    cards = ""
    for m in ("baseline", "hardened+lines"):
        bad, tot = fails(m)
        cards += f"<h3>{m}: {tot} wrong decisions (first {len(bad)} shown)</h3><div class='grid'>"
        for r, d in bad:
            cards += (f"<figure><img src='{thumb(IMG / r['file'])}'><figcaption>{r['cls']} → <b>{d}</b><br>"
                      f"anc {r['anc']['full']['inliers']} / epi {r['epi']['full']['inliers']} inl</figcaption></figure>")
        cards += "</div>" if bad else "<p>none</p></div>"
    partial = {m: (res[m]["anc_partial"]["anc"], res[m]["anc_partial"]["n"]) for m in METHOD_ORDER}
    page = f"""<!doctype html><meta charset=utf-8><title>Template ID test</title><meta name=viewport content="width=device-width,initial-scale=1">
<link rel=stylesheet href="../../static/vendor/pico.min.css">
<style>:root{{--p:#2a78d6;--n:#eb6834}}@media(prefers-color-scheme:dark){{:root{{--p:#5ea3f2;--n:#ff8a5c}}}}
svg .p{{fill:var(--p);opacity:.8}}svg .n{{fill:var(--n);opacity:.8}}svg .g{{stroke:currentColor;opacity:.12}}svg .f{{stroke:currentColor;stroke-dasharray:4 3;opacity:.6}}
svg .t{{font:12px system-ui;fill:currentColor}}svg .m{{font:10px system-ui;fill:currentColor;opacity:.6}}
.grid{{display:flex;flex-wrap:wrap;gap:.75rem}}figure{{margin:0;font-size:.75rem}}figure img{{width:220px;display:block;border:1px solid #8884}}td,th{{font-size:.85rem}}main{{max-width:960px}}</style>
<main class=container><h1>Template identification test</h1>
<p>Local test of how well a phone photo is identified as the ANC card. Everything here is on this machine. {len(rows)} images, seed {SEED}.</p>
<h2>What was tested</h2>
<table><tr><th>class</th><th>n</th><th>what</th><th>truth</th></tr>
<tr><td>anc_clean</td><td>{res['baseline']['anc_clean']['n']}</td><td>ANC scans from <code>work/raw</code>, resized only</td><td>anc</td></tr>
<tr><td>anc_phone</td><td>{res['baseline']['anc_phone']['n']}</td><td>same pages, simulated phone photo (mild): perspective, rotation, uneven light, blur, JPEG</td><td>anc</td></tr>
<tr><td>anc_hard</td><td>{res['baseline']['anc_hard']['n']}</td><td>same, strong degradation</td><td>anc</td></tr>
<tr><td>anc_partial</td><td>{res['baseline']['anc_partial']['n']}</td><td>only 50-70% of the card in frame (not scored: either answer is acceptable)</td><td>anc</td></tr>
<tr><td>epi_clean / epi_phone</td><td>{res['baseline']['epi_clean']['n']} / {res['baseline']['epi_phone']['n']}</td><td>EPI card pages (a different, similar-looking card), clean and degraded</td><td>epi</td></tr>
<tr><td>synth_table</td><td>{res['baseline']['synth_table']['n']}</td><td>random ruled tables with scribbled numbers</td><td>none</td></tr>
<tr><td>non_card</td><td>{res['baseline']['non_card']['n']}</td><td>noise, gradients, blank, text, shapes</td><td>none</td></tr></table>
<p><b>Not real phone photos.</b> The degradations are simulated from scans. I did not have labeled real phone photos. Two templates are used: the ANC pack template and one EPI page (itself excluded from the test images; a filled page, not a blank card).</p>
<h2>Headline (ANC identification)</h2>
<table><thead><tr><th>matcher</th><th>ANC accepted as ANC<br>(clean+phone+hard) [Wilson 95%]</th><th>not-ANC accepted as ANC<br>(epi+table+non-card) [95%]</th><th>EPI accepted as ANC</th><th>EPI identified as EPI</th><th>ambiguous (all)</th></tr></thead><tbody>{h_rows}</tbody></table>
<p>Page-level correlation: several images come from the same page, so the intervals are narrower than the truth. Rules for <i>hardened</i> and <i>lines</i> were fixed before scoring; only the floor sweep below is a sweep, and it is run on this same set.</p>
<h2>Score separation (inliers against the ANC template)</h2>
{strip_svg(rows, 'full', FLOOR, 'Full-image SIFT inliers (blue = ANC, orange = not ANC)')}
{strip_svg(rows, 'lines', LINE_FLOOR, 'Ruled-lines-only SIFT inliers')}
<h2>Floor sweep (baseline)</h2><table><thead><tr><th>floor (inliers)</th><th>ANC accepted</th><th>not-ANC accepted</th></tr></thead><tbody>{sw}</tbody></table>
<h2>Per class decisions</h2><table><thead><tr><th>matcher</th><th>class</th><th>truth</th><th>n</th><th>→ anc</th><th>→ epi</th><th>→ ambiguous</th><th>→ none</th></tr></thead><tbody>{c_rows}</tbody></table>
<p>anc_partial accepted as ANC: {", ".join(f"{m} {k}/{n}" for m, (k, n) in partial.items())}.</p>
<h2>Wrong decisions</h2>{cards}
<h2>How each matcher decides</h2><ul>
<li><b>baseline</b>: SIFT (long side 1600), ratio 0.75, RANSAC 4 px, accept at ≥ {FLOOR} inliers; with two templates, the one with more inliers wins. This is the pack manifest rule today.</li>
<li><b>hardened</b>: baseline plus: mapped photo outline must be convex and not mirrored, area {HARD_AREA[0]}-{HARD_AREA[1]}× the template, inlier hull must cover ≥ {int(HARD_COVER * 100)}% of the template; with two templates the winner needs {MARGIN}× the runner-up, else <i>ambiguous</i>.</li>
<li><b>lines</b>: same SIFT/RANSAC on the ruled-line image only (handwriting removed), floor {LINE_FLOOR}.</li>
<li><b>hardened+lines</b>: both must pass; scores add for the margin test.</li>
<li><b>geometry-only (post-hoc)</b>: <i>hardened</i> without the coverage rule. Added <b>after</b> seeing that coverage 0.30 rejected 9 of 20 EPI phone photos and most partial ANC cards. It was not fixed in advance, so treat its numbers as tuned on this set.</li></ul></main>"""
    (RUN / "report.html").write_text(page)
    (RUN / "summary.json").write_text(json.dumps({"headline": [{"method": m, "anc_ok": a, "other_as_anc": fa, "epi_as_anc": we, "epi_ok": e, "ambiguous": amb} for m, a, e, fa, we, amb in head],
                                                  "sweep": sweep, "partial": partial}, indent=1))
    for m, a, e, fa, we, amb in head:
        print(f"{m:16s} anc_ok {a[0]}/{a[1]}  other_as_anc {fa[0]}/{fa[1]}  epi_as_anc {we[0]}/{we[1]}  epi_ok {e[0]}/{e[1]}  ambiguous {amb}")


def _rot_with_matrix(im, a):
    """Rotate CCW by a degrees (any angle, canvas grows). Returns (image, 3x3 matrix mapping old pixel coords to new)."""
    h, w = im.shape[:2]
    if a % 360 in (90, 180, 270):
        k = a % 360
        M = {90: [[0, -1, h - 1], [1, 0, 0], [0, 0, 1]], 180: [[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]], 270: [[0, 1, 0], [-1, 0, w - 1], [0, 0, 1]]}[k]
        # rotate 90 clockwise == a=-90 in CCW terms; here a is clockwise-on-screen for quarters, matching deskew.rotate_quarter
        import deskew
        return deskew.rotate_quarter(im, k), np.array(M, np.float64)
    M2 = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
    c, s_ = abs(M2[0, 0]), abs(M2[0, 1]); nw, nh = int(h * s_ + w * c), int(h * c + w * s_)
    M2[0, 2] += nw / 2 - w / 2; M2[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(im, M2, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(120, 120, 120)), np.vstack([M2, [0, 0, 1]])


def _homography(q, t):
    (pq, dq, _), (pt, dt, _) = q, t
    if dq is None or len(pq) < 10: return None, 0
    good = [m[0] for m in cv2.BFMatcher(cv2.NORM_L2).knnMatch(dq, dt, k=2) if len(m) == 2 and m[0].distance < .75 * m[1].distance]
    if len(good) < 10: return None, 0
    H, mask = cv2.findHomography(pq[[g.queryIdx for g in good]].reshape(-1, 1, 2), pt[[g.trainIdx for g in good]].reshape(-1, 1, 2), cv2.RANSAC, 4.0)
    return H, int(mask.sum()) if H is not None else 0


def rotation():
    """Alignment accuracy on rotated ANC pages. The error that matters for reading: where a cell centre lands after the
    recovered homography, compared with where it lands when the page is not rotated (template pixels; cells are ~18 px tall)."""
    man = json.loads((RUN / "manifest.json").read_text())["items"]
    pages = [i for i in man if i["cls"] == "anc_clean"][:10]
    T = feats(cv2.imread(str(RUN / "template_anc.png")))["full"]
    with zipfile.ZipFile(PACK) as z: ref = json.loads(z.read("grid.json"))
    centres = []
    for c in range(len(ref["col_lines_top"]) - 1):
        for r in range(len(ref["rows"])):
            ya, yb = ref["row_lines_by_col"][c][r] * .5, ref["row_lines_by_col"][c][r + 1] * .5
            ym = (ya + yb) / 2; f = lambda cc: (ref["col_lines_top"][cc] + (ref["col_lines_bot"][cc] - ref["col_lines_top"][cc]) * (ym * 2 - ref["y_ref_top"]) / (ref["y_ref_bot"] - ref["y_ref_top"])) * .5
            centres.append(((f(c) + f(c + 1)) / 2, ym))
    centres = np.float64(centres)
    out = {}
    for A in (0, 5, 15, 30, 45, 90, 135, 180, 270, -20):
        errs, acc = [], 0
        for it in pages:
            im = cv2.imread(str(IMG / it["file"]))
            H0, _ = _homography(feats(im)["full"], T)
            rim, M = _rot_with_matrix(im, A)
            H1, inl = _homography(feats(rim)["full"], T)
            if H1 is None or H0 is None or inl < FLOOR: continue
            acc += 1
            p = cv2.perspectiveTransform(centres.reshape(-1, 1, 2), np.linalg.inv(H0)).reshape(-1, 2)          # page coords of the cells
            pr = cv2.perspectiveTransform(p.reshape(-1, 1, 2), M).reshape(-1, 2)                                 # same points in the rotated photo
            back = cv2.perspectiveTransform(pr.reshape(-1, 1, 2), H1).reshape(-1, 2)                              # where the new homography puts them
            errs += list(np.linalg.norm(back - centres, axis=1))
        e = np.array(errs) if errs else np.array([np.nan])
        out[A] = {"accepted": acc, "n": len(pages), "median_px": float(np.nanmedian(e)), "p95_px": float(np.nanpercentile(e, 95)), "max_px": float(np.nanmax(e))}
        print(f"rot {A:4d} deg: accepted {acc}/{len(pages)}  alignment error vs unrotated: median {out[A]['median_px']:.2f}  p95 {out[A]['p95_px']:.2f}  max {out[A]['max_px']:.2f}  (template px; a cell is ~18 px tall)", flush=True)
    (RUN / "rotation.json").write_text(json.dumps(out, indent=1))


def columns():
    """Whole card vs a photo of only part of it: one visit column, with or without the label column, or a few columns.
    Crops are cut from clean pages along the ANC grid (full height). Reports whether the crop is identified and how far the
    cells it contains land from where the whole-card alignment puts them (template px; a cell is ~18 px tall)."""
    man = json.loads((RUN / "manifest.json").read_text())["items"]
    pages = [i for i in man if i["cls"] == "anc_clean"][:10]
    T = feats(cv2.imread(str(RUN / "template_anc.png")))["full"]
    with zipfile.ZipFile(PACK) as z: ref = json.loads(z.read("grid.json"))
    ncol = len(ref["col_lines_top"]) - 1
    # template-space x range of each data column at mid height, and of the label area (left of column 1)
    ymid = (ref["y_ref_top"] + ref["y_ref_bot"]) / 2
    xs = [(ref["col_lines_top"][c] + (ref["col_lines_bot"][c] - ref["col_lines_top"][c]) * (ymid - ref["y_ref_top"]) / (ref["y_ref_bot"] - ref["y_ref_top"])) * .5 for c in range(ncol + 1)]
    ytop, ybot = ref["row_lines_by_col"][0][0] * .5, ref["row_lines_by_col"][0][-1] * .5
    label_left = 0.0
    configs = {"whole card": (label_left, xs[-1] + 40), "label + 3 columns": (label_left, xs[3]), "label + 1 column": (label_left, xs[1]),
               "3 columns, no label": (xs[0], xs[3]), "1 column, no label": (xs[0], xs[1]), "1 column, middle (c3)": (xs[2], xs[3])}
    out = {}
    for name, (xa, xb) in configs.items():
        errs, acc, inl_all, kept = [], 0, [], []
        for it in pages:
            im = cv2.imread(str(IMG / it["file"])); H0, _ = _homography(feats(im)["full"], T)
            if H0 is None: continue
            Hi = np.linalg.inv(H0)
            quad = cv2.perspectiveTransform(np.float64([[[xa, ytop - 60], [xb, ytop - 60], [xb, ybot + 60], [xa, ybot + 60]]]), Hi).reshape(4, 2)
            x0, y0 = np.maximum(quad.min(0), 0).astype(int); x1, y1 = np.minimum(quad.max(0), [im.shape[1], im.shape[0]]).astype(int)
            crop = im[y0:y1, x0:x1]
            if crop.shape[0] < 50 or crop.shape[1] < 50: continue
            H1, inl = _homography(feats(crop)["full"], T); inl_all.append(inl)
            if H1 is None or inl < FLOOR: continue
            acc += 1
            # cell centres of the columns that are inside the crop
            pts = []
            for c in range(ncol):
                if xs[c] >= xa - 5 and xs[c + 1] <= xb + 5:
                    for r in range(len(ref["rows"])):
                        ya, yb = ref["row_lines_by_col"][c][r] * .5, ref["row_lines_by_col"][c][r + 1] * .5; pts.append(((xs[c] + xs[c + 1]) / 2, (ya + yb) / 2))
            if not pts: continue
            pts = np.float64(pts); p = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), Hi).reshape(-1, 2) - [x0, y0]
            back = cv2.perspectiveTransform(p.reshape(-1, 1, 2), H1).reshape(-1, 2); errs += list(np.linalg.norm(back - pts, axis=1)); kept.append(len(pts))
        e = np.array(errs) if errs else np.array([np.nan])
        out[name] = {"identified": acc, "n": len(pages), "median_inliers": int(np.median(inl_all)) if inl_all else 0, "cells_per_crop": int(np.median(kept)) if kept else 0,
                     "median_px": float(np.nanmedian(e)), "p95_px": float(np.nanpercentile(e, 95)), "max_px": float(np.nanmax(e))}
        o = out[name]; print(f"{name:24s} identified {acc:2d}/{len(pages)}  median inliers {o['median_inliers']:4d}  cells/crop {o['cells_per_crop']:3d}  alignment error median {o['median_px']:5.2f} p95 {o['p95_px']:5.2f} max {o['max_px']:6.2f}", flush=True)
    (RUN / "columns.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"build": build, "score": score, "report": report, "rotation": rotation, "columns": columns}[sys.argv[1]]()
