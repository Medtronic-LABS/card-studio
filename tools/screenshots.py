"""Capture UI screenshots of the running Studio for the README, in a mode that never builds patient images.

    python tools/screenshots.py            (Studio must be running on 127.0.0.1:8770; Google Chrome must be installed)

Pages are loaded with ?shots=1: the failure browser (cell crops), template previews and photo previews are not built. Every capture is then
checked: the page's DOM must contain zero <img> elements, or the screenshot is discarded. docs/screenshots/MANIFEST.json records the hash of each
file and its image count; tools/leakcheck.py refuses any PNG that is not registered there with a matching hash and zero images.
The numbers on screen are aggregate metrics (counts, accuracies, field names), not records.
"""
import hashlib, json, re, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots"
BASE = "http://127.0.0.1:8770"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
VIEWS = [("dashboard", "cards", 2300, "Dashboard: model registry, getting started, activity, needs attention, field readiness"),
         ("model-quality", "ml", 3300, "Model quality: per-field errors, calibration, risk-coverage, confusions, flag-policy tuner, provenance"),
         ("history", "history", 1400, "History: training runs, paired comparison, experiment records"),
         ("train-and-ship", "train", 1700, "Train and ship: held-out gate per field"),
         ("clean-up-lab", "lab", 700, "Photo clean-up lab (empty state)"),
         ("label", "label", 900, "Label and import corrected labels")]


def up():
    try: urllib.request.urlopen(BASE + "/api/packs", timeout=3); return True
    except Exception: return False


def trim(path, width=1200):
    from PIL import Image
    im = Image.open(path).convert("RGB"); w, h = im.size; px = im.load(); bg = px[w - 1, h - 1]; last = 0
    for y in range(h - 1, -1, -4):
        if any(abs(px[x, y][0] - bg[0]) + abs(px[x, y][1] - bg[1]) + abs(px[x, y][2] - bg[2]) > 24 for x in range(240, w, 17)): last = y; break
    im = im.crop((0, 0, w, min(h, last + 40)))
    im = im.resize((width, int(im.size[1] * width / im.size[0])), Image.LANCZOS).quantize(colors=192, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE); im.save(path, optimize=True)   # flat UI: a palette is far smaller


def main():
    if not up(): sys.exit("Studio is not running on 127.0.0.1:8770")
    if not Path(CHROME).exists(): sys.exit("Google Chrome not found at the macOS default path")
    OUT.mkdir(parents=True, exist_ok=True); entries = []
    with tempfile.TemporaryDirectory() as prof:
        for name, tab, height, caption in VIEWS:
            url = f"{BASE}/?shots=1&mode=dark#{tab}"; dest = OUT / f"{name}.png"
            base = [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run", f"--user-data-dir={prof}", f"--window-size=1360,{height}", "--virtual-time-budget=20000"]
            dom = subprocess.run(base + ["--dump-dom", url], capture_output=True, text=True, timeout=120).stdout
            body = re.sub(r"<(script|style)\b.*?</\1>", "", dom, flags=re.S | re.I)      # element count, not the text of our own scripts
            n_img = len(re.findall(r"<img\b", body))
            if n_img: print(f"SKIPPED {name}: {n_img} <img> elements on the page"); dest.unlink(missing_ok=True); continue
            subprocess.run(base + [f"--screenshot={dest}", url], capture_output=True, timeout=120); trim(dest)
            entries.append({"file": f"docs/screenshots/{name}.png", "view": name, "caption": caption, "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(), "images_on_page": 0})
            print("captured", name, dest.stat().st_size, "bytes")
    (OUT / "MANIFEST.json").write_text(json.dumps({"captured": time.strftime("%Y-%m-%d"), "mode": "?shots=1 (no patient images are built)", "screenshots": entries}, indent=1))


if __name__ == "__main__":
    main()
