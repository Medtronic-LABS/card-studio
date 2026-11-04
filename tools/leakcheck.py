"""Leak check for the public repo: refuses anything that could be patient data or a secret. Run on a directory (the tree about to be committed).

    python tools/leakcheck.py DIR      exit 0 = clean, 1 = findings printed

It is deliberately an ALLOWLIST: a file type or name not listed is a finding. The data in this project is real patient handwriting, so
images, labels, trained weights, packs, notebooks (their saved outputs contain card images), PDFs and run folders are never allowed.
The only PNGs allowed are UI screenshots listed in docs/screenshots/MANIFEST.json with a matching hash and zero <img> elements on the page.
"""
import fnmatch, hashlib, json, re, sys
from pathlib import Path

TEXT_EXT = {".py", ".md", ".html", ".css", ".txt", ".yml", ".yaml", ".toml", ".cfg"}
NAMES_OK = {".gitignore", "LICENSE", "NOTICE", "requirements.txt"}
JSON_OK = ["cardpack/schema/*.json", "model_meta.json", "docs/screenshots/MANIFEST.json", "docs/status.json"]
DENY_EXT = {".onnx", ".pt", ".pth", ".ckpt", ".cardpack", ".csv", ".tsv", ".pdf", ".ipynb", ".npy", ".npz", ".h5", ".hdf5", ".zip", ".gz", ".tar", ".jpg", ".jpeg",
            ".gif", ".bmp", ".tif", ".tiff", ".heic", ".webp", ".db", ".sqlite", ".pkl", ".pickle", ".log", ".docx", ".xlsx", ".mp4", ".mov"}
DENY_DIRS = {"work", "data", "runs", "uploads", "templates", "imports", "experiments", "out", "crops", "aligned", "raw"}
SKIP_CONTENT = {"studio/static/vendor/tabler.min.css"}          # vendored third-party CSS: allowed, not scanned
MAX_BYTES = 700_000
PATTERNS = [
    (re.compile(r"/Users/[A-Za-z0-9_.-]+"), "absolute home path"), (re.compile(r"/home/[a-z][a-z0-9_-]+/"), "absolute home path"),
    (re.compile(r"[A-Za-z]:\\Users\\"), "absolute home path"),
    (re.compile(r"\b(?!git@github\.com\b)[A-Za-z0-9._%+-]+@(?!example\.|users\.noreply\.github\.com)[A-Za-z0-9-]+\.[A-Za-z]{2,}\b"), "email address"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"), "GitHub token"), (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS key"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"), (re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), "Slack token"),
    (re.compile(r"(?i)\b(password|passwd|secret|api[_-]?key|token)\s*[:=]\s*['\"][^'\"\s]{4,}['\"]"), "hard-coded credential"),
    (re.compile(r"^page,row,col,label\b", re.M), "labels.csv header (patient labels)"),
]


def _matches(rel, globs):
    return any(fnmatch.fnmatch(rel, g) for g in globs)


def check(root):
    root = Path(root).resolve(); out = []; shots = {}
    man = root / "docs" / "screenshots" / "MANIFEST.json"
    if man.exists():
        try: shots = {e["file"]: e for e in json.loads(man.read_text())["screenshots"]}
        except (ValueError, KeyError, TypeError): out.append("docs/screenshots/MANIFEST.json: unreadable")
    for p in sorted(root.rglob("*")):
        if not p.is_file() or ".git" in p.relative_to(root).parts[:1] or "__pycache__" in p.parts: continue
        rel = p.relative_to(root).as_posix(); ext = p.suffix.lower(); parts = set(p.relative_to(root).parts[:-1])
        bad_dir = parts & DENY_DIRS
        if bad_dir: out.append(f"{rel}: inside a data directory ({', '.join(sorted(bad_dir))})"); continue
        if ext in DENY_EXT: out.append(f"{rel}: file type {ext} is never allowed (images, models, packs, labels, notebooks, PDFs)"); continue
        if p.stat().st_size > MAX_BYTES: out.append(f"{rel}: {p.stat().st_size} bytes is over the {MAX_BYTES} limit"); continue
        if ext == ".png":
            e = shots.get(rel)
            if not rel.startswith("docs/screenshots/") or e is None: out.append(f"{rel}: a PNG that is not a registered UI screenshot")
            elif hashlib.sha256(p.read_bytes()).hexdigest() != e.get("sha256"): out.append(f"{rel}: does not match the hash in the screenshot manifest")
            elif e.get("images_on_page") != 0: out.append(f"{rel}: captured from a page that had images on it")
            continue
        if ext == ".json":
            if not _matches(rel, JSON_OK): out.append(f"{rel}: JSON file not on the allowlist {JSON_OK}"); continue
            if rel == "model_meta.json":
                try:
                    if set(json.loads(p.read_text())) != {"classes", "blank_idx", "img_h", "img_w", "fields"}: out.append(f"{rel}: unexpected keys (expected only the vocabulary and field masks)")
                except ValueError: out.append(f"{rel}: not valid JSON")
        elif ext not in TEXT_EXT and p.name not in NAMES_OK: out.append(f"{rel}: file type {ext or p.name} is not on the allowlist"); continue
        if rel in SKIP_CONTENT: continue
        try: text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError: out.append(f"{rel}: binary content"); continue
        if "\x00" in text: out.append(f"{rel}: binary content"); continue
        for rx, why in PATTERNS:
            m = rx.search(text)
            if m: out.append(f"{rel}: {why}: {m.group(0)[:60]!r}")
    return out


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    found = check(target)
    for f in found: print("LEAK?", f)
    print(f"leakcheck: {'CLEAN' if not found else str(len(found)) + ' finding(s)'} ({target})")
    sys.exit(1 if found else 0)
