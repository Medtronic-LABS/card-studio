"""Build the public tree from an ALLOWLIST, check it for anything patient-like, test it, then commit and push.

    python tools/publish.py --message "what changed" [--coauthor "Name <email>"] [--init] [--no-push] [--skip-screenshots] [--skip-tests]

The public tree is a separate folder (.dist/public-repo, ignored by the private repo) with its own fresh git history, so the private repo and its
history (which contain patient data) are never involved. Order: screenshots, README numbers, export, leak check, tests, commit, push.
Any failure stops everything before the commit: nothing is pushed unless the leak check and the tests pass."""
import argparse, fnmatch, json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / ".dist" / "public-repo"
REPO = "Medtronic-LABS/card-studio"
DESC = "Local-first platform for reading handwritten health cards: Card Studio, a portable card-pack format, and held-out evaluation. Code and docs only; no patient data."
INCLUDE = ["model_meta.json", "studio/*.py", "studio/tests/*.py", "studio/static/index.html", "studio/static/vendor/tabler.min.css", "studio/static/vendor/TABLER-LICENSE.txt",
           "cardpack/CARDPACK.md", "cardpack/*.py", "cardpack/schema/*.json", "work_phaseA/*.py", "tools/*.py", "tools/README.template.md",
           "docs/screenshots/*.png", "docs/screenshots/MANIFEST.json", "docs/status.json"]
GITIGNORE = """# Patient data and anything derived from it: never commit. This protects your own data if you run Studio here.
work/
data/
studio/runs/
studio/uploads/
studio/templates/
studio/cards/
studio/.venv-labelstudio/
studio/labelstudio.json
studio/labelstudio-data/
studio/imports/
studio/experiments/
cardpack/out/
.dist/
*.cardpack
*.onnx
*.pt
*.pth
*.pdf
*.csv
*.npy
*.npz
*.h5
*.zip
*.ipynb
*.heic
*.jpg
*.jpeg
*.tif
*.tiff
*.log
*.png
!docs/screenshots/*.png
__pycache__/
.DS_Store
"""
REQUIREMENTS = "# Studio and card-pack tools\nnumpy\nopencv-python\nonnxruntime\npillow\npypdfium2\njsonschema\n# Training only (work_phaseA, studio/trainer.py)\n# torch\n"
CI = """name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install numpy opencv-python-headless onnxruntime pillow pypdfium2 jsonschema
      - run: python tools/leakcheck.py .
      - run: CARDPACK_SYNTHETIC=1 python cardpack/test_cardpack.py
      - run: python -m unittest discover -s studio/tests
      - run: python -m unittest discover -s tools -p "test_*.py"
"""


def run(cmd, **kw):
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def export():
    want = {}
    for g in INCLUDE:
        for p in ROOT.glob(g):
            if p.is_file() and "__pycache__" not in p.parts: want[p.relative_to(ROOT).as_posix()] = p
    DST.mkdir(parents=True, exist_ok=True)
    for rel, src in want.items():
        d = DST / rel; d.parent.mkdir(parents=True, exist_ok=True)
        if not d.exists() or d.read_bytes() != src.read_bytes(): shutil.copy2(src, d)
    sys.path.insert(0, str(ROOT / "tools")); import update_readme as U
    st, live = U.collect()
    generated = {"README.md": U.render(DST, st), ".gitignore": GITIGNORE, "requirements.txt": REQUIREMENTS, ".github/workflows/ci.yml": CI}
    for rel, text in generated.items():
        d = DST / rel; d.parent.mkdir(parents=True, exist_ok=True); d.write_text(text)
    keep = set(want) | set(generated)
    for p in sorted(DST.rglob("*"), reverse=True):
        rel = p.relative_to(DST).as_posix()
        if p.is_dir() or rel.startswith(".git/") or rel == ".git": continue
        if rel not in keep: p.unlink()
    for p in sorted(DST.rglob("*"), reverse=True):
        if p.is_dir() and not any(p.iterdir()) and p.name != ".git" and ".git" not in p.parts: p.rmdir()
    return len(keep), live


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--message"); ap.add_argument("--coauthor"); ap.add_argument("--init", action="store_true"); ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--skip-screenshots", action="store_true"); ap.add_argument("--skip-tests", action="store_true"); a = ap.parse_args()
    if not a.skip_screenshots:
        r = run([sys.executable, str(ROOT / "tools" / "screenshots.py")])
        print(r.stdout.strip()); 
        if r.returncode: sys.exit("screenshots failed: " + (r.stderr or r.stdout)[-400:] + "\n(start Studio, or pass --skip-screenshots to reuse the existing ones)")
    n, live = export(); print(f"exported {n} files; numbers {'live' if live else 'reused from docs/status.json'}")
    r = run([sys.executable, str(ROOT / "tools" / "leakcheck.py"), str(DST)]); print(r.stdout.strip())
    if r.returncode: sys.exit("STOPPED: the leak check found something. Nothing was committed or pushed.")
    if not a.skip_tests:
        env = {**os.environ, "CARDPACK_SYNTHETIC": "1"}
        for name, cmd in (("pack format", [sys.executable, "cardpack/test_cardpack.py"]), ("studio", [sys.executable, "-m", "unittest", "discover", "-s", "studio/tests"]),
                          ("tools", [sys.executable, "-m", "unittest", "discover", "-s", "tools", "-p", "test_*.py"])):
            r = run(cmd, cwd=DST, env=env); tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
            print(f"tests [{name}]: {'OK' if r.returncode == 0 else 'FAILED'}  {' | '.join(tail)}")
            if r.returncode: sys.exit(f"STOPPED: tests failed in the clean public tree ({name}). Nothing was committed or pushed.")
    if not (DST / ".git").exists(): run(["git", "init", "-b", "main"], cwd=DST)
    gid = run(["gh", "api", "user", "-q", ".id,.login"]).stdout.split(); login = gid[1] if len(gid) > 1 else "user"
    run(["git", "config", "user.name", login], cwd=DST); run(["git", "config", "user.email", f"{gid[0]}+{login}@users.noreply.github.com" if gid else "noreply@users.noreply.github.com"], cwd=DST)
    run(["git", "add", "-A"], cwd=DST)
    if run(["git", "diff", "--cached", "--quiet"], cwd=DST).returncode == 0 and run(["git", "rev-parse", "HEAD"], cwd=DST).returncode == 0: print("nothing changed: nothing to publish"); return
    if not a.message: sys.exit("--message is required to commit")
    msg = a.message + (f"\n\nCo-Authored-By: {a.coauthor}" if a.coauthor else "")
    r = run(["git", "commit", "-m", msg], cwd=DST); print(r.stdout.strip().splitlines()[0] if r.stdout.strip() else r.stderr.strip())
    if r.returncode: sys.exit("commit failed: " + r.stderr[-300:])
    if a.no_push: print("--no-push: committed locally only"); return
    if a.init:
        r = run(["gh", "repo", "create", REPO, "--public", "--description", DESC])
        if r.returncode: sys.exit("could not create the repository (organisation permissions?): " + r.stderr.strip())
        print("created", r.stdout.strip())
    url = f"https://github.com/{REPO}.git"
    run(["git", "remote", "remove", "origin"], cwd=DST); run(["git", "remote", "add", "origin", url], cwd=DST)
    r = run(["git", "-c", "credential.helper=", "-c", "credential.helper=!gh auth git-credential", "push", "-u", "origin", "main"], cwd=DST)
    print(r.stdout.strip() or r.stderr.strip())
    if r.returncode: sys.exit("push failed")
    v = run(["gh", "repo", "view", REPO, "--json", "visibility,url", "-q", ".visibility + \" \" + .url"]); print("published:", v.stdout.strip())
    # CI runs on a clean machine: it is the check that catches what only works here (a missing dependency, a path). Wait for it and report.
    import time
    sha = run(["git", "rev-parse", "HEAD"], cwd=DST).stdout.strip(); rid = None
    for _ in range(12):
        out = run(["gh", "run", "list", "--repo", REPO, "--commit", sha, "--json", "databaseId", "-q", ".[0].databaseId"]).stdout.strip()
        if out: rid = out; break
        time.sleep(5)
    if not rid: print("CI: no run found yet (check the Actions tab)"); return
    r = run(["gh", "run", "watch", rid, "--repo", REPO, "--exit-status"]); c = run(["gh", "run", "view", rid, "--repo", REPO, "--json", "conclusion", "-q", ".conclusion"]).stdout.strip()
    print(f"CI on GitHub: {c.upper()}  (https://github.com/{REPO}/actions/runs/{rid})")
    if c != "success": sys.exit("CI FAILED on the pushed commit: fix forward and publish again")


if __name__ == "__main__":
    main()
