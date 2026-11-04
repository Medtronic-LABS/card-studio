"""Fill the README template with live aggregate numbers (from a running Studio) and the screenshot gallery.
Numbers are aggregates only. If Studio is not running, the last saved docs/status.json is reused (so a README refresh never needs patient data present)."""
import json, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8770"


def _get(path):
    with urllib.request.urlopen(BASE + path, timeout=15) as r: return json.loads(r.read())


def collect():
    """Aggregate status for the README. Returns (status dict, True if live)."""
    try:
        ov = _get("/api/ml/overview"); packs = _get("/api/packs"); labels = _get("/api/labels/summary"); exps = _get("/api/experiments")
    except Exception:
        f = ROOT / "docs" / "status.json"
        return (json.loads(f.read_text()) if f.exists() else None), False
    ok = [p for p in packs if not p.get("error")]; best = sorted(ok, key=lambda p: (-p["ship"], -p["version"]))[0] if ok else None
    runs = [r for c in ov["card_types"] for r in c["runs"]]; last = sorted(runs, key=lambda r: r["ts"])[-1] if runs else None
    st = {"updated": time.strftime("%Y-%m-%d"), "card_types": len(ov["card_types"]), "pack_versions": len(ok), "templates": len(ov["templates"]),
          "labeled_cells": labels["cells"], "labeled_pages": labels["pages"], "runs": len(runs), "experiments": len(exps),
          "best_pack": {"name": f"{best['id']} v{best['version']}", "shipped": best["ship"], "fields": best["fields"]} if best else None,
          "last_run": {k: last[k] for k in ("name", "exact", "lo", "hi", "cer", "ece", "silent_error_rate", "coverage", "policy_accuracy", "shipped", "fields", "n_cells", "n_pages")} if last else None}
    (ROOT / "docs").mkdir(exist_ok=True); (ROOT / "docs" / "status.json").write_text(json.dumps(st, indent=1))
    return st, True


def stats_md(st):
    if not st: return "## Status\n\nNo numbers yet: run a training run in Studio, then `python tools/publish.py`.\n"
    r = st.get("last_run"); b = st.get("best_pack"); pc = lambda v: f"{100 * v:.0f}%"
    rows = [("Card types / pack versions", f"{st['card_types']} / {st['pack_versions']}"), ("Saved templates", st["templates"]),
            ("Labeled cells", f"{st['labeled_cells']} on {st['labeled_pages']} pages"), ("Training runs / experiment records", f"{st['runs']} / {st['experiments']}")]
    if b: rows.append(("Fields that meet the ship gate", f"{b['shipped']} of {b['fields']} ({b['name']})"))
    if r:
        rows += [("Held-out exact match, latest run", f"{pc(r['exact'])} (95% {pc(r['lo'])} to {pc(r['hi'])}), {r['n_cells']} cells"), ("Held-out character error rate", f"{100 * r['cer']:.1f}%"),
                 ("Calibration error (ECE)", f"{100 * r['ece']:.1f} points"),
                 ("Under the app's flag rule", f"{pc(r['coverage'])} auto-accepted at {pc(r['policy_accuracy'])} accuracy; silent errors {100 * r['silent_error_rate']:.1f}% of cells")]
    return f"## Status\n\n_Aggregate numbers from the author's own labeled data, updated {st['updated']}. Held-out = pooled 5-fold cross-validation split by page._\n\n| | |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows) + "\n"


def shots_md(root):
    f = root / "docs" / "screenshots" / "MANIFEST.json"
    if not f.exists(): return "_No screenshots yet._\n"
    out = []
    for e in json.loads(f.read_text())["screenshots"]:
        out.append(f"**{e['caption']}**\n\n![{e['view']}]({e['file']})\n")
    return "\n".join(out)


def render(root=ROOT, st=None):
    t = (ROOT / "tools" / "README.template.md").read_text()
    return t.replace("{{STATS}}", stats_md(st)).replace("{{SCREENSHOTS}}", shots_md(root))


if __name__ == "__main__":
    st, live = collect(); print("live" if live else "reused saved status"); (ROOT / "README.generated.md").write_text(render(ROOT, st))
