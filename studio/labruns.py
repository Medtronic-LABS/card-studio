"""What the command-line experiments are doing, for the History tab. Read-only; no torch; local only.

capacity.py, synthexp.py, constrained.py and xscript.py are scripts, not Studio jobs: each writes a status.json (stage and a line of detail) and
result files under studio/capacity/. This reads them, checks whether the script is still alive, and turns each into one record the page can show:
a state, a progress bar where the experiment has a known number of parts, and a table of the results so far.

A script's status.json is rewritten only when it moves to its next part, so its age says nothing about whether it is running. Whether the script
is running is read from the process list. A status that is not finished and has no process is shown as stopped.

Only counts and accuracies leave this module. The result files also hold each cell's label and reading (patient-derived values); those are used
to compute paired tests and never returned.
"""
import json, math, re, subprocess, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAP = HERE / "capacity"
TERMINAL = ("done", "error", "stopped")
# size of each reader, measured by capacity.facts() (parameters, MB as float32); kept here so this module needs no torch
SIZES = {"base": (0.54, 2.2), "big_crnn": (4.29, 17.2), "attn": (5.08, 20.3)}
NAMES = {"base": "base CRNN (shipped)", "big_crnn": "bigger CRNN", "attn": "attention encoder"}
EXPERIMENTS = [
    {"id": "capacity", "title": "Larger readers", "dir": "full", "script": "capacity.py", "total": 6,
     "question": "Does a 15 to 20 MB reader read the ANC card better than the shipped 2 MB one? Three networks, each pretrained on a small and a large synthetic set, same cross-validation folds."},
    {"id": "generator", "title": "Synthetic data v2", "dir": "v2/full", "script": "synthexp.py", "total": None,
     "question": "Does pretraining on cells drawn like the real ones (grammars, empty-cell backgrounds, damage) beat stitched digit strings?"},
    {"id": "grammar", "title": "Grammar decoding", "dir": "constrained/full", "script": "constrained.py", "total": None,
     "question": "Does reading each field against the values it can hold (valid dates, pressures, ranged numbers) beat taking the best class at every step?"},
    {"id": "crossscript", "title": "Cross-script digits", "dir": "xscript/full", "script": "xscript.py", "total": 5,
     "question": "Does a digit reader trained on one script's digits read another's? (parked: Bangla only for now)"},
]


def _json(p):
    try: return json.loads(Path(p).read_text())
    except (OSError, ValueError): return None


def _processes():
    """{script name: elapsed time as printed by ps} for the experiment scripts that are running now."""
    try: out = subprocess.run(["ps", "-ax", "-o", "etime=,command="], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError): return {}
    found = {}
    for line in out.splitlines():
        m = re.match(r"\s*(\S+)\s+(.*)", line)
        if not m: continue
        for e in EXPERIMENTS:
            if re.search(r"python[0-9.]*\s+(\S*/)?" + re.escape(e["script"]) + r"\s", m[2] + " ") and " run" in " " + m[2] and "grep" not in m[2]:
                found.setdefault(e["script"], m[1])
    return found


def _wilson(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _sign_test(a, b):
    ka = {(r["page"], r["row"], r["col"]): r["pred"] == r["label"] for r in a}; kb = {(r["page"], r["row"], r["col"]): r["pred"] == r["label"] for r in b}
    keys = ka.keys() & kb.keys(); fa = sum(ka[k] and not kb[k] for k in keys); fb = sum(kb[k] and not ka[k] for k in keys); n = fa + fb
    if n == 0: return fa, fb, 1.0
    k = min(fa, fb); return fa, fb, min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def _pct(s, g):
    return f"{s[g]['exact'] * 100:.0f}% [{s[g]['lo'] * 100:.0f}-{s[g]['hi'] * 100:.0f}]" if g in s else "-"


def _count(s, g):
    return f"{s[g]['correct']}/{s[g]['n']}" if g in s else "-"


# ---------------------------------------------------------------- one adapter per experiment
def _capacity(d):
    cols = ["Reader", "Size", "Synthetic data", "All cells", "Marks", "Numbers, dates, BP", "Dates", "Better than the shipped reader?"]; res = {}
    for p in sorted(d.glob("cv_*.json")):
        r = _json(p)
        if r: res[(r["variant"], r["cond"])] = r
    base = res.get(("base", "small")); rows = []
    for (v, c), r in sorted(res.items(), key=lambda kv: (SIZES[kv[0][0]][1], kv[0][1] != "small")):
        s = r["score"]; params, mb = SIZES[v]
        if (v, c) == ("base", "small"): verdict = "the reference"
        elif base:
            fa, fb, p = _sign_test(r["preds"], base["preds"]); verdict = f"{fa} cells right that it got wrong, {fb} the other way, p = {p:.3f}"
        else: verdict = "-"
        rows.append([NAMES[v], f"{params}M, {mb} MB", f"{r['synthetic']['sequences']:,} strings, {r['synthetic']['epochs']} epochs", _pct(s, "all"), _pct(s, "marks"), _pct(s, "numbers+dates+bp"), _count(s, "dates"), verdict])
    return cols, rows, len(res), ["Page-split 5-fold cross-validation on 369 labeled ANC cells, same folds for every reader. Cells on one page are correlated, so intervals and p-values are optimistic.",
                                 "A p-value below 0.05 across six comparisons would need to be below about 0.008 to survive a correction for testing six readers."]


def _generator(d):
    s = _json(d / "summary.json")
    if not s: return ["Pretraining", "All cells", "Marks", "Numbers, dates, BP", "Dates"], [], None, []
    rows = [[f"v2: {s['v2']['images']:,} images from synthgen", _pct(s["v2"]["score"], "all"), _pct(s["v2"]["score"], "marks"), _pct(s["v2"]["score"], "numbers+dates+bp"), _count(s["v2"]["score"], "dates")]]
    for k, v in s["others"].items(): rows.append([f"old: {v['synthetic']['sequences']:,} stitched strings ({k})", _pct(v["score"], "all"), _pct(v["score"], "marks"), _pct(v["score"], "numbers+dates+bp"), _count(v["score"], "dates")])
    notes = [f"v2 against {k.split(' vs ')[1]}: {v['only_first']} cells only v2 gets right, {v['only_second']} only the other, p = {v['p']:.3f}" for k, v in s["paired"].items()]
    return ["Pretraining", "All cells", "Marks", "Numbers, dates, BP", "Dates"], rows, None, notes


def _grammar(d):
    s = _json(d / "summary.json")
    if not s: return ["Fields", "Greedy", "With grammar"], [], None, []
    rows = [[g, f"{v['greedy']['correct']}/{v['greedy']['n']} ({v['greedy']['exact'] * 100:.0f}%)", f"{v['constrained']['correct']}/{v['constrained']['n']} ({v['constrained']['exact'] * 100:.0f}%)"] for g, v in s["results"].items() if g != "label_outside_grammar"]
    return ["Fields", "Greedy", "With grammar"], rows, None, [f"Same fine-tuned models from {s['weights']}; only the decoding differs.", f"Labels the grammar cannot produce: {s['results'].get('label_outside_grammar')}"]


def _xscript(d):
    tests = ["bangla", "kannada", "kannada_dig", "devanagari", "telugu"]; rows = []
    for p in sorted(d.glob("*.json")):
        if p.name in ("status.json", "summary.json"): continue
        r = _json(p)
        if r and "tests" in r: rows.append([p.stem] + [f"{r['tests'][t]['exact'] * 100:.0f}%" + ("" if t.split("_")[0] in r["scripts"] else " *") for t in tests])
    return ["Trained on"] + tests, rows, len(rows), ["Exact match on 1 to 4 digit strings made of held-out glyphs. * = a script the reader never trained on."]


ADAPTERS = {"capacity": _capacity, "generator": _generator, "grammar": _grammar, "crossscript": _xscript}


def one(e, procs):
    d = CAP / e["dir"]; st = _json(d / "status.json")
    if st is None and not any(d.glob("*.json")): return None
    st = st or {}; stage = st.get("stage", "unknown"); alive = e["script"] in procs
    state = "running" if alive and stage not in ("done", "error") else stage if stage in TERMINAL else "stopped" if not alive else "running"
    cols, rows, done, notes = ADAPTERS[e["id"]](d)
    total = e["total"]; progress = {"done": done, "total": total} if (total and done is not None) else None
    return {"id": e["id"], "title": e["title"], "question": e["question"], "state": state, "stage": stage, "detail": st.get("detail", ""), "elapsed_s": st.get("elapsed"),
            "process_time": procs.get(e["script"]), "progress": progress, "columns": cols, "rows": rows, "notes": notes, "command": f"python studio/{e['script']} run",
            "status_file": f"studio/capacity/{e['dir']}/status.json"}


def jobs():
    """Every experiment that has left a trace, running ones first, then the most recently active."""
    procs = _processes(); out = []
    for e in EXPERIMENTS:
        j = one(e, procs)
        if j:
            f = CAP / e["dir"] / "status.json"; out.append((j["state"] != "running", -(f.stat().st_mtime if f.exists() else 0), j))
    return [j for _, _, j in sorted(out, key=lambda t: t[:2])]
