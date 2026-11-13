"""Is one setting really better than another, or is it luck? Run each as an experiment of several seeds on the same folds, then compare.

A single training run moves with its seed (initial weights, batch order, augmentation), and at this data size that movement can be as large as the effect of a setting. So a setting is
tried as an EXPERIMENT: the same cross-validation folds, N seeds, cross-validation only (no final reader, no pack). Two experiments are compared cell by cell on the cells they both
scored: each cell's score is its fraction of seeds that read it right, the difference is averaged over cells, and its interval comes from resampling whole pages. The seed-to-seed spread
of each experiment is shown beside it: a difference smaller than that spread is not evidence of anything. The held-out test set (cards.holdout) is not used here.

Also counted: how many different settings have been tried on a card. The more that have been, the more the best cross-validation score flatters its setting (you picked the winner of a
lottery): the number belongs to the next honest test, not to this one.

    python seedexp.py run CARD NAME [--settings FILE.json] [--seeds 3] [--pretrained auto] [--max-epochs 300]
    python seedexp.py compare CARD A B          python seedexp.py list CARD          python seedexp.py tried CARD
Experiments live in studio/seedexp/<card>/<name>/ (gitignored).
"""
import hashlib, json, math, random, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "work_phaseA")); sys.path.insert(0, str(HERE.parent / "cardpack")); sys.path.insert(0, str(HERE))
import cards

ROOT = HERE / "seedexp"
NAME = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
MAX_SEEDS = 10
BOOT = 2000


def exp_dir(card, name):
    return ROOT / card / name


def settings_hash(settings):
    return hashlib.sha256(json.dumps(settings or {}, sort_keys=True).encode()).hexdigest()[:10]


def status(card, name):
    try: return json.loads((exp_dir(card, name) / "status.json").read_text())
    except (OSError, ValueError): return None


def run(card, name, settings=None, seeds=3, pretrained="auto", max_epochs=300, log=print):
    """Run `seeds` cross-validation-only trainings, one after the other. Blocks until done. -> summary"""
    import settings as S
    if not NAME.match(name or ""): raise ValueError("an experiment name is letters, digits, _ and -, at most 32")
    if not cards.exists(card): raise ValueError("no such card")
    if not 2 <= seeds <= MAX_SEEDS: raise ValueError(f"seeds is 2 to {MAX_SEEDS} (one seed cannot show the spread)")
    d = exp_dir(card, name)
    if d.exists(): raise ValueError(f"an experiment called {name} already exists for {card}")
    parsed = S.load(settings or None); import pretrained as P; P.resolve(pretrained, parsed["arch"])            # fails early, saying why
    d.mkdir(parents=True); (d / "settings.json").write_text(json.dumps(settings or {}, indent=1))
    meta = {"card": card, "name": name, "seeds": seeds, "pretrained": pretrained, "max_epochs": max_epochs, "settings": settings or {}, "settings_hash": settings_hash(settings), "arch_label": parsed["arch_label"], "started": time.strftime("%Y-%m-%d %H:%M")}
    def st(state, done, detail=""): (d / "status.json").write_text(json.dumps({**meta, "state": state, "seeds_done": done, "detail": detail}))
    try:
        for s in range(seeds):
            st("running", s, f"seed {s + 1} of {seeds}")
            cmd = [sys.executable, str(HERE / "trainer.py"), str(d / f"seed-{s}"), "--card", card, "--cv-only", "--seed", str(s), "--pretrained", pretrained, "--max-epochs", str(max_epochs)]
            if settings: cmd += ["--settings", str(d / "settings.json")]
            r = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True)
            try: ok = json.loads((d / f"seed-{s}" / "status.json").read_text()).get("stage") == "done"
            except (OSError, ValueError): ok = False
            if not ok: raise ValueError(f"seed {s} failed: {(r.stdout + r.stderr)[-300:]}")
        sm = summary(card, name); (d / "summary.json").write_text(json.dumps(sm, indent=1)); st("done", seeds, "finished"); return sm
    except Exception as e:
        st("failed", 0, str(e)[-300:]); raise


def _preds(card, name):
    """[ {(page,row,col): (label, pred)} per seed ]"""
    out = []
    for p in sorted((exp_dir(card, name)).glob("seed-*/cv_predictions.json")):
        out.append({(r["page"], r["row"], r["col"]): (r["label"], r["pred"]) for r in json.loads(p.read_text())})
    return out


def summary(card, name):
    """-> {name, seeds, exact: [per seed], mean, spread: [min, max], by_field_mean}. Exact match of whole cells in cross-validation, greedy decoding."""
    per = _preds(card, name)
    if not per: raise ValueError("no finished seeds")
    ex = [sum(l == p for l, p in s.values()) / len(s) for s in per]
    by = {}
    for s in per:
        for (page, row, col), (l, p) in s.items(): by.setdefault(row, []).append(l == p)
    return {"name": name, "seeds": len(per), "cells": len(per[0]), "exact": [round(x, 4) for x in ex], "mean": round(sum(ex) / len(ex), 4), "spread": [round(min(ex), 4), round(max(ex), 4)],
            "by_field": {k: round(sum(v) / len(v), 3) for k, v in sorted(by.items())}}


def compare(card, a, b):
    """Paired comparison of two experiments on the cells both scored. -> {a, b, cells, diff (b minus a, mean over cells and seeds), ci95 (page bootstrap), only_b_better, only_a_better,
    sign_p, verdict, noise (the larger seed spread), warning}"""
    pa, pb = _preds(card, a), _preds(card, b)
    if not pa or not pb: raise ValueError("both experiments need finished seeds")
    keys = sorted(set.intersection(*[set(s) for s in pa + pb]))
    if not keys: raise ValueError("the two experiments share no cells")
    sc = lambda ps, k: sum(s[k][0] == s[k][1] for s in ps) / len(ps)            # noqa: E731  fraction of seeds that read the cell right
    ma = {k: sc(pa, k) for k in keys}; mb = {k: sc(pb, k) for k in keys}
    d = {k: mb[k] - ma[k] for k in keys}; diff = sum(d.values()) / len(d)
    pages = {}
    for k in keys: pages.setdefault(k[0], []).append(d[k])
    rng = random.Random(20261103); names = sorted(pages); boots = []
    for _ in range(BOOT):
        pick = [pages[rng.choice(names)] for _ in names]; tot = sum(len(x) for x in pick)
        boots.append(sum(sum(x) for x in pick) / tot)
    boots.sort(); lo, hi = boots[int(0.025 * BOOT)], boots[int(0.975 * BOOT)]
    wins, losses = sum(v > 0 for v in d.values()), sum(v < 0 for v in d.values())
    n = wins + losses; p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(wins, losses) + 1)) / 2 ** n)
    sa, sb = summary(card, a), summary(card, b); noise = max(sa["spread"][1] - sa["spread"][0], sb["spread"][1] - sb["spread"][0])
    verdict = f"{b} is better" if lo > 0 else f"{a} is better" if hi < 0 else "no clear difference"
    warn = None
    if verdict != "no clear difference" and abs(diff) < noise: warn = f"The difference ({diff * 100:+.1f} points) is smaller than how much one experiment moves between seeds ({noise * 100:.1f} points): do not trust it."
    elif verdict == "no clear difference": warn = "The data cannot tell these two settings apart. Keep the simpler one, or add labels before comparing again."
    return {"a": sa, "b": sb, "cells": len(keys), "pages": len(pages), "diff": round(diff, 4), "ci95": [round(lo, 4), round(hi, 4)], "only_b_better": wins, "only_a_better": losses,
            "sign_p": round(p, 4), "verdict": verdict, "noise": round(noise, 4), "warning": warn}


def listing(card):
    out = []
    for d in sorted((ROOT / card).glob("*")) if (ROOT / card).exists() else []:
        s = status(card, d.name)
        if not s: continue
        row = {"name": d.name, "state": s["state"], "seeds": s["seeds"], "seeds_done": s.get("seeds_done", 0), "arch": s["arch_label"], "settings_hash": s["settings_hash"], "started": s["started"], "detail": s.get("detail", "")}
        if (d / "summary.json").exists(): row.update(json.loads((d / "summary.json").read_text()) | {"name": d.name})
        out.append(row)
    return out


def tried(card):
    """How many different settings have been tried on this card (finished runs and experiments, counted by their settings). -> {settings_tried, message or None}"""
    seen = set()
    for r in listing(card): seen.add(r["settings_hash"])
    runs = HERE / "runs"
    for d in sorted(runs.glob("run-*")) if runs.exists() else []:
        try: res = json.loads((d / "results.json").read_text())
        except (OSError, ValueError): continue
        if res.get("card") == card: seen.add(settings_hash((res.get("settings") or {}).get("hp") and res["settings"]))
    n = len(seen)
    msg = None if n < 4 else f"{n} different settings have been tried on this card. The best cross-validation score among them is optimistic: confirm the winner once on the test set, and do not keep tuning against it."
    return {"settings_tried": n, "message": msg}


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d              # noqa: E731
    try:
        if a[:1] == ["run"] and len(a) >= 3:
            st = json.loads(Path(opt("--settings")).read_text()) if opt("--settings") else None
            print(json.dumps(run(a[1], a[2], st, int(opt("--seeds", 3)), opt("--pretrained", "auto"), int(opt("--max-epochs", 300))), indent=1))
        elif a[:1] == ["compare"] and len(a) == 4: print(json.dumps(compare(a[1], a[2], a[3]), indent=1))
        elif a[:1] == ["list"] and len(a) == 2: print(json.dumps(listing(a[1]), indent=1))
        elif a[:1] == ["tried"] and len(a) == 2: print(json.dumps(tried(a[1]), indent=1))
        else: sys.exit("usage: python seedexp.py run CARD NAME [--settings FILE] [--seeds 3] [--pretrained auto] [--max-epochs 300] | compare CARD A B | list CARD | tried CARD")
    except ValueError as e:
        sys.exit(str(e))
