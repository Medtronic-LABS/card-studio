"""Where a line-dataset training run's own results.json lives, and the comparison table the Model quality tab reads (`/api/runs/
table`, `htr/eval/compare.py` has the richer paired/bootstrap comparison of two runs). Pure Python (no torch): a caller writes one
of these after `htr.train.loop.train()` returns; this module itself never trains anything.

A run's results.json (schema: a deliberately small subset of the full M1 plan - name, when, the architecture, the metrics of its
best epoch, size - grown later as real runs need more, not speculatively now):

    {"name": str, "created": "YYYY-MM-DD HH:MM", "arch_name": str, "arch_hash": str, "n_params": int, "model_size_bytes": int or None,
     "dataset_version": str or None, "best_epoch": int, "metrics": {"exact", "cer_grapheme", "cer_codepoint", "wer", "n"}}

This is the new line-dataset track's own run history, separate from Card Studio's existing per-card results.json (work_phaseA/
train.py, studio/trainer.py) - the two are not merged here; a card-based run never appears in this table."""
import json
from pathlib import Path

from . import tracking as TR

RUNS_ROOT = Path(__file__).resolve().parent / "runs"
REQUIRED = ("name", "arch_name", "arch_hash", "n_params", "best_epoch", "metrics")


def write_result(name, *, arch_name, arch_hash, n_params, best_epoch, metrics, model_size_bytes=None, dataset_version=None, created=None, root=None):
    """Writes `<root or RUNS_ROOT>/<name>/results.json`. `metrics`: the best epoch's own dict (at least exact/cer_grapheme/
    cer_codepoint/wer/n - whatever `htr.train.loop.evaluate` returned for it); stored as given, not recomputed here."""
    import time
    if not name or "/" in name or name in (".", ".."): raise ValueError("a run name may not be empty, contain /, or be . or ..")
    record = {"name": name, "created": created or time.strftime("%Y-%m-%d %H:%M"), "arch_name": arch_name, "arch_hash": arch_hash,
              "n_params": n_params, "model_size_bytes": model_size_bytes, "dataset_version": dataset_version,
              "best_epoch": best_epoch, "metrics": metrics}
    TR.write_json_atomic((Path(root) if root is not None else RUNS_ROOT) / name / "results.json", record)
    return record


def scan(root=None):
    """-> [{..., same shape as write_result's record, plus "ok": True}, ...] for every readable results.json under `root`
    (default RUNS_ROOT), sorted by name. A run directory whose results.json is missing a required key or is not valid JSON is
    still listed, as {"name": dirname, "ok": False, "problem": str} - a broken run should be visible, not silently dropped."""
    root = Path(root) if root is not None else RUNS_ROOT
    if not root.exists(): return []
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        f = d / "results.json"
        if not f.exists(): continue
        try:
            rec = json.loads(f.read_text())
            missing = [k for k in REQUIRED if k not in rec]
            if missing: raise ValueError(f"missing {', '.join(missing)}")
            out.append({**rec, "ok": True})
        except (json.JSONDecodeError, ValueError) as e:
            out.append({"name": d.name, "ok": False, "problem": str(e)})
    return out
