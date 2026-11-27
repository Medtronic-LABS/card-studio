"""Which epoch is the best, when to stop, and what to write to disk. `EarlyStop` is a straight port of
`work_phaseA/train.py:EarlyStop` (same monitor/min_delta/patience/tie-break contract) generalized to a lower-is-better
`cer_grapheme`/`cer_codepoint` monitor alongside the legacy `exact`/`loss`. `Checkpointer` is the same port, adapted to this
package's checkpoint dict shape (model/opt/sched/scaler/RNG/epoch/vocab sha, per M1's design)."""
import json, os, re, time
from pathlib import Path


def write_json_atomic(path, obj):
    """A temp file in the same folder, then a rename: a reader (the web UI polling) never sees half a file."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh: json.dump(obj, fh)
    os.replace(tmp, path)


class EarlyStop:
    """monitor: "exact" (higher is better, ties by lower cer_grapheme), "cer_grapheme"/"cer_codepoint" (lower is better, ties by
    higher exact), "loss" (lower validation CTC loss is better). An epoch is an improvement when it beats the best by more than
    min_delta in the monitor's own units (min_delta 0 = strictly better). Patience counts epochs since the last improvement."""
    MONITORS = ("exact", "cer_grapheme", "cer_codepoint", "loss")

    def __init__(self, monitor="cer_grapheme", min_delta=0.0, patience=20):
        assert monitor in self.MONITORS, monitor
        self.monitor, self.min_delta, self.patience = monitor, float(min_delta), patience
        self.best_ep, self.best = 0, None   # best: the full metrics dict of the best epoch

    def _key(self, metrics):   # higher is better: (primary, tie-break)
        m = self.monitor
        if m == "exact": return (metrics["exact"], -metrics.get("cer_grapheme", 0.0))
        if m == "loss": return (-metrics["loss"], metrics.get("exact", 0.0))
        return (-metrics[m], metrics.get("exact", 0.0))

    def update(self, ep, metrics):
        """Record epoch ep's validation metrics (a dict with at least the monitored key). -> True when it is a new best."""
        k = self._key(metrics)
        if self.best is None: better = True
        else:
            b = self._key(self.best)
            better = (k[0] > b[0] + self.min_delta) if self.min_delta > 0 else (k > b)
        if better: self.best_ep, self.best = ep, dict(metrics)
        return better

    def should_stop(self, ep): return self.best is not None and ep - self.best_ep >= self.patience
    def state(self): return {"best_ep": self.best_ep, "best": self.best}
    def load(self, st): self.best_ep, self.best = st["best_ep"], st["best"]


def prune_numbered(dirpath, name, keep_last):
    """Delete all but the newest keep_last numbered checkpoints <name>-eNNN.pt in dirpath (the -best and -last files never match)."""
    pat = re.compile(re.escape(name) + r"-e(\d+)\.pt$")
    found = sorted(((int(m.group(1)), f) for f in Path(dirpath).iterdir() if (m := pat.match(f.name))), reverse=True)
    gone = []
    for _, f in found[max(1, keep_last):]: f.unlink(); gone.append(f.name)
    return gone


class Checkpointer:
    """The files of one training run. cfg: {"dir", "name", "every", "keep_last", "stop", "curves", "resume"} - same shape as
    `work_phaseA/train.py`'s, so `studio/trainer.py`'s stop-file / curves-file wiring needs no change."""
    def __init__(self, cfg):
        import torch
        self._torch = torch
        self.dir, self.name = Path(cfg["dir"]), cfg["name"]
        self.every, self.keep = cfg.get("every", 0), cfg.get("keep_last", 2)
        self.stop_file, self.curves, self.resume = cfg.get("stop"), cfg.get("curves"), bool(cfg.get("resume"))
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, kind): return self.dir / f"{self.name}-{kind}.pt"

    def _save(self, obj, path):
        tmp = path.with_name(path.name + ".tmp"); self._torch.save(obj, tmp); os.replace(tmp, path)

    def save_best(self, state): self._save(state, self.path("best"))
    def save_last(self, state): self._save(state, self.path("last"))

    def save_numbered(self, ep, state):
        self._save(state, self.path(f"e{ep:03d}")); prune_numbered(self.dir, self.name, self.keep)

    def due(self, ep): return self.every > 0 and ep % self.every == 0
    def stop_requested(self): return bool(self.stop_file) and Path(self.stop_file).exists()
    def pause_requested(self): return bool(self.stop_file) and (Path(self.stop_file).parent / "PAUSE").exists()
    def wait_if_paused(self, save=None):
        """Called once an epoch is fully recorded. While RUN_DIR/PAUSE exists (and STOP does not) the process stays alive and waits here, so a run can be
        inspected and carried on without a restart. `save` writes the -last checkpoint once on entering the pause, so a pause the process does not survive can still be resumed."""
        if not self.pause_requested() or self.stop_requested(): return
        if save: save()
        while self.pause_requested() and not self.stop_requested(): time.sleep(1)

    def write_curves(self, rows):
        if self.curves: write_json_atomic(self.curves, rows)

    def load_last(self):
        p = self.path("last")
        return self._torch.load(p, map_location="cpu", weights_only=False) if self.resume and p.exists() else None
