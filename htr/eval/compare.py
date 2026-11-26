"""Compare runs: `build` turns a list of runs (predictions + size/latency) into `comparison.json` rows for a sortable table; `paired`
is a page/writer-bootstrap paired comparison of two runs on the lines they share, the line-dataset counterpart of
`studio/analysis.paired` (which does the same thing for card cells, one page at a time). Pure Python (no numpy), so it needs nothing
a training run's own environment doesn't already have.

A run's rows: [{"line_id", "ref", "hyp", "document_type", "writer_id"}, ...] - exactly `htr.data.manifest.COLS`' `line_id`/`text`
(normalized, as `ref`) plus a model's own decode (`hyp`), `document_type` and `writer_id` for the breakdowns and the bootstrap unit."""
import math
import random

from . import metrics as M
from ..data import splits as SP


def aggregate(rows):
    """-> {n, exact, cer_grapheme, cer_codepoint, wer, numeral_accuracy, conjunct_accuracy}."""
    pairs = [(r["ref"], r["hyp"]) for r in rows]
    return {"n": len(rows), "exact": M.exact(pairs), "cer_grapheme": M.cer_grapheme(pairs), "cer_codepoint": M.cer_codepoint(pairs),
            "wer": M.wer(pairs), "numeral_accuracy": M.numeral_accuracy(pairs), "conjunct_accuracy": M.conjunct_accuracy(pairs)}


def build(runs, out_path=None):
    """runs: [{"name", "precision" ("fp32"|"int8"), "rows": [...], "n_params": int, "size_bytes": int, "latency": {device, p50, p95} or
    None}, ...]. -> {"rows": [one row per run, aggregate + by_document_type + the metadata passed in]}. Writes `out_path` (JSON) when given."""
    import json
    from pathlib import Path
    out = []
    for run in runs:
        row = {"name": run["name"], "precision": run.get("precision", "fp32"), "n_params": run.get("n_params"), "size_bytes": run.get("size_bytes"),
               "latency": run.get("latency"), **aggregate(run["rows"]), "by_document_type": M.breakdown(run["rows"], "document_type")}
        out.append(row)
    result = {"rows": out}
    if out_path: Path(out_path).write_text(json.dumps(result, indent=1, ensure_ascii=False))
    return result


def mcnemar_exact(b01, b10):
    """Two-sided exact binomial test on the discordant lines (b01: only B right, b10: only A right)."""
    n = b01 + b10
    if n == 0: return 1.0
    k = min(b01, b10)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def _percentile(sorted_vals, p):
    if not sorted_vals: return float("nan")
    k = (len(sorted_vals) - 1) * p / 100
    f, c = math.floor(k), math.ceil(k)
    if f == c: return sorted_vals[int(k)]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def paired(rows_a, rows_b, boot=2000, seed=0):
    """Paired comparison of two runs on the lines (`line_id`) scored by both. -> {common, acc_a, acc_b, diff (b - a), ci_bootstrap
    (resampling whole writer/doc units, not lines - lines from one writer are correlated), p_mcnemar, only_a, only_b, both, neither,
    by_document_type, notes}. Raises ValueError when fewer than 10 lines are shared."""
    ka = {r["line_id"]: r for r in rows_a}; kb = {r["line_id"]: r for r in rows_b}
    keys = sorted(set(ka) & set(kb))
    if len(keys) < 10: raise ValueError("the two runs share fewer than 10 lines, so a paired comparison is not meaningful")
    A = [ka[k]["hyp"] == ka[k]["ref"] for k in keys]
    B = [kb[k]["hyp"] == kb[k]["ref"] for k in keys]
    units = [SP.unit_of(ka[k]) for k in keys]
    only_b = sum(b and not a for a, b in zip(A, B)); only_a = sum(a and not b for a, b in zip(A, B))
    both = sum(a and b for a, b in zip(A, B)); neither = sum(not a and not b for a, b in zip(A, B))
    unit_list = sorted(set(units)); idx_by_unit = {u: [i for i, uu in enumerate(units) if uu == u] for u in unit_list}
    rng = random.Random(seed); diffs = []
    for _ in range(boot):
        idxs = [i for u in (rng.choice(unit_list) for _ in unit_list) for i in idx_by_unit[u]]
        if not idxs: diffs.append(0.0); continue
        diffs.append(sum(B[i] for i in idxs) / len(idxs) - sum(A[i] for i in idxs) / len(idxs))
    diffs.sort()
    by_doc = {}
    for i, k in enumerate(keys):
        d = by_doc.setdefault(ka[k].get("document_type", "?"), {"n": 0, "only_a": 0, "only_b": 0, "both": 0, "neither": 0}); d["n"] += 1
        d["only_a" if A[i] and not B[i] else "only_b" if B[i] and not A[i] else "both" if A[i] else "neither"] += 1
    by_document_type = [{"document_type": t, **d, "p": mcnemar_exact(d["only_b"], d["only_a"])} for t, d in sorted(by_doc.items(), key=lambda kv: -kv[1]["n"])]
    return {"common": len(keys), "units": len(unit_list), "cells_a": len(ka), "cells_b": len(kb),
            "acc_a": sum(A) / len(A), "acc_b": sum(B) / len(B), "diff": sum(B) / len(B) - sum(A) / len(A),
            "only_a": only_a, "only_b": only_b, "both": both, "neither": neither,
            "p_mcnemar": mcnemar_exact(only_b, only_a), "ci_bootstrap": [_percentile(diffs, 2.5), _percentile(diffs, 97.5)], "bootstrap": boot,
            "by_document_type": by_document_type,
            "notes": ["McNemar looks only at the lines where the runs disagree and treats them as independent (they are not: lines from one writer are correlated), so its p-value is optimistic.",
                      "The bootstrap resamples whole writer/document units, which respects that correlation; read the interval, not the p-value.",
                      "Only lines scored by both runs are compared." + ("" if len(ka) == len(kb) == len(keys) else f" Run A has {len(ka)} lines, run B {len(kb)}, {len(keys)} in common.")]}
