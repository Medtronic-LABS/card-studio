"""Should a candidate run replace the shipped one? The gate the plan calls for: a paired bootstrap CI of the improvement that
excludes zero (or a CER improvement of at least 15% relative, for a candidate too different from the shipped one to pair cleanly),
no document type the shipped model handles regressing, and - when a device benchmark exists - the mobile targets (pack size,
peak memory, P50 latency) actually met. Never auto-promotes on accuracy alone when a device record says the pack will not fit; a
reviewer can still force it (`override`) with their name on the decision.

Pure Python; works on `htr.eval.compare`'s output, not on any particular run-storage format."""
MOBILE_TARGETS = {"size_bytes": (5_000_000, 15_000_000), "peak_rss_mb": (0, 100), "p50_ms": (0, 300)}
REGRESSION_P = 0.05                 # a document type's shipped-only-right count beating the candidate's at p < this is a regression, not noise
MIN_RELATIVE_CER_IMPROVEMENT_PCT = 15.0


def _regressions(by_document_type):
    return [d["document_type"] for d in by_document_type if d["only_a"] > d["only_b"] and d["p"] < REGRESSION_P]


def _mobile_problems(device_bench):
    problems = []
    for key, (lo, hi) in MOBILE_TARGETS.items():
        v = device_bench.get(key)
        if v is None: continue
        if not (lo <= v <= hi): problems.append(f"{key} {v:g} is outside the target {lo:g}-{hi:g}")
    return problems


def decide(comparison, candidate_agg, shipped_agg, device_bench=None, min_relative_improvement_pct=MIN_RELATIVE_CER_IMPROVEMENT_PCT):
    """comparison: `htr.eval.compare.paired(candidate_rows, shipped_rows)` (so `diff`/`ci_bootstrap` are candidate minus shipped).
    candidate_agg/shipped_agg: `htr.eval.compare.aggregate(...)` of each run's own rows (for the CER-improvement fallback and the
    record). device_bench: {size_bytes, peak_rss_mb, p50_ms} from a real device benchmark, or None when there isn't one yet (mobile
    targets are then skipped, not assumed met). -> {promoted, promoted_by, reasons, accuracy_ok, no_regression, mobile_ok,
    relative_cer_improvement_pct}."""
    reasons = []
    ci_lo, ci_hi = comparison["ci_bootstrap"]
    excludes_zero_improving = ci_lo > 0
    shipped_cer = shipped_agg["cer_grapheme"]
    relative_improvement = ((shipped_cer - candidate_agg["cer_grapheme"]) / shipped_cer * 100) if shipped_cer else 0.0
    accuracy_ok = excludes_zero_improving or relative_improvement >= min_relative_improvement_pct
    if not accuracy_ok:
        reasons.append(f"improvement not significant: 95% CI on the exact-match difference is [{ci_lo:.4f}, {ci_hi:.4f}] (does not exclude 0) "
                        f"and grapheme CER only improved {relative_improvement:.1f}% (need {min_relative_improvement_pct:.0f}%)")
    regressed = _regressions(comparison["by_document_type"])
    if regressed: reasons.append(f"regresses (at p < {REGRESSION_P}) on: {', '.join(regressed)}")
    mobile_ok = True
    if device_bench is not None:
        problems = _mobile_problems(device_bench)
        mobile_ok = not problems
        reasons.extend(problems)
    promoted = accuracy_ok and not regressed and mobile_ok
    return {"promoted": promoted, "promoted_by": "auto" if promoted else None, "reasons": reasons, "accuracy_ok": accuracy_ok,
            "no_regression": not regressed, "mobile_ok": mobile_ok, "relative_cer_improvement_pct": relative_improvement,
            "device_bench_checked": device_bench is not None}


def override(decision, reviewer):
    """A reviewer promotes (or keeps un-promoted) a candidate despite what `decide` found, with their name on the record. ->
    decision with `promoted` forced True, `promoted_by: "human"`, `overridden_by: reviewer`, and the original reasons kept."""
    if not reviewer: raise ValueError("a manual override needs the reviewer's name")
    return {**decision, "promoted": True, "promoted_by": "human", "overridden_by": reviewer}
