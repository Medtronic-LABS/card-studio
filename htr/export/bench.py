"""Benchmark a reader before shipping it: latency (P50/P95) and size, on this machine; `tools/benchmark_device.py` runs the same
idea on an actual phone over adb. Needs onnxruntime (always available - every pack ships an ONNX reader) and, for peak memory,
`resource` (POSIX only; skipped elsewhere)."""
import time


def _percentile(sorted_vals, p):
    if not sorted_vals: return float("nan")
    k = (len(sorted_vals) - 1) * p / 100
    import math
    f, c = math.floor(k), math.ceil(k)
    if f == c: return sorted_vals[int(k)]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def bench_onnx(onnx_path, img_h, img_w, runs=50, warmup=5):
    """-> {size_bytes, p50_ms, p95_ms, peak_rss_mb (None where `resource` isn't available), runs}. Runs the exported ONNX reader
    `runs` times on one all-zero cell of (img_h, img_w) with onnxruntime's CPU provider - a rough, portable stand-in for the
    on-device number; `benchmark_device.py` is what gives the real one."""
    import numpy as np
    import onnxruntime as ort
    from pathlib import Path

    size_bytes = Path(onnx_path).stat().st_size
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.zeros((1, 1, img_h, img_w), dtype=np.float32)
    name = sess.get_inputs()[0].name
    for _ in range(warmup): sess.run(None, {name: x})
    try:
        import resource
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        rss_before = None
    times = []
    for _ in range(runs):
        t0 = time.perf_counter(); sess.run(None, {name: x}); times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    peak_rss_mb = None
    if rss_before is not None:
        import resource
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_mb = rss_after / (1024 if __import__("sys").platform == "linux" else 1024 * 1024)
    return {"size_bytes": size_bytes, "p50_ms": round(_percentile(times, 50), 3), "p95_ms": round(_percentile(times, 95), 3),
            "peak_rss_mb": round(peak_rss_mb, 1) if peak_rss_mb is not None else None, "runs": runs}


def bench_buckets(onnx_path_of_bucket, buckets, img_h, runs=20):
    """onnx_path_of_bucket(bucket_width) -> an ONNX file for that width (a caller has usually exported one per bucket already). ->
    {bucket: bench_onnx(...) report}, for the per-bucket latency a multi-signature TFLite pack will show in its own report."""
    return {b: bench_onnx(onnx_path_of_bucket(b), img_h, b, runs=runs) for b in buckets}
