"""Benchmark a reader.tflite on a real Android device over adb, using LiteRT's own `benchmark_model` binary (built from the
tensorflow/tensorflow repo's `tensorflow/lite/tools/benchmark` target, or downloaded from a LiteRT release - this script does not
build or fetch it). Synthetic lines only: refuses to push a path under `cards/`, `work/` or `imports/` (where real patient crops
live), the same boundary `tools/leakcheck.py` draws for the repo.

CLI: python tools/benchmark_device.py READER.tflite --device-id entry-12nm --benchmark-model /path/to/benchmark_model
     [--serial SERIAL] [--devices cardpack/schema/devices.json]
"""
import argparse, json, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_DEVICES = ROOT / "cardpack" / "schema" / "devices.json"
REMOTE_DIR = "/data/local/tmp/htr_bench"
DENY_PATH_PARTS = {"cards", "work", "imports"}


class NoDevice(Exception):
    """No adb device attached (or the named serial is not among them)."""


def refuse_patient_paths(path):
    """Raises ValueError if `path` looks like it is under a patient-data directory (cards/, work/, imports/) rather than a
    synthetic/exported one. Not a full guarantee (a caller could rename a crop into some other folder), but it catches the
    straightforward mistake of pointing this at a real card's cells."""
    parts = set(Path(path).resolve().parts)
    bad = parts & DENY_PATH_PARTS
    if bad: raise ValueError(f"{path}: looks like it is under a patient-data directory ({', '.join(sorted(bad))}); benchmark synthetic lines only")


def load_devices(path=DEFAULT_DEVICES):
    return json.loads(Path(path).read_text())["devices"]


def device_record(device_id, path=DEFAULT_DEVICES):
    for d in load_devices(path):
        if d["id"] == device_id: return d
    raise KeyError(f"no such device id {device_id!r} in {path}")


def parse_adb_devices(output):
    """`adb devices` stdout -> [serial, ...] (only lines reporting state "device", not "offline"/"unauthorized")."""
    out = []
    for line in output.splitlines()[1:]:
        line = line.strip()
        if not line or "\t" not in line: continue
        serial, state = line.split("\t", 1)
        if state.strip() == "device": out.append(serial)
    return out


def list_devices():
    r = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=30)
    return parse_adb_devices(r.stdout)


BENCH_LINE = re.compile(r"(?P<key>Inference \(avg\)|Memory footprint.*?overall_resident|Peak memory footprint.*?KB):\s*(?P<value>[\d.]+)")


def parse_benchmark_model_output(text):
    """benchmark_model's stdout -> {avg_ms, peak_rss_mb} (None for a field its output did not have - versions vary in what they
    print). avg_ms: the "Inference (avg)" line (microseconds in the tool's own output, converted here). peak_rss_mb: the overall
    resident memory footprint line, when present (KB in the tool's output, converted here)."""
    avg_us = re.search(r"Inference \(avg\):\s*([\d.]+)", text)
    peak_kb = re.search(r"overall_resident[^:]*:\s*([\d.]+)", text) or re.search(r"Peak memory footprint[^:]*:\s*([\d.]+)\s*KB", text)
    return {"avg_ms": (float(avg_us.group(1)) / 1000) if avg_us else None, "peak_rss_mb": (float(peak_kb.group(1)) / 1024) if peak_kb else None}


def run(tflite_path, device_id, benchmark_model, serial=None, devices_path=DEFAULT_DEVICES, num_runs=50):
    """Push `benchmark_model` and `tflite_path` to the device, run it, pull back a report. -> {device_id, serial, size_bytes,
    avg_ms, peak_rss_mb, targets_met: {size_bytes, peak_rss_mb, p50_ms (compared against avg_ms as a stand-in: benchmark_model
    does not report percentiles, only the average, across num_runs)}, raw}. Raises NoDevice when no device (or the named serial)
    is attached; ValueError when the file paths look wrong or the binary fails."""
    refuse_patient_paths(tflite_path)
    rec = device_record(device_id, devices_path)
    serials = list_devices()
    if not serials: raise NoDevice("no adb device attached")
    if serial and serial not in serials: raise NoDevice(f"{serial!r} is not among attached devices: {serials}")
    target = serial or serials[0]
    tflite_path, benchmark_model = Path(tflite_path), Path(benchmark_model)
    for p in (tflite_path, benchmark_model):
        if not p.exists(): raise ValueError(f"{p} not found")
    def adb(*args): return subprocess.run(["adb", "-s", target, *args], capture_output=True, text=True, timeout=120)
    adb("shell", "mkdir", "-p", REMOTE_DIR)
    adb("push", str(benchmark_model), f"{REMOTE_DIR}/benchmark_model")
    adb("push", str(tflite_path), f"{REMOTE_DIR}/reader.tflite")
    adb("shell", "chmod", "755", f"{REMOTE_DIR}/benchmark_model")
    r = adb("shell", f"{REMOTE_DIR}/benchmark_model --graph={REMOTE_DIR}/reader.tflite --num_runs={num_runs} --warmup_runs=5 --enable_op_profiling=false")
    if r.returncode != 0: raise ValueError(f"benchmark_model failed on {target}: {(r.stderr or r.stdout)[-400:]}")
    parsed = parse_benchmark_model_output(r.stdout)
    size_bytes = tflite_path.stat().st_size
    targets = rec["targets"]
    met = {
        "size_bytes": targets["size_bytes"][0] <= size_bytes <= targets["size_bytes"][1],
        "peak_rss_mb": parsed["peak_rss_mb"] is None or parsed["peak_rss_mb"] <= targets["peak_rss_mb"],
        "p50_ms": parsed["avg_ms"] is None or parsed["avg_ms"] <= targets["p50_ms"],
    }
    return {"device_id": device_id, "serial": target, "size_bytes": size_bytes, "avg_ms": parsed["avg_ms"],
            "peak_rss_mb": parsed["peak_rss_mb"], "targets_met": met, "raw": r.stdout[-2000:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tflite_path")
    ap.add_argument("--device-id", required=True)
    ap.add_argument("--benchmark-model", required=True)
    ap.add_argument("--serial")
    ap.add_argument("--devices", default=str(DEFAULT_DEVICES))
    ap.add_argument("--num-runs", type=int, default=50)
    a = ap.parse_args()
    try:
        report = run(a.tflite_path, a.device_id, a.benchmark_model, serial=a.serial, devices_path=a.devices, num_runs=a.num_runs)
    except (NoDevice, ValueError, KeyError) as e:
        sys.exit(str(e))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
