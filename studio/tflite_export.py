"""reader.onnx -> reader.tflite, for running the reader on a phone with TensorFlow Lite (LiteRT). Local only.

The reader is trained in PyTorch and exported to ONNX (trainer.py). This turns that ONNX file into a TFLite model with the same weights:
the graph is small and fixed (4 x [Conv 3x3, ReLU, MaxPool], one bidirectional LSTM, one linear layer), so it is rebuilt here from plain
TensorFlow operations with the numbers read out of the ONNX file, then converted. No retraining, and nothing is approximated: the result is
float32, uses only built-in TFLite operations (the LSTM is written out step by step), and is checked against the ONNX model before it is
accepted. A graph that is not the expected shape is refused rather than guessed at.

    input   image   float32 [1, H, W, 1]   one cell, gray, (pixel / 255 - 0.5) / 0.5   (NHWC: the ONNX model takes [N, 1, H, W])
    output  logits  float32 [T, 1, classes]   same layout as the ONNX model, so the same CTC decoding applies

TensorFlow is only needed for this step. It lives in its own environment so that the training environment is left alone:
    python -m venv studio/.venv-tflite && studio/.venv-tflite/bin/pip install tensorflow onnx onnxruntime numpy
`export()` (used by trainer.py and by `add`) runs this file with that environment's Python; set STUDIO_TFLITE_PYTHON to use another one.

CLI: python tflite_export.py convert READER.onnx OUT.tflite [--samples CELLS.npy]     (needs TensorFlow: run it with the Python above)
     python tflite_export.py add PACK.cardpack                                        (writes the next pack version with reader.tflite added)
"""
import json, os, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV_PY = HERE / ".venv-tflite" / "bin" / "python"
MAX_DIFF = 1e-3          # largest logit difference accepted between the TFLite and ONNX models (the ONNX export itself is held to the same)
RANDOM_CELLS = 8


class Unavailable(Exception):
    """No Python with TensorFlow to run the conversion."""


# ---------------------------------------------------------------- runs under TensorFlow
def read_onnx(path):
    """The reader's numbers and shapes from its ONNX file. Raises ValueError when the graph is not the reader this file knows how to rebuild."""
    import numpy as np, onnx
    from onnx import numpy_helper
    m = onnx.load(str(path)); g = m.graph; init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    attr = lambda n: {a.name: (list(a.ints) if a.ints else a.s.decode() if a.type == 3 else a.i) for a in n.attribute}
    main = [n for n in g.node if n.op_type in ("Conv", "BatchNormalization", "Relu", "MaxPool", "LSTM", "GRU", "RNN", "MatMul", "Gemm", "Add")]
    kinds = [n.op_type for n in main]
    if kinds != ["Conv", "Relu", "MaxPool"] * 4 + ["LSTM", "MatMul", "Add"]:
        raise ValueError(f"not the reader graph this converter rebuilds (4 x Conv/Relu/MaxPool, LSTM, MatMul, Add); found {kinds}")
    dims = [d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
    if len(dims) != 4 or dims[1] != 1 or not dims[2] or not dims[3]: raise ValueError(f"expected an input of [batch, 1, H, W], found {dims}")
    convs = []
    for i in range(4):
        c, p = main[3 * i], main[3 * i + 2]; ca, pa = attr(c), attr(p)
        if ca.get("kernel_shape") != [3, 3] or ca.get("strides") != [1, 1] or ca.get("pads") != [1, 1, 1, 1] or ca.get("group", 1) != 1 or ca.get("dilations", [1, 1]) != [1, 1]:
            raise ValueError(f"convolution {i + 1} is not 3x3, stride 1, padding 1: {ca}")
        if pa.get("pads", [0, 0, 0, 0]) != [0, 0, 0, 0] or pa.get("ceil_mode", 0) or pa.get("strides") != pa.get("kernel_shape"):
            raise ValueError(f"pooling {i + 1} is not a plain non-overlapping max pool: {pa}")
        if len(c.input) != 3: raise ValueError(f"convolution {i + 1} has no bias (batch norm was not folded into it)")
        convs.append({"w": init[c.input[1]], "b": init[c.input[2]], "pool": pa["kernel_shape"]})
    lstm = main[12]; la = attr(lstm)
    if la.get("direction") != "bidirectional" or any(k in la for k in ("activations", "clip", "input_forget", "layout")):
        raise ValueError(f"expected a plain bidirectional LSTM, found {la}")
    W, R, B = (init[lstm.input[k]] for k in (1, 2, 3)); hid = la["hidden_size"]
    if W.shape[:2] != (2, 4 * hid) or R.shape != (2, 4 * hid, hid) or B.shape != (2, 8 * hid): raise ValueError("LSTM weights have unexpected shapes")
    fcw = init[main[13].input[1]]; fcb = next(init[x] for x in main[14].input if x in init)
    if fcw.shape[0] != 2 * hid or fcb.shape != (fcw.shape[1],): raise ValueError("linear layer does not follow the LSTM as expected")
    return {"h": dims[2], "w": dims[3], "convs": convs, "hidden": hid, "W": W, "R": R, "B": B, "fcw": fcw, "fcb": fcb, "classes": int(fcw.shape[1]),
            "input": g.input[0].name, "output": g.output[0].name}


def build(spec):
    """The reader as a TensorFlow function of one [1, H, W, 1] cell. Every weight is a constant taken from the ONNX file."""
    import numpy as np, tensorflow as tf
    H = spec["hidden"]; h, w = spec["h"], spec["w"]
    for c in spec["convs"]: h //= c["pool"][0]; w //= c["pool"][1]
    if h != 1: raise ValueError(f"the convolution stack leaves a height of {h}, expected 1")
    T = w
    # ONNX keeps LSTM gates in the order input, output, forget, cell; the two biases (input and recurrent) only ever appear added together
    Wt = [tf.constant(spec["W"][d].T) for d in (0, 1)]; Rt = [tf.constant(spec["R"][d].T) for d in (0, 1)]
    bias = [tf.constant(spec["B"][d][:4 * H] + spec["B"][d][4 * H:]) for d in (0, 1)]
    kernels = [(tf.constant(np.transpose(c["w"], (2, 3, 1, 0))), tf.constant(c["b"]), c["pool"]) for c in spec["convs"]]      # OIHW -> HWIO
    fcw, fcb = tf.constant(spec["fcw"]), tf.constant(spec["fcb"])

    def direction(seq, d):
        """One LSTM direction over seq [T, features], written out step by step so only built-in TFLite operations are needed. -> [T, H]"""
        xw = tf.matmul(seq, Wt[d]) + bias[d]; hs = tf.zeros([1, H]); cs = tf.zeros([1, H]); out = [None] * T
        for t in (range(T) if d == 0 else range(T - 1, -1, -1)):
            i, o, f, g = tf.split(xw[t:t + 1] + tf.matmul(hs, Rt[d]), 4, axis=1)
            cs = tf.sigmoid(f) * cs + tf.sigmoid(i) * tf.tanh(g); hs = tf.sigmoid(o) * tf.tanh(cs); out[t] = hs
        return tf.concat(out, axis=0)

    @tf.function(input_signature=[tf.TensorSpec([1, spec["h"], spec["w"], 1], tf.float32, name="image")])
    def reader(image):
        x = image
        for k, b, pool in kernels:
            x = tf.nn.relu(tf.nn.conv2d(x, k, strides=1, padding="SAME") + b)
            x = tf.nn.max_pool2d(x, ksize=pool, strides=pool, padding="VALID")
        seq = tf.reshape(x, [T, x.shape[-1]])                                           # [1, 1, T, C] -> [T, C]
        y = tf.concat([direction(seq, 0), direction(seq, 1)], axis=1)                   # [T, 2H]: forward then backward, as ONNX lays them out
        return {"logits": tf.reshape(tf.matmul(y, fcw) + fcb, [T, 1, spec["classes"]])}
    return reader, T


def convert(onnx_path, out_path, samples=None):
    """Convert and check. samples: optional .npy of real cells [N, H, W] already scaled like the reader's input; random cells are always added.
    -> a JSON-able report. Raises ValueError when the TFLite model does not reproduce the ONNX model."""
    import numpy as np, onnxruntime as ort, tensorflow as tf
    spec = read_onnx(onnx_path); fn, T = build(spec)
    conv = tf.lite.TFLiteConverter.from_concrete_functions([fn.get_concrete_function()], fn)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    blob = conv.convert()
    cells = np.random.default_rng(0).uniform(-1, 1, (RANDOM_CELLS, spec["h"], spec["w"])).astype(np.float32); n_real = 0
    if samples:
        real = np.load(samples).astype(np.float32)
        if real.ndim != 3 or real.shape[1:] != (spec["h"], spec["w"]): raise ValueError(f"samples must be [N, {spec['h']}, {spec['w']}], found {list(real.shape)}")
        cells = np.concatenate([real, cells]); n_real = len(real)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    it = tf.lite.Interpreter(model_content=blob); it.allocate_tensors(); i_in, i_out = it.get_input_details()[0], it.get_output_details()[0]
    worst, same = 0.0, True
    for c in cells:
        want = sess.run(None, {spec["input"]: c[None, None]})[0]
        it.set_tensor(i_in["index"], c[None, :, :, None]); it.invoke(); got = it.get_tensor(i_out["index"])
        if got.shape != want.shape: raise ValueError(f"TFLite output is {list(got.shape)}, ONNX output is {list(want.shape)}")
        worst = max(worst, float(np.abs(got - want).max())); same = same and bool((got.argmax(-1) == want.argmax(-1)).all())
    report = {"bytes": len(blob), "input": {"name": i_in["name"], "shape": [int(v) for v in i_in["shape"]], "layout": "NHWC"},
              "output": {"name": i_out["name"], "shape": [int(v) for v in i_out["shape"]], "layout": "TBV"}, "dtype": "float32", "ops": "TFLITE_BUILTINS",
              "parity": {"against": "reader.onnx", "cells": n_real, "random_cells": RANDOM_CELLS, "max_abs_diff": worst, "best_class_equal": same},
              "tensorflow": tf.__version__}
    if worst >= MAX_DIFF or not same:
        raise ValueError(f"TFLite model does not reproduce the ONNX model: {json.dumps(report['parity'])}")
    Path(out_path).write_bytes(blob)
    return report


# ---------------------------------------------------------------- runs anywhere
def python():
    """A Python that can import tensorflow, onnx and onnxruntime: STUDIO_TFLITE_PYTHON, then studio/.venv-tflite."""
    for p in (os.environ.get("STUDIO_TFLITE_PYTHON"), VENV_PY):
        if p and Path(p).exists(): return str(p)
    raise Unavailable(f"no TensorFlow environment: python -m venv {VENV_PY.parent.parent} && {VENV_PY.parent}/pip install tensorflow onnx onnxruntime numpy")


def export(onnx_bytes, samples=None, timeout=900):
    """reader.onnx bytes -> (reader.tflite bytes, report), by running `convert` in the TensorFlow environment.
    samples: optional numpy array [N, H, W] of real cells, scaled as the reader takes them. Raises Unavailable, or ValueError with the reason."""
    py = python()
    with tempfile.TemporaryDirectory() as d:
        d = Path(d); (d / "reader.onnx").write_bytes(onnx_bytes); cmd = [py, str(Path(__file__).resolve()), "convert", str(d / "reader.onnx"), str(d / "reader.tflite")]
        if samples is not None:
            import numpy as np
            np.save(d / "cells.npy", samples); cmd += ["--samples", str(d / "cells.npy")]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3", "CUDA_VISIBLE_DEVICES": ""})
        last = (r.stdout.strip().splitlines() or [""])[-1]
        if r.returncode != 0 or not (d / "reader.tflite").exists():
            raise ValueError("TFLite conversion failed: " + (last if last.startswith("ERROR") else (r.stderr.strip().splitlines() or [last or "no output"])[-1])[:400])
        return (d / "reader.tflite").read_bytes(), json.loads(last)


def manifest_entry(report, reader):
    """What a pack's manifest says about reader.tflite (reader = the manifest's `reader` block, for the scaling it shares with the ONNX model)."""
    i = reader["input"]
    return {"file": "reader.tflite", "input": {"name": report["input"]["name"], "layout": "NHWC", "height": i["height"], "width": i["width"], "mean": i["mean"], "std": i["std"]},
            "output": {"name": report["output"]["name"], "layout": "TBV"}, "dtype": report["dtype"], "ops": report["ops"], "parity": report["parity"]}


def add_to_pack(pack_path, out_dir=None):
    """Write the next version of a pack with reader.tflite added (everything else, including held-out metrics, is unchanged: same weights)."""
    sys.path.insert(0, str(HERE.parent / "cardpack")); import cardpack as cp
    pack_path = Path(pack_path); m = cp.read(pack_path); man = json.loads(m["manifest.json"])
    blob, report = export(m["reader.onnx"])
    m["reader.tflite"] = blob; man["reader"]["tflite"] = manifest_entry(report, man["reader"])
    out_dir = Path(out_dir) if out_dir else pack_path.parent
    have = [int(p.stem.split("_v")[1]) for p in out_dir.glob(f"{man['id']}_v*.cardpack") if p.stem.split("_v")[1].isdigit()]
    man["version"] = max(have + [man["version"]]) + 1
    man["files"] = {n: cp.sha256(m[n]) for n in cp.hashed(m)}; m["manifest.json"] = json.dumps(man, indent=1).encode()
    out = out_dir / f"{man['id']}_v{man['version']}.cardpack"; cp.build(m, out)
    return {"pack": str(out), "version": man["version"], "errors": cp.validate(out), "tflite": report}


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) >= 3 and a[0] == "convert":
        try:
            print(json.dumps(convert(a[1], a[2], a[a.index("--samples") + 1] if "--samples" in a else None)))
        except Exception as e:
            print(f"ERROR {type(e).__name__}: {e}"); sys.exit(1)
    elif len(a) == 2 and a[0] == "add":
        try: print(json.dumps(add_to_pack(a[1]), indent=1))
        except (Unavailable, ValueError) as e: sys.exit(str(e))
    else: sys.exit(__doc__.split("CLI:")[1].strip())
