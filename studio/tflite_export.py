"""reader.onnx -> reader.tflite, for running the reader on a phone with TensorFlow Lite (LiteRT). Local only.

The reader is trained in PyTorch and exported to ONNX (trainer.py). This turns that ONNX file into a TFLite model with the same weights:
the network is rebuilt here from plain TensorFlow operations with the numbers read out of the ONNX file, then converted. No retraining, and nothing
is approximated: the result is float32, uses only built-in TFLite operations (the recurrent layers are written out step by step, one step per
time step), and is checked against the ONNX model before it is accepted. A graph that is not one of the known families is refused rather than
guessed at. The families are every network work_phaseA/archnet.py can build:
    1-8 conv blocks      Conv kxk (k = 3, 5, 7; stride 1, 'same' padding), ReLU, then an optional MaxPool (heights and widths 1, 2 or 4; the last block
                         pools away the whole remaining height and may pool the width too). Batch norm is already folded into the Conv by the exporter.
    0-4 context convs    Conv 1xk along time (k = 3, 5, 7), ReLU, after the height has collapsed.
    recurrent part       none, or 1-3 stacked LSTM or GRU layers (one ONNX node per layer), bidirectional or not; a layer's output (forward then
                         backward, concatenated, for bidirectional) feeds the next layer.
    head                 one linear layer (MatMul + Add, or Gemm).

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
GLUE = {"Constant", "Squeeze", "Unsqueeze", "Transpose", "Shape", "Gather", "Concat", "ConstantOfShape", "Slice", "Reshape", "Identity", "Dropout", "Flatten"}
RNN_OPS = ("LSTM", "GRU")


def read_onnx(path):
    """The reader's numbers and shapes from its ONNX file. Raises ValueError when the graph is not a reader this file knows how to rebuild.
    Accepted: 1-8 x [Conv kxk (k 3/5/7, stride 1, same padding), Relu, optional MaxPool], 0-4 x [Conv 1xk along time, Relu], then either nothing or
    1-3 LSTM/GRU nodes (one per layer, all the same kind, size and direction), then MatMul+Add or Gemm. The glue torch writes between them is ignored."""
    import numpy as np, onnx
    from onnx import numpy_helper
    m = onnx.load(str(path)); g = m.graph; init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    attr = lambda n: {a.name: (list(a.ints) if a.ints else a.s.decode() if a.type == 3 else a.f if a.type == 1 else a.i) for a in n.attribute}
    main = [n for n in g.node if n.op_type not in GLUE]
    kinds = [n.op_type for n in main]
    expected = "[Conv, Relu, MaxPool?] x 1-8, [Conv, Relu] x 0-4, LSTM|GRU x 0-3 (one per layer), MatMul+Add or Gemm"
    bad = lambda: ValueError(f"not the reader graph this converter rebuilds ({expected}); found {kinds}")
    i = 0
    def take(op):
        nonlocal i
        if i < len(main) and main[i].op_type == op: i += 1; return main[i - 1]
        return None
    blocks, ctx = [], []
    while i < len(main) and main[i].op_type == "Conv":
        c = main[i]; ks = attr(c).get("kernel_shape"); square = bool(ks) and len(ks) == 2 and ks[0] == ks[1] and ks[0] > 1
        if not square and not ctx and not blocks: raise bad()
        if square and ctx: raise bad()
        i += 1
        if not take("Relu"): raise bad()
        if square: blocks.append((c, take("MaxPool")))
        else: ctx.append(c)
    if not 1 <= len(blocks) <= 8 or len(ctx) > 4: raise bad()
    rnns = []
    while i < len(main) and main[i].op_type in RNN_OPS: rnns.append(main[i]); i += 1
    if len(rnns) > 3: raise bad()
    if i < len(main) and main[i].op_type == "Gemm" and i == len(main) - 1: fc, add = main[i], None; i += 1
    elif i + 1 == len(main) - 1 and kinds[i:] == ["MatMul", "Add"]: fc, add = main[i], main[i + 1]; i += 2
    else: raise bad()
    if i != len(main): raise bad()
    if len({n.op_type for n in rnns}) > 1: raise ValueError(f"mixed recurrent layers are not supported: {[n.op_type for n in rnns]}")
    dims = [d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
    if len(dims) != 4 or dims[1] != 1 or not dims[2] or not dims[3]: raise ValueError(f"expected an input of [batch, 1, H, W], found {dims}")
    convs = []
    for n, (c, p) in enumerate(blocks, 1):
        ca = attr(c); k = ca["kernel_shape"][0]
        if k not in (3, 5, 7) or ca.get("kernel_shape") != [k, k] or ca.get("strides", [1, 1]) != [1, 1] or ca.get("pads", [0] * 4) != [k // 2] * 4 or ca.get("group", 1) != 1 or ca.get("dilations", [1, 1]) != [1, 1]:
            raise ValueError(f"convolution {n} is not 3x3, 5x5 or 7x7, stride 1, 'same' padding: {ca}")
        if len(c.input) != 3: raise ValueError(f"convolution {n} has no bias (batch norm was not folded into it)")
        pool = [1, 1]
        if p is not None:
            pa = attr(p); pool = pa.get("kernel_shape")
            if pa.get("pads", [0, 0, 0, 0]) != [0, 0, 0, 0] or pa.get("ceil_mode", 0) or pa.get("strides") != pool or pa.get("dilations", [1, 1]) != [1, 1] or not pool or len(pool) != 2:
                raise ValueError(f"pooling {n} is not a plain non-overlapping max pool: {pa}")
        convs.append({"w": init[c.input[1]], "b": init[c.input[2]], "pool": pool})
    ctxs = []
    for n, c in enumerate(ctx, 1):
        ca = attr(c); ks = ca.get("kernel_shape"); k = ks[1] if ks and len(ks) == 2 else 0
        if ks != [1, k] or k not in (3, 5, 7) or ca.get("strides", [1, 1]) != [1, 1] or ca.get("pads", [0] * 4) != [0, k // 2, 0, k // 2] or ca.get("group", 1) != 1 or ca.get("dilations", [1, 1]) != [1, 1]:
            raise ValueError(f"context convolution {n} is not 1x3, 1x5 or 1x7 along time, stride 1, 'same' padding: {ca}")
        if len(c.input) != 3: raise ValueError(f"context convolution {n} has no bias (batch norm was not folded into it)")
        ctxs.append({"w": init[c.input[1]], "b": init[c.input[2]]})
    last = (ctxs or convs)[-1]["w"].shape[0]
    rnn = None; hid = 0; layers = []
    if rnns:
        gates = 4 if rnns[0].op_type == "LSTM" else 3; la0 = attr(rnns[0]); direction = la0.get("direction", "forward"); hid = la0.get("hidden_size")
        if direction not in ("forward", "bidirectional"): raise ValueError(f"unsupported recurrent direction {direction!r}")
        nd = 2 if direction == "bidirectional" else 1; feat = last
        for n, r in enumerate(rnns, 1):
            la = attr(r)
            if la.get("direction", "forward") != direction or la.get("hidden_size") != hid: raise ValueError("recurrent layers differ in size or direction")
            extra = ("activations", "clip", "layout") + (("input_forget",) if r.op_type == "LSTM" else ())
            if any(k in la for k in extra) or (r.op_type == "GRU" and la.get("linear_before_reset", 0) != 1):
                raise ValueError(f"expected a plain {direction} {r.op_type} as torch writes it, found {la}")
            if len(r.input) < 4 or not all(r.input[k] in init for k in (1, 2, 3)): raise ValueError(f"recurrent layer {n} has no constant weights and bias")
            if len(r.input) > 4 and r.input[4]: raise ValueError("recurrent layers with sequence_lens are not supported")
            W, R, B = (init[r.input[k]] for k in (1, 2, 3))
            if W.shape != (nd, gates * hid, feat) or R.shape != (nd, gates * hid, hid) or B.shape != (nd, 2 * gates * hid): raise ValueError(f"recurrent layer {n} weights have unexpected shapes")
            layers.append({"W": W, "R": R, "B": B}); feat = nd * hid
        rnn = {"type": rnns[0].op_type.lower(), "hidden": hid, "dirs": nd, "layers": layers}
    else: feat = last
    if fc.op_type == "Gemm":
        fa = attr(fc); fcw = init[fc.input[1]]
        if fa.get("transA", 0) or fa.get("alpha", 1.0) != 1.0 or fa.get("beta", 1.0) != 1.0 or len(fc.input) != 3: raise ValueError(f"unsupported linear layer: {fa}")
        if fa.get("transB", 0): fcw = fcw.T
        fcb = init[fc.input[2]]
    else:
        if fc.input[1] not in init: raise ValueError("linear layer has no constant weight")
        fcw = init[fc.input[1]]; fcb = next((init[x] for x in add.input if x in init), None)
        if fcb is None: raise ValueError("linear layer has no constant bias")
    if fcw.shape[0] != feat or fcb.shape != (fcw.shape[1],): raise ValueError("linear layer does not follow the previous layer as expected")
    out = {"h": dims[2], "w": dims[3], "convs": convs, "context": ctxs, "rnn": rnn, "fcw": fcw, "fcb": fcb, "classes": int(fcw.shape[1]),
           "input": g.input[0].name, "output": g.output[0].name}
    if layers: out.update(hidden=hid, W=layers[0]["W"], R=layers[0]["R"], B=layers[0]["B"])
    return out


def build(spec):
    """The reader as a TensorFlow function of one [1, H, W, 1] cell. Every weight is a constant taken from the ONNX file."""
    import numpy as np, tensorflow as tf
    h, w = spec["h"], spec["w"]
    for c in spec["convs"]: h //= c["pool"][0]; w //= c["pool"][1]
    if h != 1: raise ValueError(f"the convolution stack leaves a height of {h}, expected 1")
    T = w
    hwio = lambda a: tf.constant(np.transpose(a, (2, 3, 1, 0)))                         # OIHW -> HWIO
    kernels = [(hwio(c["w"]), tf.constant(c["b"]), c["pool"]) for c in spec["convs"]]
    ctx = [(hwio(c["w"]), tf.constant(c["b"])) for c in spec["context"]]
    fcw, fcb = tf.constant(spec["fcw"]), tf.constant(spec["fcb"])
    rnn = spec["rnn"]
    if rnn:
        H = rnn["hidden"]; lstm = rnn["type"] == "lstm"
        layers = [{"W": [tf.constant(l["W"][d].T) for d in range(rnn["dirs"])], "R": [tf.constant(l["R"][d].T) for d in range(rnn["dirs"])],
                   "Wb": [tf.constant(l["B"][d][:(4 if lstm else 3) * H]) for d in range(rnn["dirs"])], "Rb": [tf.constant(l["B"][d][(4 if lstm else 3) * H:]) for d in range(rnn["dirs"])]}
                  for l in rnn["layers"]]

    def direction(seq, L, d):
        """One recurrent direction over seq [T, features], written out step by step so only built-in TFLite operations are needed. -> [T, H]
        LSTM: ONNX gate order input, output, forget, cell (the two biases only ever appear added together). GRU: update, reset, hidden, with the
        reset gate applied after the recurrent product (linear_before_reset, as torch does)."""
        hs = tf.zeros([1, H]); out = [None] * T
        if lstm:
            xw = tf.matmul(seq, L["W"][d]) + (L["Wb"][d] + L["Rb"][d]); cs = tf.zeros([1, H])
            for t in (range(T) if d == 0 else range(T - 1, -1, -1)):
                i, o, f, g = tf.split(xw[t:t + 1] + tf.matmul(hs, L["R"][d]), 4, axis=1)
                cs = tf.sigmoid(f) * cs + tf.sigmoid(i) * tf.tanh(g); hs = tf.sigmoid(o) * tf.tanh(cs); out[t] = hs
        else:
            xw = tf.matmul(seq, L["W"][d]) + L["Wb"][d]
            for t in (range(T) if d == 0 else range(T - 1, -1, -1)):
                xz, xr, xh = tf.split(xw[t:t + 1], 3, axis=1); hz, hr, hh = tf.split(tf.matmul(hs, L["R"][d]) + L["Rb"][d], 3, axis=1)
                z = tf.sigmoid(xz + hz); r = tf.sigmoid(xr + hr); n = tf.tanh(xh + r * hh)
                hs = (1.0 - z) * n + z * hs; out[t] = hs
        return tf.concat(out, axis=0)

    @tf.function(input_signature=[tf.TensorSpec([1, spec["h"], spec["w"], 1], tf.float32, name="image")])
    def reader(image):
        x = image
        for k, b, pool in kernels:
            x = tf.nn.relu(tf.nn.conv2d(x, k, strides=1, padding="SAME") + b)
            if list(pool) != [1, 1]: x = tf.nn.max_pool2d(x, ksize=pool, strides=pool, padding="VALID")
        for k, b in ctx: x = tf.nn.relu(tf.nn.conv2d(x, k, strides=1, padding="SAME") + b)
        seq = tf.reshape(x, [T, x.shape[-1]])                                           # [1, 1, T, C] -> [T, C]
        if rnn:
            for L in layers:                                                            # each layer feeds the next; both directions: forward then backward
                seq = tf.concat([direction(seq, L, d) for d in range(rnn["dirs"])], axis=1) if rnn["dirs"] > 1 else direction(seq, L, 0)
        return {"logits": tf.reshape(tf.matmul(seq, fcw) + fcb, [T, 1, spec["classes"]])}
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
