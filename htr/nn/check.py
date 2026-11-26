"""A dry run of a spec before any training: builds it, runs one cell, exports ONNX and compares it with the PyTorch output, and times
both. Same contract as the old `archnet.check`. Works for a legacy spec/preset, a schema-2 config/preset, and a {"custom": name}
config alike."""
from .. import catalog as K
from .. import spec as S
from . import custom as CUST
from .compile import _is_custom, _is_schema2, compile


def check(config, n_classes=129, runs=20):
    """-> {ok, spec, params, time_steps, torch_ms, onnx_ms, onnx_ok, onnx_bytes, tflite, shipped_weights_fit, problems}.
    Never raises for a bad spec: the reason is in `problems`."""
    import io, time
    out = {"ok": False, "problems": []}
    custom, schema2 = _is_custom(config), _is_schema2(config)
    M = CUST if custom else __import__("htr.presets" if schema2 else "htr.spec", fromlist=["x"])
    img_w = K.IMG_W_REF if (schema2 or custom) else S.IMG_W
    try: s = M.normalize(config)
    except ValueError as e: out["problems"].append(str(e)); return out
    import numpy as np, torch
    out["spec"] = s; out["hash"] = M.spec_hash(s); d = M.describe(s, n_classes); out.update(params=d["params"], time_steps=d["time_steps"])
    net = compile(s, n_classes).eval(); x = torch.zeros(1, 1, K.IMG_H, img_w)
    with torch.no_grad():
        y, _ = net(x)
        if tuple(y.shape) != (d["time_steps"], 1, n_classes): out["problems"].append(f"output shape {tuple(y.shape)}, expected {(d['time_steps'], 1, n_classes)}"); return out
        t0 = time.time()
        for _ in range(runs): net(x)
        out["torch_ms"] = round((time.time() - t0) * 1000 / runs, 2)
    out["params_built"] = sum(p.numel() for p in net.parameters())
    if out["params_built"] != d["params"]: out["problems"].append(f"counted {d['params']:,} parameters but built {out['params_built']:,}")
    try:
        import onnxruntime as ort

        class _Wrap(torch.nn.Module):
            def __init__(self, inner): super().__init__(); self.inner = inner
            def forward(self, x): return self.inner(x)[0]

        buf = io.BytesIO(); torch.onnx.export(_Wrap(net).eval(), x, buf, input_names=["image"], output_names=["logits"], dynamic_axes={"image": {0: "batch"}, "logits": {1: "batch"}}, opset_version=17, dynamo=False)
        sess = ort.InferenceSession(buf.getvalue(), providers=["CPUExecutionProvider"]); xr = np.random.RandomState(0).randn(1, 1, K.IMG_H, img_w).astype(np.float32)
        with torch.no_grad(): ref = net(torch.from_numpy(xr))[0].numpy()
        got = sess.run(None, {"image": xr})[0]; diff = float(np.abs(ref - got).max())
        t0 = time.time()
        for _ in range(runs): sess.run(None, {"image": xr})
        out.update(onnx_ms=round((time.time() - t0) * 1000 / runs, 2), onnx_bytes=len(buf.getvalue()), onnx_max_diff=diff, onnx_ok=diff < 1e-3)
        if diff >= 1e-3: out["problems"].append(f"ONNX output differs from PyTorch by {diff:.4f}")
    except Exception as e:
        out["onnx_ok"] = False; out["problems"].append(f"ONNX export failed: {str(e)[:160]}")
    if custom: out["tflite"] = {"supported": False, "reason": "the legacy TFLite rebuild backend only knows conv/context/rnn/linear networks; a custom architecture is not checked against that family"}
    elif schema2: out["tflite"] = {"supported": False, "reason": "the legacy TFLite rebuild backend only knows conv/context/rnn/linear networks (htr.spec), not a schema-2 config"}
    else: out["tflite"] = S.tflite_support(s)
    out["shipped_weights_fit"] = (not schema2) and (not custom) and S.is_default(s)
    out["ok"] = not out["problems"]
    return out
