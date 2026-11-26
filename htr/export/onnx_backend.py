"""Plain ONNX export: works for any architecture this lab can build (legacy or schema-2 - transformers included), since it is just
`torch.onnx.export` plus a parity check against the PyTorch model, the same dry run `htr.nn.check` already does before training.
This is the ONNX file every pack ships (`reader.onnx`); `rebuild_backend.py` and (planned, M4) `ai_edge_backend.py`/`onnx2tf_backend.py`
turn it, or the model directly, into a mobile format."""
MAX_DIFF = 1e-3


def export(model, img_h, img_w, path, opset=17):
    """A built (eval-mode) HTRArchitecture -> reader.onnx at `path`, fixed at one width (`img_w`) - the width a pack ships with; the
    network itself still accepts any width, this just fixes what gets traced. -> the ONNX bytes."""
    from pathlib import Path
    import torch
    x = torch.zeros(1, 1, img_h, img_w)

    class _Wrap(torch.nn.Module):
        def __init__(self, inner): super().__init__(); self.inner = inner
        def forward(self, x): return self.inner(x)[0]

    torch.onnx.export(_Wrap(model).eval(), x, str(path), input_names=["image"], output_names=["logits"],
                       dynamic_axes={"image": {0: "batch"}, "logits": {1: "batch"}}, opset_version=opset, dynamo=False)
    return Path(path).read_bytes()


def parity(model, onnx_path, img_h, img_w, seed=0):
    """-> {max_abs_diff, ok}: the largest logit difference between the PyTorch model (eval mode) and its ONNX export, on one random
    cell at (img_h, img_w)."""
    import numpy as np, onnxruntime as ort, torch
    x = np.random.RandomState(seed).randn(1, 1, img_h, img_w).astype(np.float32)
    with torch.no_grad(): ref = model.eval()(torch.from_numpy(x))[0].numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"image": x})[0]
    diff = float(np.abs(ref - got).max())
    return {"max_abs_diff": diff, "ok": diff < MAX_DIFF}
