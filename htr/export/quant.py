"""Width buckets (a phone ships a handful of fixed-width signatures instead of one dynamic-width graph - LiteRT's INT8 path needs a
fixed shape per signature) and INT8 post-training static quantization (PyTorch's eager-mode quantization API: PT2E is not available
in every torch build this lab runs against, and eager PTQ needs nothing beyond torch itself).

torch.ao.quantization's eager-mode API (prepare/convert/prepare_qat) is deprecated upstream in favor of torchao's pt2e path, but is
still the one that works across the torch builds this lab actually runs on (confirmed end to end: see studio/tests/test_htr_quant_bench.py).
Move to torchao.quantization.pt2e once this lab's minimum torch version has it everywhere."""
import math

BUCKETS = (128, 192, 256, 384, 512, 768, 1024)


def bucket_for(width, buckets=BUCKETS):
    """-> the smallest bucket >= width, or the largest bucket (the line is cropped/scaled to fit) when `width` exceeds all of them."""
    for b in buckets:
        if width <= b: return b
    return buckets[-1]


def bucket_counts(widths, buckets=BUCKETS):
    """widths: an iterable of line pixel widths from the training data. -> {bucket: count}, for choosing which signatures are worth
    shipping (a bucket nothing maps to wastes space in the pack)."""
    out = {b: 0 for b in buckets}
    for w in widths: out[bucket_for(w, buckets)] += 1
    return out


def time_steps_for(bucket, width_stride):
    return math.ceil(bucket / width_stride)


def default_engine():
    """The quantized backend this machine's torch build actually supports ("qnnpack" on most arm64 builds including Apple
    Silicon, "fbgemm" on most x86_64 ones). Raises RuntimeError when torch has neither (quantization needs its own torch build on
    this platform)."""
    import torch
    supported = torch.backends.quantized.supported_engines
    for pref in ("qnnpack", "fbgemm"):
        if pref in supported: return pref
    raise RuntimeError(f"no quantized engine available in this torch build (supported_engines: {supported})")


def _quant_wrap(model):
    """Eager static/QAT quantization needs explicit QuantStub/DeQuantStub around whatever is to be quantized, so `prepare`/`convert`
    know where the float <-> int8 boundary is; `htr.nn.compile`'s networks have none, so this wraps one around any model."""
    import torch
    import torch.ao.quantization as tq

    class _Wrapped(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.quant, self.inner, self.dequant = tq.QuantStub(), inner, tq.DeQuantStub()

        def forward(self, x):
            return self.dequant(self.inner(self.quant(x)))

    return _Wrapped(model)


def prepare_qat(model, backend=None):
    """Prepare `model` (eval or train mode) for quantization-aware training: wraps it in Quant/DeQuant stubs, fuses Conv+BN+ReLU
    where present, inserts fake-quant observers. -> the prepared (wrapped) model - train it for the QAT tail of a run, then
    `calibrate_and_convert` (skipping its own calibration pass, since QAT already saw real data). Needs torch's eager quantization
    API (torch.ao.quantization) and a supported engine (`default_engine`)."""
    import torch
    import torch.ao.quantization as tq
    backend = backend or default_engine()
    torch.backends.quantized.engine = backend
    wrapped = _quant_wrap(model)
    wrapped.qconfig = tq.get_default_qat_qconfig(backend)
    wrapped.train()
    return tq.prepare_qat(wrapped)


def calibrate_and_convert(model, calibration_loader, backend=None, forward=None, already_prepared=False):
    """Post-training static quantization: `model` is wrapped in Quant/DeQuant stubs and given observers (unless `already_prepared`:
    pass a model `prepare_qat` already returned, after training it, and this skips re-wrapping), run over `calibration_loader`
    batches to collect activation ranges, then converted to a true INT8 model. forward(model, batch) defaults to
    `model(batch["images"])` (htr.data.dataset's collate shape); pass your own for a different calibration loader. -> the INT8
    model, whose forward still takes the same input and returns the same output shape as the float model (the stubs are internal)."""
    import torch
    import torch.ao.quantization as tq
    backend = backend or default_engine()
    torch.backends.quantized.engine = backend
    prepared = model
    if not already_prepared:
        wrapped = _quant_wrap(model)
        wrapped.eval(); wrapped.qconfig = tq.get_default_qconfig(backend)
        prepared = tq.prepare(wrapped)
    forward = forward or (lambda m, batch: m(batch["images"]))
    prepared.eval()
    with torch.no_grad():
        for batch in calibration_loader: forward(prepared, batch)
    return tq.convert(prepared)
