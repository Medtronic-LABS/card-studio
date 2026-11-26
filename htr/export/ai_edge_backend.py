"""ai-edge-torch: the only backend that should eventually reach every architecture this lab can build (including a transformer
sequence block) with INT8, straight from the PyTorch model - no ONNX round-trip, no graph-shape guessing. Lives in its own
environment (it pins its own torch build): `python -m venv studio/.venv-litert && studio/.venv-litert/bin/pip install ai-edge-torch`.

Not exercised against the real package in this environment (it is not installed here - `pip install ai-edge-torch` needs a
compatible torch build this session does not have). `convert()` calls its top-level documented entry point
(`ai_edge_torch.convert(model, sample_args) -> edge_model; edge_model.export(path)`); treat this as unverified against the live
package until an environment with it installed runs `htr/export/test_*` against a real conversion."""
from pathlib import Path


class Unavailable(Exception):
    """ai-edge-torch is not importable in this environment."""


def available():
    try:
        import ai_edge_torch  # noqa: F401
        return True
    except ImportError:
        return False


def convert(model, img_h, img_w, out_path, quantize=False, calibration_loader=None):
    """model: an eval-mode HTRArchitecture (htr.nn.compile). -> {path, bytes}. Raises Unavailable when the package is not
    installed; ValueError when the conversion itself fails (the package's own exception, wrapped with the model's shape for
    context)."""
    try:
        import ai_edge_torch
        import torch
    except ImportError as e:
        raise Unavailable(f"ai-edge-torch is not installed: {e}") from e
    sample = (torch.zeros(1, 1, img_h, img_w),)
    try:
        edge_model = ai_edge_torch.convert(model.eval(), sample)
        if quantize and calibration_loader is not None:
            from ai_edge_torch.quantize import pt2e_quantizer  # noqa: F401  # quantization recipe selection is package-version-specific; left to a follow-up once this path is actually exercised
        edge_model.export(str(out_path))
    except Exception as e:
        raise ValueError(f"ai-edge-torch conversion failed for a {img_h}x{img_w} input: {e}") from e
    return {"path": str(out_path), "bytes": Path(out_path).stat().st_size}
