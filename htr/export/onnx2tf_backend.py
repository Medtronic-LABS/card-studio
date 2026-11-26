"""onnx2tf: ONNX -> TensorFlow -> TFLite, for conv/dwconv/attention-shaped graphs (it refuses recurrent configs - a plain LSTM/GRU
ONNX subgraph does not convert cleanly through it); INT8 needs a representative-data generator. Lives in its own environment:
`python -m venv studio/.venv-onnx2tf && studio/.venv-onnx2tf/bin/pip install onnx2tf tensorflow`.

Not exercised against the real package in this environment (not installed here). `convert()` calls its top-level documented entry
point (`onnx2tf.convert(input_onnx_file_path=..., output_folder_path=..., ...)`); treat this as unverified against the live package
until an environment with it installed runs a real conversion."""
from pathlib import Path


class Unavailable(Exception):
    """onnx2tf is not importable in this environment."""


def available():
    try:
        import onnx2tf  # noqa: F401
        return True
    except ImportError:
        return False


def convert(onnx_path, out_dir, quantize=False, representative_data=None):
    """onnx_path: an exported reader.onnx (htr.export.onnx_backend). -> {path, bytes} of the .tflite onnx2tf writes into `out_dir`.
    representative_data: an iterable of input arrays for INT8 calibration (onnx2tf's own representative dataset contract); required
    when quantize is True. Raises Unavailable when the package is not installed; ValueError when conversion fails."""
    try:
        import onnx2tf
    except ImportError as e:
        raise Unavailable(f"onnx2tf is not installed: {e}") from e
    if quantize and representative_data is None:
        raise ValueError("quantize=True needs representative_data (a calibration generator) for onnx2tf's INT8 path")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {"input_onnx_file_path": str(onnx_path), "output_folder_path": str(out_dir), "non_verbose": True}
    if quantize:
        kwargs.update(output_integer_quantized_tflite=True, custom_input_op_name_np_data_path=representative_data)
    try:
        onnx2tf.convert(**kwargs)
    except Exception as e:
        raise ValueError(f"onnx2tf conversion failed for {onnx_path}: {e}") from e
    produced = sorted(out_dir.glob("*.tflite"))
    if not produced: raise ValueError(f"onnx2tf reported success but wrote no .tflite file in {out_dir}")
    path = produced[-1]
    return {"path": str(path), "bytes": path.stat().st_size}
