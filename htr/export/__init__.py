"""Export a trained reader to ONNX and LiteRT (TFLite): `backends.py` picks a backend (`auto` tries each in order), `compat.py` says
which backends a given architecture can even attempt, `rebuild_backend.py` is the legacy float32 TF-rebuild converter (moved here from
`studio/tflite_export.py`, which keeps thin wrappers for its own venv-subprocess plumbing). `onnx_backend.py` is a plain ONNX export +
parity check, usable for every architecture (the catalog's `backends_for` already tags which of these each layer type supports).

ai-edge-torch and onnx2tf backends (INT8, the full schema-2 catalog including transformers) are M4 work, not built yet."""
