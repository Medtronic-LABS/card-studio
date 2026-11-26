"""Custom architectures as plain Python, one module per file (`htr/custom/<name>.py`), each defining `build(n_classes) -> nn.Module`.
Never loaded from anything but this local directory (`htr/nn/custom.py` imports `htr.custom.<name>` by name, never arbitrary code
sent over HTTP) - the plan's "no code upload over HTTP" rule. A module here is trusted the same way any other code in this repo is;
it is not a sandboxed plugin format."""
