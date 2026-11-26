"""Compatibility with the pre-`htr` world: `work_phaseA/common.py CRNN` and `work_phaseA/archnet.py`.

`CRNN_STATE_KEYS` maps `common.CRNN`'s state-dict keys to the `htr.nn.compile`d `crnn` preset's keys: both are built in the exact same
layer order (see `htr/nn/compile.py`), so the mapping is the identity. This function exists so the one place that assumes that stays
documented and tested (`studio/tests/test_htr_legacy.py`), instead of every caller silently relying on two modules staying in step.
"""
from . import spec as S


def state_dict_matches(old_state_dict, n_classes=129):
    """True if `old_state_dict` (from `common.CRNN` or a registered pretrained weights file) loads into the `crnn` preset's compiled
    network with no key renamed and no shape changed."""
    from .nn.compile import compile
    net = compile(S.DEFAULT, n_classes)
    new_keys = dict(net.state_dict())
    if set(old_state_dict) != set(new_keys): return False
    return all(tuple(old_state_dict[k].shape) == tuple(new_keys[k].shape) for k in old_state_dict)


def is_legacy_spec(config):
    """True for the old four-key shape (conv/context/rnn/dropout) or a preset name from `htr.spec.PRESETS`, as opposed to a
    schema-2 config (not yet implemented: `htr/catalog.py`)."""
    if isinstance(config, str): return True
    return isinstance(config, dict) and "schema" not in config
