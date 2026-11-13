"""The settings of one training run: a preset, every training value, and the network. Pure Python (the checks need torch only when a run starts).

    {"preset": "light_lr3e-4",                                    the starting point: learning rate and augmentation strength (trainer.py --config)
     "hp":     {"lr": 0.0005, "optimizer": "adamw", "weight_decay": 0.01, "batch": 32, "schedule": "cosine", "grad_clip": 1.0, "balance": true, "patience": 60,
                "aug": {"rot": 3, "tx": 2, "ty": 1, "bright": 0.1, "shift": 8, "blur": 0.2, "noise": 4, "erase": 0.2}},
     "arch":   "crnn-wide"  |  {"conv": [...], "context": [...], "rnn": {...}, "dropout": 0.3}  |  null (the shipped CRNN)}

Everything is optional. A run with no settings is the shipped recipe, step for step. `load` returns what the trainer needs and what to record in results.json.
`describe_all` is what the settings panels (Studio, Frappe) show: the presets, every value with its limits and default, and the architecture presets.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "work_phaseA")); sys.path.insert(0, str(HERE))
import archnet

PRESET_KEYS = ("light_lr3e-4", "light_lr1e-4", "nbaug_lr1e-4")
# the numbers behind the three presets, kept here so a panel can show them without importing torch (train.CONFIGS is the source; a test compares them)
PRESET_VALUES = {"light_lr3e-4": {"lr": 3e-4, "aug": {"rot": 2, "tx": 2, "ty": 1, "bright": 0.08, "shift": 8, "blur": 0.15}},
                 "light_lr1e-4": {"lr": 1e-4, "aug": {"rot": 2, "tx": 2, "ty": 1, "bright": 0.08, "shift": 8, "blur": 0.15}},
                 "nbaug_lr1e-4": {"lr": 1e-4, "aug": {"rot": 3, "tx": 3, "ty": 2, "bright": 0.15, "shift": 15, "blur": 0.30}}}
HP_INFO = {"lr": ("learning rate", 1e-6, 1e-1), "optimizer": ("optimizer", ("adam", "adamw", "sgd")), "weight_decay": ("weight decay", 0.0, 0.1), "batch": ("batch size", 4, 256),
           "schedule": ("learning-rate schedule", ("none", "cosine", "step")), "step_every": ("epochs between steps (step schedule)", 1, 500), "step_gamma": ("step factor (step schedule)", 0.05, 0.95),
           "grad_clip": ("gradient clip (0 = off)", 0.0, 100.0), "balance": ("balance fields when sampling", (True, False)), "patience": ("early stopping patience, epochs", 3, 500)}
AUG_INFO = {"rot": ("rotation, degrees", 0, 15), "tx": ("shift sideways, pixels", 0, 20), "ty": ("shift up/down, pixels", 0, 10), "bright": ("brightness variation", 0, 0.6), "shift": ("gray-level shift", 0, 60),
            "blur": ("chance of blur", 0, 1), "noise": ("pixel noise, gray levels", 0, 40), "erase": ("chance of a missing patch", 0, 0.8)}
HP_DEFAULT_VALUES = {"optimizer": "adam", "weight_decay": 0.0, "batch": 64, "schedule": "none", "step_every": 50, "step_gamma": 0.5, "grad_clip": 0.0, "balance": True, "patience": 80}
AUG_DEFAULT_VALUES = {"noise": 0.0, "erase": 0.0}


def load(path_or_dict, preset="light_lr3e-4"):
    """-> {preset, cfg (resolved training config), arch (a normalized spec, None for the shipped CRNN), arch_label, config_label, record (what results.json keeps)}.
    path_or_dict: a settings file, a dict, or None. Raises ValueError saying what is wrong."""
    import train as T
    raw = {}
    if isinstance(path_or_dict, (list, tuple, int, float)): raise ValueError("the settings are an object with preset, hp and arch")
    if isinstance(path_or_dict, dict): raw = path_or_dict
    elif path_or_dict:
        try: raw = json.loads(Path(path_or_dict).read_text())
        except (OSError, ValueError) as e: raise ValueError(f"cannot read the settings file: {e}")
    if not isinstance(raw, dict): raise ValueError("the settings are an object with preset, hp and arch")
    unknown = sorted(set(raw) - {"preset", "hp", "arch"})
    if unknown: raise ValueError(f"unknown settings key(s): {', '.join(unknown)}")
    preset = raw.get("preset") or preset
    if preset not in PRESET_KEYS: raise ValueError(f"unknown preset {preset!r}: {', '.join(PRESET_KEYS)}")
    hp = raw.get("hp") or {}
    if not isinstance(hp, dict): raise ValueError("hp is an object of training values")
    cfg = T.resolve_cfg(preset, hp)
    arch = raw.get("arch")
    if arch is not None:
        arch = archnet.normalize(arch)
        if archnet.is_default(arch): arch = None
    custom = bool(hp) or arch is not None
    label = "the shipped CRNN" if arch is None else (raw["arch"] if isinstance(raw["arch"], str) else "a custom architecture") + f" ({archnet.describe(arch)['params'] / 1e6:.2f}M parameters)"
    return {"preset": preset, "cfg": cfg, "arch": arch, "arch_label": label, "config_label": preset + ("+custom" if custom else ""),
            "record": {"preset": preset, "hp": cfg, "arch": archnet.normalize(archnet.DEFAULT if arch is None else arch), "arch_name": "crnn" if arch is None else (raw["arch"] if isinstance(raw["arch"], str) else "custom"),
                       "arch_hash": archnet.spec_hash(archnet.DEFAULT if arch is None else arch), "arch_params": archnet.describe(archnet.normalize(archnet.DEFAULT if arch is None else arch))["params"]}}


def describe_all():
    """What a settings panel needs, with no torch: the presets, every value with its limits and its default, and the architecture presets."""
    hp = {k: {"label": v[0], **({"min": v[1], "max": v[2]} if len(v) == 3 else {"options": list(v[1])}), "default": HP_DEFAULT_VALUES.get(k)} for k, v in HP_INFO.items()}
    aug = {k: {"label": v[0], "min": v[1], "max": v[2]} for k, v in AUG_INFO.items()}
    return {"presets": PRESET_VALUES, "default_preset": "light_lr3e-4", "hp": hp, "aug": aug,
            "arch_presets": {k: {"label": v["label"], "spec": archnet.normalize(k), "hash": archnet.spec_hash(k), **archnet.describe(archnet.normalize(k))} for k, v in archnet.PRESETS.items()}, "arch_default": archnet.DEFAULT,
            "arch_limits": {k: v for k, v in archnet.LIMITS.items()}}
