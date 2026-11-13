"""The settings of one training run: a preset, every training value, and the network. Pure Python (no torch), so a bundle check and a settings panel can use it.

    {"preset": "light_lr3e-4",                                    the starting point: learning rate and augmentation strength
     "hp":     {"lr": 0.0005, "optimizer": "adamw", "weight_decay": 0.01, "batch": 32, "schedule": "cosine", "grad_clip": 1.0, "balance": true, "patience": 60,
                "aug": {"rot": 3, "tx": 2, "ty": 1, "bright": 0.1, "shift": 8, "blur": 0.2, "noise": 4, "erase": 0.2}},
     "arch":   "crnn-wide"  |  {"conv": [...], "context": [...], "rnn": {...}, "dropout": 0.3}  |  null (the shipped CRNN),
     "run":    {"checkpoint_every": 10, "keep_last": 2}}      how often to save a checkpoint (0 = only the best) and how many numbered ones to keep: not training values, so they do not make a run "custom"

Everything is optional. A run with no settings is the shipped recipe, step for step. `load` returns what the trainer needs and what to record in results.json;
`describe_all` is what the settings panels (Card Studio, the Frappe app) show: the presets, every value with its limits and default, and the architecture presets.
This file is kept identical in work_phaseA/ and in the Frappe app (card_studio/engine/trainsettings.py).
"""
import json
from pathlib import Path

try:
    from . import archnet, hpspec
except ImportError:
    import archnet, hpspec

HP_OPTIONS = {"optimizer": list(hpspec.OPTIMIZERS), "schedule": list(hpspec.SCHEDULES), "balance": [True, False], "monitor": list(hpspec.MONITORS), "restore": list(hpspec.RESTORES)}


def load(path_or_dict, preset=hpspec.DEFAULT_PRESET):
    """-> {preset, cfg (resolved training config), arch (a normalized spec, None for the shipped CRNN), arch_label, config_label, record (what results.json keeps)}.
    path_or_dict: a settings file, a dict, or None. Raises ValueError saying what is wrong."""
    raw = {}
    if isinstance(path_or_dict, (list, tuple, int, float)): raise ValueError("the settings are an object with preset, hp and arch")
    if isinstance(path_or_dict, dict): raw = path_or_dict
    elif path_or_dict:
        try: raw = json.loads(Path(path_or_dict).read_text())
        except (OSError, ValueError) as e: raise ValueError(f"cannot read the settings file: {e}")
    if not isinstance(raw, dict): raise ValueError("the settings are an object with preset, hp, arch and run")
    unknown = sorted(set(raw) - {"preset", "hp", "arch", "run"})
    if unknown: raise ValueError(f"unknown settings key(s): {', '.join(unknown)}")
    preset = raw.get("preset") or preset
    if preset not in hpspec.PRESETS: raise ValueError(f"unknown preset {preset!r}: {', '.join(hpspec.PRESETS)}")
    hp = raw.get("hp") or {}
    if not isinstance(hp, dict): raise ValueError("hp is an object of training values")
    cfg = hpspec.resolve(preset, hp)
    if raw.get("run") is not None and not isinstance(raw.get("run"), dict): raise ValueError("run is an object of run settings (checkpoint_every, keep_last)")
    run = hpspec.resolve_run(raw.get("run"))
    arch = raw.get("arch")
    if arch is not None:
        arch = archnet.normalize(arch)
        if archnet.is_default(arch): arch = None
    custom = bool(hp) or arch is not None
    shown = archnet.normalize(archnet.DEFAULT if arch is None else arch)
    name = "crnn" if arch is None else (raw["arch"] if isinstance(raw["arch"], str) else "custom")
    label = "the shipped CRNN" if arch is None else f"{name if name != 'custom' else 'a custom architecture'} ({archnet.describe(shown)['params'] / 1e6:.2f}M parameters)"
    return {"preset": preset, "cfg": cfg, "run": run, "arch": arch, "arch_label": label, "config_label": preset + ("+custom" if custom else ""),
            "record": {"preset": preset, "hp": cfg, "run": run, "arch": shown, "arch_name": name, "arch_hash": archnet.spec_hash(shown), "arch_params": archnet.describe(shown)["params"]}}


def describe_all():
    """What a settings panel needs: the presets, every value with its limits and its default, and the architecture presets."""
    hp = {}
    for k, label in hpspec.HP_LABELS.items():
        hp[k] = {"label": label, "default": hpspec.HP_DEFAULTS.get(k)}
        if k in hpspec.HP_LIMITS: hp[k].update(min=hpspec.HP_LIMITS[k][0], max=hpspec.HP_LIMITS[k][1])
        else: hp[k]["options"] = HP_OPTIONS[k]
    aug = {k: {"label": label, "min": hpspec.AUG_LIMITS[k][0], "max": hpspec.AUG_LIMITS[k][1]} for k, label in hpspec.AUG_LABELS.items()}
    return {"presets": hpspec.PRESETS, "default_preset": hpspec.DEFAULT_PRESET, "hp": hp, "aug": aug,
            "arch_presets": {k: {"label": v["label"], "spec": archnet.normalize(k), "hash": archnet.spec_hash(k), **archnet.describe(archnet.normalize(k))} for k, v in archnet.PRESETS.items()},
            "arch_default": archnet.DEFAULT, "arch_limits": dict(archnet.LIMITS),
            "run": {k: {"default": hpspec.RUN_DEFAULTS[k], "min": hpspec.RUN_LIMITS[k][0], "max": hpspec.RUN_LIMITS[k][1], "label": {"checkpoint_every": "save a checkpoint every N epochs (0 = only the best)", "keep_last": "numbered checkpoints kept per fold"}[k]} for k in hpspec.RUN_DEFAULTS}}
