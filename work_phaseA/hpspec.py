"""Every training value the trainer takes, with its default and its limits, and the three presets. Pure Python: no torch, so a settings panel and a bundle check can use it.

`resolve(preset, overrides)` is the one place a config is built and checked (train.resolve_cfg calls it); with no overrides it is exactly the preset, so a run that sets
nothing trains as it always did. This file is kept identical in work_phaseA/ and in the Frappe app (card_studio/engine/hpspec.py).
"""
import copy

# the starting points: learning rate and how hard the cells are distorted while training (the notebook's strength kept as a reference config)
PRESETS = {
    "light_lr1e-4": dict(lr=1e-4, aug=dict(rot=2, tx=2, ty=1, bright=.08, shift=8, blur=.15)),
    "light_lr3e-4": dict(lr=3e-4, aug=dict(rot=2, tx=2, ty=1, bright=.08, shift=8, blur=.15)),
    "nbaug_lr1e-4": dict(lr=1e-4, aug=dict(rot=3, tx=3, ty=2, bright=.15, shift=15, blur=.30)),
}
DEFAULT_PRESET = "light_lr3e-4"          # chosen on validation in Phase A (selection.json)
HP_DEFAULTS = dict(optimizer="adam", weight_decay=0.0, batch=64, schedule="none", step_every=50, step_gamma=0.5, grad_clip=0.0, balance=True, patience=80)
AUG_DEFAULTS = dict(rot=0.0, tx=0.0, ty=0.0, bright=0.0, shift=0.0, blur=0.0, noise=0.0, erase=0.0)
HP_LIMITS = dict(lr=(1e-6, 1e-1), weight_decay=(0.0, 0.1), batch=(4, 256), step_every=(1, 500), step_gamma=(0.05, 0.95), grad_clip=(0.0, 100.0), patience=(3, 500))
AUG_LIMITS = dict(rot=(0.0, 15.0), tx=(0.0, 20.0), ty=(0.0, 10.0), bright=(0.0, 0.6), shift=(0.0, 60.0), blur=(0.0, 1.0), noise=(0.0, 40.0), erase=(0.0, 0.8))
OPTIMIZERS, SCHEDULES = ("adam", "adamw", "sgd"), ("none", "cosine", "step")
HP_LABELS = {"lr": "learning rate", "optimizer": "optimizer", "weight_decay": "weight decay", "batch": "batch size", "schedule": "learning-rate schedule", "step_every": "epochs between steps (step schedule)",
             "step_gamma": "step factor (step schedule)", "grad_clip": "gradient clip (0 = off)", "balance": "balance fields when sampling", "patience": "early stopping patience, epochs"}
AUG_LABELS = {"rot": "rotation, degrees", "tx": "shift sideways, pixels", "ty": "shift up/down, pixels", "bright": "brightness variation", "shift": "gray-level shift", "blur": "chance of blur",
              "noise": "pixel noise, gray levels", "erase": "chance of a missing patch"}


def resolve(base, overrides=None, presets=None):
    """A preset name (a key of `presets`, default PRESETS) or a config dict, plus overrides -> the full, validated training config. Raises ValueError naming the bad value."""
    presets = presets or PRESETS
    if isinstance(base, str) and base not in presets: raise ValueError(f"unknown preset {base!r}: {', '.join(presets)}")
    cfg = copy.deepcopy(presets[base] if isinstance(base, str) else base)
    o = dict(overrides or {}); oaug = o.pop("aug", None) or {}
    if not isinstance(oaug, dict): raise ValueError("aug is an object of augmentation strengths")
    unknown = sorted(set(o) - set(HP_DEFAULTS) - {"lr"})
    if unknown: raise ValueError(f"unknown training setting(s): {', '.join(unknown)}")
    unknown = sorted(set(oaug) - set(AUG_DEFAULTS))
    if unknown: raise ValueError(f"unknown augmentation(s): {', '.join(unknown)}")
    full = {**HP_DEFAULTS, **{k: v for k, v in cfg.items() if k != "aug"}, **o}
    aug = {**AUG_DEFAULTS, **cfg.get("aug", {}), **oaug}
    def num(name, v, lo, hi, kind=float):
        if isinstance(v, bool): raise ValueError(f"{name} must be a number")
        try: x = kind(v)
        except (TypeError, ValueError): raise ValueError(f"{name} must be a number")
        if not lo <= x <= hi: raise ValueError(f"{name} must be between {lo} and {hi}")
        return x
    for k in ("lr", "weight_decay", "step_gamma", "grad_clip"): full[k] = num(k, full[k], *HP_LIMITS[k])
    for k in ("batch", "step_every", "patience"): full[k] = num(k, full[k], *HP_LIMITS[k], int)
    if full["optimizer"] not in OPTIMIZERS: raise ValueError(f"optimizer is one of {', '.join(OPTIMIZERS)}")
    if full["schedule"] not in SCHEDULES: raise ValueError(f"schedule is one of {', '.join(SCHEDULES)}")
    full["balance"] = bool(full["balance"])
    for k in aug: aug[k] = num(f"augmentation {k}", aug[k], *AUG_LIMITS[k])
    full["aug"] = aug
    return full
