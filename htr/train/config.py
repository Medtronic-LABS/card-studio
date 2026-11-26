"""A training run's config: what `loop.train` needs, with every default filled in and every value checked. Pure Python.

    {"run": {"name": str},
     "optim": {"optimizer": "adam"|"adamw"|"sgd", "lr": float, "weight_decay": float, "schedule": "none"|"cosine"|"step"|"plateau",
               "step_every": int, "step_gamma": float, "plateau_patience": int},
     "loop": {"batch_size": int, "epochs": int, "patience": int, "monitor": "cer_grapheme"|"exact"|"loss", "min_delta": float,
              "grad_clip": float, "balance_by_kind": bool}}

Unlisted keys take the default below; an unknown key, or a value out of `LIMITS`, raises ValueError naming what is wrong."""
DEFAULTS = {
    "run": {"name": "run"},
    "optim": {"optimizer": "adamw", "lr": 1e-3, "weight_decay": 1e-4, "schedule": "cosine", "step_every": 50, "step_gamma": 0.5, "plateau_patience": 10},
    "loop": {"batch_size": 32, "epochs": 100, "patience": 20, "monitor": "cer_grapheme", "min_delta": 0.0, "grad_clip": 5.0, "balance_by_kind": True},
}
LIMITS = {
    "optim.lr": (1e-6, 1.0), "optim.weight_decay": (0.0, 1.0), "optim.step_every": (1, 10_000), "optim.step_gamma": (0.0, 1.0), "optim.plateau_patience": (1, 1000),
    "loop.batch_size": (1, 1024), "loop.epochs": (1, 100_000), "loop.patience": (1, 100_000), "loop.min_delta": (0.0, 1.0), "loop.grad_clip": (0.0, 1000.0),
}
OPTIMIZERS = ("adam", "adamw", "sgd")
SCHEDULES = ("none", "cosine", "step", "plateau")
MONITORS = ("cer_grapheme", "cer_codepoint", "exact", "loss")


def _merge(defaults, overrides, path=""):
    out = dict(defaults)
    for k, v in (overrides or {}).items():
        key = f"{path}.{k}" if path else k
        if k not in defaults: raise ValueError(f"unknown config key {key!r}")
        if isinstance(defaults[k], dict):
            if not isinstance(v, dict): raise ValueError(f"{key} must be an object")
            out[k] = _merge(defaults[k], v, key)
        else:
            out[k] = v
    return out


def resolve(overrides=None):
    """-> the full config, `DEFAULTS` overlaid with `overrides` (a dict shaped like `DEFAULTS`, any subset of keys)."""
    cfg = _merge(DEFAULTS, overrides, "")
    for dotted, (lo, hi) in LIMITS.items():
        section, key = dotted.split(".")
        v = cfg[section][key]
        if not lo <= v <= hi: raise ValueError(f"{dotted} must be between {lo} and {hi}, got {v}")
    if cfg["optim"]["optimizer"] not in OPTIMIZERS: raise ValueError(f"optim.optimizer must be one of {OPTIMIZERS}")
    if cfg["optim"]["schedule"] not in SCHEDULES: raise ValueError(f"optim.schedule must be one of {SCHEDULES}")
    if cfg["loop"]["monitor"] not in MONITORS: raise ValueError(f"loop.monitor must be one of {MONITORS}")
    return cfg
