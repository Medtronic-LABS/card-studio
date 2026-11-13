"""The settings of one training run (preset, every training value, the network): see work_phaseA/trainsettings.py, which holds the code and is shared with the Frappe app.
This module keeps the name Studio's server and trainer import."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "work_phaseA"))
from trainsettings import describe_all, load  # noqa: E402,F401
import hpspec  # noqa: E402

PRESET_KEYS = tuple(hpspec.PRESETS)
PRESET_VALUES = hpspec.PRESETS
