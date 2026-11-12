"""The starting weights a training run begins from. Local only; listing and choosing need no torch, making new ones does.

A reader is pretrained on synthetic cells, then fine-tuned on labeled ones. Which pretraining a run starts from used to be a fixed file
(work_phaseA/pretrained.pt: 4,000 digit strings stitched on white squares). A run can now choose, and the choice is recorded with its results.
Weights live in studio/pretrained/ (gitignored: the generator draws on the empty cells of real cards), each with a small JSON record beside it:

    <name>.pt      state dict of the shipped CRNN (0.54M parameters), 129 classes
    <name>.json    what made it: generator, number of images and epochs, recipe, the classes it was made for, a hash of the weights, when
    DEFAULT        the name a run uses when it asks for `auto` (absent: `legacy`)

`legacy` is always available and is the original file. A name is accepted only when its record exists, the weights hash matches the record, and the
class list it was made for is the class list in model_meta.json now: weights for a different list would silently mislabel every character.

    python pretrained.py list
    python pretrained.py make NAME [--images 30000] [--epochs 4]     draws synthetic cells (synthgen.py) and pretrains; about 10 minutes, needs torch
    python pretrained.py default NAME                                 what `auto` means from now on
    python pretrained.py adopt NAME WEIGHTS.pt [--note TEXT] [--images N] [--epochs N]    register weights made elsewhere (an experiment's output)
"""
import hashlib, json, re, shutil, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "work_phaseA")); sys.path.insert(0, str(HERE))
DIR = HERE / "pretrained"
LEGACY = ROOT / "work_phaseA" / "pretrained.pt"
NAME = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def classes_sha():
    import cards
    return hashlib.sha256(json.dumps(cards.meta()["classes"], ensure_ascii=False).encode()).hexdigest()[:16]


def default_name():
    p = DIR / "DEFAULT"
    return p.read_text().strip() if p.exists() else "legacy"


def _record(name):
    try: return json.loads((DIR / f"{name}.json").read_text())
    except (OSError, ValueError): return None


def list_all():
    """[{name, made, images, epochs, generator, note, bytes, default, ok, problem}], legacy first."""
    out = [{"name": "legacy", "made": time.strftime("%Y-%m-%d", time.localtime(LEGACY.stat().st_mtime)) if LEGACY.exists() else None, "images": 4000, "epochs": 15,
            "generator": "stitched digit strings on white squares (the original)", "note": "the weights every reader before this feature started from", "bytes": LEGACY.stat().st_size if LEGACY.exists() else 0,
            "ok": LEGACY.exists(), "problem": "" if LEGACY.exists() else "work_phaseA/pretrained.pt is missing"}]
    for p in sorted(DIR.glob("*.pt")) if DIR.exists() else []:
        r = _record(p.stem); problem = ""
        if r is None: problem = "no record beside the weights"
        elif r.get("sha256") != sha256(p): problem = "the weights do not match their record"
        elif r.get("classes_sha256") != classes_sha(): problem = "made for a different class list than model_meta.json has now"
        out.append({"name": p.stem, "made": (r or {}).get("made"), "images": (r or {}).get("images"), "epochs": (r or {}).get("epochs"), "generator": (r or {}).get("generator", ""),
                    "note": (r or {}).get("note", ""), "bytes": p.stat().st_size, "ok": not problem, "problem": problem})
    d = default_name()
    for r in out: r["default"] = r["name"] == d
    return out


def resolve(choice="auto"):
    """-> (path, record) for `auto`, `legacy` or a registered name. Raises ValueError saying what is wrong."""
    name = default_name() if choice in (None, "", "auto") else choice
    if name == "auto": name = "legacy"
    if name != "legacy" and not NAME.match(name): raise ValueError(f"bad name {name!r}")
    hit = next((r for r in list_all() if r["name"] == name), None)
    if hit is None: raise ValueError(f"no starting weights named {name!r}: have {', '.join(r['name'] for r in list_all())}")
    if not hit["ok"]: raise ValueError(f"starting weights {name!r} cannot be used: {hit['problem']}")
    return (LEGACY if name == "legacy" else DIR / f"{name}.pt"), hit


def adopt(name, weights, note="", extra=None):
    """Register a weights file under `name` with a record. -> the record."""
    if name == "legacy" or not NAME.match(name): raise ValueError("a name is letters, digits, _ and -, at most 32, and not `legacy`")
    if (DIR / f"{name}.pt").exists(): raise ValueError(f"{name!r} already exists")
    import synthgen
    DIR.mkdir(parents=True, exist_ok=True); shutil.copy2(weights, DIR / f"{name}.pt")
    rec = {"name": name, "made": time.strftime("%Y-%m-%d %H:%M"), "note": note, "sha256": sha256(DIR / f"{name}.pt"), "classes_sha256": classes_sha(), "architecture": "base CRNN, 0.54M parameters",
           "generator": "synthgen: grammars, empty-cell backgrounds, two-line layouts, damage", **(extra or {})}
    (DIR / f"{name}.json").write_text(json.dumps(rec, indent=1))
    if not (DIR / "DEFAULT").exists(): (DIR / "DEFAULT").write_text(name)
    return rec


def make(name, images=30000, epochs=4, log=print):
    """Draw synthetic cells and pretrain the shipped reader on them. Needs torch. -> the record."""
    import tempfile, synthexp
    if name == "legacy" or not NAME.match(name): raise ValueError("a name is letters, digits, _ and -, at most 32, and not `legacy`")
    if (DIR / f"{name}.pt").exists(): raise ValueError(f"{name!r} already exists")
    with tempfile.TemporaryDirectory() as d:
        w = Path(d) / "w.pt"; synthexp.pretrain(str(w), images, epochs, log=log)
        return adopt(name, w, extra={"images": images, "epochs": epochs, "generator_file_sha256": sha256(HERE / "synthgen.py")[:16], "recipe": {"learning_rate": synthexp.K.PRE_LR, "batch": synthexp.K.PRE_BATCH, "warmup_steps": synthexp.K.WARMUP, "grad_clip": synthexp.K.CLIP},
                                     "backgrounds_from": list(synthexp.BG_CARDS)})


def set_default(name):
    resolve(name)
    if name != "legacy": DIR.mkdir(parents=True, exist_ok=True)
    (DIR / "DEFAULT").write_text(name)


if __name__ == "__main__":
    a = sys.argv[1:]
    try:
        if a[:1] == ["list"]:
            for r in list_all(): print(f"{'*' if r['default'] else ' '} {r['name']:10s} {r['made'] or '-':16s} images {r['images']} epochs {r['epochs']}  {'ok' if r['ok'] else 'UNUSABLE: ' + r['problem']}  {r['generator']}")
        elif a[:1] == ["make"] and len(a) >= 2:
            opt = lambda k, d: int(a[a.index(k) + 1]) if k in a else d
            print(json.dumps(make(a[1], opt("--images", 30000), opt("--epochs", 4)), indent=1))
        elif a[:1] == ["default"] and len(a) == 2: set_default(a[1]); print("default is now", a[1])
        elif a[:1] == ["adopt"] and len(a) >= 3:
            opt = lambda k: {k[2:]: int(a[a.index(k) + 1])} if k in a else {}
            print(json.dumps(adopt(a[1], a[2], a[a.index("--note") + 1] if "--note" in a else "", {**opt("--images"), **opt("--epochs")}), indent=1))
        else: sys.exit("usage: python pretrained.py list | make NAME [--images N] [--epochs N] | default NAME | adopt NAME WEIGHTS.pt [--note TEXT]")
    except ValueError as e:
        sys.exit(str(e))
