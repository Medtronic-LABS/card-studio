"""Card pack: build, read and validate `.cardpack` archives. See CARDPACK.md.
A pack is a zip with exactly: manifest.json, template.png, grid.json, fields.json, reader.onnx, fallback.json, and optionally reader.tflite.
Local only; no network. Usage: python cardpack.py validate PACK [PACK...]"""
import hashlib, io, json, sys, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REQUIRED = ["manifest.json", "template.png", "grid.json", "fields.json", "reader.onnx", "fallback.json"]
HASHED = [n for n in REQUIRED if n != "manifest.json"]
OPTIONAL = ["reader.tflite"]      # the same reader for TensorFlow Lite; described in manifest.reader.tflite


def hashed(members):
    """The members whose sha256 the manifest must list: every required one, plus the optional ones this pack carries."""
    return HASHED + [n for n in OPTIONAL if n in members]
MAX_MEMBER = 200 * 1024 * 1024
SCHEMAS = {n: json.loads((HERE / "schema" / f"{n}.schema.json").read_text()) for n in ("manifest", "fields", "grid", "fallback")}


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def build(members: dict, out_path):
    """members: name -> bytes (all of REQUIRED). Deterministic: sorted, fixed timestamps, so a round trip is byte-identical."""
    missing = [n for n in REQUIRED if n not in members]
    if missing:
        raise ValueError(f"missing members: {missing}")
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in sorted(members):
            zi = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, members[name])


def read(path) -> dict:
    """-> name -> bytes. Rejects unsafe names and oversized members before reading them."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for zi in z.infolist():
            n = zi.filename
            if n not in REQUIRED and n not in OPTIONAL:
                raise ValueError(f"unexpected member {n!r} (only {REQUIRED + OPTIONAL} allowed)")
            if n in out:
                raise ValueError(f"duplicate member {n!r}")
            if zi.file_size > MAX_MEMBER:
                raise ValueError(f"member {n!r} too large ({zi.file_size} bytes)")
            out[n] = z.read(n)
    return out


def _schema_errors(name, doc):
    import jsonschema
    v = jsonschema.Draft202012Validator(SCHEMAS[name])
    return [f"{name}: {'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in sorted(v.iter_errors(doc), key=str)]


NORMALIZE = ("flatten", "contrast_match", "cell_stretch", "flatten_cell_stretch", "none")   # see CARDPACK.md "Photo normalization"


def scan_reference(image_paths, stride=4):
    """Gray-level mean and std over scans (every `stride`th pixel each way): the reference `contrast_match` pulls a photo towards.
    Use the scans the reader was trained on."""
    import cv2, numpy as np
    vals = [cv2.imread(str(p), 0)[::stride, ::stride].reshape(-1).astype(np.float64) for p in image_paths]
    a = np.concatenate(vals)
    return round(float(a.mean()), 2), round(float(a.std()), 2)


def derive(src, out_path, version, normalize):
    """New pack = src with a different `preprocess.normalize` and a new version. Reader, grid, fields and held-out metrics are
    copied unchanged (the held-out numbers describe the reader, not the normalization: see CARDPACK.md)."""
    m = read(src); man = json.loads(m["manifest.json"])
    if version <= man["version"]:
        raise ValueError(f"version must be above {man['version']}")
    man["version"] = version; man["preprocess"]["normalize"] = normalize
    m["manifest.json"] = json.dumps(man, indent=1).encode()
    man["files"] = {n: sha256(m[n]) for n in hashed(m)}
    m["manifest.json"] = json.dumps(man, indent=1).encode()
    build(m, out_path)
    return validate(out_path, check_reader=False)


def _tflite_interpreter():
    """A TFLite interpreter class if one is installed here (LiteRT, tflite-runtime or TensorFlow), else None."""
    for mod, attr in (("ai_edge_litert.interpreter", "Interpreter"), ("tflite_runtime.interpreter", "Interpreter")):
        try: return getattr(__import__(mod, fromlist=[attr]), attr)
        except Exception: pass
    try:
        import tensorflow as tf; return tf.lite.Interpreter
    except Exception: return None


def _check_tflite(blob, t, ncls):
    """Load reader.tflite and run one blank cell. With no TFLite interpreter installed this is skipped (TensorFlow is large and only the
    export step needs it); the parity numbers recorded in the manifest at export time are then the only evidence."""
    if not blob.startswith(b"TFL3", 4): return ["reader.tflite: not a TFLite model"]
    I = _tflite_interpreter()
    if I is None: return []
    try:
        import numpy as np
        it = I(model_content=blob); it.allocate_tensors(); i, o = it.get_input_details()[0], it.get_output_details()[0]
        want = [1, t["input"]["height"], t["input"]["width"], 1]
        if [int(v) for v in i["shape"]] != want: return [f"reader.tflite input is {[int(v) for v in i['shape']]}, manifest says {want}"]
        it.set_tensor(i["index"], np.zeros(want, np.float32)); it.invoke(); out = it.get_tensor(o["index"])
        return [] if out.ndim == 3 and out.shape[2] == ncls else [f"reader.tflite output {list(out.shape)} does not end in {ncls} classes"]
    except Exception as e:
        return [f"reader.tflite: {type(e).__name__}: {e}"]


def validate(path, check_reader=True) -> list:
    """-> list of error strings; empty list means the pack is valid."""
    try:
        m = read(path)
    except Exception as e:
        return [f"archive: {e}"]
    errs = [f"missing member {n}" for n in REQUIRED if n not in m]
    if errs:
        return errs
    docs = {}
    for n, key in (("manifest.json", "manifest"), ("fields.json", "fields"), ("grid.json", "grid"), ("fallback.json", "fallback")):
        try:
            docs[key] = json.loads(m[n])
        except Exception as e:
            errs.append(f"{n}: not valid JSON ({e})")
    if errs:
        return errs
    for key, d in docs.items():
        errs += _schema_errors(key, d)
    if errs:
        return errs  # structural errors first; cross-checks assume the shapes are right
    man, fld, grid = docs["manifest"], docs["fields"], docs["grid"]
    nz = man["preprocess"].get("normalize")
    if nz and nz["default"] == "contrast_match" and "reference" not in nz:
        errs.append("preprocess.normalize: contrast_match needs a reference (mean, std, source)")

    # integrity
    if set(man["files"]) != set(hashed(m)):
        errs.append(f"manifest.files must list exactly {hashed(m)}, got {sorted(man['files'])}")
    if ("reader.tflite" in m) != ("tflite" in man["reader"]):
        errs.append("reader.tflite and manifest.reader.tflite go together: one is present without the other")
    for n in hashed(m):
        if n in man["files"] and sha256(m[n]) != man["files"][n]:
            errs.append(f"{n}: sha256 does not match manifest")

    # template
    try:
        import cv2, numpy as np
        im = cv2.imdecode(np.frombuffer(m["template.png"], np.uint8), cv2.IMREAD_GRAYSCALE)
        if im is None:
            errs.append("template.png: not a decodable image")
        else:
            t = man["template"]
            if (im.shape[1], im.shape[0]) != (t["width"], t["height"]):
                errs.append(f"template.png is {im.shape[1]}x{im.shape[0]}, manifest says {t['width']}x{t['height']}")
            exp = [round(t["width"] * t["grid_scale"]), round(t["height"] * t["grid_scale"])]
            if grid["image_size"] != exp:
                errs.append(f"grid.image_size {grid['image_size']} != template size * grid_scale {exp}")
    except ImportError:
        errs.append("cannot check template.png: opencv not installed")

    # grid geometry
    inc = lambda xs: all(b > a for a, b in zip(xs, xs[1:]))
    nrows, ncols = len(grid["rows"]), len(grid["col_lines_top"]) - 1
    if len(grid["col_lines_bot"]) != ncols + 1:
        errs.append("grid: col_lines_top and col_lines_bot differ in length")
    if not inc(grid["col_lines_top"]) or not inc(grid["col_lines_bot"]):
        errs.append("grid: column lines must increase left to right")
    if len(grid["row_lines_by_col"]) != ncols:
        errs.append(f"grid: row_lines_by_col has {len(grid['row_lines_by_col'])} columns, expected {ncols}")
    for c, rl in enumerate(grid["row_lines_by_col"]):
        if len(rl) != nrows + 1 or not inc(rl):
            errs.append(f"grid: column {c + 1} needs {nrows + 1} increasing row lines")
    if grid["y_ref_bot"] <= grid["y_ref_top"]:
        errs.append("grid: y_ref_bot must be below y_ref_top")
    W, H = grid["image_size"]
    for rl in grid["row_lines_by_col"]:
        if rl and (rl[0] < 0 or rl[-1] > H):
            errs.append("grid: row lines fall outside the image"); break
    if max(grid["col_lines_top"] + grid["col_lines_bot"]) > W:
        errs.append("grid: column lines fall outside the image")

    # fields
    ncls, blank = len(fld["classes"]), fld["blank_idx"]
    if blank >= ncls:
        errs.append("fields: blank_idx outside classes")
    keys = [f["key"] for f in fld["fields"]]
    if len(set(keys)) != len(keys):
        errs.append("fields: duplicate field keys")
    by_col = grid.get("axis", "rows") == "columns"
    if (man["grid"]["type"] == "ruled-rows-v1") != by_col:
        errs.append(f"manifest grid type {man['grid']['type']} does not agree with grid axis {grid.get('axis', 'rows')!r} (ruled-rows-v1 goes with axis columns)")
    if by_col and (len(grid.get("cols") or []) != ncols or len(set(grid.get("cols") or [])) != ncols):
        errs.append(f"grid: axis columns needs {ncols} unique names in cols")
    names = (grid.get("cols") or []) if by_col else grid["rows"]
    if set(keys) != set(names):
        errs.append(f"fields keys and grid {'cols' if by_col else 'rows'} differ: only in fields {sorted(set(keys) - set(names))}, only in grid {sorted(set(names) - set(keys))}")
    held = (man["metrics"].get("heldout") or {}).get("per_field", {})
    for f in fld["fields"]:
        k = f["key"]
        if any(i >= ncls for i in f["allowed_idx"]):
            errs.append(f"{k}: allowed_idx outside classes")
        if blank not in f["allowed_idx"]:
            errs.append(f"{k}: allowed_idx must include the blank class")
        if f["range"] is not None and (f["kind"] != "number" or f["range"][0] >= f["range"][1]):
            errs.append(f"{k}: range only valid for kind number with lo < hi")
        if f["kind"] == "bp" and "bp" not in f:
            errs.append(f"{k}: kind bp needs a bp rule")
        if f["kind"] == "mark":
            vv = f.get("valid_values")
            if not vv:
                errs.append(f"{k}: kind mark needs valid_values")
            else:
                allowed = {fld["classes"][i] for i in f["allowed_idx"] if i < ncls}
                bad = [v for v in vv if v not in allowed]
                if bad:
                    errs.append(f"{k}: valid_values {bad} not in the field's allowed classes")
        if f["status"] == "ship":
            if f["min_confidence"] is None:
                errs.append(f"{k}: a shipped field needs min_confidence")
            if k not in held:
                errs.append(f"{k}: status ship needs held-out evidence in manifest.metrics.heldout.per_field")

    # reader
    if check_reader:
        try:
            import numpy as np, onnxruntime as ort
            s = ort.InferenceSession(m["reader.onnx"], providers=["CPUExecutionProvider"])
            r = man["reader"]
            ins = {i.name: i.shape for i in s.get_inputs()}
            if r["input"]["name"] not in ins:
                errs.append(f"reader.onnx has no input {r['input']['name']!r} (has {list(ins)})")
            else:
                x = np.zeros((1, 1, r["input"]["height"], r["input"]["width"]), np.float32)
                out = s.run([r["output"]["name"]], {r["input"]["name"]: x})[0]
                if out.ndim != 3 or out.shape[2] != ncls:
                    errs.append(f"reader.onnx output {out.shape} does not end in {ncls} classes")
        except ImportError:
            errs.append("cannot check reader.onnx: onnxruntime not installed")
        except Exception as e:
            errs.append(f"reader.onnx: {type(e).__name__}: {e}")
        if "reader.tflite" in m and "tflite" in man["reader"]:
            errs += _check_tflite(m["reader.tflite"], man["reader"]["tflite"], ncls)
    return errs


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "normalize":
        import argparse
        ap = argparse.ArgumentParser(prog="cardpack.py normalize"); ap.add_argument("pack"); ap.add_argument("--version", type=int, required=True)
        ap.add_argument("--default", choices=NORMALIZE, required=True); ap.add_argument("--scans", nargs="*", default=[], help="scan images that give the reference")
        a = ap.parse_args(sys.argv[2:]); nz = {"default": a.default}
        if a.scans:
            mean, std = scan_reference(a.scans); nz["reference"] = {"mean": mean, "std": std, "source": f"gray mean/std over {len(a.scans)} scans the reader was trained on"}
        man = json.loads(read(a.pack)["manifest.json"]); out = Path(a.pack).with_name(f"{man['id']}_v{a.version}.cardpack")
        e = derive(a.pack, out, a.version, nz); print(out, "OK" if not e else e); sys.exit(1 if e else 0)
    if len(sys.argv) < 3 or sys.argv[1] != "validate":
        sys.exit("usage: python cardpack.py validate PACK [PACK...] | normalize PACK --version N --default METHOD [--scans IMG...]")
    bad = 0
    for p in sys.argv[2:]:
        e = validate(p)
        print(f"{p}: {'OK' if not e else 'INVALID'}")
        for x in e:
            print("  -", x)
        bad += bool(e)
    sys.exit(1 if bad else 0)
