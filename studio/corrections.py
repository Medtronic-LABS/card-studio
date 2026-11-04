"""Import corrected labels into Studio. Local only; never touches work/labels.csv.

Two sources, one destination (studio/imports/labels_imported.csv; crops that arrive with a correction go to
studio/imports/crops/<field>/<col>/<page>.png, the same layout as work/crops so the trainer reads them the same way):

1. Phone corrections: a zip (or a folder pulled off the phone) with
     corrections.jsonl    one JSON object per line: {"id", "pack_id", "pack_version", "field", "col", "label", "crop", "created"}
     crops/<name>.png     the cell image the person looked at (crop = member name)
   `col` is the visit column as "c<N>" (c1 = first visit). `field` is the pack's field key.
2. Label Studio JSON export: tasks whose image file is named <field>_<col>_<page>.png (as the Studio's own Label Studio
   project names them); the cell image is the existing work/crops file, so only the label is imported.

A re-import is safe: rows are keyed by (page, field, col) and the newest wins.
CLI: python corrections.py import-phone <zip|folder>   |   python corrections.py import-ls <export.json>   |   python corrections.py summary
"""
import csv, json, re, sys, time, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMPORTS = HERE / "imports"
CSV_COLS = ["page", "row", "col", "label", "crop", "source", "pack", "pack_version", "created"]
FIELD = re.compile(r"^[A-Za-z0-9_]{1,40}$")
COL = re.compile(r"^c[1-9][0-9]?$")
ID = re.compile(r"^[0-9A-Za-z-]{8,64}$")
CROPNAME = re.compile(r"^[A-Za-z0-9_.-]{1,80}\.png$")
PNG = b"\x89PNG\r\n\x1a\n"
MAX_RECORDS, MAX_CROP, MAX_LABEL = 5000, 400_000, 40
LSNAME = re.compile(r"^(?:[0-9a-f]{8}-)?(?P<row>[A-Za-z0-9_]+?)_(?P<col>c[1-9][0-9]?)_(?P<page>p-\d+)\.png$")


def _paths():
    return IMPORTS / "labels_imported.csv", IMPORTS / "crops"


def read_imported():
    csvp, _ = _paths()
    if not csvp.exists(): return []
    with open(csvp, newline="") as fh:
        return list(csv.DictReader(fh))


def _merge(new_rows):
    """Write imported labels keyed by (page,row,col); newest wins. Returns (added, changed)."""
    csvp, _ = _paths(); csvp.parent.mkdir(parents=True, exist_ok=True)
    cur = {(r["page"], r["row"], r["col"]): r for r in read_imported()}
    added = changed = 0
    for r in new_rows:
        k = (r["page"], r["row"], r["col"])
        if k not in cur: added += 1
        elif cur[k]["label"] != r["label"]: changed += 1
        cur[k] = {c: r.get(c, "") for c in CSV_COLS}
    with open(csvp, "w", newline="") as fh:
        w = csv.DictWriter(fh, CSV_COLS); w.writeheader(); w.writerows(cur.values())
    return added, changed


def _clean_label(v):
    if not isinstance(v, str): raise ValueError("label must be text")
    v = v.strip()
    if not v or len(v) > MAX_LABEL: raise ValueError(f"label empty or longer than {MAX_LABEL}")
    if any(ord(c) < 32 for c in v): raise ValueError("label has control characters")
    return v


def import_phone(read_member, names, jsonl_name="corrections.jsonl"):
    """read_member(name)->bytes, names = member names. Returns a report; bad records are skipped with a reason."""
    if jsonl_name not in names: raise ValueError("no corrections.jsonl in the upload")
    lines = [l for l in read_member(jsonl_name).decode("utf-8").splitlines() if l.strip()]
    if len(lines) > MAX_RECORDS: raise ValueError(f"too many records (limit {MAX_RECORDS})")
    _, cropdir = _paths(); good, bad = [], []
    for n, line in enumerate(lines, 1):
        try:
            r = json.loads(line)
            if not isinstance(r, dict): raise ValueError("not an object")
            if not ID.match(str(r.get("id", ""))): raise ValueError("bad id")
            if not FIELD.match(str(r.get("field", ""))): raise ValueError("bad field key")
            if not COL.match(str(r.get("col", ""))): raise ValueError("col must be c1, c2, ...")
            if not isinstance(r.get("pack_version"), int) or not FIELD.match(str(r.get("pack_id", ""))): raise ValueError("pack_id / pack_version missing")
            label = _clean_label(r.get("label"))
            cn = str(r.get("crop", ""))
            member = f"crops/{cn}"
            if not CROPNAME.match(cn) or member not in names: raise ValueError("crop missing or has an unsafe name")
            png = read_member(member)
            if not png.startswith(PNG) or len(png) > MAX_CROP: raise ValueError("crop is not a small PNG")
            page = "ph-" + str(r["id"]).replace("-", "")[:12]
            out = cropdir / r["field"] / r["col"] / f"{page}.png"
            out.parent.mkdir(parents=True, exist_ok=True); out.write_bytes(png)
            good.append({"page": page, "row": r["field"], "col": r["col"], "label": label, "crop": str(out.relative_to(IMPORTS)),
                         "source": "phone", "pack": r["pack_id"], "pack_version": r["pack_version"], "created": str(r.get("created", ""))[:32]})
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as e:
            bad.append({"line": n, "reason": str(e)})
    added, changed = _merge(good)
    return {"records": len(lines), "imported": len(good), "new": added, "changed": changed, "rejected": bad}


def import_phone_zip(data: bytes):
    import io
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ValueError("not a zip file")
    names = set()
    for i in z.infolist():
        if i.filename.startswith("/") or ".." in Path(i.filename).parts or i.is_dir(): continue
        if i.file_size > 5_000_000: raise ValueError(f"{i.filename} is too large")
        names.add(i.filename)
    return import_phone(z.read, names)


def import_phone_folder(folder):
    folder = Path(folder).resolve()
    names = {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()}
    return import_phone(lambda n: (folder / n).read_bytes(), names)


def import_label_studio(export: bytes, crops_root=None):
    """Label Studio JSON export -> labels for existing crops (work/crops by default)."""
    crops_root = Path(crops_root) if crops_root else HERE.parent / "work" / "crops"
    tasks = json.loads(export)
    if not isinstance(tasks, list): raise ValueError("expected the JSON export (a list of tasks)")
    rows, skipped = [], []
    for t in tasks:
        name = Path(str((t.get("data") or {}).get("image", ""))).name
        m = LSNAME.match(name)
        if not m: skipped.append({"task": t.get("id"), "reason": f"file name {name!r} is not <field>_<col>_<page>.png"}); continue
        anns = [a for a in t.get("annotations", []) if not a.get("was_cancelled")]
        if not anns: skipped.append({"task": t.get("id"), "reason": "no annotation"}); continue
        try:
            text = next(v for r in anns[-1].get("result", []) if r.get("type") == "textarea" for v in r["value"]["text"])
            label = _clean_label(text)
        except (StopIteration, KeyError, ValueError) as e:
            skipped.append({"task": t.get("id"), "reason": f"no usable text ({e})"}); continue
        if not (crops_root / m["row"] / m["col"] / f"{m['page']}.png").exists():
            skipped.append({"task": t.get("id"), "reason": "no matching crop in work/crops"}); continue
        rows.append({"page": m["page"], "row": m["row"], "col": m["col"], "label": label, "crop": "", "source": "label-studio",
                     "pack": "", "pack_version": "", "created": time.strftime("%Y-%m-%d %H:%M")})
    added, changed = _merge(rows)
    return {"tasks": len(tasks), "imported": len(rows), "new": added, "changed": changed, "skipped": skipped}


def summary():
    rows = read_imported(); by = {}
    for r in rows:
        k = (r["source"], r["pack"] or "-", r["pack_version"] or "-"); by[k] = by.get(k, 0) + 1
    return {"total": len(rows), "by_source": [{"source": s, "pack": p, "pack_version": v, "labels": n} for (s, p, v), n in sorted(by.items())]}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "summary"
    if cmd == "import-phone":
        a = Path(sys.argv[2]); print(json.dumps(import_phone_folder(a) if a.is_dir() else import_phone_zip(a.read_bytes()), indent=1))
    elif cmd == "import-ls": print(json.dumps(import_label_studio(Path(sys.argv[2]).read_bytes()), indent=1))
    else: print(json.dumps(summary(), indent=1))
