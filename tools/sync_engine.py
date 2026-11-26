"""Checks that the Frappe app's (frappe_card_studio) `engine/` byte-copies still match their source files here, so a change made
on one side is never silently missed on the other. The manifest of which files are meant to match - and which have intentionally
diverged, or aren't tracked yet - lives in the Frappe app itself: card_studio/card_studio/engine/SYNC.json.

    python tools/sync_engine.py [FRAPPE_APP_DIR]   (default: ../frappe_card_studio, or $FRAPPE_APP_DIR)

A pair is compared by content with each line's leading whitespace stripped first: the two apps use different indentation styles
(this repo: spaces; the Frappe app: tabs) by house convention, which is not drift, but a real wording or logic difference still is.
Exits 1 (and prints every mismatch) if a "synced" pair no longer matches, or a declared source/engine file is missing."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _normalized(path):
    return "\n".join(line.lstrip() for line in path.read_text().splitlines())


def check(frappe_dir):
    engine = Path(frappe_dir) / "card_studio" / "card_studio" / "engine"
    manifest_path = engine / "SYNC.json"
    if not manifest_path.is_file():
        sys.exit(f"no {manifest_path} - pass the frappe_card_studio checkout's root, or run this from a checkout that has it")
    manifest = json.loads(manifest_path.read_text())
    drift, missing, info = [], [], []
    for name, entry in manifest.items():
        status, source = entry.get("status"), entry.get("source")
        if status != "synced":
            info.append(f"{name}: {status}" + (f" - {entry['note']}" if entry.get("note") else ""))
            continue
        src, dst = ROOT / source, engine / name
        if not src.is_file():
            missing.append(f"{source}: not found in {ROOT}"); continue
        if not dst.is_file():
            missing.append(f"engine/{name}: not found in {engine}"); continue
        if _normalized(src) != _normalized(dst):
            drift.append(f"{source} <-> engine/{name}: content differs (beyond indentation style)")
    return drift, missing, info


def main():
    frappe_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FRAPPE_APP_DIR", str(ROOT.parent / "frappe_card_studio"))
    drift, missing, info = check(frappe_dir)
    if info:
        print("not tracked for drift (see SYNC.json):")
        for line in info:
            print(" ", line)
    if missing:
        print("MISSING:")
        for line in missing:
            print(" ", line)
    if drift:
        print("DRIFT:")
        for line in drift:
            print(" ", line)
        sys.exit(1)
    if not missing:
        print(f"sync_engine: every \"synced\" pair in {frappe_dir}/card_studio/card_studio/engine/SYNC.json matches")


if __name__ == "__main__":
    main()
