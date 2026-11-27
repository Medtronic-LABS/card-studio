"""One table for every card type: how much labeled data it has and whether it is ready to train on, so nobody has to switch cards to compare them.
Read-only: it calls the same functions the per-card views use (cards.describe, writer_status, holdout_status, datachecks.checklist) and writes nothing.
(studio/datasets/ without a .py is where autotrain.py keeps its label snapshots; that directory is local data and is gitignored.)
overview() -> {"cards": [{id, labels, pages, writers, split_unit, test_pages, pack_version, level, failing, blocker, error}], "totals": {...}}"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cards, datachecks

LEVELS = ("ok", "info", "warn", "bad")          # worst wins


def _row(c):
    row = {"id": c["id"], "labels": c["labels"]["cells"], "pages": c["labels"]["pages"], "writers": 0, "split_unit": None, "test_pages": 0, "pack_version": max(c["pack_versions"], default=None),
           "level": "ok", "failing": [], "blocker": c["blocker"], "error": None}
    try:
        w = cards.writer_status(c["id"]); row["writers"], row["split_unit"] = w["writers"], w["split_unit"]
        row["test_pages"] = cards.holdout_status(c["id"])["test_pages"]
        ch = datachecks.checklist(c["id"])["checks"]
        row["level"] = max((x["level"] for x in ch), key=LEVELS.index, default="ok")
        row["failing"] = [{"id": x["id"], "level": x["level"], "message": x["message"]} for x in ch if x["level"] in ("warn", "bad")]
    except (ValueError, OSError, KeyError) as e:
        row["error"] = f"{type(e).__name__}: {e}"; row["level"] = "bad"
    return row


def overview():
    rows = [_row(c) for c in cards.list_cards()]
    levels = {k: sum(1 for r in rows if r["level"] == k) for k in LEVELS}
    return {"cards": rows, "totals": {"cards": len(rows), "labels": sum(r["labels"] for r in rows), "pages": sum(r["pages"] for r in rows), "writers": sum(r["writers"] for r in rows), "levels": levels}}
