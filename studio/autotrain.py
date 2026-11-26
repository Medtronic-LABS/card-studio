"""When has a card accumulated enough newly-approved labels to be worth training again? `should_trigger` answers that by comparing
today's approved labels against the last snapshot; `snapshot` records what a run would train on (so the next check has a baseline,
and a run can later be reproduced). Triggers (any one): >=50 newly approved cells since the last snapshot, >=5 newly approved
pages, a writer new since the last snapshot contributing >=3 of those new pages, or "manual" (always fires).

This module decides WHEN and records WHAT (studio/datasets/<card>/v<N>.json) - it does not itself start a training run (an
auto-started background training process is a bigger decision than this module should make unsupervised: a caller - a cron job,
a Studio server route, the Frappe bridge - calls `check_and_snapshot`, and if it returns a snapshot, starts `trainer.py`/`htr.train`
itself, logging that it was autotrain-triggered)."""
import json, time
from pathlib import Path

import cards

HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"
NEW_CELLS_TRIGGER = 50
NEW_PAGES_TRIGGER = 5
NEW_WRITER_PAGES_TRIGGER = 3


def _dir(card):
    return DATASETS / card


def snapshot_versions(card):
    d = _dir(card)
    return sorted(int(p.stem[1:]) for p in d.glob("v*.json") if p.stem[1:].isdigit()) if d.is_dir() else []


def latest_snapshot(card):
    vs = snapshot_versions(card)
    return json.loads((_dir(card) / f"v{vs[-1]}.json").read_text()) if vs else None


def snapshot(card, reason="manual"):
    """Record today's approved labels as a new dataset version. -> the record written."""
    rows, _ = cards.labels(card, min_status="approved")
    pages = sorted({r["page"] for r in rows})
    writers = cards.writers(card)
    writer_pages = {}
    for p in pages:
        w = writers.get(p)
        if w: writer_pages[w] = writer_pages.get(w, 0) + 1
    v = (snapshot_versions(card) or [0])[-1] + 1
    rec = {"version": v, "card": card, "created": time.strftime("%Y-%m-%d %H:%M"), "reason": reason,
           "n_cells": len(rows), "n_pages": len(pages), "pages": pages, "writer_pages": writer_pages}
    d = _dir(card); d.mkdir(parents=True, exist_ok=True)
    (d / f"v{v}.json").write_text(json.dumps(rec, indent=1))
    return rec


def should_trigger(card):
    """-> (should: bool, reason: str or None)."""
    rows, _ = cards.labels(card, min_status="approved")
    pages = {r["page"] for r in rows}
    last = latest_snapshot(card)
    if last is None:
        if len(rows) >= NEW_CELLS_TRIGGER: return True, f"no snapshot yet and {len(rows)} approved cells already (>= {NEW_CELLS_TRIGGER})"
        if len(pages) >= NEW_PAGES_TRIGGER: return True, f"no snapshot yet and {len(pages)} approved pages already (>= {NEW_PAGES_TRIGGER})"
        return False, None
    new_cells = len(rows) - last["n_cells"]
    if new_cells >= NEW_CELLS_TRIGGER: return True, f"{new_cells} new approved cells since v{last['version']}"
    new_pages = pages - set(last["pages"])
    if len(new_pages) >= NEW_PAGES_TRIGGER: return True, f"{len(new_pages)} new approved pages since v{last['version']}"
    writers = cards.writers(card); known_writers = set(last.get("writer_pages", {}))
    new_writer_pages = {}
    for p in new_pages:
        w = writers.get(p)
        if w and w not in known_writers: new_writer_pages[w] = new_writer_pages.get(w, 0) + 1
    for w, n in new_writer_pages.items():
        if n >= NEW_WRITER_PAGES_TRIGGER: return True, f"new writer {w!r} contributed {n} new pages (>= {NEW_WRITER_PAGES_TRIGGER})"
    return False, None


def check_and_snapshot(card, manual=False):
    """-> the new snapshot record when a trigger fired (or `manual`), else None. Does not start training; see the module
    docstring for why that is the caller's job."""
    if manual: return snapshot(card, reason="manual")
    should, reason = should_trigger(card)
    return snapshot(card, reason=reason) if should else None
