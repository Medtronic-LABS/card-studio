"""Review status for labeled cells: pending (not yet looked at by a reviewer), approved (fit to train on), rejected (wrong, never
trains), flagged (surfaced for a human to look at, e.g. disagrees with the AI, fails a grammar check, or is a length outlier). Local
only, per card: studio/cards/<card>/review.csv (anc: work/review.csv), columns page,row,col,status,reviewer,reviewed,reason,label_hash.

A label is tied to its review by `label_hash` (sha256 of the normalized label, first 12 hex digits): editing a reviewed label after
the fact changes its hash, so the next `sync` call resets that cell back to pending - a stale approval can never silently keep
training on a label nobody reviewed.

`cards.labels(card, min_status=...)` filters on this file. A cell this file has no row for is `pending`."""
import csv, hashlib, statistics, threading
from pathlib import Path

import cards

COLS = ["page", "row", "col", "status", "reviewer", "reviewed", "reason", "label_hash"]
STATUSES = ("pending", "approved", "rejected", "flagged")
MIN_STATUS_ALLOWS = {"approved": {"approved"}, "pending": {"approved", "pending", "flagged"}}
MAD_K = 2.5            # a label's length more than this many MADs from its field's median is a length-outlier flag
LAZY_ACCEPT_CONF = 0.5  # an accepted suggestion the AI itself was this unsure about is flagged, not trusted silently
_lock = threading.Lock()


def _path(card):
    return (cards.WORK if card == cards.LEGACY else cards.CARDS / card) / "review.csv"


def label_hash(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()[:12]


def read(card):
    p = _path(card)
    if not p.exists(): return {}
    with open(p, newline="") as fh:
        return {(r["page"], r["row"], r["col"]): {"status": r["status"], "reviewer": r.get("reviewer", ""), "reviewed": r.get("reviewed", ""),
                                                    "reason": r.get("reason", ""), "label_hash": r.get("label_hash", "")}
                for r in csv.DictReader(fh)}


def write(card, rows):
    p = _path(card); p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, COLS); w.writeheader()
        w.writerows({"page": k[0], "row": k[1], "col": k[2], **v} for k, v in sorted(rows.items()))
    tmp.replace(p)


def _current_labels(card):
    """{(page,row,col): label} for every labeled cell (cards._labels: imports included, unlabeled rows left out)."""
    rows, _ = cards._labels(card)
    return {(r["page"], r["row"], r["col"]): r["label"] for r in rows}


def sync(card):
    """Bring review.csv in step with the current labels: a labeled cell missing from review.csv is added as pending; a reviewed cell
    whose label changed since (label_hash no longer matches) is reset to pending with reason "label changed since review"; a review
    row for a cell that no longer has a label is dropped. -> {added, reset, dropped}."""
    with _lock:
        current = _current_labels(card); existing = read(card)
        added = reset = dropped = 0
        out = {}
        for key, label in current.items():
            h = label_hash(label)
            if key not in existing:
                out[key] = {"status": "pending", "reviewer": "", "reviewed": "", "reason": "", "label_hash": h}; added += 1
            elif existing[key]["label_hash"] != h:
                out[key] = {"status": "pending", "reviewer": "", "reviewed": "", "reason": "label changed since review", "label_hash": h}; reset += 1
            else:
                out[key] = existing[key]
        dropped = len(existing) - len(set(existing) & set(current))
        write(card, out)
        return {"added": added, "reset": reset, "dropped": dropped}


def set_status(card, page, row, col, status, reviewer="", reason=""):
    """Record a reviewer's decision on one cell. -> the row written. Raises KeyError if the cell has no label to review."""
    if status not in STATUSES: raise ValueError(f"status must be one of {STATUSES}")
    labels = _current_labels(card)
    key = (page, row, col)
    if key not in labels: raise KeyError("no labeled cell there")
    with _lock:
        rows = read(card)
        rows[key] = {"status": status, "reviewer": reviewer, "reviewed": __import__("time").strftime("%Y-%m-%d %H:%M"), "reason": reason, "label_hash": label_hash(labels[key])}
        write(card, rows)
        return rows[key]


def migrate(card):
    """One-time: build review.csv for a card that has none yet, from what is already known.
    - `anc` (and any card with no per-label provenance): every current label -> approved (it shipped as ground truth already).
    - A card labeled through Studio (labeling.py's `how` column): typed/corrected -> approved (a person wrote or fixed it);
      accepted -> approved when the AI's own confidence was >= `suggest.SURE`-equivalent (0.8), else pending (a lazily confirmed
      suggestion should get a second look); labelstudio/imported -> pending.
    No-op (returns None) when review.csv already exists for this card - migrate once, review from then on with `set_status`."""
    if _path(card).exists(): return None
    current = _current_labels(card)
    if card == cards.LEGACY:
        out = {k: {"status": "approved", "reviewer": "", "reviewed": "", "reason": "migrated: legacy ground truth", "label_hash": label_hash(v)} for k, v in current.items()}
        write(card, out); return {"approved": len(out), "pending": 0}
    p = cards.CARDS / card / "labels.csv"
    how = {}
    if p.exists():
        with open(p, newline="") as fh:
            how = {(r["page"], r["row"], r["col"]): r.get("how", "") for r in csv.DictReader(fh)}
    out = {}
    for key, label in current.items():
        h = how.get(key, "")
        status = "approved" if h in ("typed", "corrected") else "pending"
        reason = f"migrated: how={h or 'unknown'}"
        out[key] = {"status": status, "reviewer": "", "reviewed": "", "reason": reason, "label_hash": label_hash(label)}
    write(card, out)
    return {"approved": sum(v["status"] == "approved" for v in out.values()), "pending": sum(v["status"] == "pending" for v in out.values())}


def _length_outliers(card, labeled):
    """{(page,row,col): "n.n sigma from the field's median length (k chars, field median m)"} for a label whose length is more than
    MAD_K median-absolute-deviations from its field's median length (a robust "this looks too short/long for this field" flag)."""
    by_field = {}
    for key, label in labeled.items(): by_field.setdefault(key[1], []).append((key, label))
    out = {}
    for field, items in by_field.items():
        lens = [len(lab) for _, lab in items]
        if len(lens) < 5: continue
        med = statistics.median(lens); mad = statistics.median(abs(l - med) for l in lens) or 1
        for key, lab in items:
            dev = abs(len(lab) - med) / (1.4826 * mad)
            if dev > MAD_K: out[key] = f"{dev:.1f} MADs from the field's median length ({len(lab)} chars, field median {med:.0f})"
    return out


def queue(card, show="needs_attention"):
    """Cells for the review screen. show: "needs_attention" (pending+flagged, default), "approved", "rejected", or "all".
    -> [{"page","field","col","label","status","reviewer","reviewed","reason"}, ...], sorted by field then page then col.
    Calls `sync` first, so a freshly labeled or re-labeled cell is always current. "reason" comes from the review row when the
    cell was flagged with one, else from `suggest_flags` for a cell this call is about to show anyway (an approved/rejected cell
    already has a human decision recorded, so it is never worth the extra work)."""
    if show not in ("needs_attention", "approved", "rejected", "all"): raise ValueError('show must be needs_attention, approved, rejected or all')
    sync(card)
    labeled = _current_labels(card)
    statuses = read(card)
    want = {"needs_attention": {"pending", "flagged"}, "approved": {"approved"}, "rejected": {"rejected"}, "all": set(STATUSES)}[show]
    flags = suggest_flags(card) if show in ("needs_attention", "all") else {}
    out = []
    for key, label in labeled.items():
        st = statuses.get(key, {"status": "pending", "reviewer": "", "reviewed": "", "reason": ""})
        if st["status"] not in want: continue
        page, field, col = key
        out.append({"page": page, "field": field, "col": col, "label": label, "status": st["status"], "reviewer": st["reviewer"],
                    "reviewed": st["reviewed"], "reason": st["reason"] or flags.get(key, "")})
    out.sort(key=lambda r: (r["field"], r["page"], r["col"]))
    return out


def suggest_flags(card):
    """Candidates for review attention, not auto-applied: -> {(page,row,col): reason}.
    - a label that disagrees with what the AI currently suggests for that cell
    - a label that fails `labeling.check`'s grammar check for its field kind
    - a lazily accepted suggestion (how=accepted) the AI itself was under LAZY_ACCEPT_CONF confident about
    - a length outlier within its field (see `_length_outliers`)"""
    import labeling, suggest
    if card == cards.LEGACY: return {}
    labeled = _current_labels(card)
    fl = {f["key"]: f for f in cards.fields(card)}
    sug = suggest.read(card)
    done_path = cards.CARDS / card / "labels.csv"
    done = {}
    if done_path.exists():
        with open(done_path, newline="") as fh:
            done = {(r["page"], r["row"], r["col"]): r for r in csv.DictReader(fh)}
    out = {}
    for key, label in labeled.items():
        page, field, col = key
        s = sug.get(key)
        if s and s["text"] and s["text"] != label: out[key] = f"disagrees with current AI suggestion {s['text']!r}"
        elif field in fl:
            err, warn = labeling.check(fl[field], label)
            if err: out[key] = f"grammar: {err}"
            elif warn: out[key] = f"grammar: {warn}"
        if key not in out and done.get(key, {}).get("how") == "accepted" and s and s["confidence"] < LAZY_ACCEPT_CONF:
            out[key] = f"accepted at low AI confidence ({s['confidence']:.2f})"
    out.update({k: v for k, v in _length_outliers(card, labeled).items() if k not in out})
    return out
