"""The "before you train" checklist of a card: plain-language checks of the labels, writers, test set and tuning history. Pure Python (no torch).
Rules are identical in the Frappe app; keep the wording close.   checklist(card) -> {"checks": [{id, level (ok|info|warn|bad), message, hint}], "ready": bool}
`ready` is true when no check is "bad" (a warning is advice, not a stop)."""
import csv, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cards, seedexp

MIN_PAGES, GOOD_PAGES, MIN_WRITERS, MIN_FIELD_CELLS, TRIED_WARN = 10, 30, 5, 20, 4


def _how(card):
    """{(page, row, col): how the label was made} from labels.csv ('ai-read' = an AI suggestion accepted without typing or correcting). The legacy card has no such column."""
    if card == cards.LEGACY: return {}
    p = cards.CARDS / card / "labels.csv"
    if not p.exists(): return {}
    with open(p, newline="") as fh:
        return {(r["page"], r["row"], r["col"]): (r.get("how") or "") for r in csv.DictReader(fh)}


def _c(id, level, message, hint=""):
    return {"id": id, "level": level, "message": message, "hint": hint}


def checklist(card):
    rows, _ = cards.labels(card)
    how = _how(card)
    ai = lambda r: how.get((r["page"], r["row"], r["col"])) == "ai-read"      # noqa: E731
    out = []
    pages = {r["page"] for r in rows if not ai(r)}
    n = len(pages)
    if n < MIN_PAGES: out.append(_c("pages", "bad", f"{n} person-checked labeled pages: training needs at least {MIN_PAGES} labeled pages", "Label more pages (type or correct each cell) before you train."))
    elif n < GOOD_PAGES: out.append(_c("pages", "warn", f"{n} person-checked labeled pages: it will train, but {GOOD_PAGES} or more give a steadier result", "More labeled pages narrow the held-out intervals."))
    else: out.append(_c("pages", "ok", f"{n} person-checked labeled pages"))
    nai = sum(ai(r) for r in rows)
    if nai: out.append(_c("ai_reads", "warn", f"{nai} of {len(rows)} labels are AI reads accepted unchecked: the held-out numbers measure agreement with the suggesting reader, not accuracy",
                          "Open each AI-read cell in the labeler and confirm or correct it."))
    else: out.append(_c("ai_reads", "ok", "No labels are unchecked AI reads"))
    lab_pages = {r["page"] for r in rows}; w = cards.writers(card)
    coded = [p for p in lab_pages if p in w]; nw = len({w[p] for p in coded})
    if lab_pages and len(coded) == len(lab_pages) and nw >= MIN_WRITERS: out.append(_c("writers", "ok", f"Every labeled page has a writer code ({nw} writers): the split keeps each writer on one side"))
    else:
        out.append(_c("writers", "warn", "no writer codes: the split is by page, so accuracy likely overstates what a new writer gets" if not coded else
                      f"{len(coded)} of {len(lab_pages)} labeled pages have a writer code and there are {nw} writers (need every page coded and at least {MIN_WRITERS}): the split is by page, so accuracy likely overstates what a new writer gets",
                      "Assign writer codes (a code such as W07, never a name) in the Writers panel."))
    count = {}
    for r in rows: count[r["row"]] = count.get(r["row"], 0) + 1
    nontext = [f["key"] for f in cards.fields(card) if f["kind"] != "text"]
    thin = [k for k in nontext if 0 < count.get(k, 0) < MIN_FIELD_CELLS]
    none = [k for k in nontext if count.get(k, 0) == 0]
    if thin:
        shown = ", ".join(thin[:8]) + (f" and {len(thin) - 8} more" if len(thin) > 8 else "")
        out.append(_c("fields", "warn", f"{len(thin)} non-text field(s) have fewer than {MIN_FIELD_CELLS} labeled cells: {shown}", "A field with so few cells cannot be told apart from luck; label more of them."))
    if none:
        shown = ", ".join(none[:8]) + (f" and {len(none) - 8} more" if len(none) > 8 else "")
        out.append(_c("fields", "info", f"{len(none)} field(s) have no labels at all: {shown}", "They will be marked needs-review in the pack."))
    if not thin and not none: out.append(_c("fields", "ok", f"Every non-text field has at least {MIN_FIELD_CELLS} labeled cells"))
    if cards.holdout_status(card)["exists"]: out.append(_c("test_set", "ok", "A test set is reserved"))
    else: out.append(_c("test_set", "warn", "no test set reserved: reserve one before you start tuning", "Reserve it in Honest evaluation below, once, before comparing settings."))
    t = seedexp.tried(card)
    if t["settings_tried"] >= TRIED_WARN: out.append(_c("tried", "warn", t["message"] or f"{t['settings_tried']} different settings have been tried on this card", "Confirm the winner once on the test set instead of tuning further."))
    else: out.append(_c("tried", "ok", f"{t['settings_tried']} different settings tried so far"))
    return {"checks": out, "ready": all(c["level"] != "bad" for c in out)}
