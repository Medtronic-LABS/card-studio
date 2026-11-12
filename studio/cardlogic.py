"""Card-level checks: what the card's own structure says about the dates that were read, after every cell has been read. Pure Python, no model code.

  same_day     On an EPI card the vaccines in one dose column are usually given on the same day, so the same date is written in each of those cells. When at
               least two cells of a visit read the same real date and it strictly outnumbers every other reading, a cell that reads something else (or
               something that is not a date) gets that date as a SUGGESTION. The cell's own reading is never replaced: a person decides, because doses can
               really fall on different days.
  increasing   Visit dates on an ANC card go up from visit to visit. A date earlier than the one before it gets a note; nothing is suggested.

Both only ever add a note (and for same_day a suggestion). They do not change a read, and a blank cell is never touched. The rules a card uses are
named by its fields: checks = {"same_day": ["bcg", "penta1", ...], "increasing": ["visit_date"]}. Reads are {(field, visit): text}; a cell whose read is
not trusted yet (low confidence, repaired by a grammar) goes in `weak`: it can be corrected by the others, but it does not vote.

This file is kept identical in Card Studio (studio/cardlogic.py) and the Frappe app (card_ocr/engine/cardlogic.py).
"""
import collections
import datetime
import re

DATE = re.compile(r"(\d{1,2})[./-](\d{1,2})[./-](\d{4}|\d{2})")


def date_key(text):
    """(day, month, year) of a real calendar date written as 22/2/26, 22.02.2026 ...; None for anything else, including 31/4/26 and 29/2/25."""
    m = DATE.fullmatch((text or "").strip())
    if not m:
        return None
    d, mo, y = int(m[1]), int(m[2]), int(m[3])
    y += 2000 if len(m[3]) == 2 else 0
    try:
        datetime.date(y, mo, d)
    except ValueError:
        return None
    return d, mo, y


def parse_fields(value):
    """'bcg, penta1' or ['bcg', 'penta1'] -> ['bcg', 'penta1']"""
    if isinstance(value, str):
        value = value.split(",")
    return [f.strip() for f in (value or []) if f and f.strip()]


def same_day(reads, fields, weak=()):
    out = []
    fields = parse_fields(fields)
    for visit in sorted({v for (f, v) in reads if f in fields}):
        cells = [(f, reads[f, visit]) for f in fields if (f, visit) in reads and (reads[f, visit] or "").strip()]
        votes = collections.Counter(date_key(t) for f, t in cells if (f, visit) not in weak and date_key(t))
        ranked = votes.most_common(2)
        if not ranked or ranked[0][1] < 2 or (len(ranked) > 1 and ranked[1][1] == ranked[0][1]):
            continue
        winner, n = ranked[0]
        shown = collections.Counter(t.strip() for f, t in cells if date_key(t) == winner).most_common(1)[0][0]
        for f, t in cells:
            if date_key(t) != winner:
                out.append({"field": f, "visit": visit, "rule": "same_day", "suggestion": shown,
                            "note": f"{n} other doses of visit {visit} read {shown}; this one reads {t.strip()!r}. Doses can fall on different days: check the crop"})
    return out


def increasing(reads, fields):
    out = []
    for f in parse_fields(fields):
        prev = None
        for visit in sorted(v for (g, v) in reads if g == f):
            key = date_key(reads[f, visit])
            if not key:
                continue
            stamp = (key[2], key[1], key[0])
            if prev and stamp < prev[0]:
                out.append({"field": f, "visit": visit, "rule": "increasing", "suggestion": None,
                            "note": f"This date is earlier than the one in visit {prev[1]}: check both"})
            prev = (stamp, visit)
    return out


def check(reads, checks, weak=()):
    """-> [{field, visit, rule, suggestion, note}] for every cell the card's rules doubt."""
    checks = checks or {}
    return same_day(reads, checks.get("same_day"), weak) + increasing(reads, checks.get("increasing"))
