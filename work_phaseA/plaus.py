"""A5: Python port of leapwell_ocr_demo/lib/bangla_digit/plausibility.dart (checkAncCell / checkDate)."""
import re
MIN_CONF, MIN_MARK_CONF = 0.7, 0.5
RANGES = {"weight_kg": (30, 150), "pulse": (40, 180), "fundal_height_cm": (5, 45), "hemoglobin": (4, 18), "glucose": (2, 30),
          "gestational_month": (1, 10), "height_cm": (120, 200), "temp_f": (94, 106), "fetal_movement": (0, 100)}
MARKS = {"edema", "urine_albumin", "urine_bilirubin", "usg", "tt_vaccine", "iron_tablet", "calcium_tablet",
         "discharge_pain", "perineal_stitch", "excess_bleeding"}
_NUM = re.compile(r"^[0-9]+(\.[0-9]+)?$"); _BP = re.compile(r"^([0-9]{2,3})/([0-9]{2,3})$")
_DATE = re.compile(r"^([0-9]{1,2})[./]([0-9]{1,2})[./]([0-9]{2,4})$")

def check_date(t):
    m = _DATE.match(t.replace(" ", ""))
    if not m: return "bad date format"
    d, mo = int(m[1]), int(m[2])
    if not 1 <= mo <= 12: return "month"
    if not 1 <= d <= 31: return "day"
    return None

def check_cell(field, text, conf):
    """-> reason string if flagged for review, else None. Blank text never flagged."""
    t = text.replace(" ", "")
    if not t: return None
    if conf < (MIN_MARK_CONF if field in MARKS else MIN_CONF): return "low confidence"
    if field in MARKS: return None if t in {"y", "x", "-"} else "unexpected mark"
    if field in RANGES:
        if not _NUM.match(t): return "not a clean number"
        lo, hi = RANGES[field]; v = float(t)
        return "out of range" if v < lo or v > hi else None
    if field == "bp":
        m = _BP.match(t)
        if not m: return "bp format"
        s, d = int(m[1]), int(m[2])
        if s < 70 or s > 220 or d < 40 or d > 130 or s <= d: return "bp range/order"
        return None
    if field == "visit_date": return check_date(t)
    return None
