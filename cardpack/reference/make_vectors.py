"""Golden vectors for ports of grammar decoding (cardpack/grammar.py). `python cardpack/reference/make_vectors.py` writes grammar_vectors.json.

Each vector is made-up reader output (no patient data, no model): T steps of scores over a small set of classes, with the answer cardpack/grammar.py gives.
A port passes when, for every vector, it returns the same text and a margin within 1e-6 of the one stored. The logits are drawn from a fixed seed, with
some vectors shaped like a clear read of a known string, so the expected text is not an accident of noise."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import grammar

DIGITS = list("0123456789")
CASES = [("date", None, None, "20/9/26", "clear read"), ("date", None, None, "8.5.24", "clear read with dots"), ("date", None, [24, 27], "31/12/27", "clear read, years 24 to 27"),
         ("date", None, [24, 27], "20/9/31", "year outside the range: pulled inside it"), ("date", None, None, "20/9/2", "broken read: year cut short"), ("date", None, None, "", "noise only"),
         ("bp", None, None, "120/80", "clear read"), ("bp", None, None, "26", "broken read"), ("number", [30, 150], None, "57.5", "clear read, decimal"),
         ("number", [30, 150], None, "160", "out of range: pulled inside it"), ("number", [30, 150], None, "5", "below the range")]
CHARS = {"date": ["<blank>"] + DIGITS + list("/.-"), "bp": ["<blank>"] + DIGITS + ["/"], "number": ["<blank>"] + DIGITS + list(".-")}


def logits_for(rng, chars, text, T=48):
    lg = rng.normal(0, 0.8, (T, len(chars))); lg[:, 0] += 3.5; t = 2
    for ch in text:
        lg[t, chars.index(ch)] += 9.0; t += 3 if t < T - 4 else 0
    return np.round(lg, 4)


def main():
    rng = np.random.default_rng(20261112); out = []
    for kind, rngv, years, text, note in CASES:
        chars = CHARS[kind]; lg = logits_for(rng, chars, text); best, margin = grammar.decode(lg, chars, 0, kind, rngv, years)
        out.append({"note": note, "kind": kind, "range": rngv, "years": years, "chars": chars, "blank": 0, "logits": lg.tolist(), "text": best, "margin": margin})
    (HERE / "grammar_vectors.json").write_text(json.dumps({"about": "see make_vectors.py", "tolerance": 1e-6, "vectors": out}))
    for v in out: print(f"{v['kind']:7s} {v['note']:42s} -> {v['text']!r:12} margin {v['margin']:.3f}")


if __name__ == "__main__":
    main()
