"""Reference implementation of `decode: grammar` (CARDPACK.md, "Decoding: greedy or grammar"). numpy only, no model code, no torch.

The reader outputs, for each of T steps, a score for every class the field may emit and a blank. Greedy decoding takes the best class per step. Grammar
decoding instead asks, for every string the field can hold, how likely the reader's output makes that exact string, and returns the likeliest.

The likelihood of a string is the CTC one: the probability that the per-step choices, after merging repeats and dropping blanks, spell the string. It is
computed by the prefix recursion used in CTC prefix beam search, run down a trie of the candidate strings so that strings sharing a start share the work:
for a prefix l, b_l[t] is the probability of having emitted exactly l by step t with the last step a blank, and n_l[t] the same with the last step a
character. Extending l by character c, with p_c[t] and p_blank[t] the step-t probabilities:

    n_new[t] = p_c[t] * ( n_new[t-1] + b_l[t-1] + (n_l[t-1] if the last character of l is not c else 0) )
    b_new[t] = p_blank[t] * ( b_new[t-1] + n_new[t-1] )
    the empty prefix has b[t] = p_blank[0] * ... * p_blank[t] and n[t] = 0, with b[-1] = 1

and the likelihood of the finished string is b[T-1] + n[T-1]. The confidence the pack reports is min(1, exp(margin)), margin = log(winner) minus the log
probability of the single best path (the sum over steps of the best class's log-probability).

Ports (Dart: cardpack/reference/grammar_decode.dart) must agree with this module on the vectors in cardpack/reference/grammar_vectors.json, which are made
by `python cardpack/reference/make_vectors.py`.
"""
import math

import numpy as np

DAYS_IN_MONTH = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)      # February at 29: the year is read on another line, so a leap day is checked later by date_ok
YEARS = (20, 27)          # the default two-digit years a date may carry (a pack field's date.years overrides)
_cache = {}


def candidates(kind, rng=None, years=None):
    """Every string a field of this kind can hold, sorted (sorting is what lets the trie walk share prefixes). rng: a number's [low, high]; years: [first, last]."""
    years = tuple(years) if years else YEARS; key = (kind, tuple(rng) if rng else None, years if kind == "date" else None)
    if key in _cache: return _cache[key]
    if kind == "date":
        days = [str(d) for d in range(1, 32)] + [f"{d:02d}" for d in range(1, 10)]; mons = [str(m) for m in range(1, 13)] + [f"{m:02d}" for m in range(1, 10)]
        out = [f"{d}{s}{m}{s}{y}" for d in days for m in mons for y in range(years[0], years[1] + 1) for s in "/."]
    elif kind == "daymonth":       # the top line of a date written on two lines (day.month over the year): only days that month has
        out = []
        for m, nd in enumerate(DAYS_IN_MONTH, 1):
            for d in range(1, nd + 1):
                out += [f"{dd}{sep}{mm}" for dd in {str(d), f"{d:02d}"} for mm in {str(m), f"{m:02d}"} for sep in "/."]
    elif kind == "bp":
        out = [f"{s}/{d}" for s in range(70, 221) for d in range(40, 131)]
    elif kind == "number":
        lo, hi = int(rng[0]), int(rng[1]); out = [str(v) for v in range(lo, hi + 1)] + [f"{v}.{t}" for v in range(lo, hi) for t in range(10)]
    else: raise ValueError(f"no grammar for kind {kind!r}")
    _cache[key] = sorted(out); return _cache[key]


def log_softmax(logits):
    """Per-step log-softmax over the columns given (the caller passes only the classes the field may emit, plus the blank)."""
    z = np.asarray(logits, np.float64); z = z - z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))


def scores(logp, chars, blank, cands):
    """-> {candidate: log-likelihood} for every candidate that can be written with `chars` (candidates with another character are skipped).
    logp [T, K] log-probabilities, chars the K class strings in column order, blank the column of the blank."""
    p = np.exp(np.asarray(logp, np.float64)); T = p.shape[0]; col = {c: i for i, c in enumerate(chars) if i != blank}
    pb = p[:, blank].tolist(); pc = {c: p[:, i].tolist() for c, i in col.items()}
    b0 = []; run = 1.0
    for t in range(T): run *= pb[t]; b0.append(run)
    stack = [("", b0, [0.0] * T, True)]          # (prefix, b, n, is_empty) for the prefix being extended
    out = {}; prev = ""
    for cand in cands:
        if any(ch not in col for ch in cand): continue
        k = 0
        while k < len(cand) and k < len(prev) and cand[k] == prev[k]: k += 1
        del stack[k + 1:]
        for ch in cand[k:]:
            par = stack[-1]; last = par[0][-1] if par[0] else None; b_l, n_l = par[1], par[2]; p_c = pc[ch]
            n = [0.0] * T; b = [0.0] * T
            for t in range(T):
                prior_b = b_l[t - 1] if t > 0 else (1.0 if par[3] else 0.0)
                prior_n = n_l[t - 1] if t > 0 else 0.0
                src = (n[t - 1] if t > 0 else 0.0) + prior_b + (0.0 if last == ch else prior_n)
                n[t] = p_c[t] * src
                b[t] = pb[t] * ((b[t - 1] if t > 0 else 0.0) + (n[t - 1] if t > 0 else 0.0))
            stack.append((par[0] + ch, b, n, False))
        prev = cand
        top = stack[-1]; lik = top[1][T - 1] + top[2][T - 1]
        if lik > 0: out[cand] = math.log(lik)
    return out


def posterior(sc, winner):
    """The winner's share of the probability of every candidate that can be written with the field's characters: exp(L_winner) / sum over candidates of exp(L_c). {candidate: log-likelihood} in, a number in (0, 1] out."""
    m = max(sc.values()); return math.exp(sc[winner] - m) / sum(math.exp(v - m) for v in sc.values())


def decode_full(logits, chars, blank, kind, rng=None, years=None):
    """-> (best string, margin, posterior), or ("", -inf, 0.0) when no candidate has any likelihood. Same winner and margin as decode(); posterior is what a pack's field reports as its confidence
    when it says `grammar_confidence: posterior` (see CARDPACK.md): unlike min(1, exp(margin)), which is 1 for nearly every read, it falls when other strings are nearly as likely."""
    logp = log_softmax(logits); sc = scores(logp, chars, blank, candidates(kind, rng, years))
    if not sc: return "", float("-inf"), 0.0
    best = max(sc, key=sc.get); return best, float(sc[best] - logp.max(1).sum()), float(posterior(sc, best))


def confidence(margin, post, kind="margin"):
    """The confidence a grammar read reports: min(1, exp(margin)) for kind "margin" (packs before the change), the posterior for "posterior"."""
    if kind == "posterior": return float(post)
    return min(1.0, math.exp(margin)) if math.isfinite(margin) else 0.0


def decode(logits, chars, blank, kind, rng=None, years=None):
    """-> (best string, margin) or ("", -inf) when no candidate has any likelihood. logits [T, K] raw scores over `chars` (one of them the blank, at
    column `blank`); margin = log-likelihood of the winner minus the log-probability of the best single path."""
    logp = log_softmax(logits); sc = scores(logp, chars, blank, candidates(kind, rng, years))
    if not sc: return "", float("-inf")
    best = max(sc, key=sc.get); return best, float(sc[best] - logp.max(1).sum())


def date_ok(top, year):
    """True when `top` ("22.2" or "22/02") and `year` ("26" or "2026") together name a real calendar day. A two-digit year is 20xx. Catches what each line
    passes alone: a 29 February in a year that has none, or a year line that did not read as a year."""
    import datetime, re
    m = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})", (top or "").strip()); y = (year or "").strip()
    if not m or not re.fullmatch(r"\d{2}|\d{4}", y): return False
    try: datetime.date(int(y) + (2000 if len(y) == 2 else 0), int(m[2]), int(m[1]))
    except ValueError: return False
    return True
