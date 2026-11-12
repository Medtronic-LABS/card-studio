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

YEARS = (20, 27)          # the default two-digit years a date may carry (a pack field's date.years overrides)
_cache = {}


def candidates(kind, rng=None, years=None):
    """Every string a field of this kind can hold, sorted (sorting is what lets the trie walk share prefixes). rng: a number's [low, high]; years: [first, last]."""
    years = tuple(years) if years else YEARS; key = (kind, tuple(rng) if rng else None, years if kind == "date" else None)
    if key in _cache: return _cache[key]
    if kind == "date":
        days = [str(d) for d in range(1, 32)] + [f"{d:02d}" for d in range(1, 10)]; mons = [str(m) for m in range(1, 13)] + [f"{m:02d}" for m in range(1, 10)]
        out = [f"{d}{s}{m}{s}{y}" for d in days for m in mons for y in range(years[0], years[1] + 1) for s in "/."]
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


def decode(logits, chars, blank, kind, rng=None, years=None):
    """-> (best string, margin) or ("", -inf) when no candidate has any likelihood. logits [T, K] raw scores over `chars` (one of them the blank, at
    column `blank`); margin = log-likelihood of the winner minus the log-probability of the best single path."""
    logp = log_softmax(logits); sc = scores(logp, chars, blank, candidates(kind, rng, years))
    if not sc: return "", float("-inf")
    best = max(sc, key=sc.get); return best, float(sc[best] - logp.max(1).sum())
