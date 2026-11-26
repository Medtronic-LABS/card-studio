"""Reader quality: CER at codepoint and grapheme-cluster level (a grapheme is what a person counts as one Bengali letter - বাং is one
cluster, three codepoints), WER, exact match, and two content-restricted accuracies (numeral, conjunct) that matter more than the bulk
of CER, since a wrong digit in a blood-pressure reading or a wrong conjunct in a name is a worse error than a wrong vowel-sign. Pure
Python, ported from `work_phaseA/common.edit_distance`/`wilson`."""
import math

from .. import text as T


def edit_distance(a, b):
    """Levenshtein distance between two sequences of anything hashable (characters, or grapheme-cluster strings as tokens)."""
    a, b = list(a), list(b)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev if ca == cb else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[len(b)]


def wilson(k, n, z=1.96):
    """95% Wilson interval for k successes out of n trials; (nan, nan) when n is 0."""
    if n == 0: return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _rate(pairs, tokenize):
    """Micro-averaged edit-distance rate over a set of (ref, hyp) string pairs, after tokenizing each side with `tokenize`. -> (rate,
    total_edits, total_ref_tokens). A pair whose reference tokenizes to nothing is skipped (it contributes no denominator)."""
    edits = tokens = 0
    for ref, hyp in pairs:
        r, h = tokenize(ref), tokenize(hyp)
        if not r: continue
        edits += edit_distance(r, h); tokens += len(r)
    return (edits / tokens if tokens else float("nan")), edits, tokens


def cer_codepoint(pairs):
    """pairs: [(ref, hyp), ...] normalized strings. -> character error rate over NFC codepoints (the CTC's own unit)."""
    return _rate(pairs, list)[0]


def cer_grapheme(pairs):
    """-> character error rate over grapheme clusters (what a reader perceives as one letter)."""
    return _rate(pairs, T.graphemes)[0]


def wer(pairs):
    """-> word error rate (space-separated tokens)."""
    return _rate(pairs, T.words)[0]


def exact(pairs):
    """-> fraction of pairs where ref == hyp exactly."""
    pairs = list(pairs)
    if not pairs: return float("nan")
    return sum(ref == hyp for ref, hyp in pairs) / len(pairs)


def numeral_accuracy(pairs):
    """-> 1 - CER restricted to digit characters (০-৯ and 0-9) in the reference. nan when no pair has any digits."""
    digits_only = lambda s: [c for c in s if T.is_digit(c)]
    rate, _, tokens = _rate(pairs, digits_only)
    return float("nan") if tokens == 0 else 1 - rate


def conjunct_accuracy(pairs):
    """-> 1 - (edit distance over the reference's conjunct clusters only). nan when no pair has a conjunct."""
    rate, _, tokens = _rate(pairs, T.conjuncts)
    return float("nan") if tokens == 0 else 1 - rate


def breakdown(rows, key):
    """rows: [{"ref": str, "hyp": str, key: value, ...}, ...]. -> {group value: {n, exact, exact_ci, cer_grapheme, cer_codepoint, wer}},
    one entry per distinct value of `rows[i][key]`, each with a Wilson CI on its exact-match rate."""
    groups = {}
    for r in rows: groups.setdefault(r[key], []).append(r)
    out = {}
    for value, grp in groups.items():
        pairs = [(r["ref"], r["hyp"]) for r in grp]
        k = sum(ref == hyp for ref, hyp in pairs)
        out[value] = {
            "n": len(grp), "exact": k / len(grp), "exact_ci": wilson(k, len(grp)),
            "cer_grapheme": cer_grapheme(pairs), "cer_codepoint": cer_codepoint(pairs), "wer": wer(pairs),
        }
    return out
