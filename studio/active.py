"""Which unlabeled cell teaches the reader most. A per-cell score combines the signals the plan names - how unsure the AI's own
suggestion is, how hard the Bengali text in it looks (conjuncts cost more to get right), whether its writer or document type is
one the reader has not trained on yet, and whether two model versions disagree about it - then `queue_order` round-robins across
fields so one hard field never buries every other field at the back of the queue. Pure Python; no dependency on how the caller
actually got these signals (`compute_signals` is one way to gather them for Studio's current card-cell labeling; a line-dataset
caller can build the same dict differently)."""
WEIGHTS = {"low_conf": 1.0, "hard_graphemes": 0.8, "unseen_writer": 0.7, "new_doc_type": 0.5, "version_disagree": 0.9}


def score(signals):
    """signals: {"low_conf", "hard_graphemes", "unseen_writer", "new_doc_type", "version_disagree"}, each in [0, 1] (a plain 0/1
    flag is fine for the latter three). Missing keys count as 0. -> the weighted sum."""
    return sum(WEIGHTS[k] * signals.get(k, 0.0) for k in WEIGHTS)


def compute_signals(suggestion, writer_id, document_type, seen_writers, seen_document_types, other_version_text=None):
    """Build one cell's signals from what Studio already tracks. suggestion: {"text", "confidence"} or None (nothing suggested
    yet - scored as maximally unsure and not graded for grapheme hardness). seen_writers/seen_document_types: sets of writer/doc
    ids the current training data already covers. other_version_text: another model version's read of the same cell, when one
    exists (None: version_disagree is always 0 - this signal needs a second stored suggestion, which is not wired in everywhere
    yet)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import htr.text as T
    if suggestion is None:
        low_conf, hard = 1.0, 0.0
    else:
        low_conf = 1.0 - max(0.0, min(1.0, suggestion.get("confidence", 0.0)))
        clusters = T.graphemes(T.normalize(suggestion.get("text", "")))
        hard = (sum(1 for g in clusters if len(T.conjuncts(g)) or len(g) > 1) / len(clusters)) if clusters else 0.0
    return {
        "low_conf": low_conf, "hard_graphemes": hard,
        "unseen_writer": 0.0 if writer_id is None or writer_id in seen_writers else 1.0,
        "new_doc_type": 0.0 if document_type is None or document_type in seen_document_types else 1.0,
        "version_disagree": 1.0 if (suggestion and other_version_text is not None and other_version_text != suggestion.get("text")) else 0.0,
    }


def queue_order(cells, field_key, score_key="score"):
    """cells: [{..., field_key: value, score_key: float}, ...], already the candidates for the queue. -> the same cells reordered:
    round-robin across distinct field values (each field's own cells sorted by score, highest first, then one cell taken from
    each field in turn) so a field with many hard cells does not push every other field to the back."""
    by_field = {}
    for c in cells: by_field.setdefault(c[field_key], []).append(c)
    for f in by_field: by_field[f].sort(key=lambda c: -c[score_key])
    fields = list(by_field)
    out, i = [], 0
    while any(by_field[f] for f in fields):
        f = fields[i % len(fields)]
        if by_field[f]: out.append(by_field[f].pop(0))
        i += 1
    return out
