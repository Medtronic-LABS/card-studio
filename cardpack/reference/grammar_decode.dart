// Grammar decoding for a card pack field with `decode: "grammar"` (cardpack/CARDPACK.md, "Decoding: greedy or grammar").
// A port of cardpack/grammar.py, no packages. The reference vectors are cardpack/reference/grammar_vectors.json;
// run `dart run cardpack/reference/grammar_decode_test.dart` from the repository root to check this file against them.
//
// The reader gives, for each of T steps, a score for every class the field may emit plus the blank. Greedy decoding takes the best class per step.
// This instead scores every string the field can hold by its CTC likelihood and returns the likeliest, using the prefix recursion
// of CTC prefix beam search down a sorted list of the candidate strings, so strings that share a start share the work.

import 'dart:math' as math;
import 'dart:typed_data';

class GrammarRead {
  /// The chosen string, or '' when no candidate could be written with the classes given.
  final String text;

  /// log-likelihood of [text] minus the log-probability of the single best path. Above 0 is possible (a string sums over alignments).
  /// The confidence a pack reports is min(1, exp(margin)).
  final double margin;
  const GrammarRead(this.text, this.margin);
  double get confidence => margin.isFinite ? math.min(1.0, math.exp(margin)) : 0.0;
}

final Map<String, List<String>> _cache = {};

/// Every string a field of this [kind] ('date', 'bp' or 'number') can hold, sorted. [range] is a number's [low, high],
/// [years] a date's first and last two-digit year (default 20 to 27).
List<String> candidates(String kind, {List<int>? range, List<int>? years}) {
  final y = years ?? const [20, 27];
  final key = '$kind|${range?.join(",")}|${kind == "date" ? y.join(",") : ""}';
  final hit = _cache[key];
  if (hit != null) return hit;
  final out = <String>[];
  if (kind == 'date') {
    final days = [for (var d = 1; d <= 31; d++) '$d', for (var d = 1; d <= 9; d++) '0$d'];
    final mons = [for (var m = 1; m <= 12; m++) '$m', for (var m = 1; m <= 9; m++) '0$m'];
    for (final d in days) {
      for (final m in mons) {
        for (var yy = y[0]; yy <= y[1]; yy++) {
          for (final s in const ['/', '.']) {
            out.add('$d$s$m$s$yy');
          }
        }
      }
    }
  } else if (kind == 'bp') {
    for (var s = 70; s <= 220; s++) {
      for (var d = 40; d <= 130; d++) {
        out.add('$s/$d');
      }
    }
  } else if (kind == 'number') {
    if (range == null) throw ArgumentError('a number grammar needs a range');
    for (var v = range[0]; v <= range[1]; v++) {
      out.add('$v');
    }
    for (var v = range[0]; v < range[1]; v++) {
      for (var t = 0; t < 10; t++) {
        out.add('$v.$t');
      }
    }
  } else {
    throw ArgumentError('no grammar for kind $kind');
  }
  out.sort();
  return _cache[key] = out;
}

/// [logits] is T rows of K raw scores over [chars] (class strings in column order); [blank] is the column of the blank.
GrammarRead decode(List<List<double>> logits, List<String> chars, int blank, String kind, {List<int>? range, List<int>? years}) {
  final t = logits.length, k = chars.length;
  // per-step log-softmax over the classes given, and the log-probability of the best single path
  final p = List.generate(k, (_) => Float64List(t));
  var bestPath = 0.0;
  for (var s = 0; s < t; s++) {
    var mx = -double.infinity;
    for (var c = 0; c < k; c++) {
      if (logits[s][c] > mx) mx = logits[s][c];
    }
    var sum = 0.0;
    for (var c = 0; c < k; c++) {
      sum += math.exp(logits[s][c] - mx);
    }
    final lse = mx + math.log(sum);
    var mlp = -double.infinity;
    for (var c = 0; c < k; c++) {
      final lp = logits[s][c] - lse;
      if (lp > mlp) mlp = lp;
      p[c][s] = math.exp(lp);
    }
    bestPath += mlp;
  }
  final col = <String, int>{for (var c = 0; c < k; c++) if (c != blank) chars[c]: c};
  final pb = p[blank];
  final b0 = Float64List(t);
  var run = 1.0;
  for (var s = 0; s < t; s++) {
    run *= pb[s];
    b0[s] = run;
  }
  // stack[i] holds the prefix of length i: its characters, the blank-ending and character-ending probabilities by step
  final prefixes = <String>[''];
  final bs = <Float64List>[b0];
  final ns = <Float64List>[Float64List(t)];
  var prev = '';
  String? best;
  var bestLik = 0.0;
  for (final cand in candidates(kind, range: range, years: years)) {
    if (cand.split('').any((ch) => !col.containsKey(ch))) continue;
    var l = 0;
    while (l < cand.length && l < prev.length && cand[l] == prev[l]) {
      l++;
    }
    prefixes.length = l + 1;
    bs.length = l + 1;
    ns.length = l + 1;
    for (var i = l; i < cand.length; i++) {
      final ch = cand[i];
      final pc = p[col[ch]!];
      final bl = bs.last, nl = ns.last;
      final last = i == 0 ? null : cand[i - 1];
      final isEmpty = i == 0;
      final n = Float64List(t), b = Float64List(t);
      for (var s = 0; s < t; s++) {
        final priorB = s > 0 ? bl[s - 1] : (isEmpty ? 1.0 : 0.0);
        final priorN = s > 0 ? nl[s - 1] : 0.0;
        final src = (s > 0 ? n[s - 1] : 0.0) + priorB + (last == ch ? 0.0 : priorN);
        n[s] = pc[s] * src;
        b[s] = pb[s] * ((s > 0 ? b[s - 1] : 0.0) + (s > 0 ? n[s - 1] : 0.0));
      }
      prefixes.add(prefixes.last + ch);
      bs.add(b);
      ns.add(n);
    }
    prev = cand;
    final lik = bs.last[t - 1] + ns.last[t - 1];
    if (lik > bestLik) {
      bestLik = lik;
      best = cand;
    }
  }
  if (best == null) return const GrammarRead('', double.negativeInfinity);
  return GrammarRead(best, math.log(bestLik) - bestPath);
}
