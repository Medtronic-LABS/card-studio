// Checks grammar_decode.dart against grammar_vectors.json (made by make_vectors.py from cardpack/grammar.py).
// From the repository root: dart run cardpack/reference/grammar_decode_test.dart
import 'dart:convert';
import 'dart:io';
import 'grammar_decode.dart';

void main() {
  final file = File('cardpack/reference/grammar_vectors.json').existsSync() ? File('cardpack/reference/grammar_vectors.json') : File('grammar_vectors.json');
  final doc = jsonDecode(file.readAsStringSync()) as Map<String, dynamic>;
  final tol = (doc['tolerance'] as num).toDouble();
  var bad = 0;
  for (final v in doc['vectors'] as List) {
    final logits = [for (final row in v['logits'] as List) [for (final x in row as List) (x as num).toDouble()]];
    final sw = Stopwatch()..start();
    final r = decode(logits, [for (final c in v['chars'] as List) c as String], v['blank'] as int, v['kind'] as String,
        range: v['range'] == null ? null : [for (final x in v['range'] as List) x as int], years: v['years'] == null ? null : [for (final x in v['years'] as List) x as int]);
    final ok = r.text == v['text'] && (r.margin - (v['margin'] as num).toDouble()).abs() <= tol;
    if (!ok) bad++;
    print('${ok ? "ok  " : "FAIL"} ${(v['kind'] as String).padRight(7)} ${(v['note'] as String).padRight(42)} got "${r.text}" margin ${r.margin.toStringAsFixed(6)}'
        ' want "${v['text']}" ${(v['margin'] as num).toStringAsFixed(6)}  ${sw.elapsedMilliseconds} ms');
  }
  print(bad == 0 ? 'all vectors agree' : '$bad vector(s) disagree');
  exit(bad == 0 ? 0 : 1);
}
