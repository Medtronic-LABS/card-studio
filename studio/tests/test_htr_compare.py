"""htr.eval.compare: aggregate/build for comparison.json, and paired (writer-bootstrap paired comparison, the line-dataset
counterpart of studio/analysis.paired)."""
import sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import htr.eval.compare as C


def _rows(n, writer, doc_type="card_cell", wrong=0):
    """n lines from one writer; the first `wrong` are misread."""
    out = []
    for i in range(n):
        ref = f"line{i}"
        hyp = ref if i >= wrong else "WRONG"
        out.append({"line_id": f"{writer}:{i}", "ref": ref, "hyp": hyp, "document_type": doc_type, "writer_id": writer})
    return out


class Aggregate(unittest.TestCase):
    def test_all_correct_is_zero_cer_and_full_exact(self):
        rows = _rows(10, "w1")
        agg = C.aggregate(rows)
        self.assertEqual(agg["exact"], 1.0); self.assertEqual(agg["cer_grapheme"], 0.0)

    def test_build_writes_comparison_json(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "comparison.json"
            result = C.build([{"name": "run-a", "precision": "fp32", "rows": _rows(10, "w1"), "n_params": 100, "size_bytes": 1000, "latency": None}], out_path=out)
            self.assertTrue(out.exists())
            self.assertEqual(result["rows"][0]["name"], "run-a")
            self.assertIn("by_document_type", result["rows"][0])


class Paired(unittest.TestCase):
    def test_needs_at_least_ten_shared_lines(self):
        with self.assertRaises(ValueError): C.paired(_rows(3, "w1"), _rows(3, "w1"))

    def test_candidate_better_gives_positive_diff(self):
        # paired(a, b): diff = acc_b - acc_a. a=shipped (50% exact), b=candidate (90% exact, same line_ids) -> diff > 0.
        shipped = _rows(20, "w1", wrong=10)
        candidate = _rows(20, "w1", wrong=2)
        r = C.paired(shipped, candidate, boot=200, seed=0)
        self.assertEqual(r["common"], 20)
        self.assertAlmostEqual(r["diff"], 0.4, places=5)
        self.assertGreater(r["only_b"], r["only_a"])   # candidate (b) right where shipped (a) is wrong, more often than the reverse

    def test_identical_runs_have_zero_diff_and_full_overlap(self):
        rows = _rows(15, "w1")
        r = C.paired(rows, rows, boot=100, seed=0)
        self.assertEqual(r["diff"], 0.0); self.assertEqual(r["only_a"], 0); self.assertEqual(r["only_b"], 0)
        self.assertEqual(r["both"], 15)


if __name__ == "__main__":
    unittest.main()
