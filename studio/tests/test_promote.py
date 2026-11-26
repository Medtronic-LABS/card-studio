"""studio/promote.py: the promotion gate - significant improvement (CI excludes 0, or a 15% relative CER drop), no regressed
document type, and mobile targets met when a device benchmark exists. Drawn/synthetic numbers only."""
import sys, unittest
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import promote as P


def comparison(ci_bootstrap, by_document_type=()):
    return {"ci_bootstrap": ci_bootstrap, "by_document_type": list(by_document_type)}


class Decide(unittest.TestCase):
    def test_ci_excluding_zero_promotes(self):
        c = comparison([0.02, 0.10])
        d = P.decide(c, candidate_agg={"cer_grapheme": 0.10}, shipped_agg={"cer_grapheme": 0.12})
        self.assertTrue(d["promoted"]); self.assertEqual(d["promoted_by"], "auto"); self.assertEqual(d["reasons"], [])

    def test_ci_including_zero_and_small_improvement_refuses(self):
        c = comparison([-0.01, 0.03])
        d = P.decide(c, candidate_agg={"cer_grapheme": 0.11}, shipped_agg={"cer_grapheme": 0.12})
        self.assertFalse(d["promoted"]); self.assertIsNone(d["promoted_by"]); self.assertTrue(d["reasons"])

    def test_large_relative_cer_drop_promotes_even_with_ci_including_zero(self):
        c = comparison([-0.01, 0.05])
        d = P.decide(c, candidate_agg={"cer_grapheme": 0.05}, shipped_agg={"cer_grapheme": 0.10})   # 50% relative drop
        self.assertTrue(d["promoted"])

    def test_regressed_document_type_blocks_even_with_good_ci(self):
        c = comparison([0.05, 0.10], [{"document_type": "card_cell", "only_a": 20, "only_b": 2, "both": 50, "neither": 5, "p": 0.001}])
        d = P.decide(c, candidate_agg={"cer_grapheme": 0.05}, shipped_agg={"cer_grapheme": 0.12})
        self.assertFalse(d["promoted"]); self.assertFalse(d["no_regression"])
        self.assertIn("card_cell", d["reasons"][0])

    def test_regression_must_be_significant_not_just_more_errors(self):
        # only_a > only_b but p is high (not significant) -> not a regression
        c = comparison([0.05, 0.10], [{"document_type": "card_cell", "only_a": 3, "only_b": 2, "both": 50, "neither": 5, "p": 0.8}])
        d = P.decide(c, candidate_agg={"cer_grapheme": 0.05}, shipped_agg={"cer_grapheme": 0.12})
        self.assertTrue(d["no_regression"])

    def test_device_bench_outside_targets_blocks_promotion(self):
        c = comparison([0.05, 0.10])
        d = P.decide(c, candidate_agg={"cer_grapheme": 0.05}, shipped_agg={"cer_grapheme": 0.12}, device_bench={"size_bytes": 30_000_000, "peak_rss_mb": 50, "p50_ms": 100})
        self.assertFalse(d["promoted"]); self.assertFalse(d["mobile_ok"])
        self.assertTrue(d["device_bench_checked"])

    def test_device_bench_within_targets_allows_promotion(self):
        c = comparison([0.05, 0.10])
        d = P.decide(c, candidate_agg={"cer_grapheme": 0.05}, shipped_agg={"cer_grapheme": 0.12}, device_bench={"size_bytes": 8_000_000, "peak_rss_mb": 60, "p50_ms": 150})
        self.assertTrue(d["promoted"]); self.assertTrue(d["mobile_ok"])

    def test_no_device_bench_does_not_block(self):
        c = comparison([0.05, 0.10])
        d = P.decide(c, candidate_agg={"cer_grapheme": 0.05}, shipped_agg={"cer_grapheme": 0.12}, device_bench=None)
        self.assertTrue(d["promoted"]); self.assertFalse(d["device_bench_checked"])


class Override(unittest.TestCase):
    def test_override_forces_promotion_and_records_reviewer(self):
        refused = {"promoted": False, "promoted_by": None, "reasons": ["not significant"]}
        d = P.override(refused, "reviewer-1")
        self.assertTrue(d["promoted"]); self.assertEqual(d["promoted_by"], "human"); self.assertEqual(d["overridden_by"], "reviewer-1")
        self.assertEqual(d["reasons"], ["not significant"])   # kept, not erased

    def test_override_needs_a_reviewer_name(self):
        with self.assertRaises(ValueError): P.override({"promoted": False}, "")


if __name__ == "__main__":
    unittest.main()
