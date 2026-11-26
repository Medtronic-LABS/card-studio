"""The default Bengali vocabulary: stable hash, kind masks cover both digit sets for number/date/bp, round-trip encode/decode, and
`Vocabulary.from_classes` reproduces the exact class order of the shipped pack's model_meta.json (so old weights still fit)."""
import json, sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import htr.text as T
import htr.vocab as V


class Default(unittest.TestCase):
    def setUp(self): self.v = V.Vocabulary.default()

    def test_blank_and_unk_are_first(self):
        self.assertEqual(self.v.classes[0], V.BLANK); self.assertEqual(self.v.classes[1], V.UNK)

    def test_sha_is_stable_and_changes_with_content(self):
        v2 = V.Vocabulary.default()
        self.assertEqual(self.v.sha256, v2.sha256)
        v3 = V.Vocabulary(self.v.classes + ["ঁঁ"], self.v.kinds)   # a two-character string: not a duplicate of any single class
        self.assertNotEqual(self.v.sha256, v3.sha256)

    def test_number_date_bp_masks_hold_both_digit_sets(self):
        for kind in ("number", "date", "bp"):
            chars = self.v.chars(kind)
            for d in "0123456789" + T.BN_DIGITS: self.assertIn(d, chars, kind)

    def test_text_kind_allows_everything_but_unk(self):
        allowed = set(self.v.allowed("text"))
        self.assertNotIn(self.v.unk, allowed)
        self.assertEqual(len(allowed), len(self.v) - 1)

    def test_roundtrip_bengali_line(self):
        s = T.normalize("বাংলাদেশ")
        self.assertEqual(self.v.decode(self.v.encode(s)), s)

    def test_unknown_character_maps_to_unk_unless_strict(self):
        ids = self.v.encode("☃", strict=False)
        self.assertEqual(ids, [self.v.unk])
        with self.assertRaises(ValueError): self.v.encode("☃", strict=True)

    def test_save_and_load_roundtrip(self, tmp_path=Path("/tmp/htr_vocab_test.json")):
        self.v.save(tmp_path)
        loaded = V.Vocabulary.load(tmp_path)
        self.assertEqual(loaded.classes, self.v.classes)
        self.assertEqual(loaded.sha256, self.v.sha256)
        tmp_path.unlink()


class FromLegacyMeta(unittest.TestCase):
    def test_matches_shipped_model_meta(self):
        meta = json.loads((ROOT / "model_meta.json").read_text())
        v = V.Vocabulary.from_classes(meta["classes"])
        self.assertEqual(v.classes, meta["classes"])
        self.assertEqual(v.blank, meta["blank_idx"])
        self.assertEqual(len(v), len(meta["classes"]))


if __name__ == "__main__":
    unittest.main()
