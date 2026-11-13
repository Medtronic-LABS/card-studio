"""Choosing a run's starting weights. Stand-in files only; no torch. Run: python -m unittest discover -s tests"""
import hashlib, json, sys, tempfile, unittest
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "cardpack"))
import pretrained as P


class Registry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); t = Path(self.tmp.name); self.old = (P.DIR, P.LEGACY); P.DIR = t / "pretrained"; P.LEGACY = t / "legacy.pt"; P.LEGACY.write_bytes(b"legacy weights")
        self.w = t / "w.pt"; self.w.write_bytes(b"some weights")

    def tearDown(self):
        P.DIR, P.LEGACY = self.old; self.tmp.cleanup()

    def test_with_nothing_registered_only_the_original_is_there_and_auto_means_it(self):
        self.assertEqual([r["name"] for r in P.list_all()], ["legacy"]); path, rec = P.resolve("auto"); self.assertEqual((path, rec["name"], rec["default"]), (P.LEGACY, "legacy", True))

    def test_adopting_registers_checks_and_becomes_the_default_once(self):
        rec = P.adopt("v9", self.w, "a note", {"images": 1000, "epochs": 2}); self.assertEqual((rec["images"], rec["note"], rec["classes_sha256"]), (1000, "a note", P.classes_sha()))
        self.assertEqual(rec["sha256"], hashlib.sha256(b"some weights").hexdigest()); self.assertEqual(P.default_name(), "v9")
        P.adopt("v10", self.w); self.assertEqual(P.default_name(), "v9")                       # the second one does not take the default
        self.assertEqual(P.resolve("auto")[0], P.DIR / "v9.pt"); self.assertEqual(P.resolve("v10")[1]["name"], "v10"); self.assertEqual(P.resolve("legacy")[0], P.LEGACY)
        self.assertEqual({r["name"]: r["default"] for r in P.list_all()}, {"legacy": False, "v9": True, "v10": False})
        P.set_default("legacy"); self.assertEqual(P.resolve("auto")[1]["name"], "legacy")

    def test_bad_names_and_duplicates(self):
        for bad in ("legacy", "../x", "a b", "x" * 33, ""):
            with self.assertRaises(ValueError, msg=bad): P.adopt(bad, self.w)
        P.adopt("v1", self.w)
        with self.assertRaises(ValueError): P.adopt("v1", self.w)
        for bad in ("nosuch", "../v1"):
            with self.assertRaises(ValueError, msg=bad): P.resolve(bad)

    def test_weights_that_changed_or_were_made_for_another_class_list_are_refused(self):
        P.adopt("v1", self.w); (P.DIR / "v1.pt").write_bytes(b"tampered")
        with self.assertRaises(ValueError) as e: P.resolve("v1")
        self.assertIn("do not match their record", str(e.exception))
        P.adopt("v2", self.w); r = json.loads((P.DIR / "v2.json").read_text()); r["classes_sha256"] = "0" * 16; (P.DIR / "v2.json").write_text(json.dumps(r))
        with self.assertRaises(ValueError) as e: P.resolve("v2")
        self.assertIn("different class list", str(e.exception))
        P.adopt("v3", self.w); (P.DIR / "v3.json").unlink()
        with self.assertRaises(ValueError) as e: P.resolve("v3")
        self.assertIn("no record", str(e.exception)); self.assertEqual([r["name"] for r in P.list_all() if r["ok"]], ["legacy"])

    def test_a_missing_original_is_reported_not_hidden(self):
        P.LEGACY.unlink()
        with self.assertRaises(ValueError): P.resolve("legacy")


if __name__ == "__main__":
    unittest.main()

class Networks(unittest.TestCase):
    """Weights belong to one network: another architecture starts from scratch, or from weights pretrained for it."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); t = Path(self.tmp.name); self.old = (P.DIR, P.LEGACY); P.DIR = t / "pretrained"; P.LEGACY = t / "legacy.pt"; P.LEGACY.write_bytes(b"legacy weights")
        self.w = t / "w.pt"; self.w.write_bytes(b"some weights")

    def tearDown(self):
        P.DIR, P.LEGACY = self.old; self.tmp.cleanup()

    def test_the_shipped_network_refuses_weights_made_for_another_and_the_other_way_round(self):
        P.adopt("small1", self.w, arch="crnn-small"); P.adopt("shipped1", self.w)
        self.assertEqual(P.resolve("shipped1")[1]["name"], "shipped1"); self.assertEqual(P.resolve("small1", "crnn-small")[1]["name"], "small1")
        with self.assertRaisesRegex(ValueError, "different|not the one"): P.resolve("small1")
        with self.assertRaisesRegex(ValueError, "not the one"): P.resolve("shipped1", "crnn-wide")
        with self.assertRaisesRegex(ValueError, "not the one"): P.resolve("legacy", "crnn-small")

    def test_another_network_with_no_weights_trains_from_scratch_and_auto_finds_its_own(self):
        path, rec = P.resolve("auto", "crnn-wide"); self.assertEqual((path, rec["name"]), (None, "none")); self.assertIn("from scratch", rec["note"])
        self.assertEqual(P.resolve("none")[0], None); self.assertEqual(P.resolve("none", "gru-crnn")[0], None)
        P.adopt("wide1", self.w, arch="crnn-wide"); path, rec = P.resolve("auto", "crnn-wide"); self.assertEqual((path, rec["name"]), (P.DIR / "wide1.pt", "wide1"))
        self.assertEqual(P.default_name(), "legacy"); self.assertEqual(P.resolve("auto")[1]["name"], "legacy")          # weights for another network never become the default of the shipped one

    def test_the_record_says_which_network_it_is_for(self):
        rec = P.adopt("g1", self.w, arch="gru-crnn"); self.assertEqual((rec["arch_name"], rec["arch_sha256"]), ("gru-crnn", P.arch_hash("gru-crnn"))); self.assertIn("gru-crnn", rec["architecture"])
        by = {r["name"]: r for r in P.list_all()}; self.assertEqual(by["g1"]["arch"], "gru-crnn"); self.assertEqual(by["legacy"]["arch"], "crnn")

