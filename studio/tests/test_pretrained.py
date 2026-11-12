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
