"""The registry of pretrained models from outside: import checks, the record, scoring a card, and the licence gate. No network: models are tiny ONNX files made in the test."""
import json, sys, tempfile, unittest
from pathlib import Path
from unittest import mock

import cv2, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import external, extreader as X
from test_extreader import tiny_onnx

CHARS = list("0123456789/.")


@unittest.skipUnless(all(__import__("importlib").util.find_spec(m) for m in ("onnx", "onnxruntime")), "needs the onnx and onnxruntime packages to build and run a test model")
class Registry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        p = mock.patch.object(external, "DIR", Path(self.tmp.name) / "external"); p.start(); self.addCleanup(p.stop); external._readers.clear()
        s = X.make_spec("paddle-rec", CHARS); self.onnx = Path(self.tmp.name) / "m.onnx"; self.onnx.write_bytes(tiny_onnx(s["classes"], "22/3/26"))
        self.chars = json.dumps(CHARS)

    def add(self, name="tiny", **kw):
        return external.add(name, self.onnx, "paddle-rec", self.chars, source="https://example.test/m", licence="apache-2.0", commercial_use=True, note="test", **kw)

    def test_add_list_use_remove(self):
        rec = self.add(); self.assertEqual(rec["spec"]["classes"][0], "<blank>")
        m, = external.list_all(); self.assertEqual((m["name"], m["licence"], m["commercial_use"], m["ok"], m["classes"]), ("tiny", "apache-2.0", True, True, len(CHARS) + 2))
        self.assertEqual(external.reader("tiny").read(np.full((40, 120), 200, np.uint8))[0], "22/3/26")
        external.remove("tiny"); self.assertEqual(external.list_all(), []); self.assertRaises(KeyError, external.reader, "tiny")

    def test_refusals(self):
        self.add()
        with self.assertRaisesRegex(ValueError, "already"): self.add()
        for bad in ("", "a b", "x" * 41, "../x"):
            with self.assertRaises(ValueError): external.add(bad, self.onnx, "paddle-rec", self.chars)
        with self.assertRaisesRegex(ValueError, "columns"): external.add("wrong", self.onnx, "paddle-rec", json.dumps(CHARS[:-3]))          # class count does not match the model
        with self.assertRaisesRegex(ValueError, "no such model file"): external.add("gone", self.onnx.with_name("nope.onnx"), "paddle-rec", self.chars)
        self.assertEqual([m["name"] for m in external.list_all()], ["tiny"])           # a refused import leaves nothing behind
        self.assertFalse([p for p in external.DIR.iterdir() if p.name.startswith(".")])

    def test_a_changed_model_file_is_noticed(self):
        self.add(); (external.DIR / "tiny" / "model.onnx").write_bytes(b"changed")
        self.assertFalse(external.list_all()[0]["ok"])

    def test_benchmark_scores_reads_against_labels_and_is_saved(self):
        self.add(); crop = Path(self.tmp.name) / "c.png"; cv2.imwrite(str(crop), np.full((40, 120), 200, np.uint8))
        cells = [("weight", "22/3/26", str(crop), "p1", "c1"), ("weight", "99", str(crop), "p2", "c1"), ("date", "22/3/26", str(crop), "p3", "c1")]
        with mock.patch.object(external, "_cells", lambda card: (cells, {"two_line_cells": 4, "free_text_cells": 0})):
            r = external.benchmark("anc", "tiny")
        self.assertEqual((r["overall"]["n"], r["overall"]["correct"]), (3, 2)); self.assertEqual(r["by_field"]["weight"]["correct"], 1); self.assertEqual(r["skipped"]["two_line_cells"], 4)
        self.assertEqual(external.list_all()[0]["benchmarks"][0]["card"], "anc")

    def test_hugging_face_import_needs_the_licence_accepted_first(self):
        with self.assertRaisesRegex(ValueError, "licence must be accepted"): external.hf_import("o/n", "x", "m.onnx", "c.json", "paddle-rec")
        with mock.patch.object(external, "hf_info", lambda repo: {"repo": repo, "licence": "mit", "gated": False, "url": "u", "files": [{"name": "m.onnx", "bytes": 10 ** 9}, {"name": "c.json", "bytes": 10}]}):
            with self.assertRaisesRegex(ValueError, "too large"): external.hf_import("o/n", "x", "m.onnx", "c.json", "paddle-rec", accept_licence=True)
            with self.assertRaisesRegex(ValueError, "no file"): external.hf_import("o/n", "x", "nope.onnx", "c.json", "paddle-rec", accept_licence=True)


if __name__ == "__main__":
    unittest.main()
