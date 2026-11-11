"""TFLite reader: the pack rules for the optional reader.tflite, and (only where the TensorFlow environment and a trained reader exist) the
conversion itself. Run: python -m unittest discover -s tests"""
import json, sys, tempfile, unittest
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "cardpack"))
import cardpack as cp, synthetic, tflite_export as T

FAKE = b"\x1c\x00\x00\x00TFL3" + b"\x00" * 64            # starts like a TFLite file; never loaded (check_reader=False)
ENTRY = {"file": "reader.tflite", "input": {"name": "image", "layout": "NHWC", "height": 48, "width": 192, "mean": 0.5, "std": 0.5},
         "output": {"name": "Identity", "layout": "TBV"}, "dtype": "float32", "ops": "TFLITE_BUILTINS",
         "parity": {"against": "reader.onnx", "cells": 0, "random_cells": 8, "max_abs_diff": 4e-6, "best_class_equal": True}}
READERS = sorted((HERE / "runs").glob("*/reader.onnx"))
try:
    import onnx; HAVE_ONNX = True
except Exception: HAVE_ONNX = False
CAN_CONVERT = bool(READERS) and T.VENV_PY.exists()


class PackRules(unittest.TestCase):
    def pack(self, tflite=FAKE, entry=ENTRY, hash_it=True, change=None):
        m = synthetic.build_members(); man = json.loads(m["manifest.json"])
        if tflite is not None: m["reader.tflite"] = tflite
        if entry is not None: man["reader"]["tflite"] = json.loads(json.dumps(entry))
        if change: change(man)
        man["files"] = {n: cp.sha256(m[n]) for n in (cp.hashed(m) if hash_it else cp.HASHED)}; m["manifest.json"] = json.dumps(man).encode()
        self.tmp = tempfile.TemporaryDirectory(); out = Path(self.tmp.name) / "synthetic_v1.cardpack"; cp.build(m, out)
        return cp.validate(out, check_reader=False), out

    def tearDown(self):
        if getattr(self, "tmp", None): self.tmp.cleanup()

    def test_a_pack_may_carry_a_tflite_reader(self):
        errs, out = self.pack(); self.assertEqual(errs, [])
        self.assertEqual(cp.read(out)["reader.tflite"], FAKE); self.assertIn("reader.tflite", cp.hashed(cp.read(out)))

    def test_a_pack_without_one_is_still_valid(self):
        self.assertEqual(self.pack(tflite=None, entry=None)[0], [])

    def test_file_and_manifest_block_go_together(self):
        self.assertTrue(any("go together" in e for e in self.pack(entry=None)[0]))
        self.assertTrue(any("go together" in e for e in self.pack(tflite=None)[0]))

    def test_its_hash_must_be_listed(self):
        self.assertTrue(any("manifest.files" in e for e in self.pack(hash_it=False)[0]))

    def test_the_manifest_block_is_checked(self):
        for change in (lambda m: m["reader"]["tflite"]["input"].update(layout="NCHW"), lambda m: m["reader"]["tflite"].pop("parity"),
                       lambda m: m["reader"]["tflite"]["parity"].update(best_class_equal=False), lambda m: m["reader"]["tflite"].update(dtype="int8")):
            self.assertTrue(self.pack(change=change)[0])

    def test_something_that_is_not_a_tflite_file_is_refused_when_the_reader_is_checked(self):
        self.assertEqual(cp._check_tflite(b"not a model at all", ENTRY, 129), ["reader.tflite: not a TFLite model"])

    def test_other_extra_members_are_still_refused(self):
        m = synthetic.build_members(); m["notes.txt"] = b"x"
        self.tmp = tempfile.TemporaryDirectory(); out = Path(self.tmp.name) / "x.cardpack"; cp.build(m, out)
        self.assertTrue(any("unexpected member" in e for e in cp.validate(out, check_reader=False)))


class Environment(unittest.TestCase):
    def test_no_tensorflow_environment_is_reported_not_guessed(self):
        import os
        old = (T.VENV_PY, os.environ.pop("STUDIO_TFLITE_PYTHON", None)); T.VENV_PY = Path("/nonexistent/python")
        try:
            with self.assertRaises(T.Unavailable): T.export(b"anything")
        finally:
            T.VENV_PY = old[0]
            if old[1] is not None: os.environ["STUDIO_TFLITE_PYTHON"] = old[1]

    def test_manifest_entry_shares_the_onnx_scaling(self):
        rep = {"input": {"name": "image"}, "output": {"name": "Identity"}, "dtype": "float32", "ops": "TFLITE_BUILTINS", "parity": ENTRY["parity"]}
        self.assertEqual(T.manifest_entry(rep, {"input": {"name": "image", "height": 48, "width": 192, "mean": 0.5, "std": 0.5}}), ENTRY)


@unittest.skipUnless(HAVE_ONNX and READERS, "needs the onnx package and a trained reader in studio/runs")
class Graph(unittest.TestCase):
    def test_the_reader_graph_is_read(self):
        s = T.read_onnx(READERS[-1])
        self.assertEqual((s["h"], s["w"], s["hidden"], s["classes"], [c["pool"] for c in s["convs"]], s["W"].shape, s["fcw"].shape),
                         (48, 192, 128, 129, [[2, 2], [2, 2], [2, 1], [6, 1]], (2, 512, 128), (256, 129)))

    def test_a_different_graph_is_refused(self):
        import onnx
        m = onnx.load(str(READERS[-1])); relu = next(n for n in m.graph.node if n.op_type == "Relu"); relu.op_type = "Sigmoid"
        with tempfile.TemporaryDirectory() as d:
            onnx.save(m, f"{d}/other.onnx")
            with self.assertRaises(ValueError) as e: T.read_onnx(f"{d}/other.onnx")
        self.assertIn("not the reader graph", str(e.exception))


@unittest.skipUnless(CAN_CONVERT, "needs studio/.venv-tflite (TensorFlow) and a trained reader in studio/runs")
class Convert(unittest.TestCase):
    def test_the_tflite_reader_reproduces_the_onnx_reader(self):
        import numpy as np
        cells = np.random.default_rng(1).uniform(-1, 1, (5, 48, 192)).astype(np.float32)
        blob, rep = T.export(READERS[-1].read_bytes(), samples=cells)
        self.assertTrue(blob.startswith(b"TFL3", 4))
        self.assertEqual((rep["input"]["shape"], rep["output"]["shape"], rep["parity"]["cells"], rep["parity"]["best_class_equal"]), ([1, 48, 192, 1], [48, 1, 129], 5, True))
        self.assertLess(rep["parity"]["max_abs_diff"], T.MAX_DIFF)

    def test_garbage_in_is_an_error_with_a_reason(self):
        with self.assertRaises(ValueError) as e: T.export(b"this is not an onnx file")
        self.assertIn("TFLite conversion failed", str(e.exception))


if __name__ == "__main__":
    unittest.main()
