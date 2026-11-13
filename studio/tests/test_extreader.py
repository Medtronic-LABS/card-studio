"""Reading cells with an outside CTC model: the spec, preparation, decoding and scoring. Uses a tiny ONNX model built in the test (no download)."""
import sys, unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import extreader as X

CHARS = list("0123456789/.")


def tiny_onnx(classes, text):
    """An ONNX 'recogniser' that ignores its picture and always outputs frames spelling `text` (probabilities, [1, T, classes])."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    frames = []
    for ch in text:
        for hot in (classes.index(ch), 0):
            z = np.full(len(classes), 0.001, np.float32); z[hot] = 1.0; frames.append(z / z.sum())
    const = numpy_helper.from_array(np.stack(frames)[None].astype(np.float32), "out")
    g = helper.make_graph([helper.make_node("Constant", [], ["c"], value=const), helper.make_node("Identity", ["c"], ["y"])], "t",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3, 48, "W"])], [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, len(frames), len(classes)])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]); m.ir_version = 8
    return m.SerializeToString()


class Spec(unittest.TestCase):
    def test_blank_and_trailing_space_go_where_the_profile_says(self):
        s = X.make_spec("paddle-rec", CHARS)
        self.assertEqual(s["classes"][0], "<blank>"); self.assertEqual(s["classes"][-1], " "); self.assertEqual(len(s["classes"]), len(CHARS) + 2)
        g = X.make_spec("gray-ctc", CHARS, blank=len(CHARS)); self.assertEqual(g["classes"][-1], "<blank>"); self.assertEqual(g["channels"], 1)

    def test_bad_input_is_refused(self):
        for bad in (lambda: X.make_spec("nope", CHARS), lambda: X.make_spec("paddle-rec", [])): self.assertRaises(ValueError, bad)

    @unittest.skipUnless(__import__("importlib").util.find_spec("yaml"), "needs PyYAML")
    def test_charset_files(self):
        self.assertEqual(X.parse_charset('["a", "b"]'), ["a", "b"]); self.assertEqual(X.parse_charset("a\nb\n"), ["a", "b"])
        self.assertEqual(X.parse_charset("PostProcess:\n  name: CTCLabelDecode\n  character_dict:\n  - '0'\n  - '1'\n  - c\n"), ["0", "1", "c"])
        with self.assertRaises(ValueError): X.parse_charset('[1, 2]')


class Prepare(unittest.TestCase):
    def test_height_channels_and_range(self):
        s = X.make_spec("paddle-rec", CHARS); x = X.prepare(np.full((30, 90), 255, np.uint8), s)
        self.assertEqual(x.shape, (1, 3, 48, 144)); self.assertAlmostEqual(float(x.max()), 1.0, places=5)
        self.assertEqual(X.prepare(np.zeros((30, 9000), np.uint8), s).shape[3], s["max_width"])
        self.assertEqual(X.prepare(np.zeros((30, 90), np.uint8), X.make_spec("gray-ctc", CHARS)).shape, (1, 1, 32, 96))


class Decode(unittest.TestCase):
    def test_collapse_blank_and_confidence(self):
        s = X.make_spec("paddle-rec", CHARS); ix = {c: i for i, c in enumerate(s["classes"])}
        def frame(c, p=0.9): z = np.full(len(s["classes"]), (1 - p) / (len(s["classes"]) - 1), np.float32); z[ix[c]] = p; return z
        out = np.stack([frame("2"), frame("2"), frame("<blank>"), frame("2", 0.6), frame("/"), frame("3")])
        text, conf = X.decode(out[None], s); self.assertEqual(text, "22/3"); self.assertAlmostEqual(conf, 0.6, places=5)
        self.assertEqual(X.decode(np.stack([frame("<blank>")] * 3), s), ("", 0.0))

    def test_raw_scores_and_transposed_layouts_and_a_wrong_class_count(self):
        s = X.make_spec("paddle-rec", CHARS); k = len(s["classes"]); z = np.full((4, k), -5.0, np.float32); z[0, s["classes"].index("7")] = 5; z[2, s["classes"].index("8")] = 5; z[1, 0] = z[3, 0] = 5
        self.assertEqual(X.decode(z, s)[0], "78"); self.assertEqual(X.decode(z.T, s)[0], "78")
        with self.assertRaisesRegex(ValueError, "columns"): X.decode(np.zeros((4, k + 3), np.float32), s)

    def test_bengali_digits_can_be_mapped_to_latin(self):
        s = X.make_spec("paddle-rec", list(X.BENGALI_DIGITS + "/"), digit_map=True); ix = {c: i for i, c in enumerate(s["classes"])}
        out = np.zeros((3, len(s["classes"])), np.float32); out[0, ix["২"]] = out[1, 0] = out[2, ix["৩"]] = 1
        self.assertEqual(X.decode(out, s)[0], "23")


@unittest.skipUnless(all(__import__("importlib").util.find_spec(m) for m in ("onnx", "onnxruntime")), "needs the onnx and onnxruntime packages to build and run a test model")
class Reader(unittest.TestCase):
    def test_a_model_runs_end_to_end_and_a_wrong_spec_is_caught(self):
        s = X.make_spec("paddle-rec", CHARS); r = X.ExternalReader(tiny_onnx(s["classes"], "22/3/26"), s)
        self.assertEqual(r.read(np.full((40, 120), 200, np.uint8))[0], "22/3/26"); r.check()
        with self.assertRaises(ValueError): X.ExternalReader(tiny_onnx(s["classes"], "22/3/26"), X.make_spec("paddle-rec", CHARS[:-2])).check()


class Score(unittest.TestCase):
    def test_exact_cer_and_punctuation_blind_comparison(self):
        r = X.score([("d", "22/3/26", "22.3.26"), ("d", "1/2/26", "1/2/27"), ("w", "57.5", "57"), ("w", "60", "60")])
        self.assertEqual(r["overall"]["correct"], 2); self.assertEqual(r["by_field"]["d"]["correct"], 1); self.assertEqual(r["by_field"]["w"]["n"], 2)
        self.assertAlmostEqual(r["by_field"]["w"]["cer"], 2 / 6, places=3); self.assertLess(r["overall"]["lo"], r["overall"]["exact"])


if __name__ == "__main__":
    unittest.main()
