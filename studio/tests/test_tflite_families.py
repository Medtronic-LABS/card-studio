"""Every network family archnet can build converts to TFLite and matches ONNX. Needs torch (to export the ONNX files) and studio/.venv-tflite
(TensorFlow, to convert); skipped when either is missing. Run: python -m unittest discover -s tests -p "test_tflite_families.py" """
import json, subprocess, sys, tempfile, unittest
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "work_phaseA"))
import tflite_export as T
try:
    import torch, archnet; HAVE_TORCH = True
except Exception: HAVE_TORCH = False

B = lambda out, k=3, pool=None: {"out": out, "k": k, **({"pool": pool} if pool is not None else {})}
BASE = [B(8, 3, [2, 2]), B(8, 3, [2, 2]), B(8, 3, [2, 1]), B(8, 3, "collapse")]
RNN = lambda t="lstm", layers=1, bi=True: {"type": t, "hidden": 16, "layers": layers, "bidirectional": bi, "dropout": 0.0}
CUSTOM = {
    "k5-k7": {"conv": [B(8, 5, [2, 2]), B(8, 7, [2, 2]), B(8, 3, [2, 1]), B(8, 5, "collapse")], "context": [], "rnn": RNN()},
    "pool-1x1-blocks": {"conv": [B(8, 3, [1, 1]), B(8, 3, [2, 2]), B(8, 3, [1, 1]), B(8, 3, [2, 2]), B(8, 3, [2, 1]), B(8, 3, "collapse")], "context": [], "rnn": RNN()},
    "last-width-pool-2": {"conv": [B(8, 3, [2, 1]), B(8, 3, [2, 2]), B(8, 3, [2, 1]), B(8, 3, ["collapse", 2])], "context": [], "rnn": RNN()},
    "context-convs": {"conv": BASE, "context": [{"out": 16, "k": 3}, {"out": 20, "k": 7}, {"out": 16, "k": 5}, {"out": 16, "k": 3}], "rnn": RNN()},
    "gru-bi": {"conv": BASE, "context": [], "rnn": RNN("gru")},
    "gru-uni": {"conv": BASE, "context": [], "rnn": RNN("gru", 1, False)},
    "lstm-uni": {"conv": BASE, "context": [], "rnn": RNN("lstm", 1, False)},
    "lstm-2": {"conv": BASE, "context": [], "rnn": RNN("lstm", 2)},
    "lstm-3-uni": {"conv": BASE, "context": [], "rnn": RNN("lstm", 3, False)},
    "gru-3": {"conv": BASE, "context": [{"out": 16, "k": 3}], "rnn": RNN("gru", 3)},
    "no-rnn": {"conv": BASE, "context": [{"out": 16, "k": 5}], "rnn": None},
    "8-blocks": {"conv": [B(4, 3, [1, 1])] * 3 + [B(4, 3, [2, 1]), B(4, 3, [2, 1]), B(4, 3, [2, 2]), B(4, 3, [2, 1]), B(4, 3, "collapse")], "context": [], "rnn": RNN()},
}


def export_onnx(spec, path):
    import numpy as np
    torch.manual_seed(0); net = archnet.build(spec); net.eval()
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.uniform_(-0.5, 0.5); m.running_var.uniform_(0.5, 1.5); m.weight.uniform_(0.5, 1.5); m.bias.uniform_(-0.3, 0.3)
    torch.onnx.export(net, torch.zeros(1, 1, 48, 192), str(path), input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {1: "batch"}}, opset_version=17, dynamo=False)


def convert(onnx, out):
    r = subprocess.run([str(T.VENV_PY), str(HERE / "tflite_export.py"), "convert", str(onnx), str(out)], capture_output=True, text=True, timeout=900,
                       env={"PATH": "/usr/bin:/bin", "TF_CPP_MIN_LOG_LEVEL": "3", "CUDA_VISIBLE_DEVICES": ""})
    last = (r.stdout.strip().splitlines() or [""])[-1]
    return r.returncode, last


@unittest.skipUnless(HAVE_TORCH, "needs torch to export the ONNX files")
@unittest.skipUnless(T.VENV_PY.exists(), "needs studio/.venv-tflite (TensorFlow)")
class Families(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); cls.dir = Path(cls.tmp.name); cls.diffs = {}

    @classmethod
    def tearDownClass(cls):
        print("\nmax abs diff per spec:", json.dumps(cls.diffs, indent=1)); cls.tmp.cleanup()

    def check(self, name, spec):
        onnx = self.dir / f"{name}.onnx"; export_onnx(spec, onnx)
        code, last = convert(onnx, self.dir / f"{name}.tflite")
        self.assertEqual(code, 0, last); rep = json.loads(last)["parity"]
        self.assertLess(rep["max_abs_diff"], T.MAX_DIFF); self.assertTrue(rep["best_class_equal"]); self.diffs[name] = rep["max_abs_diff"]
        self.assertTrue((self.dir / f"{name}.tflite").exists())

    def test_every_preset(self):
        for name in archnet.PRESETS: self.check("preset-" + name, name)

    def test_custom_specs(self):
        for name, spec in CUSTOM.items():
            with self.subTest(name): self.check(name, spec)

    def test_support_flag_agrees(self):
        for name in list(archnet.PRESETS) + list(CUSTOM):
            s = archnet.normalize(CUSTOM.get(name, name)); self.assertTrue(archnet.tflite_support(s)["supported"], name)

    def test_unsupported_graph_is_refused(self):
        import onnx
        export_onnx("crnn", self.dir / "base.onnx")
        m = onnx.load(str(self.dir / "base.onnx")); next(n for n in m.graph.node if n.op_type == "Relu").op_type = "Sigmoid"; onnx.save(m, str(self.dir / "sig.onnx"))
        code, last = convert(self.dir / "sig.onnx", self.dir / "sig.tflite"); self.assertNotEqual(code, 0); self.assertIn("not the reader graph", last)
        m = onnx.load(str(self.dir / "base.onnx")); c = next(n for n in m.graph.node if n.op_type == "Conv")
        next(a for a in c.attribute if a.name == "strides").ints[:] = [2, 2]; onnx.save(m, str(self.dir / "stride.onnx"))
        code, last = convert(self.dir / "stride.onnx", self.dir / "stride.tflite"); self.assertNotEqual(code, 0); self.assertIn("stride 1", last)
        self.assertFalse((self.dir / "sig.tflite").exists())


if __name__ == "__main__":
    unittest.main()
