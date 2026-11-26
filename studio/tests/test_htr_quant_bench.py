"""htr.export.quant (width buckets, real PTQ/QAT end to end) and htr.export.bench (real ONNX latency/size benchmark)."""
import sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import htr.export.quant as Q

try:
    import torch
except ImportError:
    torch = None

try:
    import onnxruntime  # noqa: F401
    HAVE_ORT = True
except ImportError:
    HAVE_ORT = False


class Buckets(unittest.TestCase):
    def test_bucket_for_picks_smallest_fit(self):
        self.assertEqual(Q.bucket_for(100), 128)
        self.assertEqual(Q.bucket_for(128), 128)
        self.assertEqual(Q.bucket_for(129), 192)

    def test_bucket_for_caps_at_largest(self):
        self.assertEqual(Q.bucket_for(10_000), Q.BUCKETS[-1])

    def test_bucket_counts_tallies_each_bucket(self):
        counts = Q.bucket_counts([50, 150, 150, 2000])
        self.assertEqual(counts[128], 1); self.assertEqual(counts[192], 2); self.assertEqual(counts[Q.BUCKETS[-1]], 1)

    def test_time_steps_for_matches_ceil_division(self):
        self.assertEqual(Q.time_steps_for(192, 4), 48)
        self.assertEqual(Q.time_steps_for(190, 4), 48)


@unittest.skipIf(torch is None, "torch not installed")
class Quantize(unittest.TestCase):
    def test_default_engine_is_one_torch_actually_supports(self):
        eng = Q.default_engine()
        self.assertIn(eng, torch.backends.quantized.supported_engines)

    def test_post_training_quantization_runs_and_keeps_output_shape(self):
        from htr.nn.compile import compile
        net = compile("crnn-small", 37).eval()

        class Wrap(torch.nn.Module):
            def __init__(self, inner): super().__init__(); self.inner = inner
            def forward(self, x): return self.inner(x)[0]

        w = Wrap(net).eval()
        batches = [{"images": torch.randn(2, 1, 48, 96)} for _ in range(2)]
        q = Q.calibrate_and_convert(w, batches, forward=lambda m, b: m(b["images"]))
        with torch.no_grad():
            out = q(torch.zeros(1, 1, 48, 96))
        self.assertEqual(tuple(out.shape), (24, 1, 37))   # crnn-small's width_stride is 4: 96/4=24 time steps


@unittest.skipIf(torch is None or not HAVE_ORT, "needs torch and onnxruntime")
class Bench(unittest.TestCase):
    def test_bench_onnx_reports_real_timing_and_size(self):
        import htr.export.bench as B
        import htr.export.onnx_backend as O
        from htr.nn.compile import compile
        net = compile("crnn-small", 37).eval()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "reader.onnx"
            O.export(net, 48, 192, p)
            r = B.bench_onnx(p, 48, 192, runs=5, warmup=1)
            self.assertGreater(r["size_bytes"], 0)
            self.assertGreaterEqual(r["p50_ms"], 0)
            self.assertGreaterEqual(r["p95_ms"], r["p50_ms"])
            self.assertEqual(r["runs"], 5)

    def test_bench_arch_builds_exports_benches_and_cleans_up(self):
        import htr.export.bench as B
        before = set(Path(tempfile.gettempdir()).glob("*.onnx"))
        r = B.bench_arch("crnn-small", n_classes=37, width=96, runs=3, warmup=1)
        self.assertGreater(r["size_bytes"], 0)
        self.assertEqual(r["runs"], 3)
        after = set(Path(tempfile.gettempdir()).glob("*.onnx"))
        self.assertEqual(before, after)   # no leftover .onnx temp file

    def test_bench_arch_refuses_a_bad_spec(self):
        import htr.export.bench as B
        with self.assertRaises(ValueError): B.bench_arch({"conv": []}, n_classes=37)


if __name__ == "__main__":
    unittest.main()
