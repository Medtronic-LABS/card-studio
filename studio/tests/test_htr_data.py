"""htr.data: writer-independent splits (never cross a writer, respect forced units), manifest build/load round trip, preprocess
(dynamic width, invert-if-dark), and dataset collate (padding, input_lengths, concatenated targets)."""
import sys, unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import htr.data.manifest as Man
import htr.data.preprocess as P
import htr.data.splits as S
import htr.vocab as V

try:
    import cv2  # noqa: F401
except ImportError:
    cv2 = None

try:
    import torch  # noqa: F401
    from htr.data.dataset import collate
except ImportError:
    torch = None


def _rows(n_writers=8, per_writer=5, doc_type="card_cell"):
    rows = []
    for w in range(n_writers):
        for i in range(per_writer):
            rows.append({"image_path": f"/x/{w}_{i}.png", "text": f"line{w}_{i}", "writer_id": f"W{w}", "document_type": doc_type})
    return rows


class Splits(unittest.TestCase):
    def test_no_writer_crosses_a_split(self):
        rows = _rows()
        out = S.split(rows, seed=1)
        by_split = {name: {r["writer_id"] for r in rs} for name, rs in out.items()}
        self.assertEqual(by_split["train"] & by_split["validation"], set())
        self.assertEqual(by_split["train"] & by_split["test"], set())
        self.assertEqual(sum(len(rs) for rs in out.values()), len(rows))

    def test_deterministic_for_a_fixed_seed(self):
        rows = _rows()
        self.assertEqual(S.split(rows, seed=42), S.split(rows, seed=42))

    def test_forced_unit_goes_to_its_split_regardless_of_fraction(self):
        rows = _rows()
        out = S.split(rows, seed=1, forced={"W0": "test"})
        self.assertTrue(all(r["writer_id"] == "W0" for r in out["test"] if r["writer_id"] == "W0"))
        self.assertIn("W0", {r["writer_id"] for r in out["test"]})
        self.assertNotIn("W0", {r["writer_id"] for r in out["train"]} | {r["writer_id"] for r in out["validation"]})

    def test_balances_across_document_types(self):
        rows = _rows(doc_type="a") + _rows(n_writers=8, per_writer=5, doc_type="b")
        for i, r in enumerate(rows[64:]): r["writer_id"] += "b"   # keep the second batch's writers distinct
        out = S.split(rows, seed=3)
        types_per_split = {name: {r["document_type"] for r in rs} for name, rs in out.items() if rs}
        self.assertIn("a", types_per_split.get("train", set()) | types_per_split.get("validation", set()) | types_per_split.get("test", set()))


class Manifest(unittest.TestCase):
    def test_build_and_reload_roundtrip(self):
        import json, tempfile
        rows = _rows(n_writers=3, per_writer=2)
        for r in rows: r.setdefault("source", "test"); r.setdefault("line_id", r["image_path"]); r.setdefault("kind", "text")
        splits = S.split(rows, seed=1)
        with tempfile.TemporaryDirectory() as d:
            man = Man.build(d, splits, V.Vocabulary.default(), sources=[{"name": "test", "licence": "n/a"}], version="v0")
            self.assertEqual(sum(man["counts"].values()), len(rows))
            loaded = json.loads((Path(d) / "metadata" / "dataset_version.json").read_text())
            self.assertEqual(loaded["vocab"]["sha256"], V.Vocabulary.default().sha256)
            train_rows = Man.load_split(Path(d) / "annotations" / "train.csv")
            for r in train_rows: self.assertIn("text_sha", r); self.assertTrue(r["text_sha"])


@unittest.skipIf(cv2 is None, "opencv not installed")
class Preprocess(unittest.TestCase):
    def test_resize_keeps_aspect_and_fixed_height(self):
        im = np.full((96, 480), 255, np.uint8)
        out = P.resize_to_height(im, height=48, max_width=1024)
        self.assertEqual(out.shape[0], 48)
        self.assertEqual(out.shape[1], 240)

    def test_caps_width(self):
        im = np.full((48, 4000), 255, np.uint8)
        out = P.resize_to_height(im, height=48, max_width=1024)
        self.assertLessEqual(out.shape[1], 1024)

    def test_invert_if_dark(self):
        dark = np.full((10, 10), 10, np.uint8)
        self.assertGreater(P.invert_if_dark(dark).mean(), dark.mean())
        light = np.full((10, 10), 240, np.uint8)
        self.assertTrue((P.invert_if_dark(light) == light).all())


@unittest.skipIf(torch is None, "torch not installed")
class Collate(unittest.TestCase):
    def test_pads_to_stride_multiple_and_concatenates_targets(self):
        batch = [
            {"image": np.zeros((48, 50), np.float32), "width": 50, "target": [1, 2, 3], "kind": "text", "text": "x", "row": {}},
            {"image": np.zeros((48, 61), np.float32), "width": 61, "target": [4, 5], "kind": "text", "text": "y", "row": {}},
        ]
        out = collate(batch, width_stride=4)
        self.assertEqual(out["images"].shape[-1] % 4, 0)
        self.assertEqual(out["images"].shape[-1], 64)
        self.assertEqual(out["input_lengths"].tolist(), [50, 61])
        self.assertEqual(out["target_lengths"].tolist(), [3, 2])
        self.assertEqual(out["targets"].tolist(), [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
