"""htr.train end to end on a tiny synthetic dataset: config resolution, EarlyStop/Checkpointer semantics, one epoch of masked CTC
training through htr.nn.compile's dynamic-length forward, resume from a checkpoint, and that a lower cer_grapheme is what wins for
the default monitor."""
import sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

try:
    import torch
except ImportError:
    torch = None

import htr.train.config as C
import htr.train.tracking as TR
import htr.vocab as V


class Config(unittest.TestCase):
    def test_defaults_resolve_clean(self):
        cfg = C.resolve()
        self.assertEqual(cfg["loop"]["monitor"], "cer_grapheme")

    def test_unknown_key_rejected(self):
        with self.assertRaises(ValueError): C.resolve({"loop": {"nope": 1}})

    def test_out_of_range_rejected(self):
        with self.assertRaises(ValueError): C.resolve({"optim": {"lr": 5.0}})

    def test_override_merges_not_replaces(self):
        cfg = C.resolve({"loop": {"epochs": 3}})
        self.assertEqual(cfg["loop"]["epochs"], 3)
        self.assertEqual(cfg["loop"]["batch_size"], C.DEFAULTS["loop"]["batch_size"])


class EarlyStop(unittest.TestCase):
    def test_lower_cer_is_better(self):
        es = TR.EarlyStop(monitor="cer_grapheme", patience=5)
        self.assertTrue(es.update(1, {"cer_grapheme": 0.5, "exact": 0.1}))
        self.assertFalse(es.update(2, {"cer_grapheme": 0.6, "exact": 0.1}))
        self.assertTrue(es.update(3, {"cer_grapheme": 0.3, "exact": 0.1}))
        self.assertEqual(es.best_ep, 3)

    def test_should_stop_after_patience_with_no_improvement(self):
        es = TR.EarlyStop(monitor="cer_grapheme", patience=2)
        es.update(1, {"cer_grapheme": 0.5, "exact": 0.0})
        self.assertFalse(es.should_stop(2))
        self.assertTrue(es.should_stop(3))


@unittest.skipIf(torch is None, "torch not installed")
class TrainLoop(unittest.TestCase):
    def _loaders(self):
        from torch.utils.data import DataLoader
        from htr.data.dataset import collate
        vocab = V.Vocabulary.default()
        rows = [{"text": t, "kind": "text"} for t in ["ab", "ba", "aab", "bba"]] * 4

        class Toy(torch.utils.data.Dataset):
            def __len__(self): return len(rows)
            def __getitem__(self, i):
                r = rows[i]
                import numpy as np
                ids = vocab.encode(r["text"], kind="text")
                w = 40 + 4 * len(r["text"])
                return {"image": np.random.RandomState(i).randn(48, w).astype("float32") * 0.1, "width": w, "target": ids, "kind": "text", "text": r["text"], "row": r}

        dl = DataLoader(Toy(), batch_size=4, shuffle=False, collate_fn=lambda b: collate(b, width_stride=4))
        return dl, vocab

    def test_one_epoch_runs_and_loss_is_finite(self):
        import htr.train.loop as L
        from htr.nn.compile import compile as build_net
        dl, vocab = self._loaders()
        net = build_net("crnn-small", len(vocab))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        loss = L.train_epoch(net, dl, opt, vocab)
        self.assertTrue(loss == loss and loss < 1e6)   # not nan, not exploded

    def test_evaluate_returns_expected_keys(self):
        import htr.train.loop as L
        from htr.nn.compile import compile as build_net
        dl, vocab = self._loaders()
        net = build_net("crnn-small", len(vocab))
        out = L.evaluate(net, dl, vocab)
        self.assertEqual(set(out), {"loss", "exact", "cer_grapheme", "cer_codepoint", "wer", "n"})
        self.assertEqual(out["n"], 16)

    def test_train_checkpoints_and_resumes(self):
        import tempfile
        import htr.train.loop as L
        from htr.nn.compile import compile as build_net
        dl, vocab = self._loaders()
        with tempfile.TemporaryDirectory() as d:
            ckpt = {"dir": d, "name": "test", "resume": True}
            net = build_net("crnn-small", len(vocab))
            out1 = L.train({"loop": {"epochs": 2, "patience": 100}}, net, dl, dl, vocab, ckpt=ckpt, log=lambda *a: None)
            self.assertEqual(len(out1["history"]), 2)
            self.assertTrue((Path(d) / "test-last.pt").exists())
            self.assertTrue((Path(d) / "test-best.pt").exists())
            net2 = build_net("crnn-small", len(vocab))
            out2 = L.train({"loop": {"epochs": 3, "patience": 100}}, net2, dl, dl, vocab, ckpt=ckpt, log=lambda *a: None)
            self.assertEqual(len(out2["history"]), 3)   # resumed from epoch 2, ran one more
            self.assertEqual(out2["history"][0]["epoch"], 1)
            self.assertEqual(out2["history"][-1]["epoch"], 3)


if __name__ == "__main__":
    unittest.main()
