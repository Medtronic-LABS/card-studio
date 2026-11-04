"""The leak check must catch every kind of thing that could expose patient data, and must not flag ordinary code. Run: python -m unittest discover -s tools"""
import hashlib, json, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import leakcheck as L

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 30


class Check(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory(); self.r = Path(self.t.name)

    def tearDown(self): self.t.cleanup()

    def put(self, rel, data="x"):
        p = self.r / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data if isinstance(data, bytes) else data.encode()); return p

    def found(self): return L.check(self.r)

    def test_ordinary_code_is_clean(self):
        self.put("studio/server.py", "import json\nprint('hello')\n"); self.put("README.md", "# Title\nSee docs.\n"); self.put(".gitignore", "work/\n")
        self.put("cardpack/schema/grid.schema.json", "{}"); self.put("requirements.txt", "numpy\n")
        self.assertEqual(self.found(), [])

    def test_each_kind_of_patient_artifact_is_refused(self):
        for rel in ("a/crop.jpg", "a/scan.jpeg", "reader.onnx", "w/model.pt", "anc.cardpack", "labels.csv", "card.pdf", "nb.ipynb", "x.npy", "d.h5", "bundle.zip", "photo.heic", "run.log"):
            self.put(rel, b"x")
        out = self.found(); self.assertEqual(len(out), 13, out)
        for rel in ("crop.jpg", "reader.onnx", "labels.csv", "nb.ipynb", "card.pdf", "photo.heic"): self.assertTrue(any(rel in o for o in out), rel)

    def test_data_directories_are_refused_even_for_innocent_files(self):
        for d in ("work", "data", "studio/runs", "studio/uploads", "studio/imports", "studio/experiments", "cardpack/out"): self.put(f"{d}/note.txt", "hello")
        self.assertEqual(len(self.found()), 7)

    def test_unregistered_png_is_refused_registered_screenshot_allowed(self):
        self.put("docs/stray.png", PNG); self.put("studio/crop.png", PNG)
        self.assertEqual(len(self.found()), 2)
        shot = self.put("docs/screenshots/home.png", PNG)
        self.put("docs/screenshots/MANIFEST.json", json.dumps({"screenshots": [{"file": "docs/screenshots/home.png", "sha256": hashlib.sha256(PNG).hexdigest(), "images_on_page": 0}]}))
        self.assertEqual(len(self.found()), 2)                                     # the two strays only
        shot.write_bytes(PNG + b"tampered"); self.assertTrue(any("does not match the hash" in o for o in self.found()))
        shot.write_bytes(PNG); self.put("docs/screenshots/MANIFEST.json", json.dumps({"screenshots": [{"file": "docs/screenshots/home.png", "sha256": hashlib.sha256(PNG).hexdigest(), "images_on_page": 2}]}))
        self.assertTrue(any("had images on it" in o for o in self.found()))

    def test_json_must_be_allowlisted(self):
        self.put("results.json", "{}"); self.put("predictions.json", "[]"); self.put("model_meta.json", json.dumps({"classes": [], "blank_idx": 0, "img_h": 1, "img_w": 1, "fields": {}}))
        out = self.found(); self.assertEqual(len(out), 2, out)
        self.put("model_meta.json", json.dumps({"classes": [], "blank_idx": 0, "img_h": 1, "img_w": 1, "fields": {}, "labels": ["98"]}))
        self.assertTrue(any("unexpected keys" in o for o in self.found()))

    def test_secrets_paths_emails_and_label_headers(self):
        # built at runtime so this file's own source never contains the patterns it tests for
        self.put("a.py", "p = '/Us" + "ers/someone/work/x'\n"); self.put("b.md", "mail me: person" + "@" + "company.org\n"); self.put("c.py", "t = 'gh" + "p_" + "a" * 30 + "'\n")
        self.put("d.py", "pass" + "word = 'hunter2hunter2'\n"); self.put("e.txt", "page,row" + ",col,label\np-000,pulse,c3,98\n"); self.put("f.py", "-----BEGIN RSA " + "PRIVATE KEY-----\n")
        out = self.found(); self.assertEqual(len(out), 6, out)

    def test_normal_things_that_look_like_these_are_fine(self):
        self.put("a.py", "def f(token_count):\n    password_field = None  # not a literal\n    return 'git@github.com' if False else 1\n")
        self.put("b.md", "Contact: noreply@users.noreply.github.com or user@example.com\n")
        self.assertEqual(self.found(), [], self.found())

    def test_oversize_and_binary_content(self):
        self.put("big.txt", "x" * (L.MAX_BYTES + 1)); self.put("bin.txt", b"abc\x00def")
        self.assertEqual(len(self.found()), 2)

    def test_vendored_css_is_allowed_but_other_css_is_scanned(self):
        self.put("studio/static/vendor/tabler.min.css", "a{} /* mail x" + "@y.org /Us" + "ers/z */"); self.put("studio/static/own.css", "/* /Us" + "ers/z */")
        out = self.found(); self.assertEqual(len(out), 1); self.assertIn("own.css", out[0])


if __name__ == "__main__":
    unittest.main()
