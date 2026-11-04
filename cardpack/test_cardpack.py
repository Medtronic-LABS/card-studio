import copy, hashlib, io, json, tempfile, unittest, zipfile
from pathlib import Path
import numpy as np, cv2, onnxruntime as ort
import os
import cardpack as cp, export_anc as ex, synthetic

ROOT = cp.HERE.parent


class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); cls.d = Path(cls.tmp.name)
        # Real mode needs the app's model assets (patient-trained, not in the public repo). Otherwise a synthetic pack exercises the same rules.
        cls.real = False
        if not os.environ.get("CARDPACK_SYNTHETIC"):
            try: cls.members = ex.build_members(); cls.real = True
            except (FileNotFoundError, OSError): pass
        if not cls.real: cls.members = synthetic.build_members()
        cls.pack = cls.d / "anc_v3.cardpack"; cp.build(cls.members, cls.pack)

    def val(self, path):
        return cp.validate(path, check_reader=self.real)

    def mutated(self, name, fn, rebuild_hashes=True):
        m = {k: v for k, v in self.members.items()}
        if name.endswith(".json"):
            doc = json.loads(m[name]); fn(doc); m[name] = json.dumps(doc).encode()
        else:
            m[name] = fn(m[name])
        if rebuild_hashes and name != "manifest.json":
            man = json.loads(m["manifest.json"]); man["files"] = {n: cp.sha256(m[n]) for n in cp.HASHED}; m["manifest.json"] = json.dumps(man).encode()
        p = self.d / f"bad_{abs(hash((name, id(fn))))}.cardpack"; cp.build(m, p); return self.val(p)

    def test_valid(self):
        self.assertEqual(self.val(self.pack), [])

    def test_round_trip_byte_identical(self):
        r1 = cp.read(self.pack); p2 = self.d / "rt.cardpack"; cp.build(r1, p2)
        self.assertEqual(self.pack.read_bytes(), p2.read_bytes())         # same archive bytes
        r2 = cp.read(p2); self.assertEqual(r1.keys(), r2.keys())
        for k in r1: self.assertEqual(hashlib.sha256(r1[k]).hexdigest(), hashlib.sha256(r2[k]).hexdigest(), k)
        self.assertEqual(self.val(p2), [])

    def test_sources_match_work_dir(self):
        if not self.real: self.skipTest('needs the real app assets')  # app assets == work/ originals
        for a, b in [("crnn.onnx", "work/crnn.onnx"), ("grid.json", "work/grid.json"), ("model_meta.json", "model_meta.json")]:
            self.assertEqual((ex.APP / a).read_bytes(), (ROOT / b).read_bytes(), a)

    def test_reader_output_identical_to_app_asset(self):
        if not self.real: self.skipTest('needs the real app assets')
        s_app = ort.InferenceSession(str(ex.APP / "crnn.onnx")); s_pack = ort.InferenceSession(cp.read(self.pack)["reader.onnx"])
        tests = sorted((ROOT.parent / "leapwell_ocr_demo" / "assets" / "test_set").glob("*.png"))
        self.assertTrue(tests)
        for p in tests:
            im = cv2.imread(str(p), 0); h, w = im.shape
            im = cv2.resize(im, (min(192, max(1, int(w * 48 / h))), 48), interpolation=cv2.INTER_AREA)
            c = np.full((48, 192), 255, np.uint8); c[:, :im.shape[1]] = im
            x = ((c.astype(np.float32) / 255 - .5) / .5)[None, None]
            np.testing.assert_array_equal(s_app.run(None, {"image": x})[0], s_pack.run(None, {"image": x})[0])

    def test_fields_match_plausibility_dart(self):
        if not self.real: self.skipTest('needs the real app assets and the frozen Dart fixture')
        import re
        dart = (cp.HERE / "fixtures" / "plausibility_before_packs.dart.txt").read_text()  # the app rules as they were before packs
        got = {k: tuple(float(x) for x in (a, b)) for k, a, b in re.findall(r"'(\w+)': \(([\d.]+), ([\d.]+)\)", dart)}
        fl = {f["key"]: f for f in json.loads(self.members["fields.json"])["fields"]}
        for k, (lo, hi) in got.items(): self.assertEqual(tuple(fl[k]["range"]), (lo, hi), k)
        marks = set(re.findall(r"'(\w+)'", dart.split("kMarkFields = {")[1].split("}")[0]))
        self.assertEqual({k for k, f in fl.items() if f["kind"] == "mark"}, marks)

    # ---- the validator must reject broken packs ----
    def test_rejects_tampered_reader(self):
        m = dict(self.members); m["reader.onnx"] = m["reader.onnx"] + b"x"; p = self.d / "t1.cardpack"; cp.build(m, p)
        self.assertTrue(any("sha256" in e for e in self.val(p)))

    def test_rejects_missing_member(self):
        p = self.d / "t2.cardpack"
        with zipfile.ZipFile(p, "w") as z:
            for n, b in self.members.items():
                if n != "fallback.json": z.writestr(n, b)
        self.assertTrue(any("missing" in e for e in self.val(p)))

    def test_rejects_unsafe_member_name(self):
        p = self.d / "t3.cardpack"
        with zipfile.ZipFile(p, "w") as z:
            for n, b in self.members.items(): z.writestr(n, b)
            z.writestr("../evil.txt", b"x")
        self.assertTrue(any("unexpected member" in e for e in self.val(p)))

    def test_rejects_ship_without_heldout(self):
        def f(d): d["fields"][0]["status"] = "ship"
        self.assertTrue(any("held-out" in e for e in self.mutated("fields.json", f)))

    def test_rejects_bad_range(self):
        def f(d): [x.__setitem__("range", [10, 5]) for x in d["fields"] if x["key"] == "pulse"]
        self.assertTrue(any("range" in e for e in self.mutated("fields.json", f)))

    def test_rejects_grid_row_mismatch(self):
        def f(d): d["rows"] = d["rows"][:-1]
        errs = self.mutated("grid.json", f); self.assertTrue(any("row" in e for e in errs), errs)

    def test_rejects_wrong_template_size(self):
        def f(d): d["template"]["width"] += 1
        self.assertTrue(any("template.png is" in e for e in self.mutated("manifest.json", f)))

    def test_rejects_wrong_format_version(self):
        def f(d): d["format"] = "cardpack/9"
        self.assertTrue(self.mutated("manifest.json", f))

    def test_rejects_garbage_archive(self):
        p = self.d / "junk.cardpack"; p.write_bytes(b"not a zip"); self.assertTrue(self.val(p))


    # ---- photo normalization block (preprocess.normalize)
    def with_normalize(self, nz):
        def f(doc):
            doc["preprocess"]["normalize"] = nz
        return self.mutated("manifest.json", f)

    def test_normalize_accepts_every_method_that_needs_no_reference(self):
        for m in ("flatten", "cell_stretch", "flatten_cell_stretch", "none"):
            self.assertEqual(self.with_normalize({"default": m}), [], m)
        ref = {"mean": 226.7, "std": 60.2, "source": "scans"}
        self.assertEqual(self.with_normalize({"default": "contrast_match", "reference": ref}), [])

    def test_normalize_rejections(self):
        self.assertTrue(any("needs a reference" in e for e in self.with_normalize({"default": "contrast_match"})))
        self.assertTrue(self.with_normalize({"default": "clahe"}))                                        # proven harmful: not an option
        self.assertTrue(self.with_normalize({"default": "contrast_match", "reference": {"mean": 300, "std": 60, "source": "x"}}))
        self.assertTrue(self.with_normalize({"default": "contrast_match", "reference": {"mean": 200, "std": 0, "source": "x"}}))
        self.assertTrue(self.with_normalize({"default": "flatten", "extra": 1}))

    def test_pack_without_normalize_is_still_valid_and_unchanged(self):
        self.assertNotIn("normalize", json.loads(cp.read(self.pack)["manifest.json"])["preprocess"])      # v3 behaviour does not change

    def test_derive_keeps_the_reader_and_checks_the_version(self):
        out = self.d / "anc_v9.cardpack"
        nz = {"default": "contrast_match", "reference": {"mean": 226.7, "std": 60.2, "source": "scans"}}
        self.assertEqual(cp.derive(self.pack, out, 9, nz), [])
        a, b = cp.read(self.pack), cp.read(out)
        for k in ("reader.onnx", "grid.json", "fields.json", "template.png", "fallback.json"): self.assertEqual(a[k], b[k], k)
        mb = json.loads(b["manifest.json"]); self.assertEqual((mb["version"], mb["preprocess"]["normalize"]["default"]), (9, "contrast_match"))
        self.assertEqual(json.loads(a["manifest.json"])["metrics"], mb["metrics"])
        cur = json.loads(a["manifest.json"])["version"]
        with self.assertRaises(ValueError): cp.derive(self.pack, self.d / "x.cardpack", cur, nz)         # the version must go up

    def test_scan_reference_is_the_gray_mean_and_std(self):
        img = np.zeros((40, 40), np.uint8); img[:, 20:] = 200; p = self.d / "s.png"; cv2.imwrite(str(p), img)
        mean, std = cp.scan_reference([p], stride=1); self.assertAlmostEqual(mean, 100.0, 1); self.assertAlmostEqual(std, 100.0, 1)

if __name__ == "__main__":
    unittest.main(verbosity=2)
