"""Clean-up lab: normalization maths, the port of the app's ANC steps, the endpoint. Run: python -m unittest discover -s tests"""
import http.client, json, sys, tempfile, threading, unittest
from pathlib import Path
import cv2, numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import normalize as N

CASES = HERE.parent / "cardpack" / "dart_check" / "normalization_cases.json"
SCAN = HERE.parent / "work" / "raw" / "p-000.png"
PACKS = HERE.parent / "cardpack" / "out"
HAVE = SCAN.exists() and (PACKS / "anc_v5.cardpack").exists() and (PACKS / "anc_v3.cardpack").exists()
NEWEST = "anc v%d" % max([int(p.stem.split("_v")[1]) for p in PACKS.glob("anc_v*.cardpack")] or [0])   # auto picks the newest pack on ties


@unittest.skipUnless(CASES.exists(), "run cardpack/dart_check/make_normalization_cases.py first")
class MathMatchesDartReference(unittest.TestCase):
    def test_luts_and_stats(self):
        for c in json.loads(CASES.read_text()):
            px = np.array(c["pixels"], np.uint8); mask = None if c["mask"] is None else np.array(c["mask"], np.uint8)
            h = N.hist256(px, mask)
            self.assertEqual(h, c["hist"], c["name"])
            self.assertEqual(N.contrast_lut(h, 227.01, 59.53).tolist(), c["contrast_lut"], c["name"])
            self.assertEqual(N.stretch_lut(h).tolist(), c["stretch_lut"], c["name"])
            self.assertAlmostEqual(N.percentile(h, 98), c["p98"], places=9)


class PageSteps(unittest.TestCase):
    def test_contrast_match_without_reference_falls_back_to_flatten(self):
        page = np.random.RandomState(0).randint(100, 240, (200, 300)).astype(np.uint8)
        a, used = N.normalize_page("contrast_match", page, np.ones_like(page), 31, None)
        self.assertEqual(used, "flatten"); self.assertTrue(np.array_equal(a, N.flatten(page, 31)))

    def test_covered_mask_keeps_white_fill_out_of_the_statistics(self):
        rs = np.random.RandomState(1)
        page = np.full((100, 100), 255, np.uint8); page[:, :40] = np.clip(rs.normal(128, 20, (100, 40)), 0, 255).astype(np.uint8)
        cov = np.zeros_like(page); cov[:, :40] = 1
        masked, _ = N.normalize_page("contrast_match", page, cov, 31, (128.0, 40.0))
        unmasked, _ = N.normalize_page("contrast_match", page, np.ones_like(page), 31, (128.0, 40.0))
        self.assertAlmostEqual(float(masked[:, :40].std()), 40.0, delta=3)         # the photographed part lands on the reference spread
        self.assertGreater(abs(float(unmasked[:, :40].std()) - 40.0), 5)           # the fill would have skewed it

    def test_too_little_coverage_leaves_the_page_alone(self):
        page = np.random.RandomState(2).randint(0, 255, (100, 100)).astype(np.uint8); cov = np.zeros_like(page); cov[:2, :2] = 1
        out, _ = N.normalize_page("contrast_match", page, cov, 31, (227.0, 60.0)); self.assertTrue(np.array_equal(out, page))

    def test_line_removal_characterisation(self):
        """What the app's step does today, pinned so a change is deliberate: long thin lines go, a stroke taller than 60% of the cell goes too."""
        import cleanup as L
        crop = np.full((60, 120), 255, np.uint8); crop[:, 5] = 20                      # a cell border
        crop[10:25, 60] = 20                                                           # short pen stroke (25% of the height)
        out = L.remove_lines(crop)
        self.assertEqual(int(out[:, 5].min()), 255)                                   # border erased
        self.assertLess(int(out[10:25, 60].min()), 60)                                # short stroke survives
        crop2 = np.full((60, 120), 255, np.uint8); crop2[8:52, 60] = 20               # tall digit stroke (73% of the height)
        self.assertEqual(int(L.remove_lines(crop2)[8:52, 60].min()), 255)             # erased: the known cost on tall digits


@unittest.skipUnless(HAVE, "needs work/raw/p-000.png and the v3 + v5 packs")
class Lab(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cleanup as L
        cls.L = L; cls.packs = L.installed(); cls.photo = cv2.imread(str(SCAN)); cls.tmp = tempfile.TemporaryDirectory()

    def test_auto_picks_newest_on_ties_and_runs_every_method(self):
        rep = self.L.analyze(self.photo, self.tmp.name, packs=self.packs)
        self.assertEqual(rep["pack"], NEWEST); self.assertGreater(rep["inliers"], 100)
        self.assertEqual([m["method"] for m in rep["methods"]], list(N.METHODS)); self.assertTrue(all(m["available"] for m in rep["methods"]))
        self.assertGreater(min(m["read"] for m in rep["methods"]), 40)
        for m in rep["methods"]: self.assertTrue((Path(self.tmp.name) / f"{rep['id']}_{m['method']}.png").exists())

    def test_pack_without_reference_marks_contrast_match_unavailable(self):
        v3 = next(p for p in self.packs if p.name == "anc v3")
        rep = self.L.analyze(self.photo, self.tmp.name, pack=v3, packs=self.packs)
        cm = next(m for m in rep["methods"] if m["method"] == "contrast_match")
        self.assertFalse(cm["available"]); self.assertIn("reference", cm["note"])

    def test_rotated_photo_reads_about_as_many_cells(self):
        import deskew
        up = self.L.analyze(self.photo, self.tmp.name, packs=self.packs)
        rot = self.L.analyze(deskew.rotate_by(self.photo, 30), self.tmp.name, packs=self.packs)
        a, b = up["methods"][0]["read"], rot["methods"][0]["read"]
        self.assertLess(abs(a - b), 0.15 * a, (a, b))

    def test_not_a_card_is_refused(self):
        noise = np.random.RandomState(3).randint(0, 255, (900, 1300, 3)).astype(np.uint8)
        with self.assertRaises(ValueError): self.L.analyze(noise, self.tmp.name, packs=self.packs)

    def test_reads_on_a_scan_are_mostly_right_for_marks(self):
        """Smoke test of the port on a training page: tick/cross fields should mostly read correctly."""
        import csv
        v5 = next(p for p in self.packs if p.name == "anc v5"); H, inl, bad = self.L.align(self.photo, v5); warped, cov = self.L.warp(self.photo, H, v5)
        _, _, reads, _ = self.L.run_method(v5, warped, cov, "none")
        lab = {(r["row"], r["col"]): r["label"] for r in csv.DictReader(open(HERE.parent / "work" / "labels.csv")) if r["page"] == "p-000" and r["label"].strip()}
        marks = [(k, v) for k, v in lab.items() if k[0] in ("edema", "urine_albumin", "urine_bilirubin", "calcium_tablet", "iron_tablet", "usg", "tt_vaccine", "fetal_movement")]
        hit = sum(reads[(v5.grid["rows"].index(k[0]), int(k[1][1:]) - 1)][0] == v for k, v in marks)
        self.assertGreaterEqual(hit, max(1, int(0.6 * len(marks))), (hit, len(marks)))


@unittest.skipUnless(HAVE, "needs the scan and packs")
class Endpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import server
        cls.httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.H); cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls): cls.httpd.shutdown()

    def call(self, method, path, body=None, hdr=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=120)
        c.request(method, path, body=body, headers={"Host": f"127.0.0.1:{self.port}", **(hdr or {})})
        r = c.getresponse(); d = r.read(); return r.status, d, r.getheader("Content-Type", "")

    def test_cleanup_round_trip(self):
        png = cv2.imencode(".png", cv2.imread(str(SCAN)))[1].tobytes()
        st, d, _ = self.call("POST", "/api/cleanup", png, {"X-Card-Studio": "1"}); self.assertEqual(st, 200, d[:200])
        rep = json.loads(d); self.assertEqual(rep["pack"], NEWEST)
        st, img, ct = self.call("GET", rep["methods"][0]["image"]); self.assertEqual((st, ct), (200, "image/png")); self.assertTrue(img.startswith(b"\x89PNG"))
        self.assertEqual(self.call("GET", "/api/cleanup/" + "0" * 32 + "/none.png")[0], 404)
        self.assertEqual(self.call("POST", "/api/cleanup", b"x")[0], 403)                                     # needs the guard header (small body: the server refuses before reading it)
        self.assertEqual(self.call("POST", "/api/cleanup?pack=../x", png, {"X-Card-Studio": "1"})[0], 400)
        self.assertEqual(self.call("POST", "/api/cleanup?pack=nope.cardpack", png, {"X-Card-Studio": "1"})[0], 404)
        st, d, _ = self.call("GET", "/api/cleanup/packs"); self.assertEqual(st, 200); self.assertGreaterEqual(len(json.loads(d)), 3)


if __name__ == "__main__":
    unittest.main()
