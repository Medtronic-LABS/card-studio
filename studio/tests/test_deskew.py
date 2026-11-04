"""Straightening rotated card photos before layout detection. Run: python -m unittest discover -s tests"""
import http.client, json, sys, threading, unittest, zipfile
from pathlib import Path
import cv2, numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "cardpack"))
import deskew as D, detect, gridconv

PACK = HERE.parent / "cardpack" / "out" / "anc_v4.cardpack"


def table(rows=12, cols=6, w=1300, h=900):
    im = np.full((h, w, 3), 245, np.uint8)
    xs = np.linspace(100, w - 100, cols + 1).astype(int); ys = np.linspace(100, h - 100, rows + 1).astype(int)
    for x in xs: cv2.line(im, (x, 100), (x, h - 100), (30, 30, 30), 3)
    for y in ys: cv2.line(im, (100, y), (w - 100, y), (30, 30, 30), 3)
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 3 == 0: cv2.putText(im, str(10 + i * 7 + j), (xs[j] + 15, ys[i + 1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 40, 30), 2)
    return im


class Tilt(unittest.TestCase):
    def test_estimate_matches_applied_rotation(self):
        im = table()
        for A in (0, 1, 3, 7, 15, 25, 40, -4, -20, -40):
            est, conf = D.estimate_tilt(D.rotate_by(im, A))
            self.assertAlmostEqual(est, -A, delta=0.6, msg=f"applied {A}")      # est is the angle to undo
            self.assertGreater(conf, 0.5)

    def test_straighten_restores_layout(self):
        im = table(); base = detect.detect_layout(im)["stats"]
        self.assertGreater(base["cells"], 60)
        for A in (2, 5, 12, 30, -15, 44):
            out, info = D.straighten(D.rotate_by(im, A))
            st = detect.detect_layout(out)["stats"]
            self.assertTrue(info["straightened"], (A, info))
            self.assertGreaterEqual(st["cells"], 0.85 * base["cells"], (A, st, base))
            self.assertLessEqual(abs(st["rows"] - base["rows"]), 2, (A, st, base))

    def test_unrotated_image_is_left_alone(self):
        out, info = D.straighten(table())
        self.assertFalse(info["straightened"]); self.assertEqual(out.shape, table().shape)

    def test_no_lines_is_not_trusted(self):
        rs = np.random.RandomState(1)
        for im in (rs.randint(0, 255, (700, 1000, 3)).astype(np.uint8), np.full((700, 1000, 3), 200, np.uint8)):
            out, info = D.straighten(im)
            self.assertFalse(info["straightened"], info)

    def test_quarter_turns(self):
        im = table(w=1300, h=900)
        self.assertEqual(D.rotate_quarter(im, 90).shape[:2], (1300, 900)); self.assertEqual(D.rotate_quarter(im, 180).shape[:2], (900, 1300))
        self.assertTrue(np.array_equal(D.rotate_quarter(D.rotate_quarter(im, 90), 270), im))
        with self.assertRaises(ValueError): D.rotate_quarter(im, 45)

    def test_quarter_then_level_recovers_cells(self):
        im = table(); base = detect.detect_layout(im)["stats"]["cells"]
        for quarter in (90, 180, 270):
            sideways = D.rotate_quarter(im, quarter)
            fixed, _ = D.straighten(sideways, quarter={90: 270, 180: 180, 270: 90}[quarter])   # the user turns it back
            self.assertTrue(np.array_equal(fixed, im) or detect.detect_layout(fixed)["stats"]["cells"] >= 0.9 * base)


@unittest.skipUnless(PACK.exists(), "needs cardpack/out/anc_v4.cardpack")
class RealCard(unittest.TestCase):
    def test_rotated_anc_template_still_gives_the_table(self):
        tpl = cv2.imdecode(np.frombuffer(zipfile.ZipFile(PACK).read("template.png"), np.uint8), 1)
        for A in (0, 3, 10, -20, 30):
            out, _ = D.straighten(D.rotate_by(tpl, A))
            lay = detect.detect_layout(out)
            grid, _ = gridconv.layout_to_grid({"image": [out.shape[1], out.shape[0]], "fields": lay["fields"]}, label_cols=0, data_cols=6)
            self.assertEqual(gridconv.check_grid(grid), [], A)
            self.assertEqual(len(grid["col_lines_top"]) - 1, 6, A)
            self.assertGreaterEqual(len(grid["rows"]), 20, (A, len(grid["rows"])))      # the card has 22 rows


class Endpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import server
        cls.httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.H); cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls): cls.httpd.shutdown()

    def post(self, png, query=""):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        c.request("POST", "/api/analyze" + query, body=png, headers={"Host": f"127.0.0.1:{self.port}", "X-Card-Studio": "1"})
        r = c.getresponse(); return r.status, json.loads(r.read())

    def test_tilted_and_sideways_uploads(self):
        base = detect.detect_layout(table())["stats"]
        tilted = cv2.imencode(".png", D.rotate_by(table(), 12))[1].tobytes()
        st, r = self.post(tilted)
        self.assertEqual(st, 200); self.assertTrue(r["orientation"]["straightened"])
        self.assertGreaterEqual(r["layout"]["stats"]["cells"], 0.85 * base["cells"])
        st2, r2 = self.post(tilted, "?straighten=0")                   # without levelling the old failure returns
        self.assertLess(r2["layout"]["stats"]["cells"], 0.5 * base["cells"])
        st3, r3 = self.post(cv2.imencode(".png", table())[1].tobytes(), "?rotate=90")
        self.assertEqual((r3["size"][0], r3["size"][1]), (900, 1300)); self.assertEqual(r3["orientation"]["quarter"], 90)

    def test_bad_rotate_value(self):
        self.assertEqual(self.post(cv2.imencode(".png", table())[1].tobytes(), "?rotate=45")[0], 400)


if __name__ == "__main__":
    unittest.main()
