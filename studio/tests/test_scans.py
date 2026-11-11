"""Filled scans: identify the card, align, cut cells, ink gate, re-cut, remove. Drawn cards only (no patient data). Run: python -m unittest discover -s tests"""
import http.client, json, random, sys, tempfile, threading, unittest
from pathlib import Path
import cv2, numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "cardpack"))
import cards, corrections, gridconv, scans, server
from test_gridconv import table

ROWS, COLS = 5, 3


def draw_card(seed):
    """A blank printed card: a ruled table with a title and row labels in made-up words (the words are what SIFT matches on)."""
    rng = random.Random(seed); layout = table(rows=ROWS, cols=COLS, cw=120, rh=44, x0=180, y0=120, names=[f"f{seed}_{i}" for i in range(ROWS)])
    W, H = layout["image"]; img = np.full((H, W), 250, np.uint8)
    word = lambda n: "".join(rng.choice("ABCDEFGHKMNPRSTWXYZ23456789") for _ in range(n))
    for f in layout["fields"]: cv2.rectangle(img, (f["x"], f["y"]), (f["x"] + f["w"], f["y"] + f["h"]), 30, 2)
    for i in range(3): cv2.putText(img, word(22), (30, 35 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    for f in layout["fields"]:
        if f["x"] < 180: cv2.putText(img, word(6), (f["x"] + 6, f["y"] + 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, 0, 2)
    for i in range(4): cv2.putText(img, word(30), (20, H - 80 + 22 * i), cv2.FONT_HERSHEY_PLAIN, 1.3, 0, 2)
    return img, layout


def fill(img, layout, cells, turn=None, seed=0):
    """A filled scan of the card: digits written in the given (row, col) cells, then the page is turned and slightly skewed like a real scan."""
    out = img.copy(); data = [f for f in layout["fields"] if f["x"] >= 180]
    for r, c in cells:
        f = data[r * COLS + c]; cv2.putText(out, "47", (f["x"] + 25, f["y"] + 32), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, 0, 2)
    h, w = out.shape; rng = random.Random(seed); j = lambda: rng.uniform(-8, 8)
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), np.float32([[20 + j(), 20 + j()], [w + 20 + j(), 24 + j()], [w + 18 + j(), h + 22 + j()], [22 + j(), h + 20 + j()]]))
    out = cv2.warpPerspective(out, M, (w + 40, h + 40), borderValue=255)
    return cv2.rotate(out, turn) if turn is not None else out


def png(img):
    return cv2.imencode(".png", img)[1].tobytes()


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); t = Path(self.tmp.name)
        self.old = (cards.TEMPLATES, cards.CARDS, cards.PACKS, corrections.IMPORTS)
        cards.TEMPLATES, cards.CARDS, cards.PACKS, corrections.IMPORTS = t / "templates", t / "cards", t / "out", t / "imports"
        scans._targets.clear(); self.card = {}
        for name, seed in (("alpha", 1), ("beta", 2)): self.card[name] = self.template(name, seed)

    def tearDown(self):
        cards.TEMPLATES, cards.CARDS, cards.PACKS, corrections.IMPORTS = self.old; scans._targets.clear(); self.tmp.cleanup()

    def template(self, name, seed, version=1, data_cols=COLS):
        img, layout = draw_card(seed); d = cards.TEMPLATES / name / f"v{version}"; d.mkdir(parents=True)
        cv2.imwrite(str(d / "template.png"), img); (d / "layout.json").write_text(json.dumps(layout))
        g, _ = gridconv.layout_to_grid(layout, label_cols=1, data_cols=data_cols); (d / "grid.json").write_text(json.dumps(g))
        return img, layout

    def scan(self, name, cells=((0, 0), (2, 1)), turn=cv2.ROTATE_90_CLOCKWISE, seed=0):
        return png(fill(*self.card[name], cells, turn, seed))


class Identify(Base):
    def test_auto_routes_each_scan_to_its_own_card_whatever_way_up(self):
        for name in ("alpha", "beta"):
            for turn in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180):
                r = scans.add(scans.AUTO, self.scan(name, turn=turn), f"{name}.png")[0]
                self.assertEqual((r["card"], r["status"]), (name, "ok"), r)
                self.assertGreater(r["inliers"], scans.CHECK_BELOW)

    def test_something_that_is_no_card_is_rejected_and_nothing_is_stored(self):
        noise = np.random.default_rng(0).integers(0, 255, (600, 800), dtype=np.uint8)
        r = scans.add(scans.AUTO, png(noise), "noise.png")[0]
        self.assertEqual((r["status"], r["page"], r["card"]), ("rejected", None, None)); self.assertIn("closest is", r["reason"])
        self.assertFalse((cards.CARDS / "alpha").exists() or (cards.CARDS / "beta").exists())

    def test_a_scan_of_another_card_is_not_forced_onto_the_chosen_card(self):
        self.assertEqual(scans.add("alpha", self.scan("beta"), "b.png")[0]["status"], "rejected")

    def test_unknown_card_and_unreadable_file(self):
        with self.assertRaises(KeyError): scans.add("nosuch", self.scan("alpha"), "a.png")
        with self.assertRaises(KeyError): scans.add("anc", self.scan("alpha"), "a.png")
        with self.assertRaises(ValueError): scans.add("alpha", b"not an image", "a.png")


class Cut(Base):
    def test_cells_are_cut_and_only_written_cells_have_ink(self):
        r = scans.add("alpha", self.scan("alpha", cells=((0, 0), (2, 1), (4, 2))), "p-001.png")[0]
        self.assertEqual((r["page"], r["cells"], r["with_ink"]), ("p-001", ROWS * COLS, 3))
        cells = scans._read("alpha", "cells.csv", scans.CELL_COLS); rows = [f"f1_{i}" for i in range(ROWS)]
        self.assertEqual({(c["row"], c["col"]) for c in cells if c["ink"] == "1"}, {(rows[0], "c1"), (rows[2], "c2"), (rows[4], "c3")})
        for c in cells: self.assertTrue((cards.crops_dir("alpha") / c["row"] / c["col"] / "p-001.png").exists())
        g, k = cards.pack_grid("alpha"); al = cv2.imread(str(cards.aligned_dir("alpha") / "p-001.png"), 0)
        self.assertEqual([al.shape[1], al.shape[0]], g["image_size"]); self.assertGreaterEqual(k, 2)        # read at a higher resolution than the small template
        crop = cv2.imread(str(cards.crops_dir("alpha") / rows[0] / "c1" / "p-001.png"), 0)
        self.assertAlmostEqual(crop.shape[1], (120 - 2) * k + 2 * cards.CELL_MARGIN, delta=8)

    def test_same_file_again_replaces_and_same_name_with_other_content_gets_its_own_page(self):
        a = self.scan("alpha", seed=1); b = self.scan("alpha", cells=((1, 1),), seed=2)
        self.assertEqual(scans.add("alpha", a, "IMG 1.jpg")[0]["page"], "IMG-1")
        self.assertEqual(scans.add("alpha", a, "renamed.jpg")[0]["page"], "IMG-1")
        p2 = scans.add("alpha", b, "IMG 1.jpg")[0]["page"]
        self.assertRegex(p2, r"^IMG-1-[0-9a-f]{6}$")
        s = scans.summary("alpha"); self.assertEqual((s["totals"]["scans"], s["totals"]["ok"], s["totals"]["cells"], s["totals"]["with_ink"]), (2, 2, 2 * ROWS * COLS, 3))

    def test_labels_for_cut_cells_make_the_card_trainable(self):
        for i in range(cards.MIN_PAGES): scans.add("alpha", self.scan("alpha", seed=i), f"p-{i:03d}.png")
        d = cards.CARDS / "alpha"; (d / "fields.json").write_text(json.dumps({"fields": {"f1_0": {"kind": "number"}}}))
        (d / "labels.csv").write_text("page,row,col,label\n" + "".join(f"p-{i:03d},f1_0,c1,47\n" for i in range(cards.MIN_PAGES)))
        c = cards.describe("alpha"); self.assertEqual((c["trainable"], c["labels"], c["label_problems"]), (True, {"cells": cards.MIN_PAGES, "pages": cards.MIN_PAGES}, {}))
        self.assertEqual(scans.summary("alpha")["totals"]["labeled"], cards.MIN_PAGES)


class Lifecycle(Base):
    def test_recut_after_a_grid_edit_and_stale_after_a_new_template_image(self):
        scans.add("alpha", self.scan("alpha"), "p-001.png")
        d = cards.TEMPLATES / "alpha" / "v1"; g, _ = gridconv.layout_to_grid(self.card["alpha"][1], label_cols=1, data_cols=2); (d / "grid.json").write_text(json.dumps(g))
        import os; os.utime(d / "grid.json", (1, 1))                                   # make the edit visible whatever the clock resolution
        self.assertEqual(scans.summary("alpha")["needs_recut"], 1)
        self.assertEqual(scans.recut("alpha"), {"card": "alpha", "recut": 1, "stale": 0})
        s = scans.summary("alpha"); self.assertEqual((s["needs_recut"], s["totals"]["cells"]), (0, ROWS * 2))
        self.assertFalse(list(cards.crops_dir("alpha").glob("*/c3/p-001.png")))
        self.template("alpha", 1, version=2)
        self.assertEqual(scans.summary("alpha")["totals"]["stale"], 1)
        self.assertEqual(scans.recut("alpha")["stale"], 1); self.assertFalse(list(cards.crops_dir("alpha").glob("*/*/p-001.png")))

    def test_remove(self):
        scans.add("alpha", self.scan("alpha"), "p-001.png")
        self.assertEqual(scans.remove("alpha", "p-001"), {"card": "alpha", "removed": "p-001"})
        self.assertEqual(scans.summary("alpha")["totals"]["scans"], 0)
        self.assertFalse((cards.aligned_dir("alpha") / "p-001.png").exists() or list(cards.crops_dir("alpha").glob("*/*/p-001.png")))
        with self.assertRaises(KeyError): scans.remove("alpha", "p-001")
        with self.assertRaises(ValueError): scans.remove("alpha", "../x")

    def test_overlay_is_a_png(self):
        scans.add("alpha", self.scan("alpha"), "p-001.png")
        self.assertEqual(scans.overlay("alpha", "p-001")[:8], b"\x89PNG\r\n\x1a\n")
        with self.assertRaises(KeyError): scans.overlay("alpha", "p-999")


class Endpoint(Base):
    def setUp(self):
        super().setUp(); self.httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.H); self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        self.httpd.shutdown(); self.httpd.server_close(); super().tearDown()

    def call(self, method, path, body=b""):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        c.request(method, path, body=body, headers={"X-Card-Studio": "1"}); r = c.getresponse(); data = r.read(); c.close()
        return r.status, (json.loads(data) if r.getheader("Content-Type", "").startswith("application/json") else data)

    def test_upload_list_view_recut_delete(self):
        st, r = self.call("POST", "/api/scans?card=auto&name=p-001.png", self.scan("beta"))
        self.assertEqual((st, r["results"][0]["card"], r["results"][0]["status"]), (200, "beta", "ok"))
        st, s = self.call("GET", "/api/scans"); self.assertEqual((st, {x["card"]: x["totals"]["scans"] for x in s}), (200, {"alpha": 0, "beta": 1}))
        st, img = self.call("GET", "/api/card/beta/scan/p-001.png"); self.assertEqual((st, img[:4]), (200, b"\x89PNG"))
        self.assertEqual(self.call("GET", "/api/card/beta/scan/nope.png")[0], 404)
        self.assertEqual(self.call("POST", "/api/card/beta/recut", b"{}"), (200, {"card": "beta", "recut": 1, "stale": 0}))
        self.assertEqual(self.call("POST", "/api/card/beta/scan/delete", json.dumps({"page": "p-001"}).encode())[0], 200)
        self.assertEqual(self.call("POST", "/api/card/nosuch/recut", b"{}")[0], 404)
        self.assertEqual(self.call("POST", "/api/scans?card=../x&name=a.png", self.scan("beta"))[0], 400)
        self.assertEqual(self.call("POST", "/api/scans?card=alpha&name=a.png", b"junk")[0], 400)


if __name__ == "__main__":
    unittest.main()
