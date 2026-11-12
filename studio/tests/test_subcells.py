"""Two-line cells: split into pieces, label parts, joining, training rows and the synthetic generator. Drawn cells only (no patient data). Run: python -m unittest discover -s tests"""
import csv, json, random, sys, tempfile, unittest
from pathlib import Path
import cv2, numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "cardpack"))
import cards, corrections, labeling, scans, subcells


def stacked_cell(h=120, w=200, tilt=0.0):
    c = np.full((h, w), 250, np.uint8); y = h // 2
    cv2.line(c, (10, int(y - tilt * 90)), (w - 10, int(y + tilt * 90)), 20, 3)                   # the fraction bar
    cv2.putText(c, "22.2", (40, y - 14), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 3); cv2.putText(c, "26", (70, y + 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 3)
    return c


def diagonal_cell(h=110, w=220):
    c = np.full((h, w), 250, np.uint8); cv2.line(c, (0, h - 1), (w - 1, 0), 20, 2)
    cv2.putText(c, "120", (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 3); cv2.putText(c, "80", (125, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 3)
    return c


def ink(im): return int((im < 128).sum())


class Geometry(unittest.TestCase):
    def test_stacked_pieces_hold_one_line_each_and_the_bar_goes_to_neither(self):
        for tilt in (0.0, 0.1, -0.1):
            top, bot = subcells.parts(stacked_cell(tilt=tilt), "stacked")
            bar = lambda x: (60 - tilt * 90) + tilt * (x - 10)             # where stacked_cell drew the bar
            yt, xt = np.where(top < 128); yb, xb = np.where(bot < 128)
            self.assertTrue((yt < bar(xt)).all() and (yb > bar(xb)).all(), f"tilt {tilt}: ink on the wrong side of the bar")
            self.assertGreater(ink(top), 150); self.assertGreater(ink(bot), 150)
            self.assertLess(max(ink(top), ink(bot)), 4 * min(ink(top), ink(bot)))        # neither piece swallowed the other line or the bar

    def test_no_bar_means_a_cut_through_the_middle(self):
        c = np.full((100, 160), 250, np.uint8); cv2.putText(c, "12", (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 3); cv2.putText(c, "34", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 3)
        top, bot = subcells.parts(c, "stacked"); self.assertGreater(ink(top), 100); self.assertGreater(ink(bot), 100)

    def test_diagonal_pieces_split_the_two_numbers_and_a_known_blank_removes_the_printed_line(self):
        cell = diagonal_cell(); blank = diagonal_cell(); blank[:] = 250; cv2.line(blank, (0, 109), (219, 0), 20, 2)
        for bl in (None, blank):
            a, b = subcells.parts(cell, "diagonal", bl)
            ya, xa = np.where(a < 128); yb, xb = np.where(b < 128)
            self.assertTrue(len(ya) > 100 and len(yb) > 100)
            self.assertLess(ya.mean(), 55); self.assertGreater(yb.mean(), 55); self.assertLess(xa.mean(), xb.mean())     # systolic above and left, diastolic below and right
        with_line = ink(subcells.parts(cell, "diagonal", None)[0]); without = ink(subcells.parts(cell, "diagonal", blank)[0])
        self.assertLess(without, with_line)

    def test_unknown_layout_and_tiny_cells(self):
        with self.assertRaises(ValueError): subcells.parts(stacked_cell(), "single")
        with self.assertRaises(ValueError): subcells.parts(np.full((5, 5), 250, np.uint8), "stacked")


class Labels(unittest.TestCase):
    def test_split_join_and_canonical(self):
        self.assertEqual(subcells.split_label("stacked", "22.2.26"), ("22.2", "26")); self.assertEqual(subcells.split_label("stacked", "5/11/2026"), ("5/11", "2026"))
        self.assertEqual(subcells.split_label("diagonal", "120/80"), ("120", "80"))
        for lay, bad in (("stacked", "22/2"), ("stacked", "-"), ("diagonal", "120"), ("diagonal", "1/2"), ("single", "1/2")): self.assertIsNone(subcells.split_label(lay, bad), bad)
        self.assertEqual(subcells.join("stacked", "22.2", "26"), "22/2/26"); self.assertEqual(subcells.join("diagonal", "120", "80"), "120/80"); self.assertEqual(subcells.join("diagonal", "", "80"), "")
        self.assertEqual(subcells.canonical("stacked", "22.2.26"), "22/2/26"); self.assertEqual(subcells.canonical("diagonal", "120/80"), "120/80")

    def test_layout_must_fit_the_kind(self):
        subcells.check_layout("date", "stacked"); subcells.check_layout("bp", "diagonal"); subcells.check_layout("number", "single")
        for kind, lay in (("number", "stacked"), ("date", "diagonal"), ("bp", "stacked"), ("date", "spiral")):
            with self.assertRaises(ValueError): subcells.check_layout(kind, lay)

    def test_piece_names_and_masks(self):
        self.assertEqual(subcells.piece_names("bp"), ["bp#1", "bp#2"]); self.assertEqual(subcells.base_key("bp#2"), ("bp", 2)); self.assertEqual(subcells.base_key("bp"), ("bp", None))
        m = cards.meta(); top = {m["classes"][i] for i in subcells.piece_masks("stacked", m["classes"], m["blank_idx"])["#1"]}; bot = {m["classes"][i] for i in subcells.piece_masks("stacked", m["classes"], m["blank_idx"])["#2"]}
        self.assertTrue({".", "/"} <= top and "." not in bot and set("0123456789") <= bot)

    def test_compose_scores_whole_cells_and_ignores_punctuation_style(self):
        preds = [{"page": "p1", "row": "d#1", "col": "c1", "label": "22.2", "pred": "22/2", "conf": 0.9}, {"page": "p1", "row": "d#2", "col": "c1", "label": "26", "pred": "26", "conf": 0.6},
                 {"page": "p2", "row": "d#1", "col": "c1", "label": "1.1", "pred": "1.1", "conf": 0.9}, {"page": "p1", "row": "w", "col": "c1", "label": "57", "pred": "57", "conf": 0.8}]
        out = subcells.compose(preds, {"d": "stacked"}, {("p1", "d", "c1"): "22.2.26", ("p2", "d", "c1"): "1.1.26"})
        got = {(p["page"], p["row"]): p for p in out}
        self.assertEqual(len(out), 2)                                                          # p2 has no second piece: left out, not guessed
        self.assertEqual((got[("p1", "d")]["pred"], got[("p1", "d")]["label"], got[("p1", "d")]["conf"]), ("22/2/26", "22/2/26", 0.6)); self.assertIn(("p1", "w"), got)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); t = Path(self.tmp.name)
        self.old = (cards.TEMPLATES, cards.CARDS, cards.PACKS, corrections.IMPORTS)
        cards.TEMPLATES, cards.CARDS, cards.PACKS, corrections.IMPORTS = t / "templates", t / "cards", t / "out", t / "imports"
        scans._targets.clear()

    def tearDown(self):
        cards.TEMPLATES, cards.CARDS, cards.PACKS, corrections.IMPORTS = self.old; scans._targets.clear(); self.tmp.cleanup()


class Fields(Base):
    def make(self):
        import gridconv
        from test_gridconv import table
        layout = table(rows=3, cols=2, names=["bp", "visit", "weight"]); W, H = layout["image"]; d = cards.TEMPLATES / "nc" / "v1"; d.mkdir(parents=True)
        cv2.imwrite(str(d / "template.png"), np.full((H, W, 3), 240, np.uint8)); (d / "layout.json").write_text(json.dumps(layout))
        g, _ = gridconv.layout_to_grid(layout, label_cols=1, data_cols=2); (d / "grid.json").write_text(json.dumps(g))

    def test_layout_is_saved_validated_and_kept_across_a_kind_edit(self):
        self.make()
        self.assertEqual(labeling.set_field("nc", "bp", "bp", None, "diagonal")["layout"], "diagonal")
        self.assertEqual({f["key"]: f["layout"] for f in cards.fields("nc")}["bp"], "diagonal")
        self.assertEqual(labeling.set_field("nc", "bp", "bp")["layout"], "diagonal")                   # editing the kind without saying layout keeps it
        self.assertEqual(labeling.set_field("nc", "bp", "date")["layout"], "single")                   # but not when it no longer fits
        with self.assertRaises(ValueError): labeling.set_field("nc", "weight", "number", None, "stacked")
        with self.assertRaises(ValueError): labeling.set_field("nc", "visit", "date", None, "diagonal")

    def test_masks_pack_fields_and_the_overview_know_about_pieces(self):
        self.make(); labeling.set_field("nc", "bp", "bp", None, "diagonal"); labeling.set_field("nc", "visit", "date", None, "stacked")
        m = cards.masks("nc"); self.assertTrue({"bp", "bp#1", "bp#2", "visit", "visit#1", "visit#2"} <= set(m)); self.assertNotIn("weight#1", m)
        mem = cards.base_members("nc"); fl = {f["key"]: f for f in json.loads(mem["fields.json"])["fields"]}
        self.assertEqual((fl["bp"].get("layout"), fl["visit"].get("layout"), "layout" in fl["weight"]), ("diagonal", "stacked", False))
        import cardpack as cp
        man = json.loads(mem["manifest.json"]); man["version"] = 1; mem["reader.onnx"] = b"x"; man["files"] = {n: cp.sha256(mem[n]) for n in cp.hashed(mem)}; mem["manifest.json"] = json.dumps(man).encode()
        out = cards.PACKS / "nc_v1.cardpack"; cards.PACKS.mkdir(); cp.build(mem, out); self.assertEqual(cp.validate(out, check_reader=False), [])
        bad = json.loads(mem["fields.json"]); bad["fields"][0]["layout"] = "spiral"; mem["fields.json"] = json.dumps(bad).encode()
        man["files"] = {n: cp.sha256(mem[n]) for n in cp.hashed(mem)}; mem["manifest.json"] = json.dumps(man).encode(); cp.build(mem, out)
        self.assertTrue(any("layout" in e for e in cp.validate(out, check_reader=False)))

    def test_a_label_that_does_not_split_draws_a_warning(self):
        self.make(); labeling.set_field("nc", "bp", "bp", None, "diagonal")
        d = cards.CARDS / "nc"; (d / "crops" / "bp" / "c1").mkdir(parents=True); cv2.imwrite(str(d / "crops" / "bp" / "c1" / "p-1.png"), diagonal_cell())
        (d / "cells.csv").write_text("page,row,col,ink\np-1,bp,c1,1\n")
        r = labeling.save("nc", "p-1", "bp", "c1", "120"); self.assertFalse(r["saved"]); self.assertIn("systolic/diastolic", r["warning"])
        self.assertTrue(labeling.save("nc", "p-1", "bp", "c1", "120/80")["saved"])

    def test_training_rows_become_pieces_and_odd_labels_are_counted(self):
        self.make(); labeling.set_field("nc", "bp", "bp", None, "diagonal"); d = cards.CARDS / "nc"
        for page, lab in (("p-1", "120/80"), ("p-2", "140/90"), ("p-3", "-")):
            p = d / "crops" / "bp" / "c1" / f"{page}.png"; p.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(p), diagonal_cell())
        p = d / "crops" / "weight" / "c1" / "p-1.png"; p.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(p), np.full((50, 100), 250, np.uint8))
        rows = [{"page": "p-1", "row": "bp", "col": "c1", "label": "120/80", "crop": str(d / "crops/bp/c1/p-1.png")}, {"page": "p-2", "row": "bp", "col": "c1", "label": "140/90", "crop": str(d / "crops/bp/c1/p-2.png")},
                {"page": "p-3", "row": "bp", "col": "c1", "label": "-", "crop": str(d / "crops/bp/c1/p-3.png")}, {"page": "p-1", "row": "weight", "col": "c1", "label": "57", "crop": str(d / "crops/weight/c1/p-1.png")}]
        dropped = {}; out, layouts = subcells.expand("nc", rows, dropped)
        self.assertEqual(layouts, {"bp": "diagonal"}); self.assertEqual(dropped, {"label_does_not_split": 1})
        self.assertEqual(sorted((r["page"], r["row"], r["label"]) for r in out), [("p-1", "bp#1", "120"), ("p-1", "bp#2", "80"), ("p-1", "weight", "57"), ("p-2", "bp#1", "140"), ("p-2", "bp#2", "90")])
        self.assertTrue(all(Path(r["crop"]).exists() for r in out))


class Synthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import synthgen
        cls.G = synthgen
        try: cls.pool = synthgen.load_pool()
        except OSError: raise unittest.SkipTest("needs the NumtaDB digits in data/bangla_digits")

    def bgs(self, layout):
        return [{"field": "f", "visit": "c1", "layout": layout, "kind": {"stacked": "date", "diagonal": "bp"}.get(layout, "number"), "image": np.full((100, 190), 248, np.uint8)}]

    def test_values_follow_the_grammars(self):
        rng = random.Random(1)
        for _ in range(200):
            lab, parts = self.G.value("date", rng); self.assertIsNotNone(subcells.split_label("stacked", lab)); self.assertEqual(parts[0].replace("|", "/") + lab[len(parts[0])] + parts[1], lab)
            lab, parts = self.G.value("bp", rng); self.assertIsNotNone(subcells.split_label("diagonal", lab))
            lab, _ = self.G.value("number", rng); float(lab)

    def test_samples_are_cells_and_pieces_carry_the_matching_labels(self):
        rng = random.Random(2)
        for kind, lay in (("number", "single"), ("date", "single"), ("bp", "single"), ("date", "stacked"), ("bp", "diagonal")):
            s = self.G.sample(rng, self.pool, self.bgs(lay), {(kind, lay): 1}); self.assertEqual(s["image"].shape, (100, 190)); self.assertLess(int(s["image"].min()), 120)      # something was written
            pcs = self.G.pieces(s); self.assertEqual(len(pcs), 1 if lay == "single" else 2)
            if lay != "single": self.assertEqual([p[1] for p in pcs], list(subcells.split_label(lay, s["label"]))); self.assertTrue(all(ink(p[0]) > 40 for p in pcs))

    def test_same_seed_same_cell_and_no_background_is_an_error(self):
        a = self.G.sample(random.Random(5), self.pool, self.bgs("single")); b = self.G.sample(random.Random(5), self.pool, self.bgs("single"))
        self.assertEqual((a["label"], a["image"].tobytes()), (b["label"], b["image"].tobytes()))
        with self.assertRaises(ValueError): self.G.sample(random.Random(5), self.pool, [])

    def test_shaded_backgrounds_are_brought_up_to_white_and_ghosts_are_left_out(self):
        grey = np.full((60, 120), 150, np.uint8); self.assertGreater(float(np.median(self.G._flatten(grey))), 235)
        clean = np.full((60, 120), 250, np.uint8); ghost = clean.copy(); ghost[10:50, 20:100] = 120
        cells = [{"field": "f", "visit": "c1", "image": clean}, {"field": "f", "visit": "c2", "image": ghost}]
        self.assertEqual([c["visit"] for c in self.G._clean_cells(cells)], ["c1"])
        self.assertEqual([c["visit"] for c in self.G._clean_cells(cells, {("f", "c1"): 0.0, ("f", "c2"): 0.9})], ["c1"])


if __name__ == "__main__":
    unittest.main()
