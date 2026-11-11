"""Card registry: field kinds -> class masks, per-card labels, pack built from a template. Synthetic data only. Run: python -m unittest discover -s tests"""
import csv, json, sys, tempfile, unittest
from pathlib import Path
import cv2, numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "cardpack"))
import cards, corrections, gridconv, server
import cardpack as cp
from test_gridconv import table, visit_table

ROWS = ["weight", "seen", "bp", "visit", "note", "row_06"]


class Kinds(unittest.TestCase):
    def test_kinds_give_the_masks_the_anc_reader_was_trained_with(self):
        m = cards.meta()
        for key, f in m["fields"].items():
            self.assertEqual(cards.allowed_idx(cards.META_KIND[f["type"]], m["classes"], m["blank_idx"]), sorted(f["allowed_idx"]), key)

    def test_unknown_kind(self):
        with self.assertRaises(ValueError): cards.allowed_idx("barcode", ["<blank>", "1"], 0)

    def test_anc_masks_are_model_meta(self):
        self.assertEqual(cards.masks("anc"), {k: sorted(v["allowed_idx"]) for k, v in cards.meta()["fields"].items()})


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); t = Path(self.tmp.name)
        self.old = (cards.TEMPLATES, cards.CARDS, cards.PACKS, corrections.IMPORTS)
        cards.TEMPLATES, cards.CARDS, cards.PACKS, corrections.IMPORTS = t / "templates", t / "cards", t / "out", t / "imports"
        cards.PACKS.mkdir()

    def tearDown(self):
        cards.TEMPLATES, cards.CARDS, cards.PACKS, corrections.IMPORTS = self.old; self.tmp.cleanup()

    def template(self, name="demo", grid=True):
        layout = table(rows=6, cols=3, names=ROWS[:5] + ["r6_c1"]); W, H = layout["image"]
        d = cards.TEMPLATES / name / "v1"; d.mkdir(parents=True)
        cv2.imwrite(str(d / "template.png"), np.full((H, W, 3), 240, np.uint8))
        (d / "layout.json").write_text(json.dumps(layout))
        if grid:
            g, _ = gridconv.layout_to_grid(layout, label_cols=1, data_cols=3); (d / "grid.json").write_text(json.dumps(g))

    def kinds(self, name="demo", **over):
        spec = {"weight": {"kind": "number", "range": [30, 150]}, "seen": {"kind": "mark"}, "bp": {"kind": "bp"}, "visit": {"kind": "date"}, **over}
        (cards.CARDS / name).mkdir(parents=True, exist_ok=True); (cards.CARDS / name / "fields.json").write_text(json.dumps({"fields": spec}))

    def label(self, rows, name="demo", crop=True):
        d = cards.CARDS / name; d.mkdir(parents=True, exist_ok=True)
        with open(d / "labels.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["page", "row", "col", "label"]); w.writerows(rows)
        for page, row, col, _ in rows if crop else []:
            p = d / "crops" / row / col / f"{page}.png"; p.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(p), np.full((40, 120), 255, np.uint8))


class Fields(Base):
    def test_rows_come_from_the_grid_and_default_to_text(self):
        self.template(); self.kinds()
        self.assertEqual([(f["key"], f["kind"], f["range"]) for f in cards.fields("demo")],
                         [("weight", "number", [30, 150]), ("seen", "mark", None), ("bp", "bp", None), ("visit", "date", None), ("note", "text", None), ("row_06", "text", None)])
        m = cards.meta(); self.assertEqual(len(cards.masks("demo")["note"]), len(m["classes"]))

    def test_bad_field_specs_are_refused(self):
        self.template()
        for over in ({"nosuchrow": {"kind": "mark"}}, {"seen": {"kind": "barcode"}}, {"seen": {"kind": "mark", "range": [1, 2]}}, {"weight": {"kind": "number", "range": [9, 3]}}):
            self.kinds(**over)
            with self.assertRaises(ValueError, msg=str(over)): cards.fields("demo")


class Labels(Base):
    def test_rows_that_cannot_be_trained_are_left_out_and_counted(self):
        self.template(); self.kinds()
        self.label([("p-001", "weight", "c1", "57.5"), ("p-001", "seen", "c2", "y"), ("p-002", "weight", "c1", "৫৭"), ("p-002", "seen", "c1", ""),
                    ("p-003", "weight", "c1", "5y"), ("p-003", "gone", "c1", "1"), ("p-003", "bp", "c0", "120/80")])
        (cards.CARDS / "demo" / "crops" / "seen" / "c2" / "p-001.png").unlink()
        rows, bad = cards.labels("demo")
        self.assertEqual([(r["page"], r["row"], r["label"]) for r in rows], [("p-001", "weight", "57.5"), ("p-002", "weight", "৫৭")])
        self.assertTrue(all(Path(r["crop"]).exists() for r in rows))
        self.assertEqual(bad, {"unknown_field": 1, "bad_cell": 1, "missing_crop": 1, "bad_characters": 1})

    def test_phone_corrections_go_to_the_card_their_pack_names(self):
        self.template(); self.kinds()
        from test_corrections import rec, zip_of
        corrections.import_phone_zip(zip_of([rec(1, pack_id="demo", field="weight", col="c1", label="61"), rec(2, pack_id="anc", field="pulse", col="c3", label="98")]))
        self.assertEqual([(r["row"], r["label"]) for r in cards.labels("demo")[0]], [("weight", "61")])
        self.assertEqual([(r["row"], r["label"]) for r in cards.labels("anc")[0] if r["page"].startswith("ph-")], [("pulse", "98")])


class Pack(Base):
    def test_pack_from_template_validates_once_a_reader_is_added(self):
        self.template(); self.kinds()
        mem = cards.base_members("demo"); man = json.loads(mem["manifest.json"])
        self.assertEqual((man["id"], man["version"], mem["reader.onnx"]), ("demo", 0, b""))
        man["version"] += 1; mem["reader.onnx"] = b"placeholder"; man["files"] = {n: cp.sha256(mem[n]) for n in cp.HASHED}; mem["manifest.json"] = json.dumps(man).encode()
        out = cards.PACKS / "demo_v1.cardpack"; cp.build(mem, out)
        self.assertEqual(cp.validate(out, check_reader=False), [])
        self.assertEqual(cards.pack_versions("demo"), [1])
        self.assertEqual(json.loads(cards.base_members("demo")["manifest.json"])["version"], 1)      # the next run numbers after it
        f = {x["key"]: x for x in json.loads(mem["fields.json"])["fields"]}
        self.assertEqual((f["weight"]["range"], f["seen"]["valid_values"], f["bp"]["bp"]["sys_above_dia"], f["note"]["kind"]), ([30, 150], ["y", "x", "-"], True, "text"))
        self.assertTrue(all(x["status"] == "needs-review" for x in f.values()))

    def test_a_card_with_one_row_per_visit_gets_a_ruled_rows_pack(self):
        layout = visit_table(); W, H = layout["image"]; d = cards.TEMPLATES / "visits" / "v1"; d.mkdir(parents=True)
        cv2.imwrite(str(d / "template.png"), np.full((H, W, 3), 240, np.uint8))
        g, _ = gridconv.layout_to_grid(layout, axis="columns", label_rows=1); (d / "grid.json").write_text(json.dumps(g))
        (cards.CARDS / "visits").mkdir(parents=True); (cards.CARDS / "visits" / "fields.json").write_text(json.dumps({"fields": {"date": {"kind": "date"}, "bp": {"kind": "bp"}, "weight": {"kind": "number"}}}))
        self.assertEqual([(f["key"], f["kind"]) for f in cards.fields("visits")], [("date", "date"), ("bp", "bp"), ("weight", "number"), ("note", "text")])
        self.assertEqual(cards.describe("visits")["grid"], {"rows": 5, "cols": 4, "axis": "columns", "fields": 4, "visits": 5})
        mem = cards.base_members("visits"); man = json.loads(mem["manifest.json"]); grid = json.loads(mem["grid.json"])
        self.assertEqual((man["grid"]["type"], grid["axis"], grid["cols"]), ("ruled-rows-v1", "columns", ["date", "bp", "weight", "note"]))
        def packed(man, grid):
            m = dict(mem); m["reader.onnx"] = b"placeholder"; m["grid.json"] = json.dumps(grid).encode()
            man = {**man, "version": 1}; man["files"] = {n: cp.sha256(m[n]) for n in cp.HASHED}; m["manifest.json"] = json.dumps(man).encode()
            out = cards.PACKS / "visits_v1.cardpack"; cp.build(m, out); return cp.validate(out, check_reader=False)
        self.assertEqual(packed(man, grid), [])
        self.assertTrue(any("does not agree" in e for e in packed({**man, "grid": {**man["grid"], "type": "ruled-columns-v1"}}, grid)))
        self.assertTrue(any("cols" in e for e in packed(man, {**grid, "cols": grid["cols"][:-1]})))

    def test_names_that_cannot_be_pack_ids(self):
        self.template("_odd")
        with self.assertRaises(ValueError): cards.base_members("_odd")
        self.assertIn("pack id", cards.describe("_odd")["blocker"])


class Readiness(Base):
    def test_blockers_in_order(self):
        self.template("nogrid", grid=False); self.template(); self.kinds()
        self.assertIn("no grid", cards.describe("nogrid")["blocker"])
        self.assertIn("0 labeled page", cards.describe("demo")["blocker"])
        self.label([(f"p-{i:03d}", "weight", "c1", "57") for i in range(cards.MIN_PAGES)])
        d = cards.describe("demo")
        self.assertEqual((d["trainable"], d["blocker"], d["labels"], d["grid"]), (True, "", {"cells": cards.MIN_PAGES, "pages": cards.MIN_PAGES}, {"rows": 6, "cols": 3, "axis": "rows", "fields": 6, "visits": 3}))
        self.assertEqual([c["id"] for c in cards.list_cards()], ["anc", "demo", "nogrid"])

    def test_server_refuses_to_train_a_card_that_is_not_ready(self):
        self.template()
        with self.assertRaises(ValueError) as e: server.start_training({"card": "demo"})
        self.assertIn("cannot be trained yet", str(e.exception))
        with self.assertRaises(ValueError): server.start_training({"card": "../x"})


if __name__ == "__main__":
    unittest.main()
