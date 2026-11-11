"""Layout -> grid conversion. Run: python -m unittest discover -s tests"""
import json, sys, zipfile, unittest
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "cardpack"))
import gridconv as g

PACK = HERE.parent / "cardpack" / "out" / "anc_v4.cardpack"


def table(rows=6, cols=4, label=True, x0=100, y0=100, cw=80, rh=30, names=None):
    """A flat, regular ruled table as detect.py would return it (label column first)."""
    fields = []
    for r in range(rows):
        for c in range(cols + (1 if label else 0)):
            nm = f"r{r + 1}_c{c + 1}"
            if label and c == 0 and names: nm = names[r]
            fields.append({"id": f"f{len(fields)}", "name": nm, "kind": "text", "x": x0 + (c - (1 if label else 0)) * cw, "y": y0 + r * rh, "w": cw - 2, "h": rh - 2})
    return {"image": [x0 + cols * cw + 100, y0 + rows * rh + 100], "fields": fields}


class Flat(unittest.TestCase):
    def test_regular_table(self):
        grid, warn = g.layout_to_grid(table(), label_cols=1, data_cols=4)
        self.assertEqual(g.check_grid(grid), [])
        self.assertEqual((len(grid["rows"]), len(grid["col_lines_top"]) - 1), (6, 4))
        self.assertEqual(grid["col_lines_top"], grid["col_lines_bot"])          # no skew in a flat table
        self.assertAlmostEqual(grid["row_lines_by_col"][0][0], 100, delta=1)
        self.assertAlmostEqual(grid["row_lines_by_col"][0][-1], 100 + 6 * 30 - 2, delta=1)

    def test_row_names_from_renamed_label_cells(self):
        names = ["weight_kg", "pulse", "r3_c1", "bp", "hemoglobin", "pulse"]      # r3_c1 is default; last repeats
        grid, _ = g.layout_to_grid(table(names=names), label_cols=1, data_cols=4)
        self.assertEqual(grid["rows"], ["weight_kg", "pulse", "row_03", "bp", "hemoglobin", "row_06"])

    def test_scale(self):
        grid, _ = g.layout_to_grid(table(), label_cols=1, data_cols=4, scale=2.0)
        self.assertEqual(grid["image_size"], [table()["image"][0] * 2, table()["image"][1] * 2])
        self.assertAlmostEqual(grid["col_lines_top"][0], 2 * 100 - 0, delta=2)

    def test_marks_inside_cells_are_ignored(self):
        t = table(); t["fields"].append({"id": "m", "name": "r2_c3_box", "kind": "mark", "x": 200, "y": 135, "w": 20, "h": 20})
        grid, _ = g.layout_to_grid(t, label_cols=1, data_cols=4)
        self.assertEqual((len(grid["rows"]), len(grid["col_lines_top"]) - 1), (6, 4))

    def test_errors(self):
        with self.assertRaises(ValueError): g.layout_to_grid({"image": [100, 100], "fields": table(2, 1)["fields"][:3]})
        with self.assertRaises(ValueError): g.layout_to_grid(table(), label_cols=9)


def visit_table(fields=("date", "bp", "weight", "note"), visits=5):
    """A card with one row per visit: a heading row whose cells the user named, then the visit rows."""
    L = table(rows=visits + 1, cols=len(fields), label=False)
    for f in L["fields"]:
        r, c = (int(v[1:]) for v in f["name"].split("_"))
        if r == 1: f["name"] = fields[c - 1]
    return L


class FieldsDownColumns(unittest.TestCase):
    def test_columns_are_named_from_the_heading_row_and_rows_are_visits(self):
        grid, warn = g.layout_to_grid(visit_table(), axis="columns", label_rows=1)
        self.assertEqual(g.check_grid(grid), [])
        self.assertEqual((grid["axis"], grid["cols"], grid["rows"]), ("columns", ["date", "bp", "weight", "note"], ["r1", "r2", "r3", "r4", "r5"]))
        self.assertEqual(g.field_names(grid), grid["cols"])
        self.assertAlmostEqual(grid["row_lines_by_col"][0][0], 100 + 30 - 1, delta=2)       # the heading row is not a visit

    def test_unnamed_columns_and_row_limits(self):
        L = visit_table(); next(f for f in L["fields"] if f["name"] == "bp")["name"] = "r1_c2"
        grid, _ = g.layout_to_grid(L, axis="columns", label_rows=1, data_rows=3, label_cols=1)
        self.assertEqual((grid["cols"], grid["rows"]), (["col_01", "weight", "note"], ["r1", "r2", "r3"]))
        with self.assertRaises(ValueError): g.layout_to_grid(L, axis="columns", label_rows=9)
        with self.assertRaises(ValueError): g.layout_to_grid(L, axis="diagonal")

    def test_heading_rows_can_be_skipped_on_a_row_field_card_too(self):
        grid, _ = g.layout_to_grid(table(), label_cols=1, data_cols=4, label_rows=1, data_rows=4)
        self.assertEqual((len(grid["rows"]), "axis" in grid, g.field_names(grid)), (4, False, grid["rows"]))

    def test_checker_wants_one_unique_name_per_column(self):
        grid, _ = g.layout_to_grid(visit_table(), axis="columns", label_rows=1)
        self.assertTrue(g.check_grid({**grid, "cols": grid["cols"][:-1]})); self.assertTrue(g.check_grid({**grid, "cols": ["a", "a", "b", "c"]}))
        plain, _ = g.layout_to_grid(table(), label_cols=1); self.assertTrue(g.check_grid({**plain, "cols": ["a"]}))


class Checker(unittest.TestCase):
    def good(self):
        return g.layout_to_grid(table(), 1, 4)[0]

    def test_mutations_are_caught(self):
        base = self.good()
        for mut, want in ((lambda x: x["col_lines_top"].reverse(), "increase"),
                          (lambda x: x["rows"].append("extra"), "row lines"),
                          (lambda x: x.update(y_ref_bot=x["y_ref_top"]), "y_ref_bot"),
                          (lambda x: x.update(image_size=[50, 50]), "outside"),
                          (lambda x: x["rows"].__setitem__(1, x["rows"][0]), "unique")):
            d = json.loads(json.dumps(base)); mut(d)
            errs = g.check_grid(d)
            self.assertTrue(any(want in e for e in errs), (want, errs))


@unittest.skipUnless(PACK.exists(), "needs cardpack/out/anc_v4.cardpack")
class AgainstRealGrid(unittest.TestCase):
    def test_rebuild_the_anc_grid_from_cells_made_from_it(self):
        """Cut cells out of the pack's own grid (with its skew), convert back, compare."""
        ref = json.loads(zipfile.ZipFile(PACK).read("grid.json"))
        fields = []
        nrows, ncols = len(ref["rows"]), len(ref["col_lines_top"]) - 1
        xat = lambda c, y: ref["col_lines_top"][c] + (ref["col_lines_bot"][c] - ref["col_lines_top"][c]) * (y - ref["y_ref_top"]) / (ref["y_ref_bot"] - ref["y_ref_top"])
        for r in range(nrows):
            for c in range(ncols):
                ya, yb = ref["row_lines_by_col"][c][r] + 2, ref["row_lines_by_col"][c][r + 1] - 2
                ym = (ya + yb) / 2
                fields.append({"id": "x", "name": f"r{r + 1}_c{c + 1}", "kind": "text", "x": xat(c, ym) + 3, "y": ya, "w": xat(c + 1, ym) - xat(c, ym) - 6, "h": yb - ya})
        layout = {"image": ref["image_size"], "fields": fields}
        grid, warn = g.layout_to_grid(layout, label_cols=0, data_cols=ncols)
        self.assertEqual(g.check_grid(grid), [])
        self.assertEqual((len(grid["rows"]), len(grid["col_lines_top"]) - 1), (nrows, ncols))
        # every recovered cell centre lies inside the original cell (3% of its size)
        for c in range(ncols):
            for r in range(nrows):
                ya, yb = ref["row_lines_by_col"][c][r], ref["row_lines_by_col"][c][r + 1]
                ga, gb = grid["row_lines_by_col"][c][r], grid["row_lines_by_col"][c][r + 1]
                self.assertLess(abs((ga + gb) / 2 - (ya + yb) / 2), 0.03 * (yb - ya) + 2, (r, c))


if __name__ == "__main__":
    unittest.main()
