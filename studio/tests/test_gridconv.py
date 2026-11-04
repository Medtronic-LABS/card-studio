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
