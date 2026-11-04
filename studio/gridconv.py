"""Studio layout (free rectangles from detect.py / the editor) -> cardpack `ruled-columns-v1` grid.

A Studio layout is a list of cell rectangles. A pack grid is a table: named rows x visit columns, with column lines at
the top and bottom of the table and row lines per column (so a slightly skewed card is still described). This module
finds the table in the rectangles and writes that description. Pure Python, no OpenCV.

Assumptions (stated so they can be checked, not hidden):
- the card is a ruled table, rows are horizontal bands, columns are vertical bands;
- `label_cols` leading columns are printed labels and `data_cols` columns after them are the visits;
- a cell left of the data columns that the user renamed (e.g. `weight_kg`) names its row, otherwise rows are row_01...
"""
import re
from statistics import median

ROWNAME = re.compile(r"^[A-Za-z0-9_]{1,40}$")
DEFAULT = re.compile(r"^r\d+_c\d+$")


# tolerances as fractions of the image size (6 / 3 / 8 / 5 px on a 1169 x 827 template), so they hold at any resolution (with small floors for tiny images)
X_FRAC, Y_FRAC, EDGE_FRAC, NEAR_FRAC = 6 / 1169, 3 / 827, 8 / 1169, 5 / 827
SLACK_FRAC = 0.0   # extra room for the global vote (0 = tight; looser values pulled stray cells in and merged real lines on the ANC card)
X_VOTE = 0.4     # a column line needs this fraction of the best-supported line's votes
LINE_VOTE = 0.34  # a row line needs votes from this fraction of the columns (tuned on the ANC card only; 0.5 missed its bottom line)


def _cells(layout):
    return [f for f in layout["fields"] if not f["name"].endswith("_box")]


def _cluster(vals, tol):
    """Group sorted 1-D values whose neighbours are within tol. Returns lists of values."""
    out = []
    for v in sorted(vals):
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return out


def _near(vals, target, tol):
    close = [v for v in vals if abs(v - target) <= tol]
    return median(close) if close else target


def layout_to_grid(layout, label_cols=0, data_cols=None, scale=1.0):
    """Returns (grid, warnings). Raises ValueError when no table can be found.

    Column lines are the x positions where many cells start or end; row lines are the y positions where cells in the
    chosen columns start or end (a line is the bottom of the cell above and the top of the cell below, so it is voted
    for twice per column). This tolerates merged cells and rows that detect.py split differently per column.
    label_cols: leading columns to skip (printed row labels). data_cols: keep only this many columns after that.
    """
    cells = _cells(layout)
    if len(cells) < 4:
        raise ValueError("too few cells to form a table")
    warnings = []
    W, H = layout["image"]
    X_TOL, Y_TOL, EDGE_TOL, NEAR_TOL = max(4, W * X_FRAC), max(3, H * Y_FRAC), max(5, W * EDGE_FRAC), max(4, H * NEAR_FRAC)
    xe = _cluster([c["x"] for c in cells] + [c["x"] + c["w"] for c in cells], X_TOL)
    thr = max(3, X_VOTE * max(len(c) for c in xe))
    xl = [median(c) for c in xe if len(c) >= thr]
    if len(xl) < 2:
        raise ValueError("no vertical table lines found")
    cols = list(zip(xl, xl[1:]))[label_cols:]
    if data_cols:
        cols = cols[:data_cols]
    if not cols:
        raise ValueError("no data columns left after label_cols/data_cols")
    x0, x1 = cols[0][0], cols[-1][1]
    slack = max(X_TOL, SLACK_FRAC * median(b - a for a, b in cols))   # a skewed card moves cell edges by tens of px, but never by half a column
    inside = [c for c in cells if c["x"] >= x0 - slack and c["x"] + c["w"] <= x1 + slack]
    ye = _cluster([v for c in inside for v in (c["y"], c["y"] + c["h"])], Y_TOL)
    yl = [median(c) for c in ye if len(c) >= max(2, round(LINE_VOTE * len(cols)))]
    if len(yl) < 3:
        raise ValueError("no horizontal row lines found in the data columns")
    nrows = len(yl) - 1
    # row names: a renamed cell to the left of the data columns, else row_01...
    left = [c for c in cells if c["x"] + c["w"] / 2 < x0 and ROWNAME.match(c["name"]) and not DEFAULT.match(c["name"])]
    names, seen = [], set()
    for i in range(nrows):
        hit = [c["name"] for c in left if yl[i] <= c["y"] + c["h"] / 2 <= yl[i + 1] and c["name"] not in seen]
        nm = hit[0] if hit else f"row_{i + 1:02d}"
        names.append(nm); seen.add(nm)
    # row lines per column (follows a skewed card): this column's own edges near each global line, within
    # almost half a row so a line is never matched to its neighbour
    NEAR_TOL = max(NEAR_TOL, 0.45 * median(b - a for a, b in zip(yl, yl[1:])))
    rl = []
    for (xa, xb) in cols:
        mine = [c for c in cells if xa <= c["x"] + c["w"] / 2 <= xb and c["w"] <= 1.4 * (xb - xa)]   # by centre, from all cells: skew can push edge cells out of the vote
        edges = [v for c in mine for v in (c["y"], c["y"] + c["h"])]
        col = []
        for i, y in enumerate(yl):
            v = _near(edges, y, NEAR_TOL)
            if col and v <= col[-1] + 1:      # two global lines picked the same edge: use the global line instead
                v = max(y, col[-1] + 1)
            col.append(v)
        rl.append(col)
    # column lines at the top and the bottom of the table (first and last row band)
    xs_all = [cols[0][0]] + [b for _, b in cols]
    def col_lines(i):
        band = [c for c in inside if yl[i] <= c["y"] + c["h"] / 2 <= yl[i + 1]]
        edges = [v for c in band for v in (c["x"], c["x"] + c["w"])]
        return [_near(edges, x, EDGE_TOL) for x in xs_all]
    top, bot = col_lines(0), col_lines(nrows - 1)
    f = lambda v: round(v * scale, 1)
    grid = {"image_size": [round(W * scale), round(H * scale)], "rows": names,
            "y_ref_top": f(rl[0][0]), "y_ref_bot": f(rl[0][-1]),
            "col_lines_top": [f(v) for v in top], "col_lines_bot": [f(v) for v in bot],
            "row_lines_by_col": [[f(v) for v in col] for col in rl]}
    thin = [i + 1 for i in range(nrows) if yl[i + 1] - yl[i] < 8]
    if thin:
        warnings.append(f"row(s) {thin} are under 8 px tall: probably a spurious line")
    return grid, warnings


def check_grid(grid):
    """The geometry rules cardpack.validate applies to grid.json (kept in step with it; see test_gridconv.py)."""
    errs = []
    inc = lambda a: all(b > c for c, b in zip(a, a[1:]))
    nrows, ncols = len(grid["rows"]), len(grid["col_lines_top"]) - 1
    if len(set(grid["rows"])) != nrows: errs.append("row names must be unique")
    if len(grid["col_lines_bot"]) != ncols + 1: errs.append("col_lines_top and col_lines_bot differ in length")
    if not inc(grid["col_lines_top"]) or not inc(grid["col_lines_bot"]): errs.append("column lines must increase left to right")
    if len(grid["row_lines_by_col"]) != ncols: errs.append("row_lines_by_col has the wrong number of columns")
    for c, rl in enumerate(grid["row_lines_by_col"]):
        if len(rl) != nrows + 1 or not inc(rl): errs.append(f"column {c + 1} needs {nrows + 1} increasing row lines")
    if grid["y_ref_bot"] <= grid["y_ref_top"]: errs.append("y_ref_bot must be below y_ref_top")
    W, H = grid["image_size"]
    if any(v < 0 or v > H for rl in grid["row_lines_by_col"] for v in rl): errs.append("row lines fall outside the image")
    if min(grid["col_lines_top"] + grid["col_lines_bot"]) < 0 or max(grid["col_lines_top"] + grid["col_lines_bot"]) > W: errs.append("column lines fall outside the image")
    return errs
