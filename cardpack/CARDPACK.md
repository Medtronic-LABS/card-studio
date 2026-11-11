# Card pack format (`cardpack/1`)

A card pack is everything the phone needs to recognise one type of card and read it, in one file. Packs are independent: adding a card type means adding a pack, not changing app code.

A pack is a zip file named `<id>_v<version>.cardpack` containing exactly these six files and nothing else:

| File | What it is |
|---|---|
| `manifest.json` | identity, version, match settings, preprocessing settings, reader I/O contract, metrics, sha256 of every other file |
| `template.png` | the blank card the phone matches photos against (grayscale) |
| `grid.json` | row and column line positions in template space (`ruled-columns-v1`, or `ruled-rows-v1`: see Grid types) |
| `fields.json` | class list and, per field: kind, allowed classes, range, minimum confidence, ship / needs-review |
| `reader.onnx` | the recogniser (CRNN, input `1x1xHxW`, output `T x B x classes`, CTC) |
| `fallback.json` | when and how the slow second reader (Qwen) is used |

Schemas are in `schema/`. The validator is `cardpack.py`.

## Rules

- **Integrity.** `manifest.files` holds the sha256 of every file except the manifest. A pack with any mismatch, missing file, extra file, or an unsafe member name (for example `../x`) is rejected before use.
- **Cross-file consistency.** Field keys must equal the grid's field names (row names, or column names for `ruled-rows-v1`). `grid.image_size` must equal the template size times `template.grid_scale`. Grid lines must increase and stay inside the image. Allowed classes must exist in the class list and include the blank class. The reader must load and its output class count must equal the class list.
- **Ship needs evidence.** A field may be `ship` only if `manifest.metrics.heldout.per_field` has an entry for it (held-out count, correct count, 95% interval, from a page-disjoint split). With no held-out evidence every field is `needs-review`.
- **Unknown card.** `match.min_inliers` is the floor. Below it the app reports an unknown card type and must not read with this pack.
- **Determinism.** `cardpack.build` writes members sorted, with fixed timestamps, so unpacking and re-packing gives a byte-identical archive.
- **Privacy.** A template can contain handwriting. Packs stay on local machines and are gitignored (`cardpack/out/`).

## Grid types

A grid is a ruled table: column lines at the top and bottom of the table, and row lines for each column. What differs is which way the fields run.

| `manifest.grid.type` | `grid.json` | a field is | a visit is | example |
|---|---|---|---|---|
| `ruled-columns-v1` | `rows` = field keys | a row | a column | ANC, EPI |
| `ruled-rows-v1` | `axis: "columns"`, `cols` = field keys, `rows` = `r1`, `r2`, ... | a column | a row | NCD (one line per visit) |

Field keys must equal the grid's field names (`rows`, or `cols` for `ruled-rows-v1`). The cell geometry is the same in both. The phone app only reads `ruled-columns-v1` so far.

## Field kinds

`number` (optional inclusive `range`), `date` (D/M/YY or D.M.YY with a real day and month), `bp` (`sys`/`dia` ranges, optional systolic above diastolic), `mark` (`valid_values`, the only reads that are not flagged), `text` (no checks).

A cell is flagged for a person when its confidence is below the field's `min_confidence`, or when the read breaks the field's rules. Blank cells are never flagged.

## Photo normalization (`preprocess.normalize`, optional)

The reader takes raw gray levels, so a washed-out, shadowed or colour-cast phone photo can throw it off. A pack may say how the app should clean a photo up before cutting cells:

```json
"normalize": {"default": "contrast_match", "reference": {"mean": 227.01, "std": 59.53, "source": "gray mean/std over 50 scans the reader was trained on"}}
```

| method | what it does |
|---|---|
| `flatten` | divide by the blurred page background (`flatten_kernel`). The behaviour before this block existed, so **a pack without it means `flatten`**. |
| `contrast_match` | move the page's gray mean and std to `reference` (only over the part of the card the photo covered). Needs `reference`. |
| `cell_stretch` | stretch each cell between its own 2nd and 98th percentile (a cell with almost no range is left alone). |
| `flatten_cell_stretch` | flatten, then `cell_stretch`. |
| `none` | raw gray, exactly as trained. |

The app lets a person override the pack's default; an option the pack cannot honour (`contrast_match` without a reference) falls back to `flatten`.
CLAHE and a min-channel gray were tested and are not options: both were worse than doing nothing.

**Evidence and its limits.** Fixed `anc_v4` reader, colour aligned scans with simulated bad photos (warm cast, shadow, low contrast, all combined), cells cut with the recorded boxes. Choices made on 7 validation pages (57 cells), then confirmed once on the 7 test pages (62 cells, exact match, Wilson 95%):

| | clean scan | cast | shadow | low contrast | combined |
|---|---|---|---|---|---|
| raw gray | 58% | 58% | 39% | 55% | 21% |
| Reinhard to gray | 58% | 58% | 61% | 58% | 59% |
| gray + cell stretch | 57% | 55% | 60% | 59% | 49% |

`contrast_match` is the gray-level version of that Reinhard step; on validation it was within 2 points of full Reinhard in every condition. The shadow gain is likely but its intervals touch; the combined gain is clear. The degradations are simulated, not real phone photos; the intervals are 20+ points wide; the reader was not retrained with the preprocessing; held-out numbers in `metrics` describe the reader on gray input, not on any normalization. Reproduce with `work_phaseA/preproc_test.py`.

`python cardpack.py normalize PACK --version N --default METHOD --scans IMG...` writes a new pack version with this block (reader, grid, fields and metrics copied unchanged). Studio's trainer adds it automatically to packs it builds (`contrast_match`, reference from the aligned scans it trained on; `flatten` when fewer than 10 scans are available).

## Use

```sh
python export_anc.py                          # writes out/anc_v3.cardpack from the app's current model assets
python cardpack.py validate out/anc_v3.cardpack
python test_cardpack.py                       # round trip, parity with the app asset, and 10 rejection cases
```

`anc_v3` is the reader the app ships today. It has no held-out evaluation, so every field in it is `needs-review`.
