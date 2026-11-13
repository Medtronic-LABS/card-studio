# Card Studio

A local-first platform for reading **handwritten health cards**: teach it a card, measure how well it reads each field on pages it has not seen, and ship the result to a phone as a self-contained **card pack**.

Everything runs on one machine and binds to `127.0.0.1` only. There are no cloud OCR or LLM calls, and photos are never uploaded.

> **Privacy.** This repository contains **code and documentation only**. The system was built on real patient records, which are never committed: no scans, crops, labels, trained weights, packs, notebooks (their saved outputs hold card images), or PDFs. `tools/leakcheck.py` enforces that with an allowlist before every commit, and the `.gitignore` keeps your own data out if you run this on yours. The screenshots below show aggregate metrics only and are captured in a mode that never renders card images.

{{STATS}}

## Screenshots

{{SCREENSHOTS}}

## What is in it

| Piece | What it does |
|---|---|
| **Card Studio** (`studio/`) | A local web app: upload and identify a card, detect and edit its layout, add filled scans (each is matched to its template, straightened and cut into cells), label cells, train and gate a reader, compare runs, inspect model quality, and try photo clean-up options. Stdlib `http.server`, OpenCV, no framework. UI: [Tabler](https://tabler.io) (MIT), vendored, no CDN. |
| **Card pack format** (`cardpack/`) | One zip holding everything a phone needs to recognise and read one card type: template, grid, field rules, reader model (ONNX, and the same weights as TensorFlow Lite), preprocessing. Versioned, hashed, validated. See [`cardpack/CARDPACK.md`](cardpack/CARDPACK.md). |
| **Training recipe** (`work_phaseA/`) | CRNN + CTC reader (about 538k parameters), page-level splits, field-masked decoding, Wilson intervals. |
| **Tools** (`tools/`) | The leak check, screenshot capture, README generator and publish script. |

Studio tabs: **Dashboard** (model registry, getting started, activity, what needs attention), **Label** (set each field's kind, then type what each cut cell says, one cell at a time from the keyboard, with a trained reader's suggestion to confirm or correct once the card has one; optionally send the cells to a local [Label Studio](https://labelstud.io) and pull its labels back; import corrected labels), **Train & Ship** (choose the starting weights, held-out gate per field; what each field still needs to pass; a measured learning curve that says roughly how many labeled cells the target would take), **History** (runs, paired comparison, experiment records, and live progress and results of the experiments started from the command line), **Model quality** (failure browser, calibration, risk-coverage, confusions, flag-policy tuner, provenance), **Playground** (pick a card and any model: a pack's reader, a training run's reader or a registered outside model; drop a photo and see every cell with its crop, read and confidence; add a `.cardpack`, which brings its template and reader together), **Clean-up lab** (try each photo clean-up on one photo), **How to use** (these steps, and how the reader works).

## How to use

From a new card to a model on the phone. Start Studio with `python studio/server.py` and open http://127.0.0.1:8770. The same steps are on the **How to use** tab.

1. **Add the card type** (Cards tab). Under **Add a new card type**, drop one clean, flat picture of the card. Use the same kind of capture as the cards you will read later: a scan for scans, a phone photo for phone photos. Turn it the right way up, fix the detected cells, and give a name to the cell that labels each field (the label at the left of a row, or the heading above a column). Approve and save the template.
2. **Save its grid** (Cards tab). In **Layouts**, press **Grid**. Say whether each field is a row (a column per visit, like ANC and EPI cards) or a column (a row per visit, like the NCD card). Skip printed label columns and heading rows until the boxes in the preview sit exactly on the cells people write in, then save.
3. **Add filled cards** (Cards tab). In **Filled scans**, leave the card on Auto-identify and add your scans, photos or a PDF. Each one is matched to its template, straightened and cut into cells. Open **Scans** and press **View grid** on anything marked *check*. A scan that matches no template is refused, not guessed. Give each batch a **writer code** (W07 or a clinic id, never a name): with codes for at least five writers, training is tested on people it never saw.
4. **Set each field's kind** (Label tab). Pick the card, then choose a kind for every field: number, date, bp, mark or text, and an expected range for numbers. Do this before labeling: the kind decides which characters the reader may produce. A date written on two lines (day.month over the year) or a blood pressure either side of a printed diagonal has a **layout** to set as well: the cell is then split into two pieces, each read as one line. You still label the whole value, such as 22/2/26 or 120/80.
5. **Label the cells** (Label tab). Press **Label** on a field and type what each cell says. Enter saves and moves on, Ctrl+B means nothing is written, Ctrl+U means it cannot be read. Training needs labels on at least 10 pages; more pages and more writers give a better reader. Once the card has been trained, press **Read the cells with AI**: the card's own reader fills in what it is sure of and you only confirm or correct, and the queue can put the cells it is least sure of first. To share the work, send the cells to Label Studio from the strip below the table and pull the labels back.
6. **Train and check** (Train & Ship tab). Choose the card and press **Train + gate + export**. Every labeled cell is read once by a model that never saw its page. A field ships only if the low end of its 95% interval meets the target; the rest stay needs-review. Below it, **Is there enough labeled data?** says for each field whether it only needs more checked cells (and about how many) or is not accurate enough yet, and can measure a learning curve.
7. **Take the model** (Cards tab). In **Model registry**, press **View** on the new pack. Download the pack (template, grid, field rules, and the reader as ONNX and TensorFlow Lite), or only **reader.tflite**.
8. **Improve it** (Model quality tab). Look at which fields and which mistakes are worst, label more of those, train again, and compare the two runs in History.

From the command line: `python studio/scans.py add auto FOLDER` adds a large batch of scans, `python studio/labelstudio.py start` runs the optional Label Studio, and `python studio/tflite_export.py add PACK.cardpack` adds a TensorFlow Lite reader to an existing pack.

## Starting weights

A reader is pretrained on synthetic cells before it sees your labels. The choice is a dropdown on the Train & Ship tab and is recorded with every run: `legacy` (the original 4,000 stitched digit strings) or weights made by `python studio/pretrained.py make NAME` (synthetic cells drawn on the empty cells of the real cards: grammars, two-line layouts, damage). Held-out results on the ANC card (369 cells, page split) for the same reader: 72% with `legacy` weights against 77% with generator v2 weights (sign test p = 0.006). `python studio/pretrained.py list` shows what is registered; weights for a different class list, or whose file changed, are refused.

## How "good" is decided

A field is shipped only if the **lower end of its 95% interval**, measured on pages the model never saw (pooled 5-fold cross-validation, split by page), meets a target with enough cells. Otherwise it stays **needs-review**. Only held-out numbers are called accuracy; numbers on training pages are labelled as such. Intervals are optimistic because cells on a page are correlated.

## Quick start

```sh
pip install numpy opencv-python onnxruntime pillow pypdfium2 jsonschema    # Studio
pip install torch                                              # only for training
python -m venv studio/.venv-tflite && studio/.venv-tflite/bin/pip install tensorflow onnx onnxruntime numpy   # only for the TFLite reader
python studio/server.py                                        # http://127.0.0.1:8770
```

Studio starts empty on a fresh clone: follow How to use above with your own cards. A small synthetic pack for tests lives in `cardpack/synthetic.py`.

```sh
CARDPACK_SYNTHETIC=1 python cardpack/test_cardpack.py          # pack format
python -m unittest discover -s studio/tests                    # Studio backend
python -m unittest discover -s tools                           # leak check
python tools/leakcheck.py .                                    # nothing patient-like in this tree
```

## Layout

```
studio/        web app: server.py, trainer.py, detect.py, cleanup.py, mlstats.py, analysis.py, history.py, experiments.py, ... and tests/
cardpack/      card pack format: CARDPACK.md, schema/, cardpack.py (build, read, validate), synthetic.py
work_phaseA/   training recipe and evaluation scripts (code only)
tools/         leakcheck.py, screenshots.py, update_readme.py, publish.py
docs/          screenshots/ (registered in MANIFEST.json), status.json
```

## Keeping this page current

`python tools/publish.py` regenerates the numbers and screenshots from a running Studio, runs the leak check and the tests, then commits and pushes. Nothing is pushed if the leak check or a test fails.

## Limits

- Evidence comes from a small labeled set; most fields have under 40 held-out cells, so differences of a few points are noise.
- The phone app is not part of this repository. Phone-photo accuracy has not been measured on real labeled phone photos.
- A new card's pack copies the ANC card's matching and photo clean-up settings; they have not been tuned on other cards. Cells written on two lines (a date stacked over its year, a blood pressure split by a printed diagonal) are split into two pieces, each read as one line (`studio/subcells.py`); how well that reads has not been measured, because no card with such cells has labels yet.
- A grid can run either way: one field per row with a column per visit (`ruled-columns-v1`), or one visit per row with a column per field (`ruled-rows-v1`). Studio cuts, trains and packs both; the phone app only reads the first so far.
- Held-out accuracy is split by page unless writer codes are recorded (`studio/cards/<card>/writers.csv`); with codes for five or more writers it is split by writer. The ANC pages have no writer codes, so its figures are split by page and likely overstate accuracy on a new writer.
- A template should come from the same kind of capture as the scans it will match: a colour phone photo used as the template did not match photocopied scans of the same card.
- License: not chosen yet. Until the maintainers add a `LICENSE` file, all rights are reserved.
