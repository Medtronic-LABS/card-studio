# Card OCR Platform

A local-first platform for reading **handwritten health cards**: teach it a card, measure how well it reads each field on pages it has not seen, and ship the result to a phone as a self-contained **card pack**.

Everything runs on one machine and binds to `127.0.0.1` only. There are no cloud OCR or LLM calls, and photos are never uploaded.

> **Privacy.** This repository contains **code and documentation only**. The system was built on real patient records, which are never committed: no scans, crops, labels, trained weights, packs, notebooks (their saved outputs hold card images), or PDFs. `tools/leakcheck.py` enforces that with an allowlist before every commit, and the `.gitignore` keeps your own data out if you run this on yours. The screenshots below show aggregate metrics only and are captured in a mode that never renders card images.

{{STATS}}

## Screenshots

{{SCREENSHOTS}}

## What is in it

| Piece | What it does |
|---|---|
| **Card Studio** (`studio/`) | A local web app: upload and identify a card, detect and edit its layout, label cells, train and gate a reader, compare runs, inspect model quality, and try photo clean-up options. Stdlib `http.server`, OpenCV, no framework. UI: [Tabler](https://tabler.io) (MIT), vendored, no CDN. |
| **Card pack format** (`cardpack/`) | One zip holding everything a phone needs to recognise and read one card type: template, grid, field rules, reader model, preprocessing. Versioned, hashed, validated. See [`cardpack/CARDPACK.md`](cardpack/CARDPACK.md). |
| **Training recipe** (`work_phaseA/`) | CRNN + CTC reader (about 538k parameters), page-level splits, field-masked decoding, Wilson intervals. |
| **Tools** (`tools/`) | The leak check, screenshot capture, README generator and publish script. |

Studio tabs: **Dashboard** (model registry, getting started, activity, what needs attention), **Label** (import corrected labels), **Train & Ship** (held-out gate per field), **History** (runs, paired comparison, experiments), **Model quality** (failure browser, calibration, risk-coverage, confusions, flag-policy tuner, provenance), **Clean-up lab** (try each photo clean-up on one photo).

## How "good" is decided

A field is shipped only if the **lower end of its 95% interval**, measured on pages the model never saw (pooled 5-fold cross-validation, split by page), meets a target with enough cells. Otherwise it stays **needs-review**. Only held-out numbers are called accuracy; numbers on training pages are labelled as such. Intervals are optimistic because cells on a page are correlated.

## Quick start

```sh
pip install numpy opencv-python onnxruntime pillow pypdfium2 jsonschema    # Studio
pip install torch                                              # only for training
python studio/server.py                                        # http://127.0.0.1:8770
```

Studio starts empty on a fresh clone: it needs a card pack in `cardpack/out/` and, for training, your own labeled crops. A small synthetic pack for tests lives in `cardpack/synthetic.py`.

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
- The trainer currently handles ANC-style ruled tables; other card types can be laid out and saved as templates but not yet trained.
- License: not chosen yet. Until the maintainers add a `LICENSE` file, all rights are reserved.
