"""Training: one entry (`loop.train`) for every architecture the lab can build, length-aware (dynamic width, `input_lengths` from
`htr.data.dataset.collate`), masked by field kind (`htr.vocab`). `config.py` resolves a run config; `tracking.py` is the pure-Python
early-stop rule and the (torch) checkpoint writer, ported from `work_phaseA/train.py`'s `EarlyStop`/`Checkpointer`."""
