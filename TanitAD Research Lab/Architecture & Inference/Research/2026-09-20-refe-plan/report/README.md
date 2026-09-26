# REFe training report (live page)

Published privately at https://claude.ai/artifact/PfymZvPmm6Vu3ePiCdcufi. Rebuild:

1. Fetch from the pod into this folder: `metrics.jsonl`, `train.log`, `config.json` from
   `/workspace/data/refe_runs/vitl16_navtrain10_grow_tau0.3/` (scp, BatchMode).
   Also fetch `/workspace/data/refe_navtrain/train_grow/assembler.log` (the bank totals come from it).
2. `python build_data.py` -> `report_data.json` (training series, per-epoch table, bank events,
   and the learning-curve points in `D:/Projects/TanitAD/data/refe_navtest/points/`).
3. `python inject.py` -> `refe-training-run.html`; republish that file to the same artifact URL.

Data files and the built page are NOT committed; every number on the page comes from the pod's
`metrics.jsonl` / `train.log` / `config.json` and the eval point JSONs at build time.
