# HANDOFF -- trigger probe 4 (spatial readout), written 2026-10-29 ~05:00 by the Research Lab agent

Package: `E:/Projects/TanitAD-artifacts/dagland/trigger_probe4/` (P4).  Big arrays (held-out attention weights): `F:/TanitAD-scratch/trigger_probe4_scratch/`.  Python: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I`.  No git add / commit / push; nothing deleted.
Design frozen first: `PREREG_FROZEN.md` (sha256 a8f5d785...327f, `FROZEN_SHA.txt`); one post-freeze edit, `POSTFREEZE_CHANGES.md`.

## State
* Code + pytest done (`raw/pytest_output.txt`, synthetic only).  Known answer AUC 0.973, uniform-attention mutation 0.568.
* **RUNNING DETACHED** (WMI, cmd pid 33072 at launch, `chain4.sh`): waits for ALL FIVE probe-3 markers (`trigger_probe3/raw/CHAIN_c2band|mlpctl|evalmlp|mutate|pf.done`, gives up after 36 h), RAM guard >= 4 GB, then two per-population workers (3 threads each):
  decision arms G8att, G4att (3 seeds) -> controls C1/C2 (+ seeds 3-6 if needed) -> G8max, G4max; then strata + attention maps, eval-141, tables, pytest, manifest.  Resumable (per arm-seed banking).
* Progress: `raw/chain4.log` (START/END/rc per step), `raw/<step>.log`.  Completion marker `raw/CHAIN4.done` (content = names of failed steps; empty = none; `waiting-timeout` = probe 3 never finished).

## If the chain died (no python p4_run process, `CHAIN4.done` absent, probe-3 markers present)
`powershell -NoProfile -ExecutionPolicy Bypass -File launch_chain4.ps1` (re-launch is safe: finished arms are skipped).  Do NOT launch it twice while one is alive (check `raw/chain4.log` for a recent END line).

## EXACT NEXT STEPS for the finalising agent (fresh, Sonnet is enough; no model-turn polling)
1. Wait: `until [ -f raw/CHAIN4.done ]; do sleep 60; done` as a background task.
2. Read `raw/VERDICT.json` (`verdict`, `decision.cells`, `detail.<pop>.controls`, `detail.<pop>.input_check`, `eval141`) and `raw/tables4.txt` (every number of RESULTS.md).  Confirm `input_check.ok` is true for both populations; if false, say so in the headline.
3. Rewrite the PENDING sections of `RESULTS.md` from `tables4.txt`: verdict by the committed rule, the 4 decision cells with seed-spread clause and controls, the arm table (seeds and spread), controls, strata, attention maps (view `raw/attn_maps_<pop>_G8att.png` and `maps4_<pop>_descriptors.json`), eval-141 (or "NOT RUN"), the R-MAX beside arms.  A CONTROL-FAILED or missing cell is stated, never promoted.
4. `python -I make_manifest.py` (chain already ran it; re-run after RESULTS.md changes) -> `MANIFEST.txt`, `raw/INPUTS_SHA256.json`.  Pytest final output is `raw/pytest_output_final.txt`.
5. Escalate in the headline: nothing is merged; the package lives only on E: (+ F: scratch).  Master Mind lands it via the lander; register an `H-CRAWL-06` note only after the PI reads RESULTS.
