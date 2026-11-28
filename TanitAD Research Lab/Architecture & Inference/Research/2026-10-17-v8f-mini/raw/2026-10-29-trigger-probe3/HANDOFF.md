# HANDOFF -- trigger probe 3 (finer grid), written 2026-10-29 ~04:20 by the Research Lab agent

Package: `E:/Projects/TanitAD-artifacts/dagland/trigger_probe3/` (call it P3).  Big arrays: `F:/TanitAD-scratch/trigger_probe3_scratch/`.  No git add/commit/push was done; nothing deleted.
Card frozen first: `PREREG_FROZEN.md` (sha256 f73b8a5a...5728, `FROZEN_SHA.txt`).  Python: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I`.  CUDA hidden.

## DONE (banked)
1. PREREG_FROZEN.md + FROZEN_SHA.txt.  2. `G8_<pop>.npy` [12000,163840] fp16 on F: for all 24,000 TRAIN windows, gates PASSED (`raw/extract_g8_meta.json`: G4 vs cached pool4x10 rel L2 median 3.03 %, max 3.19 %; G4 == loop reference bit-for-bit, max abs diff 0.0).
3. Per-token PCA-16 readout fitted for both pops (`raw/pcaG_*.json/npz`, `S8_/S4_<pop>.npy` on F:).  4. P-STOP linear arms + C1/C3 controls (`raw/resg_pstop_linear.json`); P-STOP MLP G4+E, G8+E.  pytest 16 passed (`raw/pytest_output.txt`).

## RUNNING DETACHED (WMI; survive the agent; all resumable, per-arm banking; do NOT edit chain.sh while they run)
* chain A `chain.sh linear eval mutate` (P-STOP done; P-MOVE pca+linear; strata; eval-141 extract+linear; C5b mutation).  Markers `raw/CHAIN_linear.done`, `CHAIN_eval.done`, `CHAIN_mutate.done`.
* chain B `chain.sh mlp evalmlp mlpctl` (MLP G4+E,G8+E,G4,G8 both pops; eval-141 MLP; MLP C1 shuffles).  Markers `CHAIN_mlp.done`, `CHAIN_evalmlp.done`, `CHAIN_mlpctl.done`.
* chain C `chainC.sh` POST-FREEZE sensitivity arms (`postfreeze_g.py`; `CHAIN_pf.done`).  chain D `chainD.sh`: after `CHAIN_linear.done`, runs `c2band.py` (the PRE-REGISTERED conditional C2 seeds 3-6 for any arm whose seed-2 CI misses 0.5; P-STOP G8+E needs it: seed-2 0.5426 [0.5012, 0.5839], lower bound 0.0012 above 0.5), then `tables_g.py`; marker `CHAIN_c2band.done`.
* Progress: `raw/chain.log` (START/END/rc per step), `raw/<step>.log`.  If a step died (rc != 0, or no python run_g/pca_g/eval_g in the process list for a stage whose marker is absent) just re-run the same `bash chain.sh <stage>` via `launch_chain.ps1` logic (WMI) -- finished arms are skipped.

## EXACT NEXT STEPS for the finalising agent (fresh, Sonnet is enough)
1. Wait for all markers (do not poll by model turns: one background `until [ -f raw/CHAIN_c2band.done ] && [ -f raw/CHAIN_mlpctl.done ] && [ -f raw/CHAIN_evalmlp.done ] && [ -f raw/CHAIN_mutate.done ] && [ -f raw/CHAIN_pf.done ]; do sleep 30; done`).
2. `python -I tables_g.py` (writes `raw/tables3.txt`, `raw/VERDICT.json`, `readout_heatmap_G8.png`), `python -I tables_pf.py` (`raw/tables_pf.txt`), `python -I strata_g.py` only if the chain's strata step failed.
3. Read `raw/VERDICT.json["verdict"]` (committed rule, PREREG 4) and the per-cell `void` / `controls_unverified` flags; rewrite RESULTS.md sections marked PENDING from `tables3.txt`; if any cell is INCOMPLETE say so, never promote.
4. `python -I make_manifest.py` (MANIFEST.txt + raw/INPUTS_SHA256.json; hashes the 7.9 GB G8 files, about 2 min), `python -I -m pytest -q -p no:cacheprovider test_trigger_probe3.py > raw/pytest_output.txt`.
5. Escalate in the headline: nothing here is merged; the package lives only on E: (and F: scratch).  Register an `H-CRAWL-06` note only after the PI reads RESULTS.

## What the interim numbers say (P-STOP, fold-mean AUC, MEASURED; full set in `raw/tables3.txt` once produced)
G8+E - G4+E linear **-0.042 [-0.060, -0.022]** (G4+E 0.650, G8+E 0.609); MLP G4+E 0.631, G8+E 0.609.  The inner CV chose the grid-minimum C in most folds (edge of the frozen grid), post-freeze extended grid in chain C: two-stage PCA-64 readout G4w64+E 0.668, G8w64+E 0.660 (probe 1's whole-vector V+E: 0.680).

## Notes added at hand-off (04:26)
* `MANIFEST_interim.txt` = sha256 of the stable code/docs; `make_manifest.py` (step 4) replaces it with the full `MANIFEST.txt`.  `raw/G8_sha256.txt` = sha256 of the two G8 memmaps (P-STOP 59cd06e9...3b3d; P-MOVE line appears when its background hash ends).
* Post-freeze (chain C) so far: P-STOP PF:G4k4+E 0.676, PF:G8k4+E 0.656, PF:G4w64+E 0.668, PF:G8w64+E 0.660 -- flat readouts are feature-count limited; none shows an 8x20 gain.  They never enter the verdict.
* Memory: the box has ~5 GB available with 3 chains + MLP running; if `require_ram` (>= 3000 MB free) stalls a step for 15 min it exits and the chain moves on -- check `raw/chain.log` for rc != 0 and re-run that stage.
