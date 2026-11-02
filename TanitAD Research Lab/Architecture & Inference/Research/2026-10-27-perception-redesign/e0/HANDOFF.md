# HANDOFF -- E0 probes (hold_r9_e0), 2026-11-02 ~22:00 Berlin

Nothing is staged or committed. Nothing was written to G:. Steering edits were landed separately (hold_r9_decisions = c62dce2); this hold only ADDS files under `e0/`.
⚠️ The E0h chain runs its scripts FROM this hold's `e0/` folder: do not delete or move the hold until the E0h marker below appears.

| item | state | where |
|---|---|---|
| E0i DEV/HOLD split | **DONE** (DEV 70 / HOLD 69 episodes; sha256 `ca5af020…db0a`) | `e0/DEV_HOLD_SPLIT.json`, `E0i_CARD.md` |
| E0b range census, E0c pitch/roll | **DONE** (Thor CPU, `ZZE0BCdone-0ZZ`) | `E0bc_CARD.md`, `raw/E0BC_RESULT.json`; Thor copy `/home/nvidia/r8ns/r9e0/E0BC_RESULT.json` |
| E0a over-fit gap | **COMPUTING** on the dev box (CPU) | `F:/r9e0/e0a/E0A_RESULT.json` appears when done; card `E0a_CARD.md` |
| E0h dose-response | **RUNNING** (10 runs, 4 at a time, ~15-20 min per wave while the CPU is shared) | `F:/r9e0/e0h/runs/<tag>/`, logs `F:/r9e0/logs/e0h_*.log`, marker `ZZE0Hdone-<n>-10ZZ` in `F:/r9e0/logs/e0h_chain.log` |
| E0d map re-score | **NOT STARTED** | see below |

## Next steps, in order
1. **E0a:** `ls F:/r9e0/e0a/E0A_RESULT.json`. If absent and no python process runs `e0a_gap.py`, re-run (command in `E0a_CARD.md`; packs: `train_diag.packs.pkl` md5 985537b0895ec5ea882836a3835f99a6, `perc_det.pkl` md5 a4f537a24c3641f59226140a27941c1b, both copied from Thor `perc8/out`). Read `K_REDERIVE_*`, `K_NULL_*` first (a failed control voids the gap), then `equal_count.all.gap_ci95` -> `A1_runs`. Fill the Result paragraph of `E0a_CARD.md`.
2. **E0h:** when the marker shows `ZZE0Hdone-10-10ZZ` (a smaller n = a failed run: read its `F:/r9e0/logs/e0h_<tag>.log`; re-running `bash run_e0h.sh` skips finished runs): `PYTHONPATH= python -X utf8 e0h_read.py --runs F:/r9e0/e0h/runs --out-dir <this e0 dir>`. Read the controls FIRST (K-P1, K-ALL, K-CAL; `VOID` true voids the curve), then `dose_curve`, `p_half`, `p_MAP1`. Replace the "Status" paragraph of `E0h_CARD.md` with the result; the number feeds MAP-2's 5-20 m value (design §5.2, provisional 0.85).
3. **E0d (not started):** re-score refcv8's banked map rasters (`F:/refcv9_mapsel/rast_s0/e###_w####.npz`: `psw pdrv pedge code gt`, 1000x600 at 0.1 m, 4,754 windows; refcv8 EMA-48,852) under the PUBLISHED protocol from `LIT_BEVMAP.md`: thin GT classes dilated to 0.5 m, per-class threshold fitted on a FIT split only, tolerance @0.3 m, range bands 0-20 / 20-40 / ... m. Use the E0h cache `F:/r9e0/e0h/gt_cache.npy` for GT planes. Controls required by the TanitAD probe rule: a constant-only map must read the no-information value; thresholds fit on FIT, scored on the other split (n and d printed). Tier-2 deliverable: one-page card + known-answer test. Decides the MAP-3 / MAP-4 baselines.

## Disclosures
* Two clip ids were printed into this agent's transcript by exploratory probes (a dict-key listing and a one-line cache dump); none was written to any file, and every artifact here carries counts, sha12 or distributions only (checked: no UUID pattern in `e0/`).
* E0c reads two statistics because the clip-mean one is contaminated by frame-tilt × yaw (tested); R10 is IN on both, but whether the egomotion pitch is physical is UNVERIFIED (`E0bc_CARD.md`).
