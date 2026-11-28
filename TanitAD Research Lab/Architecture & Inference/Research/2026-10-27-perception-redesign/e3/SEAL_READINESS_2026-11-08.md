# SEAL_READINESS - what remains before REFCV9_DESIGN section 5 can be sealed (2026-11-08, read from mirror tip 449eda31)

Nothing was changed, committed or run on Thor (only `ls` and one `md5sum`). Bars and values are untouched. Paths: PR = `TanitAD Research Lab/Architecture & Inference/Research/2026-10-27-perception-redesign`, EF = `FlyWheels/TanitAD_EvalFlyWheel/incoming`.

**Headline.** 16 items (W1 and W2 counted separately). 5 READY (K-BASE, FAM-1, MAP-5, W4, W5). 7 NEEDS-MEASUREMENT (MAP-1, MAP-4, BOX-1..4, FAM-2). 4 NEEDS-MM-DECISION (MAP-2, MAP-3, W1, W2; MAP-2 and MAP-3 also need a measurement after the decision). One INSTRUMENT-MISSING item inside MAP-1 and FAM-2: an interval for "separated below refcv8" on S6c-HOLD (`s6c_dev.py` emits a point rate only). No sealing item needs the Thor GPU; the only GPU item is refcv8 inference seed 1 for S6c (dev box). No E2 read tool has a HOLD switch: all filter to DEV and refuse HOLD by design.

## 1. Bar table
Status: R = READY, M = NEEDS-MEASUREMENT, D = NEEDS-MM-DECISION, I = INSTRUMENT-MISSING. "all" = all eval139 / all S6c. Costs are ESTIMATED unless stated.

| bar | reference named | artifact holding it | split of artifact | class | status | instrument and cost to get what is missing |
|---|---|---|---|---|---|---|
| K-BASE | registered PF table sha256 | `C:/Users/Admin/qland/hold_r8pf/REGISTERED_PF_TABLE.sha256` = `2aacc9ff...6d80` | n/a (train139 fit) | MEASURED, re-hashed today | **R** (section 2) | none |
| MAP-1 | 17.9 % = 0.17917, n 3198 | `EF/2026-10-25-refcv9-mapsel/result/s0/MAPSEL_RESULT.json` (map_compliance.S6c.BASE.off_drivable), seed 0 | all S6c | MEASURED file; INHERITED in the design | **M + I** | HOLD number: CPU, from `F:/refcv9_mapsel/trace_s0.npz` + `rast_s0` (4,754 files, both exist) (section 3). Interval vs refcv9: none in `stack/ops/r9e2_read/s6c_dev.py` (`build_flat`, point only). Seed-1 refcv8 trace: not found (only `trace_s0.npz`); a roll took 2,903 s on the dev-box GPU (E0d card, limit 1) |
| MAP-2 | 0.251 / 0.733 / 0.716 / 0.744 | design s1 table; source `H-R9-MAPFT-1` (register prose) | all eval139 | INHERITED | **D**, then M | MM names the instrument (sidewalk-only `nf_prec_*` vs union lift-valid `nf_swedge_prec_lv_*`, design instrument note) and the 5-20 m value (E0h: ~0.99 needed at recall 1). Then refcv8 on that instrument on HOLD: `stack/scripts/r9_e1_eval.py --rast-dir F:/refcv9_mapsel/rast_s0 --windows DEV_HOLD_SPLIT.json --split hold --lift-valid <sidecar>`, CPU, minutes. A lift-valid sidecar: UNVERIFIED |
| MAP-3 | E0d lane tol-IoU 0-20 m; boundary recall 0.29 | `PR/e0/raw/E0D_RESULT.json`, `E0D_TABLE.md`, `PR/e0/E0d_CARD.md` | **DEV only**, HOLD never opened | MEASURED | **D + M** | Lane 0-20 m 0.232 [0.153, 0.301] primary / 0.261 cross-fit; boundary recall 0-20 m 0.377 / 0.402. The 0.29 is the 0-60 m pool (card). MM pins lane definition (card: L-DIL) and band. HOLD read: `e0/e0d_score.py` refuses HOLD, needs a HOLD mode; thresholds for it are in the result JSON (`thresholds_fit_on_all_DEV_for_a_later_HOLD_read`). CPU, ~5 min extract (card limit 6). refcv9 must bank a lane probability plane (card limit 1) |
| MAP-4 | drivable-union IoU, E0d column 1 | same | DEV only | MEASURED | **M** | same HOLD mode and cost as MAP-3 |
| MAP-5 | structural identity (own flags on `unobs` = 0) | none needed | n/a | identity | **R** | checked on refcv9 rasters at the final read |
| BOX-1 | AP@2 m 0.328 [0.301, 0.354] | `EF/2026-10-21-refcv8-perc8/PERC8_EVAL.json`; Q3 `hold_perc_diag` `raw/RESULT_Q3.json` | all eval139 | MEASURED | **M** | refcv8 on HOLD: `r9_e1_eval.py --packs C:/Users/Admin/refcv8_pf/perc8/raw/A_nosync/perc_det.pkl` (58 MB, exists, 4,754 windows) `--windows DEV_HOLD_SPLIT.json --split hold --gate <fit on DEV>`. CPU, minutes. The P=R gate is fitted on DEV (`--gate` / `--gates-json`) |
| BOX-2 | 0.539 / 0.234 / 0.092 | same | all | MEASURED | **M** | same call, by-range rows |
| BOX-3 | 2.14 boxes/object | PERC8 (R8-PERC8-1) | all | INHERITED | **M** | same call (`boxes_per_object_noNMS_*`) |
| BOX-4 | 0.822 | Q3 | all | MEASURED | **M** | same call (`tp_class_acc`) |
| FAM-1 | refcv8 SHIP-R8 table, seeds 0 and 1 | `EF/2026-10-20-refcv8-perframe-final/raw/ship_r8/SHIP_R8_step48852ema_pf.json` (overall PASS, gains hash = registered) | all eval139 | MEASURED | **R** | none for the reference |
| FAM-2 | refcv8 map-compliance + distance-keeping | compliance: `MAPSEL_RESULT.json` BASE off_drivable 0.1792, edge_contact 0.2855, lane_cross; distance-keeping: **not found** (no banked refcv8 headway / time-gap / TTC table) | all S6c | MEASURED / none | **M** | refcv8 distkeep: `stack/scripts/r9_e2_distkeep.py` requires `--dev-split` (DEV filter) and a lead block on the 0.5 s grid (`build_lead_block_b1.py --dt 0.5 --k 4`, `E:/refcv8_eval_kit/tree_de4e15b/taniteval/tools/`); in the repo only 0.2 s `..._k30.npz` blocks found (shared instants 1.0 and 2.0 s only). CPU. Compliance on HOLD = the MAP-1 path |
| W1 / W2 | false / missed decels, "refcv8 + H2cn" | metric definition and a banked reference row: **not found** (probed the SHIP_R8 json and `GOALS_AND_CLAIMS.md`; hits are register prose, e.g. H-R9-SELCOR-1, H-R9-MDFIX-1) | ? | ? | **D** | MM names the metric file and the reference artifact; nearest instrument `EF/2026-10-15-refcv8-milestone-battery/code/brake_locus_r8.py` |
| W4 | camera-blind; no number | refcv8 reads PASS: `EF/2026-10-27-refcv9-anticheat/RESULT_S01.md` (seeds 0, 1) | all 4,754 windows | MEASURED | **R** | refcv9 side is an E3-final read |
| W5 | ego-only floor; no number | same file; floor is model-free, fitted on train139 | all | MEASURED | **R** | refcv9 side is an E3-final read |

## 2. K-BASE
The registered table is the per-frame RETIME table `FROZEN_GAINS_RETIME_train_pf.json` (9,374 B), a fit on train139 windows at EMA-30k (not 48,852), seeds 0+1 pooled: `EF/2026-10-20-refcv8-perframe-final/TRAINFIT_PF_FROZEN.md`. Produced by `code/chain_trainfit_pf.sh` / `trainfit_pf.py` with fit code `retime_trainfit_r8.py`. The 48,852 eval then ran `code/chain_final_pf_r8.sh` (tag step48852ema_pf, seeds 0 and 1, `--nav-feed perframe_full`); the verdict `raw/ship_r8/SHIP_R8_step48852ema_pf.json` is PASS with `amendment3_registration.ok true`.

Verified today (MEASURED): the registration file holds `2aacc9ff...6d80`; sha256 of `C:/Users/Admin/qland/hold_r8pf/stage/.../retime_trainfit_pf/FROZEN_GAINS_RETIME_train_pf.json` equals it; md5 of `E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt` = b4364029...; md5 of `E:/refcv8_eval_kit/data/refcv8_v9labels/release/v9_labels_eval139.npz` = 6b5c7f20... (= `EV9MD5` in the chain); the eval archive `C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf/` and the train tree `C:/Users/Admin/refcv8_pf/trainfit/step30000ema_train139_pf` exist.

"Reproduce" as the design words it is a hash check, and it passes. A numeric re-derivation (re-run the forward fit on the stored train tree, CPU) was not run (UNVERIFIED). MM decides which reading applies.

## 3. refcv8 on HOLD
No existing refcv8 read on HOLD for any bar. Every E2 tool is DEV-only on purpose: `stack/ops/r9e2_read/s6c_dev.py` (`make_load_trace` refuses any HOLD window), `stack/scripts/r9_e2_families.py` and `r9_e2_distkeep.py` (drop non-DEV; `--non-dev refuse`), `run_battery_r9_e2.py` / `mapsel_roll_r9_e2.py` (`make_dev_window_filter`), `r9e2_roll.sh` (`--split dev`). E0d never opened HOLD. Only `r9_e1_dump.py` and `r9_e1_eval.py` take `--split dev|hold`, so BOX-1..4 and MAP-2 on HOLD need no code change.

Smallest changes: (a) a HOLD copy of `s6c_dev.py` (flip `isdev` to `ishold` in the three wrappers, replace the trace guard by a filter, since the trace holds all windows) plus a paired interval on the S6c rate; (b) a HOLD mode in `e0d_score.py`. Split identity: Thor `/home/nvidia/r8ns/r9dump/DEV_HOLD_SPLIT.json` md5 = repo copy md5 = 9e0e2faf... (MEASURED).

## 4. What the E3 final read needs that does not exist (does not block launch)
1. refcv9 EMA-73,280 rolls on all eval139, S2+S6, seeds 0 and 1, without the DEV filter.
2. HOLD variants of the S6c roll/score and of distkeep (section 3).
3. A camera-blind roll of refcv9 over all 4,754 windows (`camblind_roll_r9_e2.py` is the E2 launcher; refcv8 took ~0.56 s/window on the dev box, refcv9 UNVERIFIED).
4. An E3 verdict judge for section 5: only `stack/scripts/r9_e3_make_argv.py` exists; `r9_e2_verdict.py` is the E2 judge (probed `ls-tree stack/scripts` for `e3`).
5. A 0.5 s lead block for FAM-2.

## 5. Order of work (items that block sealing; CPU unless stated)
1. MM decisions, zero cost: MAP-2 instrument and 5-20 m value; MAP-3 lane definition and band; W1/W2 metric and reference; meaning of "reproduce".
2. BOX-1..4 refcv8 on HOLD: one `r9_e1_eval.py` call, minutes, inputs exist.
3. MAP-2 refcv8 on HOLD after item 1 (find or build the lift-valid sidecar).
4. MAP-3/4 refcv8 on HOLD: HOLD mode for `e0d_score.py` + ~5 min extract.
5. MAP-1 / FAM-2 compliance on HOLD, seed 0: HOLD copy of `s6c_dev.py` + paired interval, tens of minutes.
6. FAM-2 distance-keeping reference: build the 0.5 s lead block, run distkeep on the refcv8 dump (needs an all/HOLD switch).
7. Only GPU item: refcv8 seed-1 S6c roll for MAP-1 / FAM-2 (~48 min dev-box GPU, from the 2,903 s seed-0 figure). Thor is not needed for any sealing item.

## Manifest
Created: `E:/Projects/TanitAD-artifacts/dagland/e3/SEAL_READINESS.md`. No script written. No temp data created by me (two background `find` probes wrote only to the harness task folder).

## MM correction (2026-11-08, after review)
W1 / W2 are NOT missing: the metric and reference are in `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-27-refcv9-warm/DRAFT_PREREG_REFCV9_WARM.md:53-57` and `lon_indicators` (`.../2026-10-25-refcv9-l2/code/l2_common.py:144`, used by `.../2026-10-26-refcv8-ship-r8-h2cn-factored/bars_4c.py:5`). Only MAP-1 names a refcv8 re-read on HOLD before sealing: the BOX, MAP-2, MAP-3 and MAP-4 HOLD reads in sections 1 and 5 above are final-read work, not sealing blockers. Decisions: `SEAL_DECISIONS_PRE_DATA_2026-11-08.md`.
