# READ_PLAN for PREREG_V8F_ACTRSP (agent 2026-11-16): what of the ACTSENS read runs unchanged, and what must be added

Tier T0 throughout (oracle goal: a world-model diagnostic, never driving performance). Dev-box CPU only, `CUDA_VISIBLE_DEVICES=-1`; box 2 is read only by sftp. Paths: ACTSENS read = `F:/TanitAD-scratch/actsens/read/` (**RD**), code of record = `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/` (read only). Nothing below was run; no file here exists yet except this plan.

## 1 Runs unchanged (same file, same sha, driven by env or arguments it already takes)
| script | role in ACTRSP | how it is driven |
|---|---|---|
| `RD/read_actsens.sh` | per-arm driver (IA-3, windows + C6, TRAIN latents, plans, contrasts, decode_check, assemble) | `ARM_RUN_DIR=C:/Users/Admin/t2v8f_pulls/actrsp_<tag>`, chassis argument 0 / 1; ONE change needed (section 2, item M1) |
| `RD/ia_read.py` | IA-3 / IA-0 / BAND-MET (code of record `ladder_read/run_ladder_read.py` + `lr_lib.py`) | `--run-dir`, `--chassis-seed`; it checks argv `--seed` against the name suffix, which holds for RSP_s0..s3 and NULL_s4 / s5 |
| `RD/c6_windows.py` | C6 WINDOWS-EQUAL against R35's phase-1 windows | `<win dir> <chassis> <out>` |
| `RD/collect_latents_arm.py`, `RD/make_collector_arm.py` | the arm's own TRAIN latents | `COL_*` env |
| `RD/decode_check_arm.py`, `RD/make_decode_check_arm.py` | off-image diagnostic | `DC_*` env |
| `RD/c7_checks.py`, `RD/c7_run.sh` | C7 REPRO (R35's 75 phase-1 rows) and SELF | unchanged |
| `RD/hf_fetch.py` | the R35 phase-1 reference tar (HF rev `aa2d8014`) | unchanged; already fetched under `RD/hf_phase1/` |
| `hwm_plan/run_hwm.py`, `F:/TanitAD-scratch/hwm_plan/phase2_l1full/contrast_l1full.py` | planning and the validated paired contrast | `--b` / `--rep-b` = the paired baseline's plans |
| `actsens/probe_actsens.py` Part A | L1 forward response and IDM R² (beside) | per checkpoint |
| `RD/c8_check.py`, `RD/c8_run.sh` | C8 ZERO-ACT on R35_s0 for IA-3 and Part A | unchanged; item N1 adds the IA-3P and TARGET-READ halves |

Banked and reused as read, never re-read: `RD/out/NULL_s2/`, `RD/out/NULL_s3/` (plans `win/plans_{hwm,flat}_oracle_s{0,1}.npz`, `ia_read.json`, `arm_NULL_s*.json`) and the R35 references.

## 2 Must change (small, each a copy with its own sha, the original kept)
* **M1, `read_actsens.sh` → `read_actrsp.sh`.** Step 5 hard-codes the baseline as `$R35REF/s$CH` (`--b`, `--rep-b`). Add `ARM_BASE_PLANS` (default `$R35REF/s$CH`; `RD/out/NULL_s2/win` for RSP_s2, `RD/out/NULL_s3/win` for RSP_s3) and record the baseline tag in the arm JSON. `--windows` stays R35's (C6 makes them equal).
* **M2, `env.sh` → `env_actrsp.sh`.** `ACT_TREE`, `WMACT_CODE_ROOT`, `LADDER_READ_CODE_ROOT` and `PYTHONPATH` point at the ACTRSP tree: an ACTRSP checkpoint's `cfg` carries `wm_rsp_w`, which the ACTSENS tree's config refuses (G-HYG). `ACT_COMMIT` = the ACTRSP commit. R35 / NULL checkpoints load under the ACTRSP tree through the field's default (C1).
* **M3, `check_tree.py` → `check_tree_actrsp.py`.** It asserts the ACTSENS tree's shas; the copy asserts the ACTRSP tree and the unchanged code-of-record shas.
* **M4, `assemble_arm.py` → `assemble_arm_rsp.py`.** The key `vs_r35` is the paired baseline for seeds 2 and 3, so the copy writes `vs_base` plus `base_tag`, and adds the IA-3P and TARGET-READ blocks (items N1, N2).

## 3 Must be added
| id | file (new, in `F:/TanitAD-scratch/actrsp/read/`) | what it computes | validation before any ACTRSP checkpoint |
|---|---|---|---|
| N1 | `ia3p_read.py` | IA-3P: IA-3's contrasts and rollouts (`lr_lib.encode_clip`, `roll_cells`, imported unchanged), decoded by a cross-fitted LINEAR probe on `[z_{h−1}, z_h]` → (acceleration, yaw rate), fitted on the real L1 pairs of the 141 eval clips with IA-3's folds; the probe's calibration slope on real pairs; raw and calibrated gains at L1 h = 1.0 s and 2.0 s, context 1 and built context; episode-cluster bootstrap over the 141 clips (IA-3's `boot_weights`) | card C10 on R35_s0 / R35_s1 (slope > 0.3 on both channels); C8 zero-act reads exactly 0.0; R35's raw IA-3P gain printed beside IA-3's |
| N2 | `target_read.py` | TARGET-READ: the arm's own cycle gain G (the trained quantity, its own IDM and slope) on the 141 eval clips, per entry of the 2 × 2 matrix | C8 zero-act reads exactly 0.0; on R35_s0 it runs (no term was trained there) and prints the flag-off value |
| N3 | `seedmean_contrast.py` | Δ̄_k: the paired episode-cluster bootstrap of the mean of the two seed contrasts at a chassis, from four plan files, through `hwm_eval.band_metrics` and `hwm_eval.contrast` as `contrast_l1full.py` uses them | with seed B's files equal to seed A's, it reproduces `contrast_l1full.py`'s rows exactly |
| N4 | `b_cal.py` | build control B-CAL on `F:/TanitAD-scratch/hwm_plan/latents_train_R35_s0.pt` (TRAIN only): linear probe R² from `z_t` alone vs from `[z_t, z_{t+1}]`, per channel | decides the slope rule before the first run (card section 6) |
| N5 | `rsp_w_from_log.py` | C2: the loss identity with the rsp term; recovers λ per logged step | on an ACTSENS NULL log: no `rsp/L1` key, identity closes; on a synthetic log with λ 1.0 it returns 1.0 |
| N6 | `c7b_null_repro.py` | C7 extension: NULL_s2 against R35_s0 and NULL_s3 against R35_s1, re-contrasted under the ACTRSP tree, reproduce `arm_NULL_s{2,3}.json`'s rows | exact |
| N7 | `verdict_actrsp.py` + `test_verdict_actrsp.py` | the card's labels. `verdict_actsens.py` cannot run unchanged: it hard-codes `LEVERS = ("IDM", "DIV")`, `NULLS = {0: "NULL_s2", 1: "NULL_s3"}`, a single-pair training floor and R35 as the only baseline (`verdict_actsens.py:16-18`, `:33-50`). The new one reads the pair table (RSP_s0↔R35_s0, RSP_s2↔NULL_s2, RSP_s1↔R35_s1, RSP_s3↔NULL_s3), F_T,k from the three flag-off arms, the sign rule, Δ̄_k from N3, MECH-MOVED from N1, HARM from BAND-MET, and the outcome table with its next steps | the `test_verdict_actsens.py` pattern: synthetic arm JSONs that must give each label, plus a deliberate regression per rule (a sign flip on one seed must give SEEDS-DISAGREE) |
| N8 | `families_table_rsp.py` | the four-family table (STRATEGIC cells s_adh / s_false with logged rates, plus the n/a rows, card section 4.3) for the six new arms, each cell against its pair, B / W marks against the three-arm floors | reproduces ACTSENS's table when fed ACTSENS's arms and floors |

## 4 Order and cost (dev-box CPU; ESTIMATED from ACTSENS rates)
1. Before any ACTRSP checkpoint exists: N4 (B-CAL), then N1 / N2 / N3 / N5 / N6 validations on R35 and the banked NULL arms. ESTIMATED 1 h.
2. Per banked arm, in chain order (RSP_s0, RSP_s2, NULL_s4, RSP_s1, RSP_s3, NULL_s5): `read_actrsp.sh` (about 2.0-2.2 h: HWM + FLAT at two inference seeds, IA-3 341 s, latents 405 s for two checkpoints, ACTSENS card section 2, INHERITED), then N1 + N2 (ESTIMATED 10-15 min). About 14 h for six arms, under the 6 GB free-RAM guard the driver already has.
3. After both chassis: N3 for each deciding row, then N7 and N8. Minutes.

## 5 Dependencies on the build (escalated, not done here)
* N1, N2 and every read of an ACTRSP checkpoint need the ACTRSP tree (a checkout of the landed ACTRSP commit, `git -c core.autocrlf=false` export, verified by blob hash).
* N2 needs `HJEPAStack.response_loss` from that tree (it reuses the trained term's own code, so the read and the loss cannot drift apart).
