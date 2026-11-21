# BUILD_FAM_EXTRA -- W-FAM-STRAT1-TTC-TACCONF (MM ruling 12, `strat1/read/ckpt120/MM_RULINGS_CKPT120_READ.md:12`)

Built 2026-11-17. CPU only, zero training, no GPU, nothing re-dumped, no pinned evaluator or landed reader edited. Nothing committed or pushed.
Evidence class of every number below: MEASURED (ours; banked T1-kin dumps, artifact `F:/TanitAD-scratch/strat1/fam/out/RESULT_FAM_EXTRA.json`, 17 arm-dumps, 141 episodes, 2,399 windows each).

## 1. What exists (two probes each)

| item | where (file:line) | status |
|---|---|---|
| DK instrument: `distance_keeping` (headway / time-gap / min-TTC), `lead_viol`, `paired_lead_viol`, `lead_viol_verdict`, `path_arc_length` | `strat1/read/tree/taniteval/taniteval/lead_metrics.py:131, 610, 758, 873, 450` (sha256 equal to the OTBAND tree copy) | exists, landed; reused unedited |
| Lead join + per-arm DK-VIOL inside the OTBAND read: `join` (B1 lead block k30), `labels`, `arm_arrays` computing `_dk_viol` and `dk_hw_abs_err_m` | `strat1/read/theory/otband/read/read_otband.py:95, 112, 120-133`; extracted unedited by `theory/obd/read/obd_geom.py` (`build()`) | exists; the STRAT1 readers (`strat1_read.slim`) keep only the headway / time-gap ABSOLUTE ERROR cells and drop `_dk_viol`, so levels, TTC and DK-VIOL never printed |
| DK reader precedent for the 141-clip grid (v8F-mini) | `E:/Projects/TanitAD-artifacts/sfresh_inputs/code/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/code/v8f_dk_read.py` (613 lines; reads eval npz `pred`, not T1 dumps) | exists; same lead block and instrument, different input format |
| lead block | `E:/Projects/TanitAD-artifacts/dagoacc_t1/inputs/b1_eval_lead_block_k30.npz` (sha256 pinned in `read_otband.py:43`, re-checked on load) | exists |
| TAC instrument for T0 heads: `tactical_metrics` (macro-F1, confusion, 22-token F1) | `strat1/read/tree/stack/scripts/v8f_mini_eval.py:121`; its output is in each `out/t0/eval_*_real.json` under `raw.tactical` (lat / lon confusion, `goal_macro_f1`) | exists for T0 real nav only; no reader prints it, no paired CI, no floor |
| TAC from trajectories (executed vs human, 3-way lat / lon): `four_families.tactical_from_trajectory`; its text states it is NOT selected-vs-executed | `tree/taniteval/taniteval/four_families.py:~1330` | exists, not called by the STRAT1 readers |
| head outputs at every tick: `t_lat_logits` (5), `t_lon_logits` (7), `t_goal_prob` (22), `nav_in`, `tick`, `ci`, `t0` | `strat1/read/t1/ticks/ticks_STRAT1-<arm>_m10_causal_heads.npz` (written by `--dump-heads`, `run_dumps_strat1.py`) | present for all 17 dumps |
| lead / obstacle geometry, GT tactical class, executed manoeuvre inside the dumps | the `dump_*/ep*.npz` carry `g, cl, ol, fl, ha0, v0, ws, eid, clip_index` only (no lead field); lead geometry joins in from the B1 block by (clip, frame 2*ws), GT classes from `strat1/read/inputs/join_eval141.npz` | no re-dump needed |
| anchor / candidate selection | `four_families.tactical_goal` states the arm commits to ONE path per window | NOT COMPUTABLE for these arms |

## 2. What was built

* `F:/TanitAD-scratch/strat1/fam/fam_extra.py` (569 lines) -- `run()`; `dk_arm` (:132), `dk_pair` (:158), `human_control` (:186); `conf_block` (:240), `conf_pair` (:261), `goal_block` (:319), `selexe_lat` (:351), `tick0_heads` (:86), `v9_rows` (:103); `reader_rows` (:536). No new metric and no new bootstrap: DK is `lead_metrics`, intervals are `taniteval.ci` (`episode_cluster_bootstrap`, `paired_episode_cluster_bootstrap`, `stat_cluster_bootstrap`, `paired_stat_cluster_bootstrap`), B 2,000, seed 0.
  * DK family, per arm and band (operative 0-2 s, tactical 2-6 s, full 0-6 s): headway, time-gap, min-TTC means over LEAD windows with CIs, `n_closing` (TTC is censored at 30 s), `n_lead_lost`, DK-VIOL (n_viol, n_at_risk, share of at-risk with CI, lead-progress ratio, the instrument's verdict), the human reference, and paired arm deltas with the seed floor.
  * TAC family: (a) SEL: the head's argmax at the window's first tick vs the v9 label (lat 5-way, lon 7-way): confusion matrix, accuracy, kappa, macro-F1 over present classes, per-class recall, majority baseline; (b) GOAL: 22-token F1 and macro-F1, micro token accuracy, top-1 token in GT positives; (c) EXEC: manoeuvre read off the rolled plan vs the human path's (3-way lat / lon); (d) SELEXE: declared lateral decision vs executed manoeuvre (LANE_KEEP, TURN_L, TURN_R mapped, LANE_CHANGE excluded and counted) plus a descriptive 7x3 longitudinal cross-table.
  * Every block carries its tier, n, estimator and, for deltas, the floor F = |C0_rep - C0_s0| and an above-floor flag. Families are never pooled.
* `fam/test_fam_extra.py` -- 45 tests, all green (`python -m pytest test_fam_extra.py -q`, 10.5 s). Synthetic inputs reach every branch (lead present / absent / lost, all-ignored labels, absent class, unweighted goal token, missing tick-0 row, unlabelled windows). Closed-form known values: headway 38.0 m, time-gap 3.8 s, min-TTC 4.0 s; 3x3 confusion with accuracy 5/7, kappa 19/33, macro-F1 0.7; goal macro-F1 0.75. Zero controls: DK self-pair, class self-pair and goal self-pair read exactly 0.0 [0.0, 0.0]; the human path as plan reads n_viol 0 and progress 1.0. 29 formula mutations (accuracy denominator, kappa, expected agreement, recall, precision, F1, confusion orientation, ignore mask, goal F1 / fn / micro / top-1 / weighting, lateral mapping and agreement, item decode, DK band / human path / lead windows / lead lost, floor flag, control rule x2, tick-0 pick, v9 row and mask) each go RED.
* `fam/render_md.py` -- JSON to `out/RESULT_FAM_EXTRA.md`.

## 3. Controls on the run (all PASS, `problems: none`)

K-GRID (dump windows = the eval grid's); K-HUMAN (human path as plan: n_viol 0, ratio 1.0, three bands); K1 (self-pair exactly 0.0); K-T0: this module's confusion and goal code applied to the T0 evaluator's own heads reproduces the evaluator's own `raw.tactical` confusions and `goal_macro_f1` exactly (17 of 17 arms with a T0 read). Beside row (not a control): the T1 tick-0 heads differ from the T0 real-nav heads because T1 feeds dead-reckoned nav; both are printed.

## 4. Results (full 0-6 s band unless stated; MEASURED; tier T1-kin; paired episode-cluster bootstrap B 2,000 seed 0; floor F from one replicate pair)

| cell | result | evidence |
|---|---|---|
| DK, levels, human reference | headway 25.66 m, time-gap 4.00 s, min-TTC 19.79 s over 743 lead windows | `out/RESULT_FAM_EXTRA.json` families.LONGITUDINAL-DK |
| DK, C0 s0 / s1 / s10 / s11 / rep | headway 25.14 / 25.53 / 25.68 / 25.75 / 25.22 m; time-gap 3.83 / 3.94 / 4.00 / 3.98 / 3.75 s; n_closing 439 / 406 / 399 / 408 / 413 of about 720 | same |
| DK-VIOL, share of at-risk (n_at_risk 595) | C0 0.0571 / 0.0353 / 0.0471 / 0.0370 (rep 0.0655); S1 0.0639 / 0.0319 / 0.0387 / 0.0471; N1 0.0622 / 0.0353 / 0.0403 / 0.0555; S1d 0.0571 / 0.0353 / 0.0420 / 0.0571 | same |
| DK-VIOL, S1 - C0 at-risk delta | s0 +0.0067, s1 -0.0034, s10 -0.0084, s11 +0.0101, none separated; only s11 above F (F = 0.0084, a single replicate pair, smaller than the C0 seed-to-seed range 0.0319 to 0.0571) | pairs.S1_s*-C0_s*.DK |
| DK headway, S1 - C0 | s1 -0.28 m [-0.54, -0.04] separated and above F; s0 -0.05, s10 -0.02, s11 -0.21 not separated | same |
| DK lead-progress ratio (instrument's control, bar 0.90) | arms at seed 10 read 0.877-0.882 (verdict INADMISSIBLE-CRAWL for C0, N1, S1, S1d at s10) and N1_s1 0.899; the rest 0.902-0.933 | families.LONGITUDINAL-DK.*.full_0_6s.DK-VIOL.verdict |
| TAC SEL lat (5-way), C0 s0 / s1 | accuracy 0.949 / 0.959, macro-F1 0.816 / 0.862, TURN_L recall 0.81 / 0.89, TURN_R 0.77 / 0.82 (majority baseline macro-F1 0.317) | families.TACTICAL.C0_s*.TAC-SEL-LAT |
| TAC SEL lat, S1 s0 / s1 | accuracy 0.925 / 0.932, macro-F1 0.676 / 0.713, TURN_L recall 0.55 / 0.57, TURN_R 0.56 / 0.61 | same |
| TAC SEL lat, S1 - C0 (macro-F1) | s0 -0.140 [-0.240, -0.057], s1 -0.150 [-0.272, -0.059]; both separated and above F | pairs.S1_s0-C0_s0.TAC.TAC-SEL-LAT, pairs.S1_s1-C0_s1 |
| TAC SEL lon (7-way), C0 s0 / S1 s0 | accuracy 0.606 / 0.609, macro-F1 0.487 / 0.472; S1 - C0 not separated at s0 (-0.015), s1 -0.027 not separated | same |
| TAC GOAL, C0 s0 / s1 | macro-F1 0.248 / 0.272; micro token accuracy 0.957 / 0.959; top-1 token a GT positive 0.935 / 0.947 (n 2,173) | families.TACTICAL.*.TAC-GOAL |
| TAC GOAL, S1 - C0 (macro-F1) | s0 -0.036 [-0.097, 0.002] not separated; s1 -0.040 [-0.079, -0.008] separated | pairs |
| TAC EXEC lat / lon (3-way, plan vs human), C0 s0 | lat accuracy 0.795, kappa 0.593; lon accuracy 0.519, kappa 0.268 | families.TACTICAL.C0_s0.TAC-EXEC-* |
| TAC SELEXE lat (declared vs executed), S1 - C0 | s0 -0.026 [-0.046, -0.007], s1 -0.030 [-0.052, -0.008] | pairs |

Reading note: a separated interval from a one-seed arm is necessary, not sufficient; the estimator answers "another draw of EPISODES". The lateral-head drop of S1 against C0 repeats at both seeds 0 and 1; seeds 10 and 11 are in the JSON (pairs `S1_s10-C0_s10`, `S1_s11-C0_s11`) and the MM reads them against the card's seed rule.

## 5. NOT COMPUTABLE (with reason and n)

| item | reason | n |
|---|---|---|
| TAC anchor / candidate selection | these arms commit to ONE path per window: no candidate fan, no selector | 2,399 windows |
| TAC SELEXE longitudinal scalar | head vocabulary (HOLD, CREEP, STOP, FOLLOW, DECELERATE, ACCELERATE, KEEP) has no one-to-one map onto brake_stop / steady / accelerate; counts only (`TAC-SELEXE-LON`) | 2,240 labelled |
| DK on non-lead windows | NO_LEAD 552, NOT_STRAIGHT 931, NO_LABEL 173: counted, never scored as free flow | 1,656 windows |
| DK reaction of the lead to the arm | the lead follows its logged future (tier caveat, same as every T1-kin DK read) | n/a |
| lat classes LANE_CHANGE_L/R in SELEXE | no kinematic-gate class; excluded and counted per arm (`n_excluded_lane_change`) | per arm in JSON |
| TAC SEL lat/lon classes absent from the labels | no recall cell (LANE_CHANGE_L/R have 0 support in the eval labels) | support 0 |

## 6. How the readers call it (the MM wires it; nothing of the landed readers was touched)

```
import sys; sys.path.insert(0, "F:/TanitAD-scratch/strat1/fam")   # needs the strat1/read dir, torch and the e81bde01 tree on the path exactly as strat1_read.py has them
import fam_extra as FE
res = FE.run("F:/TanitAD-scratch/strat1/read/t1", out_dir, nb=2000, t0_dir="F:/TanitAD-scratch/strat1/read/out/t0")   # ~9.5 min on the dev box for 17 arms (CPU, 2 torch threads)
rows = FE.reader_rows(res, "S1_s0-C0_s0")          # (family, cell, delta, lo, hi, above_floor, n) tuples for the pair table
```
CLI: `python fam_extra.py --dumps <t1 dir> --t0 <out/t0> --out <dir> [--arms C0_s0,C0_rep,S1_s0] [--n-boot 2000]` (run `python F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 8 300` first). A new arm family or seed is picked up when its `dump_*`, `ticks_*` and `done_*` files exist (`arms_present`). For CKPT120 / S1d / N1 readers: replace the "not computable here" entry (`ckpt120/c120_read.py:386`) with the `families.LONGITUDINAL-DK` and `families.TACTICAL` blocks from this JSON, or print `reader_rows`. If a reader pins reader sha256s, add `fam_extra.py` (sha256 in `provenance.reader_sha256`).

## 7. Deliverable manifest

| artifact | where | copies |
|---|---|---|
| `fam_extra.py` | `F:/TanitAD-scratch/strat1/fam/fam_extra.py` | one place (F: scratch); not staged: the MM stages when wiring |
| `test_fam_extra.py` | `F:/TanitAD-scratch/strat1/fam/test_fam_extra.py` | one place |
| `render_md.py` | `F:/TanitAD-scratch/strat1/fam/render_md.py` | one place |
| `RESULT_FAM_EXTRA.json` (812 KB), `RESULT_FAM_EXTRA.md` (52 KB) | `F:/TanitAD-scratch/strat1/fam/out/` | one place |
| `BUILD_FAM_EXTRA.md`, `run_full.log` | `F:/TanitAD-scratch/strat1/fam/` | one place |

All five artifacts exist in ONE place (F: scratch): the MM lands them. Total written under `fam/`: about 1 MB.
