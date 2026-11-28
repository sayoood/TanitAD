# HGATE_RESULT: PREREG_V8F_HGATE executed exactly as written (2026-10-25)

Dev box, CPU only, ZERO training, no GPU, no box-2 access. Every number below is **MEASURED (ours)** from the artifact paths named; tier **T1-class open loop (own held outputs; not closed loop)**; estimator **paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0** unless a row says otherwise. The bootstrap answers 'would another draw of EPISODES say this?' only; training-seed variance enters only through the two seeds shown side by side (`H-ESTIM-SEED-1`); inference is deterministic (no inference-seed floor). `OUT` = `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/hgate`.

## 0. Outcome (the PREREG outcome table, applied verbatim)

**CONSISTENCY NOT RESTORED.** PRIMARY-C **fails at both seeds** (upper 95 % bound +0.0338 at s0 and +0.0404 at s1, against a bar of <= +0.02). PRIMARY-K **passes at both seeds** (ADE2 and ADE6 HG - B both negative and separated). The row 'CONSISTENCY NOT RESTORED' (PRIMARY-C fails) matches; 'GAIN LOST' requires PRIMARY-C to pass, so it does not apply.

**Committed next step, verbatim from the PREREG:** *T's own constraint predictions are not good enough to tell where the human constrains: the T-side ceiling again (`D-V8F-T-ATTRIB-1`). Next: no further post-processing arm. The lever moves to T, after MIXO and FLAT read.*

Read plainly, because the point estimate at s0 sits on the bar and a reader could take it for a near-pass: PRIMARY-C is a bar on the **upper 95 % bound**, not the point estimate. At s0 the point estimate is +0.0196 (under +0.02) but the bound is +0.0338; at s1 the point estimate is +0.0288 and the bound +0.0404. The bar is not moved.

What the gate bought and what it cost, against H (HBLEND), as MEASURED point estimates: CONSISTENCY(a) 'any' cost vs B fell from +0.1005 / +0.0754 (H) to +0.0196 / +0.0288 (HG), i.e. 80 % / 62 % of the cost removed (s0 / s1); the ADE gain over B shrank from ADE6 -0.2068 / -0.0697 to -0.0372 / -0.0267 (18 % / 38 % retained) and ADE2 from -0.1033 / -0.0777 to -0.0197 / -0.0214 (19 % / 28 % retained). HG is strictly between B and H on both axes: it is not a free win on either.

## 1. Step 1: the train-window GT constraint activity and the re-fit

* **Source.** The box-2 label join `join_train.npz` (the trainer's own `--join-train`, sha256 `978f41bf4f1e4c8b...`), **already on the dev box** at `hblend/box2/join_train.npz` (it carries `goal_y/goal_w`, the 26 constraint cells, `lead_valid` and `lead_gap` for all 4,572 train clips). The GT constraint activity of the training windows is therefore derivable from what is on the dev box, so **no box-2 `scp` and no box-2 access of any kind was made**.
* **Definition.** 'Active' is the eval's own: `gt_constraints(X)` + `v8f_consistency.violations(...)[rule]['active']` (hblend/deps copy of `taniteval`), applied to each fit window's join row (`off[ci] + t`) with its `lead_gap_h` (`lead_gap[off + min(t+1+0..29, T-1)]`) and `s_h` (the data module's own `human_arc` on the cached `pose_grid`). STOP = STOP_POINT bit with weight, finite stop distance, `lon_t_reach_s` <= 6. TURN = TURN_L/R bit with weight and >= 1 s overlap with the plan horizon. LEAD = `lead_valid` and a finite lead gap at some knot >= 10. SPEED is excluded (active on every window).
* **Counts (HBLEND fit set: 9,582 windows, 141 clips, k = 32; the window set is identical for both seeds: True).** Active: STOP 344 (3.6 %), TURN 786 (8.2 %), LEAD 3,350 (35.0 %), SPEED 9,582 (100.0 %). Windows with any of STOP/TURN/LEAD active: 4,041. **Windows with none active (the fit set of w^G): 5,541 (57.8 %), at BOTH seeds** (s0 5,541, s1 5,541; 107 clips). Windows without a label row: 0. Reference, eval-141 (same code): STOP 81, TURN 158, LEAD 1027 (LEAD 1,027 matches HPROJ), SPEED 2,399 of 2,399.
* **Controls on the activity code** (`train_activity.json::controls`, all pass = True): (A) `s_h` from the cached pose grid vs `s_h` re-derived from the GT waypoints, max |d| = 1.52e-05 m; (B) the masks are identical for three different plans (GT future, B's plan, P10): True; (C) the SAME builder on the 2,399 eval windows reproduces every input field of the eval's context and the eval's own STOP/TURN/LEAD/SPEED masks exactly; (D) mutation: removing the lead-gap data changes the LEAD mask (True), so control (C) can fail.
* **The fit.** HBLEND's `fit_weights` (closed form per step and axis, clip [0, 1], unweighted PAV), imported, not re-implemented, on the 5,541 inactive windows, per seed from its own checkpoint's outputs (`fit_windows_M2b_s*.npz`).

| quantity | s0 | s1 |
|---|---|---|
| fit windows (n) | 5,541 of 9,582 | 5,541 of 9,582 |
| cells clipped at 0 / at 1 (of 60) | 4 / 2 | 5 / 2 |
| cells changed by isotonic | 38 | 37 |
| w^G in [0, 1] and non-decreasing | True | True |
| mean abs (w^G - banked w) over the 60 cells | 0.0326 | 0.0380 |
| max abs (w^G - banked w) | 0.1005 | 0.1300 |
| mean signed (w^G - w), along / cross | -0.0013 / -0.0528 | +0.0216 / -0.0528 |
| descriptive: same-size RANDOM subset (50 draws), mean abs dev from banked w: mean / max over draws | 0.0109 / 0.0237 | 0.0110 / 0.0227 |

The restriction moves the weights by 3.0x / 3.5x (s0 / s1) the mean deviation of an equal-size random subset, and beyond the largest deviation of any of the 50 draws (True / True), so the constraint-aware fit is a different set of 60 numbers from HBLEND's, not sampling noise around them (descriptive). Whether that difference matters for the metrics is read in section 8, item 5.

w^G (the entire fitted artifact: 60 numbers per seed); banked HBLEND w beside it. `OUT/weights_G_M2b_s{0,1}.json` (raw, clipped, final).

| t (s) | s0 along w^G / w | s0 cross w^G / w | s1 along w^G / w | s1 cross w^G / w |
|---|---|---|---|---|
| 0.2 | 0.000 / 0.000 | 0.650 / 0.656 | 0.000 / 0.000 | 0.607 / 0.609 |
| 0.4 | 0.012 / 0.035 | 0.650 / 0.656 | 0.000 / 0.016 | 0.607 / 0.609 |
| 0.6 | 0.137 / 0.115 | 0.650 / 0.670 | 0.154 / 0.138 | 0.629 / 0.641 |
| 0.8 | 0.137 / 0.115 | 0.650 / 0.670 | 0.154 / 0.138 | 0.629 / 0.641 |
| 1.0 | 0.137 / 0.120 | 0.650 / 0.670 | 0.154 / 0.138 | 0.629 / 0.641 |
| 1.2 | 0.137 / 0.120 | 0.650 / 0.670 | 0.154 / 0.138 | 0.629 / 0.641 |
| 1.4 | 0.137 / 0.120 | 0.650 / 0.670 | 0.154 / 0.138 | 0.629 / 0.641 |
| 1.6 | 0.137 / 0.120 | 0.650 / 0.670 | 0.154 / 0.138 | 0.629 / 0.641 |
| 1.8 | 0.137 / 0.120 | 0.650 / 0.670 | 0.154 / 0.138 | 0.629 / 0.641 |
| 2.0 | 0.137 / 0.120 | 0.733 / 0.749 | 0.154 / 0.138 | 0.686 / 0.700 |
| 2.2 | 0.194 / 0.178 | 0.784 / 0.802 | 0.154 / 0.141 | 0.712 / 0.729 |
| 2.4 | 0.312 / 0.311 | 0.803 / 0.841 | 0.268 / 0.270 | 0.732 / 0.748 |
| 2.6 | 0.423 / 0.429 | 0.803 / 0.861 | 0.432 / 0.439 | 0.739 / 0.763 |
| 2.8 | 0.492 / 0.498 | 0.803 / 0.869 | 0.593 / 0.588 | 0.739 / 0.772 |
| 3.0 | 0.532 / 0.534 | 0.803 / 0.873 | 0.715 / 0.690 | 0.739 / 0.779 |
| 3.2 | 0.555 / 0.555 | 0.803 / 0.875 | 0.793 / 0.754 | 0.739 / 0.787 |
| 3.4 | 0.567 / 0.568 | 0.803 / 0.875 | 0.834 / 0.790 | 0.739 / 0.795 |
| 3.6 | 0.571 / 0.576 | 0.803 / 0.875 | 0.852 / 0.808 | 0.739 / 0.801 |
| 3.8 | 0.575 / 0.583 | 0.803 / 0.875 | 0.860 / 0.817 | 0.739 / 0.803 |
| 4.0 | 0.581 / 0.591 | 0.803 / 0.875 | 0.864 / 0.822 | 0.739 / 0.806 |
| 4.2 | 0.589 / 0.601 | 0.803 / 0.875 | 0.867 / 0.828 | 0.739 / 0.812 |
| 4.4 | 0.595 / 0.609 | 0.803 / 0.875 | 0.870 / 0.834 | 0.739 / 0.819 |
| 4.6 | 0.597 / 0.613 | 0.803 / 0.875 | 0.871 / 0.838 | 0.739 / 0.827 |
| 4.8 | 0.601 / 0.617 | 0.803 / 0.876 | 0.873 / 0.843 | 0.739 / 0.835 |
| 5.0 | 0.611 / 0.627 | 0.803 / 0.881 | 0.878 / 0.849 | 0.739 / 0.846 |
| 5.2 | 0.622 / 0.636 | 0.803 / 0.886 | 0.882 / 0.857 | 0.740 / 0.856 |
| 5.4 | 0.622 / 0.636 | 0.803 / 0.886 | 0.882 / 0.858 | 0.741 / 0.862 |
| 5.6 | 0.622 / 0.636 | 0.803 / 0.886 | 0.882 / 0.858 | 0.741 / 0.864 |
| 5.8 | 0.622 / 0.636 | 0.803 / 0.887 | 0.882 / 0.858 | 0.742 / 0.868 |
| 6.0 | 0.630 / 0.641 | 0.806 / 0.906 | 0.898 / 0.876 | 0.757 / 0.887 |

## 2. Identity controls (run BEFORE any HG read; the script refuses to read, rc 2, if any fails)

| control | bar | s0 | s1 |
|---|---|---|---|
| (i) gate forced CLOSED everywhere: max abs diff of the plan vs B's banked `pred` (m) | = 0.0 exactly | 0.0 | 0.0 |
| (i) ... per-window ADE6 max abs diff vs banked | - | 0.0 | 0.0 |
| (i) ... the eval's own `evaluate()` block (traj, consistency_a, strata, tactical, self_consistency_b, families) vs the banked JSON `raw`: mismatches | 0 | 0 | 0 |
| (ii) gate forced OPEN with HBLEND's banked w: max abs diff vs HBLEND's saved H (m) | = 0.0 exactly | 0.0 | 0.0 |
| (ii) ... P recomputed from the checkpoint vs HBLEND's saved P (m) | - | 0.0 | 0.0 |
| (ii) ... `evaluate()` block vs HBLEND's saved H block: mismatches | 0 | 0 | 0 |
| (iii) the fit on ALL windows vs HBLEND's banked weights: max abs diff | <= 1e-12 | 0.0e+00 | 0.0e+00 |
| R1 deliberate regression: gate OPEN with w^G instead of the banked w, max abs vs saved H (m) | must be > 0 (the control can fail) | 4.113 | 5.412 |
| R2 deliberate regression: the real gate vs B, windows that differ | must be > 0 | 945 | 1291 |
| banked `pred` == `traj30` | True | True | True |
| checkpoint's embedded P10 vs the P10-yaw-4512 file, max abs (m) | <= 1e-4 | 1.53e-05 | 1.53e-05 |
| w^G in [0, 1] and non-decreasing | True | True | True |
| PREREG literals (LON_FOLLOW 3, STOP 6, TURN 1/2, band knot 10) == the code tree's constants | True | True | True |
| **all identity controls pass** |  | **True** | **True** |

Artifacts: `OUT/score_M2b_s{0,1}.json::identity_controls`, `OUT/weights_G_M2b_s{0,1}.json::identity_fit_all_windows_vs_banked`.

## 3. The PREREG rows

| read | seed | value | 95 % CI | verdict | estimator | tier | artifact |
|---|---|---|---|---|---|---|---|
| **PRIMARY-C** CONSISTENCY(a) 'any' HG - B (share of windows) | s0 | +0.0196 | [+0.0075, +0.0338] | bar: upper bound <= +0.02 -> **FAIL** | paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0 | T1-class open loop (own held outputs; not closed loop) | `OUT/score_M2b_s0.json::consistency.paired.HG_minus_B.viol_any` |
| **PRIMARY-C** CONSISTENCY(a) 'any' HG - B (share of windows) | s1 | +0.0288 | [+0.0179, +0.0404] | bar: upper bound <= +0.02 -> **FAIL** | paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0 | T1-class open loop (own held outputs; not closed loop) | `OUT/score_M2b_s1.json::consistency.paired.HG_minus_B.viol_any` |
| **PRIMARY-K** ADE6 HG - B (m) | s0 | -0.0372 | [-0.0681, -0.0085] | bar: < 0 and separated -> **PASS** | paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0 | T1-class open loop (own held outputs; not closed loop) | `OUT/score_M2b_s0.json::paired.HG_minus_B.ade6` |
| **PRIMARY-K** ADE2 HG - B (m) | s0 | -0.0197 | [-0.0297, -0.0116] | bar: < 0 and separated -> **PASS** | paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0 | T1-class open loop (own held outputs; not closed loop) | `OUT/score_M2b_s0.json::paired.HG_minus_B.ade2` |
| **PRIMARY-K** ADE6 HG - B (m) | s1 | -0.0267 | [-0.0521, -0.0042] | bar: < 0 and separated -> **PASS** | paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0 | T1-class open loop (own held outputs; not closed loop) | `OUT/score_M2b_s1.json::paired.HG_minus_B.ade6` |
| **PRIMARY-K** ADE2 HG - B (m) | s1 | -0.0214 | [-0.0281, -0.0152] | bar: < 0 and separated -> **PASS** | paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0 | T1-class open loop (own held outputs; not closed loop) | `OUT/score_M2b_s1.json::paired.HG_minus_B.ade2` |

| outcome | condition | met? |
|---|---|---|
| **PASS** | PRIMARY-C and PRIMARY-K at both seeds | **False** (C: False, K: True) |
| **CONSISTENCY NOT RESTORED** | PRIMARY-C fails | **True** |
| GAIN LOST | PRIMARY-C passes, PRIMARY-K fails | False |

## 4. The gate, beside (descriptive)

'Open' = the shrunk plan tau_G is used; 'closed' = tau_B is used (a trigger fired). That reading is fixed by the PREREG's own identity controls ('forced closed -> tau_B', 'forced open -> H'). Shares are over the 2,399 eval windows, with the episode-cluster CI.

| share of windows | s0 | s1 |
|---|---|---|
| **gate OPEN (tau_G used)** | 0.3939 [0.3442, 0.4456] (945) | 0.5381 [0.4770, 0.5984] (1,291) |
| gate CLOSED (tau_B used) | 0.6061 [0.5544, 0.6558] (1,454) | 0.4619 [0.4016, 0.5230] (1,108) |
| T1 fires: longitudinal argmax FOLLOW | 0.2272 [0.1708, 0.2838] (545; exclusive 304) | 0.2472 [0.1908, 0.3058] (593; exclusive 407) |
| T2 fires: P(STOP) >= 0.5 | 0.0383 [0.0229, 0.0546] (92; exclusive 25) | 0.0150 [0.0058, 0.0259] (36; exclusive 21) |
| T3 fires: P(TURN_L or TURN_R) >= 0.5 | 0.0842 [0.0546, 0.1184] (202; exclusive 142) | 0.0905 [0.0601, 0.1263] (217; exclusive 169) |
| T4 fires: speed of tau_G > T's predicted v_hi_ms, t >= 2 s | 0.4081 [0.3560, 0.4612] (979; exclusive 628) | 0.2072 [0.1664, 0.2508] (497; exclusive 279) |
| T1 or T2 or T3 (the decision triggers) | 0.3443 [0.2873, 0.4022] | 0.3456 [0.2852, 0.4063] |
| windows where HG differs from B | 945 | 1,291 |
| predicted v_hi_ms <= 0 (T4 closes the gate by the literal reading) | 20 | 3 |

'exclusive' = windows closed by that trigger and no other. Artifact: `OUT/score_M2b_s*.json::gate`, per-window flags in `score_M2b_s*_perwindow.npz` (`trig_*`, `closed`).

**Do T's triggers fire where the human is constrained?** Cross-tab of the trigger against the eval's own GT activity on eval-141 (descriptive; this is the `D-V8F-T-ATTRIB-1` question asked directly):

| trigger vs GT | seed | fired and active | fired, not active | not fired, active | precision | recall |
|---|---|---|---|---|---|---|
| T1 vs GT LEAD active | s0 | 438 | 107 | 589 | 0.804 | 0.426 |
| T1 vs GT LEAD active | s1 | 537 | 56 | 490 | 0.906 | 0.523 |
| T2 vs GT STOP active | s0 | 30 | 62 | 51 | 0.326 | 0.370 |
| T2 vs GT STOP active | s1 | 16 | 20 | 65 | 0.444 | 0.198 |
| T3 vs GT TURN active | s0 | 123 | 79 | 35 | 0.609 | 0.778 |
| T3 vs GT TURN active | s1 | 131 | 86 | 27 | 0.604 | 0.829 |
| T1/T2/T3 any vs GT any of STOP/TURN/LEAD | s0 | 671 | 155 | 490 | 0.812 | 0.578 |
| T1/T2/T3 any vs GT any of STOP/TURN/LEAD | s1 | 730 | 99 | 431 | 0.881 | 0.629 |
| gate closed (T1-T4) vs GT any of STOP/TURN/LEAD | s0 | 849 | 605 | 312 | 0.584 | 0.731 |
| gate closed (T1-T4) vs GT any of STOP/TURN/LEAD | s1 | 826 | 282 | 335 | 0.745 | 0.711 |
| T4 vs H violates SPEED | s0 | 230 | 749 | 239 | 0.235 | 0.490 |
| T4 vs H violates SPEED | s1 | 161 | 336 | 376 | 0.324 | 0.300 |
| T4 vs B violates SPEED | s0 | 59 | 920 | 170 | 0.060 | 0.258 |
| T4 vs B violates SPEED | s1 | 70 | 427 | 231 | 0.141 | 0.233 |

GT active counts on eval-141: STOP 81, TURN 158, LEAD 1,027 (SPEED 2,399). Reading: T1 is a good LEAD detector by precision (0.80 / 0.91) but finds only 43 % / 52 % of the LEAD-active windows; T2 (STOP) is weak at both seeds; T3 finds 78 % / 83 % of GT turns at 61 % precision; T4, the gate's only SPEED handler, has low precision against the SPEED violations that matter (H: 0.24 / 0.32, B: 0.06 / 0.14).

## 5. CONSISTENCY per rule and (b), beside

CONSISTENCY(a) = share of windows whose PLAN violates a GT-derived constraint; (b) = violation of the model's OWN predicted constraints. Lower is better. Paired episode-cluster bootstrap, B=2000. `*` = CI excludes 0 (worse if positive).

| seed | rule | B | H | HG | P10 | HG - B | HG - H | HG - P10 |
|---|---|---|---|---|---|---|---|---|
| s0 | any | 0.1596 | 0.2601 | 0.1792 | 0.3110 | +0.0196 [+0.0075, +0.0338] * | -0.0809 [-0.1088, -0.0551] * | -0.1317 [-0.1755, -0.0905] * |
| s0 | SPEED | 0.0955 | 0.1955 | 0.1100 | 0.2418 | +0.0146 [+0.0042, +0.0275] * | -0.0855 [-0.1122, -0.0605] * | -0.1317 [-0.1726, -0.0900] * |
| s0 | STOP | 0.0188 | 0.0292 | 0.0213 | 0.0338 | +0.0025 [+0.0004, +0.0050] * | -0.0079 [-0.0133, -0.0033] * | -0.0125 [-0.0200, -0.0058] * |
| s0 | TURN | 0.0042 | 0.0071 | 0.0042 | 0.0342 | +0.0000 [-0.0013, +0.0013] | -0.0029 [-0.0063, -0.0004] * | -0.0300 [-0.0434, -0.0179] * |
| s0 | LEAD | 0.0725 | 0.1042 | 0.0813 | 0.1384 | +0.0088 [+0.0025, +0.0159] * | -0.0229 [-0.0363, -0.0117] * | -0.0571 [-0.0821, -0.0342] * |
| s0 | self (b) any | 0.2297 | 0.3697 | 0.2372 | n/a | +0.0075 [+0.0008, +0.0163] * | -0.1326 [-0.1617, -0.1047] * | n/a |
| s1 | any | 0.2097 | 0.2851 | 0.2384 | 0.3110 | +0.0288 [+0.0179, +0.0404] * | -0.0467 [-0.0600, -0.0342] * | -0.0725 [-0.1076, -0.0384] * |
| s1 | SPEED | 0.1255 | 0.2238 | 0.1584 | 0.2418 | +0.0329 [+0.0200, +0.0463] * | -0.0654 [-0.0838, -0.0484] * | -0.0834 [-0.1133, -0.0521] * |
| s1 | STOP | 0.0271 | 0.0296 | 0.0275 | 0.0338 | +0.0004 [+0.0000, +0.0013] | -0.0021 [-0.0042, -0.0004] * | -0.0063 [-0.0117, -0.0021] * |
| s1 | TURN | 0.0067 | 0.0071 | 0.0071 | 0.0342 | +0.0004 [+0.0000, +0.0013] | +0.0000 [-0.0021, +0.0021] | -0.0271 [-0.0405, -0.0150] * |
| s1 | LEAD | 0.1084 | 0.1263 | 0.1146 | 0.1384 | +0.0063 [+0.0025, +0.0104] * | -0.0117 [-0.0192, -0.0050] * | -0.0238 [-0.0425, -0.0042] * |
| s1 | self (b) any | 0.1430 | 0.2172 | 0.1542 | n/a | +0.0113 [+0.0046, +0.0188] * | -0.0629 [-0.0833, -0.0454] * | n/a |

Where the residual cost lives (HG - B, point estimates, s0 / s1): the SPEED excess is +0.0146 / +0.0329 against an 'any' excess of +0.0196 / +0.0288 (rule excesses overlap inside 'any', so they need not sum to it); LEAD is next (+0.0088 / +0.0063); STOP (+0.0025 / +0.0004) and TURN (+0.0000 / +0.0004) are at or near zero. SPEED is the one rule active on every window, so the constraint-aware fit cannot see it and T4 is its only handler.

Window transitions, HG vs B (violated by HG and ok in B / ok in HG and violated in B):

| seed | rule | HG viol, B ok | HG ok, B viol | both viol |
|---|---|---|---|---|
| s0 | any | 66 | 19 | 364 |
| s0 | SPEED | 49 | 14 | 215 |
| s0 | STOP | 6 | 0 | 45 |
| s0 | TURN | 1 | 1 | 9 |
| s0 | LEAD | 25 | 4 | 170 |
| s1 | any | 82 | 13 | 490 |
| s1 | SPEED | 91 | 12 | 289 |
| s1 | STOP | 1 | 0 | 65 |
| s1 | TURN | 2 | 1 | 15 |
| s1 | LEAD | 19 | 4 | 256 |

By gate state (share of windows violating 'any'; HG = B on closed windows by construction): s0 open: HG 0.246 vs B 0.196, closed: 0.136; s1 open: HG 0.257 vs B 0.204, closed: 0.217. The whole excess is on the open windows, as it must be.

## 6. ADE and FDE at 1-6 s, accel, families, dominance (beside, per PREREG)

**M2b_s0** (m; mean [95 % CI]; paired deltas paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0)

| metric | B | HG | H | P10 | HG - B | HG - H | HG - P10 |
|---|---|---|---|---|---|---|---|
| ADE@1s | 0.0819 [0.0714, 0.0953] | 0.0780 [0.0676, 0.0916] | 0.0598 [0.0511, 0.0728] | 0.0633 [0.0545, 0.0764] | -0.0039 [-0.0057, -0.0023] * | +0.0182 [+0.0140, +0.0231] * | +0.0146 [+0.0104, +0.0196] * |
| ADE@2s | 0.3155 [0.2765, 0.3621] | 0.2958 [0.2573, 0.3427] | 0.2123 [0.1916, 0.2392] | 0.2221 [0.2008, 0.2491] | -0.0197 [-0.0297, -0.0116] * | +0.0836 [+0.0558, +0.1216] * | +0.0738 [+0.0427, +0.1159] * |
| ADE@3s | 0.5772 [0.5254, 0.6401] | 0.5587 [0.5070, 0.6223] | 0.4734 [0.4324, 0.5198] | 0.5337 [0.4865, 0.5871] | -0.0186 [-0.0280, -0.0101] * | +0.0853 [+0.0567, +0.1250] * | +0.0250 [-0.0116, +0.0748] |
| ADE@4s | 0.9838 [0.9078, 1.0702] | 0.9596 [0.8826, 1.0473] | 0.8541 [0.7859, 0.9292] | 1.0129 [0.9246, 1.1071] | -0.0242 [-0.0380, -0.0116] * | +0.1055 [+0.0735, +0.1432] * | -0.0533 [-0.1062, +0.0052] |
| ADE@5s | 1.5179 [1.4027, 1.6470] | 1.4874 [1.3720, 1.6151] | 1.3530 [1.2458, 1.4681] | 1.6628 [1.5163, 1.8107] | -0.0305 [-0.0516, -0.0109] * | +0.1344 [+0.0927, +0.1808] * | -0.1754 [-0.2642, -0.0890] * |
| ADE@6s | 2.1753 [2.0071, 2.3568] | 2.1381 [1.9685, 2.3199] | 1.9685 [1.8106, 2.1306] | 2.4870 [2.2606, 2.7135] | -0.0372 [-0.0681, -0.0085] * | +0.1695 [+0.1135, +0.2292] * | -0.3489 [-0.4928, -0.2149] * |
| FDE@1s | 0.1220 [0.1066, 0.1443] | 0.1229 [0.1072, 0.1456] | 0.1209 [0.1054, 0.1429] | 0.1288 [0.1129, 0.1509] | +0.0010 [-0.0003, +0.0022] | +0.0020 [-0.0009, +0.0051] | -0.0058 [-0.0093, -0.0025] * |
| FDE@2s | 0.7022 [0.6156, 0.8066] | 0.6819 [0.5978, 0.7870] | 0.5603 [0.5105, 0.6173] | 0.6089 [0.5556, 0.6695] | -0.0203 [-0.0387, -0.0049] * | +0.1216 [+0.0592, +0.2133] * | +0.0730 [+0.0053, +0.1728] * |
| FDE@3s | 1.4754 [1.3587, 1.6069] | 1.4484 [1.3309, 1.5813] | 1.3354 [1.2289, 1.4543] | 1.5911 [1.4524, 1.7375] | -0.0270 [-0.0469, -0.0089] * | +0.1130 [+0.0712, +0.1594] * | -0.1427 [-0.2216, -0.0617] * |
| FDE@4s | 2.7303 [2.5108, 2.9629] | 2.6819 [2.4639, 2.9138] | 2.4830 [2.2813, 2.6914] | 3.0894 [2.8142, 3.3663] | -0.0484 [-0.0885, -0.0121] * | +0.1989 [+0.1229, +0.2796] * | -0.4075 [-0.5744, -0.2395] * |
| FDE@5s | 4.3146 [3.9548, 4.6900] | 4.2538 [3.9012, 4.6131] | 3.9699 [3.6405, 4.3013] | 5.1174 [4.6429, 5.5918] | -0.0607 [-0.1239, +0.0009] | +0.2840 [+0.1720, +0.4009] * | -0.8636 [-1.1666, -0.5748] * |
| FDE@6s | 6.2810 [5.7498, 6.8269] | 6.2075 [5.6687, 6.7392] | 5.8185 [5.3261, 6.3198] | 7.6673 [6.9298, 8.4116] | -0.0735 [-0.1675, +0.0162] | +0.3889 [+0.2369, +0.5438] * | -1.4598 [-1.9537, -0.9703] * |

**M2b_s1** (m; mean [95 % CI]; paired deltas paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0)

| metric | B | HG | H | P10 | HG - B | HG - H | HG - P10 |
|---|---|---|---|---|---|---|---|
| ADE@1s | 0.0706 [0.0613, 0.0832] | 0.0683 [0.0590, 0.0816] | 0.0601 [0.0516, 0.0731] | 0.0633 [0.0545, 0.0764] | -0.0023 [-0.0034, -0.0013] * | +0.0081 [+0.0055, +0.0119] * | +0.0050 [+0.0022, +0.0088] * |
| ADE@2s | 0.2914 [0.2541, 0.3397] | 0.2701 [0.2337, 0.3186] | 0.2137 [0.1931, 0.2400] | 0.2221 [0.2008, 0.2491] | -0.0214 [-0.0281, -0.0152] * | +0.0564 [+0.0322, +0.0941] * | +0.0480 [+0.0201, +0.0913] * |
| ADE@3s | 0.5525 [0.4983, 0.6177] | 0.5309 [0.4780, 0.5963] | 0.4814 [0.4414, 0.5274] | 0.5337 [0.4865, 0.5871] | -0.0216 [-0.0311, -0.0130] * | +0.0495 [+0.0245, +0.0919] * | -0.0027 [-0.0382, +0.0510] |
| ADE@4s | 0.9361 [0.8585, 1.0205] | 0.9113 [0.8354, 0.9943] | 0.8663 [0.7980, 0.9396] | 1.0129 [0.9246, 1.1071] | -0.0248 [-0.0372, -0.0136] * | +0.0450 [+0.0234, +0.0784] * | -0.1016 [-0.1514, -0.0387] * |
| ADE@5s | 1.4403 [1.3246, 1.5633] | 1.4127 [1.2997, 1.5300] | 1.3687 [1.2598, 1.4832] | 1.6628 [1.5163, 1.8107] | -0.0276 [-0.0456, -0.0120] * | +0.0440 [+0.0236, +0.0720] * | -0.2501 [-0.3280, -0.1690] * |
| ADE@6s | 2.0546 [1.8859, 2.2323] | 2.0279 [1.8615, 2.2028] | 1.9848 [1.8199, 2.1587] | 2.4870 [2.2606, 2.7135] | -0.0267 [-0.0521, -0.0042] * | +0.0431 [+0.0216, +0.0694] * | -0.4591 [-0.5827, -0.3346] * |
| FDE@1s | 0.1214 [0.1062, 0.1432] | 0.1217 [0.1065, 0.1437] | 0.1221 [0.1066, 0.1438] | 0.1288 [0.1129, 0.1509] | +0.0004 [-0.0014, +0.0020] | -0.0004 [-0.0028, +0.0027] | -0.0070 [-0.0098, -0.0038] * |
| FDE@2s | 0.7080 [0.6210, 0.8234] | 0.6712 [0.5861, 0.7843] | 0.5682 [0.5200, 0.6221] | 0.6089 [0.5556, 0.6695] | -0.0367 [-0.0538, -0.0213] * | +0.1031 [+0.0459, +0.1976] * | +0.0624 [-0.0037, +0.1672] |
| FDE@3s | 1.4102 [1.2925, 1.5347] | 1.3821 [1.2692, 1.5035] | 1.3573 [1.2506, 1.4738] | 1.5911 [1.4524, 1.7375] | -0.0282 [-0.0472, -0.0117] * | +0.0247 [+0.0054, +0.0493] * | -0.2090 [-0.2834, -0.1227] * |
| FDE@4s | 2.5848 [2.3687, 2.8044] | 2.5462 [2.3371, 2.7631] | 2.5087 [2.3004, 2.7194] | 3.0894 [2.8142, 3.3663] | -0.0386 [-0.0697, -0.0123] * | +0.0375 [+0.0154, +0.0586] * | -0.5432 [-0.6902, -0.3952] * |
| FDE@5s | 4.0775 [3.7120, 4.4422] | 4.0421 [3.6888, 4.4079] | 4.0022 [3.6567, 4.3598] | 5.1174 [4.6429, 5.5918] | -0.0353 [-0.0885, +0.0098] | +0.0399 [+0.0081, +0.0683] * | -1.0753 [-1.3421, -0.8143] * |
| FDE@6s | 5.8582 [5.3224, 6.3994] | 5.8509 [5.3159, 6.3982] | 5.8168 [5.2957, 6.3540] | 7.6673 [6.9298, 8.4116] | -0.0073 [-0.0793, +0.0556] | +0.0341 [-0.0102, +0.0747] | -1.8163 [-2.2602, -1.3816] * |

**Dominance vs P10-yaw-4512** (descriptive: 'HG dominates P10' iff ADE2, ADE6 and accel MAE point estimates are all <= P10's and ADE6 H - P10 is separated):

| seed | ADE2 HG / P10 (m) | ADE6 HG / P10 (m) | ADE6 HG - P10, paired | accel MAE HG / P10 (m/s^2) | accel HG - P10, paired | HG dominates P10 |
|---|---|---|---|---|---|---|
| s0 | 0.2958 / 0.2221 | 2.1381 / 2.4870 | -0.3489 [-0.4928, -0.2149] * | 1.1085 / 0.6376 | +0.4709 [+0.3748, +0.5814] * | **False** (ADE2 False, ADE6 True, accel False) |
| s1 | 0.2701 / 0.2221 | 2.0279 / 2.4870 | -0.4591 [-0.5827, -0.3346] * | 0.8858 / 0.6376 | +0.2482 [+0.1808, +0.3387] * | **False** (ADE2 False, ADE6 True, accel False) |

### The four families (HG - B and HG - P10 paired; lower is better except the two plan-derived tactical rows)

**M2b_s0**

| family | metric | B | HG | H | P10 | HG - B | HG - P10 |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | speed MAE (m/s) | 1.0674 | 1.0334 | 0.8709 | 0.9802 | -0.0340 [-0.0538, -0.0158] * | +0.0532 [-0.0247, +0.1381] |
| LONGITUDINAL | accel MAE (m/s^2) | 1.2058 | 1.1085 | 0.7414 | 0.6376 | -0.0973 [-0.1375, -0.0645] * | +0.4709 [+0.3748, +0.5814] * |
| LONGITUDINAL | along MAE (m) | 1.8097 | 1.7659 | 1.5925 | 1.8217 | -0.0438 [-0.0729, -0.0165] * | -0.0558 [-0.1767, +0.0727] |
| LATERAL | cross-track offset MAE (m) | 0.7662 | 0.7680 | 0.7557 | 1.1858 | +0.0018 [-0.0122, +0.0148] | -0.4177 [-0.5117, -0.3276] * |
| LATERAL | heading MAE (deg) | 4.7804 | 4.5760 | 4.0259 | 5.7302 | -0.1748 [-0.3902, -0.0066] * | -1.1572 [-1.6941, -0.5501] * |
| LATERAL | yaw-rate MAE (deg/s) | 8.1167 | 7.7085 | 6.4244 | 6.3299 | -0.4211 [-0.6217, -0.2554] * | +1.4314 [+0.7807, +2.3611] * |
| LATERAL | curvature MAE (1/m) | 0.0815 | 0.0755 | 0.0536 | 0.0462 | -0.0082 [-0.0128, -0.0044] * | +0.0308 [+0.0150, +0.0551] * |
| TACTICAL (plan-derived) | lateral decision agreement (higher better) | 0.8458 | 0.8449 | 0.8629 | 0.7728 | -0.0008 [-0.0071, +0.0054] | +0.0721 [+0.0471, +0.0976] * |
| TACTICAL (plan-derived) | longitudinal decision agreement (higher better) | 0.5790 | 0.5761 | 0.5969 | 0.5336 | -0.0029 [-0.0096, +0.0046] | +0.0425 [-0.0008, +0.0825] |

**M2b_s1**

| family | metric | B | HG | H | P10 | HG - B | HG - P10 |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | speed MAE (m/s) | 0.9112 | 0.8998 | 0.8389 | 0.9802 | -0.0114 [-0.0181, -0.0046] * | -0.0804 [-0.1390, -0.0125] * |
| LONGITUDINAL | accel MAE (m/s^2) | 0.9313 | 0.8858 | 0.7036 | 0.6376 | -0.0455 [-0.0628, -0.0300] * | +0.2482 [+0.1808, +0.3387] * |
| LONGITUDINAL | along MAE (m) | 1.6081 | 1.5878 | 1.5573 | 1.8217 | -0.0203 [-0.0333, -0.0085] * | -0.2339 [-0.3262, -0.1339] * |
| LATERAL | cross-track offset MAE (m) | 0.8644 | 0.8563 | 0.8375 | 1.1858 | -0.0081 [-0.0326, +0.0137] | -0.3295 [-0.4275, -0.2378] * |
| LATERAL | heading MAE (deg) | 4.4383 | 4.3767 | 4.3276 | 5.7302 | -0.0616 [-0.1460, +0.0159] | -1.3535 [-1.8258, -0.9052] * |
| LATERAL | yaw-rate MAE (deg/s) | 6.9864 | 6.7956 | 6.4447 | 6.3299 | -0.2051 [-0.3171, -0.1089] * | +0.5118 [+0.0757, +1.0249] * |
| LATERAL | curvature MAE (1/m) | 0.0626 | 0.0613 | 0.0572 | 0.0462 | -0.0025 [-0.0063, +0.0002] | +0.0168 [+0.0065, +0.0290] * |
| TACTICAL (plan-derived) | lateral decision agreement (higher better) | 0.8287 | 0.8395 | 0.8470 | 0.7728 | +0.0108 [+0.0008, +0.0238] * | +0.0667 [+0.0417, +0.0930] * |
| TACTICAL (plan-derived) | longitudinal decision agreement (higher better) | 0.5919 | 0.6048 | 0.6132 | 0.5336 | +0.0129 [+0.0025, +0.0237] * | +0.0713 [+0.0271, +0.1170] * |

TACTICAL (T's own heads: lat / lon macro-F1 and goal-token macro-F1): HG's block equals B's banked block exactly (json equality) at s0: True and s1: True (the gate reads T's heads, it does not change them). The plan-derived tactical rows above are read off the trajectory and do move with the plan. **STRATEGIC: PAUSED** per the PI ruling of 2026-09-19 (`PI_DECISION_QUEUE.md` ITEM 25); S is untouched and its banked checkpoint error is B's; reported as PAUSED, not as a pass.

### 6a. Distance keeping (the other half of LONGITUDINAL), K = 30 (6 s) lead block, through `v8f_dk_read.read_dk` -> `taniteval.lead_metrics`

Lead block `b1_eval_lead_block_k30.npz`, window states {'LEAD': 743, 'NO_LEAD': 552, 'NOT_STRAIGHT': 931, 'NO_LABEL': 173}, kinematic-contract check **OK**. Open loop: the lead's logged future does not react to the arm. No distance-keeping bar is registered for HGATE, so nothing here is PASS/FAIL. Min-TTC is a CENSORED mean (30 s cap); 'more gap' is not 'better'; the better/worse reading is LEAD-VIOL and the distance to the human.

| arm | headway min (m) | time-gap min (s) | min-TTC (s, censored) | LEAD-VIOL share of at-risk (n viol) |
|---|---|---|---|---|
| human (GT path) | 25.665 | 3.995 | 19.790 | by definition 0 |
| B_s0 | 25.622 | 4.022 | 19.396 | 0.0353 (21) |
| H_s0 | 25.553 | 3.920 | 20.497 | 0.0487 (29) |
| HG_s0 | 25.519 | 3.974 | 19.369 | 0.0370 (22) |
| B_s1 | 25.544 | 3.950 | 19.524 | 0.0605 (36) |
| H_s1 | 25.606 | 3.950 | 20.438 | 0.0588 (35) |
| HG_s1 | 25.559 | 3.954 | 19.724 | 0.0588 (35) |
| P10 | 25.116 | 3.832 | 19.947 | 0.0807 (48) |

| paired contrast | headway (m) | time-gap (s) | min-TTC (s) | LEAD-VIOL at-risk (share) |
|---|---|---|---|---|
| HG_s0 - B_s0 | -0.077 [-0.217, +0.073]  | -0.048 [-0.145, +0.013]  | -0.116 [-0.443, +0.219]  | +0.0017 [-0.0036, +0.0075]  |
| HG_s0 - H_s0 | +0.071 [-0.169, +0.281]  | +0.047 [+0.012, +0.095] * | -1.146 [-1.771, -0.585] * | -0.0118 [-0.0216, -0.0033] * |
| HG_s0 - P10 | +0.449 [+0.059, +0.865] * | +0.138 [+0.029, +0.282] * | -0.582 [-1.447, +0.381]  | -0.0437 [-0.0727, -0.0181] * |
| HG_s1 - B_s1 | -0.054 [-0.200, +0.058]  | -0.000 [-0.034, +0.030]  | +0.187 [+0.022, +0.388] * | -0.0017 [-0.0057, +0.0000]  |
| HG_s1 - H_s1 | -0.050 [-0.153, +0.030]  | -0.003 [-0.027, +0.021]  | -0.756 [-1.276, -0.308] * | +0.0000 [-0.0049, +0.0048]  |
| HG_s1 - P10 | +0.332 [-0.081, +0.781]  | +0.116 [+0.001, +0.255] * | -0.238 [-1.147, +0.719]  | -0.0218 [-0.0508, +0.0018]  |

`*` = CI excludes 0. Artifact: `OUT/dk_read.json` (`pairs`, `arms`), per-window `OUT/dk_perwindow.npz`.

## 7. The outcome table, applied verbatim, and the next step (Rule Zero)

**Result: CONSISTENCY NOT RESTORED.** PRIMARY-C fails at both seeds, so the PREREG's committed step applies: *T's own constraint predictions are not good enough to tell where the human constrains: the T-side ceiling again (`D-V8F-T-ATTRIB-1`). Next: no further post-processing arm. The lever moves to T, after MIXO and FLAT read.* No bar, trigger, threshold or weight was changed after any HGATE number was seen.

This is a refutation of one candidate, and per Rule Zero it is a waypoint. What is eliminated, with its cost: the restriction of the fit to unconstrained windows plus a four-trigger gate on T's own outputs recovers 80 % / 62 % of H's CONSISTENCY cost but not enough to put the upper bound under +0.02, and it spends 82 % / 62 % of H's ADE6 gain to do it. What blocks a next step inside this family is named by the PREREG itself: it rules out a further post-processing arm, and the T-side lever waits on MIXO and FLAT reading (per the task brief, box 2 is training). Not run, because the PREREG rules it out and a bar would have to be invented: a further post-processing arm (for example a better SPEED handler than T4). Lifting that ruling is the Master Mind's / PI's call. What WAS run is the cheapest informative thing that invents no bar: section 8 (post-hoc, descriptive) attributes the residual so that the T-side work is aimed at the right term, the SPEED term and T4's predicted `v_hi_ms`. It also argues for the PREREG's T-side route over another post-processing arm: even with oracle decision triggers and an oracle `v_hi_ms`, the gain is gone at s1 (O2).

## 8. Post-hoc diagnostics (descriptive; NOT a PREREG row; no bar; none can change the outcome)

Run AFTER `READ.json` was fixed (`diag_gate.py`). 'tau_G everywhere' = the blend with w^G and NO gate (not HBLEND's H, which uses the unrestricted w). The ORACLE arms are **T0** (they read a GT label at inference): a ceiling on what a perfect detector could buy, **never a result**.

**M2b_s0** (HG-style arm - B; paired episode-cluster bootstrap, B=2000)

| arm | open share | ADE2 | ADE6 | accel | CONS any | SPEED | LEAD | STOP | TURN | C (upper <= 0.02) | K (both < 0, separated) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tau_G everywhere (w^G, NO gate: all triggers off) | 1.000 | -0.102 [-0.139, -0.075] * | -0.204 [-0.277, -0.137] * | -0.478 [-0.568, -0.391] * | +0.0884 [+0.0592, +0.1196] | +0.0909 | +0.0288 | +0.0108 | +0.0008 | False | True |
| HG (the PREREG arm: T1+T2+T3+T4) | 0.394 | -0.020 [-0.030, -0.012] * | -0.037 [-0.068, -0.009] * | -0.097 [-0.138, -0.065] * | +0.0196 [+0.0075, +0.0338] | +0.0146 | +0.0088 | +0.0025 | +0.0000 | False | True |
| T1+T2+T3 only (T4 off) | 0.656 | -0.059 [-0.093, -0.037] * | -0.100 [-0.144, -0.055] * | -0.253 [-0.328, -0.192] * | +0.0609 [+0.0392, +0.0871] | +0.0579 | +0.0163 | +0.0029 | +0.0004 | False | True |
| T4 only | 0.592 | -0.032 [-0.044, -0.022] * | -0.069 [-0.115, -0.027] * | -0.169 [-0.218, -0.125] * | +0.0242 [+0.0096, +0.0409] | +0.0229 | +0.0100 | +0.0063 | +0.0004 | False | True |
| leave out T1 (FOLLOW) | 0.521 | -0.027 [-0.038, -0.018] * | -0.045 [-0.081, -0.011] * | -0.133 [-0.178, -0.095] * | +0.0188 [+0.0063, +0.0333] | +0.0196 | +0.0075 | +0.0029 | +0.0000 | False | True |
| leave out T2 (STOP) | 0.404 | -0.021 [-0.031, -0.013] * | -0.048 [-0.081, -0.017] * | -0.108 [-0.150, -0.074] * | +0.0221 [+0.0096, +0.0367] | +0.0158 | +0.0100 | +0.0046 | +0.0000 | False | True |
| leave out T3 (TURN) | 0.453 | -0.023 [-0.034, -0.014] * | -0.050 [-0.088, -0.016] * | -0.121 [-0.165, -0.084] * | +0.0221 [+0.0083, +0.0375] | +0.0167 | +0.0096 | +0.0038 | +0.0004 | False | True |
| leave out T4 (v_hi) | 0.656 | -0.059 [-0.093, -0.037] * | -0.100 [-0.144, -0.055] * | -0.253 [-0.328, -0.192] * | +0.0609 [+0.0392, +0.0871] | +0.0579 | +0.0163 | +0.0029 | +0.0004 | False | True |
| ORACLE O1: GT STOP/TURN/LEAD active + T4 | 0.303 | -0.015 [-0.025, -0.008] * | -0.026 [-0.055, +0.002] | -0.070 [-0.106, -0.041] * | +0.0104 [-0.0000, +0.0229] | +0.0104 | +0.0000 | +0.0000 | +0.0000 | False | False |
| ORACLE O2: GT STOP/TURN/LEAD active + GT-v_hi speed band | 0.314 | -0.013 [-0.019, -0.007] * | -0.038 [-0.068, -0.008] * | -0.096 [-0.127, -0.066] * | +0.0067 [+0.0021, +0.0121] | +0.0067 | +0.0000 | +0.0000 | +0.0000 | True | True |
| ORACLE O3: close wherever tau_G violates any GT rule | 0.752 | -0.042 [-0.051, -0.033] * | -0.177 [-0.240, -0.119] * | -0.272 [-0.323, -0.225] * | -0.0163 [-0.0237, -0.0096] | -0.0108 | -0.0046 | +0.0000 | -0.0008 | True | True |

**M2b_s1** (HG-style arm - B; paired episode-cluster bootstrap, B=2000)

| arm | open share | ADE2 | ADE6 | accel | CONS any | SPEED | LEAD | STOP | TURN | C (upper <= 0.02) | K (both < 0, separated) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tau_G everywhere (w^G, NO gate: all triggers off) | 1.000 | -0.077 [-0.114, -0.052] * | -0.064 [-0.099, -0.032] * | -0.215 [-0.275, -0.163] * | +0.0713 [+0.0529, +0.0901] | +0.0946 | +0.0146 | +0.0025 | +0.0004 | False | True |
| HG (the PREREG arm: T1+T2+T3+T4) | 0.538 | -0.021 [-0.028, -0.015] * | -0.027 [-0.052, -0.004] * | -0.045 [-0.063, -0.030] * | +0.0288 [+0.0179, +0.0404] | +0.0329 | +0.0063 | +0.0004 | +0.0004 | False | True |
| T1+T2+T3 only (T4 off) | 0.654 | -0.050 [-0.085, -0.028] * | -0.047 [-0.080, -0.019] * | -0.110 [-0.165, -0.069] * | +0.0417 [+0.0279, +0.0567] | +0.0517 | +0.0050 | +0.0004 | +0.0004 | False | True |
| T4 only | 0.793 | -0.033 [-0.040, -0.026] * | -0.035 [-0.063, -0.009] * | -0.098 [-0.121, -0.076] * | +0.0450 [+0.0304, +0.0604] | +0.0575 | +0.0117 | +0.0021 | +0.0013 | False | True |
| leave out T1 (FOLLOW) | 0.708 | -0.028 [-0.035, -0.022] * | -0.036 [-0.064, -0.011] * | -0.078 [-0.098, -0.059] * | +0.0367 [+0.0238, +0.0496] | +0.0467 | +0.0100 | +0.0008 | +0.0004 | False | True |
| leave out T2 (STOP) | 0.547 | -0.022 [-0.029, -0.016] * | -0.028 [-0.054, -0.006] * | -0.048 [-0.065, -0.032] * | +0.0304 [+0.0188, +0.0421] | +0.0346 | +0.0071 | +0.0017 | +0.0004 | False | True |
| leave out T3 (TURN) | 0.609 | -0.025 [-0.032, -0.018] * | -0.024 [-0.050, -0.001] * | -0.062 [-0.082, -0.043] * | +0.0350 [+0.0221, +0.0487] | +0.0417 | +0.0067 | +0.0004 | +0.0013 | False | True |
| leave out T4 (v_hi) | 0.654 | -0.050 [-0.085, -0.028] * | -0.047 [-0.080, -0.019] * | -0.110 [-0.165, -0.069] * | +0.0417 [+0.0279, +0.0567] | +0.0517 | +0.0050 | +0.0004 | +0.0004 | False | True |
| ORACLE O1: GT STOP/TURN/LEAD active + T4 | 0.430 | -0.015 [-0.019, -0.010] * | -0.021 [-0.045, -0.000] * | -0.031 [-0.044, -0.018] * | +0.0204 [+0.0108, +0.0305] | +0.0204 | +0.0000 | +0.0000 | +0.0000 | False | True |
| ORACLE O2: GT STOP/TURN/LEAD active + GT-v_hi speed band | 0.296 | -0.005 [-0.009, -0.001] * | -0.001 [-0.015, +0.013] | -0.025 [-0.038, -0.014] * | +0.0042 [+0.0008, +0.0079] | +0.0042 | +0.0000 | +0.0000 | +0.0000 | True | False |
| ORACLE O3: close wherever tau_G violates any GT rule | 0.719 | -0.020 [-0.026, -0.014] * | -0.034 [-0.057, -0.011] * | -0.090 [-0.110, -0.072] * | -0.0083 [-0.0138, -0.0042] | -0.0046 | -0.0033 | +0.0000 | -0.0008 | True | True |

Reading (descriptive, each a MEASURED statement about the arms above, not a claim about the programme):

1. **T4 is the load-bearing trigger and also the weakest (its precision against the SPEED violations that matter is 0.24 / 0.32 for H and 0.06 / 0.14 for B, section 4).** Dropping T4 (T1+T2+T3 only) leaves CONS 'any' at +0.0609 / +0.0417 (upper +0.0871 / +0.0567); T4 alone gives +0.0242 / +0.0450. Neither half passes PRIMARY-C alone at either seed. Removing any single one of T1/T2/T3 moves 'any' by at most 0.0079 (point estimate), so none of the three decision triggers is individually decisive.
2. **A perfect STOP/TURN/LEAD detector is not enough while T4 stays as it is.** O1 (GT activity replaces T1-T3, T4 kept) gives CONS 'any' +0.0104 [upper +0.0229] / +0.0204 [upper +0.0305]: the upper bound is still above +0.02 at both seeds. The residual is then the SPEED term, i.e. T's predicted `v_hi_ms`.
3. **With oracle detection of the decisions AND oracle `v_hi_ms` the consistency bar is met, at the price of the gain.** O2 gives CONS 'any' +0.0067 [upper +0.0121] / +0.0042 [upper +0.0079] (C met at both seeds), but ADE6 -0.038 (separated: True) / -0.001 (separated: False) and ADE2 -0.013 / -0.005: K holds at s0 and fails at s1 even with oracle gating. A gate family that can only choose between tau_B and tau_G has a thin margin here.
4. **Ceiling of any per-window B-or-tau_G gate.** O3 (close wherever tau_G itself violates a GT rule) gives CONS 'any' -0.0163 / -0.0083 (better than B, separated) with ADE6 -0.177 / -0.034 (separated at both). So a gate that restores consistency and keeps the gain EXISTS in this family; HGATE's triggers do not find it. This is consistent with the PREREG's T-side attribution (`D-V8F-T-ATTRIB-1`), and O3 is an oracle of the violation itself, so it is an upper bound, not a design.
5. **The constraint-aware FIT, on its own, is almost H; the GATE does the work.** tau_G everywhere (w^G, no gate) gives CONS 'any' +0.0884 / +0.0713 and ADE6 -0.204 / -0.064 against HBLEND's H at +0.1005 / +0.0754 and -0.207 / -0.070 (all vs B, s0 / s1). Fitting only on unconstrained windows removes a little of the consistency cost and a little of the gain; the restriction is not where the effect is.

## 9. Interpretations declared before any HG number (the PREREG text left them open; none changed afterwards)

1. **'Open' / 'closed'.** Closed = the plan is tau_B (a trigger fired, or forced); open = tau_G. Fixed by the PREREG's identity controls. 'The gate's open share' = the share of windows on which tau_G is used.
2. **'Active GT constraint'** on a train window = the eval's own `violations(...)[STOP|TURN|LEAD]['active']` through `gt_constraints`; SPEED excluded (active on 9,582 of 9,582 windows). Windows without a label row would be inactive (there are none).
3. **T4.** Speed of tau_G = `plan_kinematics(plan)['v']` on the 0.2 s knot plan, intervals that START at t >= 2.0 s (`v[:, 10:]`, the SPEED rule's own band, `_BAND_KNOT = 10`), compared with T's RAW predicted `v_hi_ms` (`cons_pred` cell 25; no clamp at 0, no min with the posted limit, no 0.5 m/s SPEED tolerance). The PREREG says 'exceeds T's own predicted v_hi_ms'. 20 / 3 windows (s0 / s1) have a predicted v_hi <= 0 and are closed.
4. **T2 / T3 use '>= 0.5'** (the PREREG's operator). The eval's own self-consistency uses '> 0.5'. MEASURED here (build_result.py, from the banked npz `goal_prob`): windows with a probability of exactly 0.5 in the STOP / TURN_L / TURN_R columns: s0 [0, 0, 0], s1 [0, 0, 0]; STOP fires on 92 (>=) vs 92 (>) windows at s0, 36 vs 36 at s1; TURN 202 vs 202 at s0, 217 vs 217 at s1. So the operator choice cannot change the gate on these data.
5. **Scored grid.** tau_G is formed on the 0.2 s knots (the PREREG writes up(P10), the 0.1 s upsampling; on the scored knots up(P10)[1::2] = P10, so the plan scored is identical). HBLEND's own declaration that w is fitted on 30 steps x 2 axes = 60 numbers per seed carries over.
6. **P.** P = the checkpoint's own frozen P10 evaluated on the eval kinematic features (HBLEND's route); recomputed here and identical to HBLEND's saved P (max abs 0.0).
7. **The identity controls were read as exact** (0.0), not as a tolerance: they are the same arithmetic on the same arrays, so any non-zero would have been a bug.

## 10. Escalations and caveats

* **Nothing here passes the registered bar, so nothing here is deployable under the PREREG.** HG improves on H in the consistency rows (HG - H negative and separated on 'any', SPEED, STOP and LEAD at both seeds; TURN at s0 only) while keeping a separated ADE2/ADE6 gain over B. A decision to ship HG on the strength of that is the Master Mind's / PI's call and would need its own PREREG: the pre-registered bar is not met.
* **Two seeds only; one training run each.** The separated-CI reading is necessary, not sufficient, for a lever claim (`H-ESTIM-SEED-1`). The two seeds disagree in size (open share 0.39 vs 0.54; PRIMARY-C point 0.020 vs 0.029) and agree in verdict and sign.
* **The weights are trained-checkpoint-specific and fitted in-sample on train windows**; eval-141 never enters the fit (INHERITED from HBLEND section 8 for the unrestricted w: the train-fit sat within 0.004 m of the eval-oracle weights; not re-measured for w^G).
* **The gate reads T's heads on the eval windows from the banked KIND run (`kind-own`).** If MIXO/FLAT change T's heads, the trigger statistics (section 4) change; they must be re-read per arm.
* **Section 8 is post hoc.** It was run after the outcome was fixed. It attributes; it does not decide.

## 11. Deliverable manifest (everything under `OUT`; nothing staged in the repo, nothing pushed, nothing deleted)

| what | file | note |
|---|---|---|
| executor (activity / fit / score / read / dk) | `v8f_hgate.py` | imports HBLEND's `v8f_hblend` (fit_weights, apply_w, arm_pw, paired rows, loaders) and the eval's own `evaluate()`; HBLEND/HPROJ files untouched |
| post-hoc diagnostics | `diag_gate.py, diag_gate.json, diag_gate.log` | section 8; descriptive |
| self-tests (16 pass, `--basetemp C:/Users/Admin/pt_hgate`) | `test_v8f_hgate.py, pytest.log` | trigger thresholds, T4 band, gate identities, fit restriction with a wrong-subset regression arm, activity-builder indexing, outcome table |
| result builder | `build_result.py` | reads the JSON; no typed numbers |
| step 1 activity | `train_activity.json, train_activity.npz, activity.log` | per-window STOP/TURN/LEAD/SPEED masks aligned with the fit windows |
| step 1 weights | `weights_G_M2b_s0.json, weights_G_M2b_s1.json, fit_M2b_s*.log` | w^G (60 numbers per seed) + the identity (iii) record |
| score | `score_M2b_s0.json, score_M2b_s1.json (+ _perwindow.npz, score_*.log)` | controls, gate, every PREREG row, beside rows |
| outcome | `READ.json, read.log` | the outcome table applied |
| distance keeping | `dk_read.json, dk_perwindow.npz, dk.log` | via `v8f_dk_read.read_dk` |
| this report | `HGATE_RESULT.md` |  |

Reproduce (dev box, CPU): `PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe`; `$PY v8f_hgate.py activity`; `$PY v8f_hgate.py fit --seed M2b_s0` (and `M2b_s1`); `$PY v8f_hgate.py score --seed M2b_s0` (about 70 s each, identity controls first); `$PY v8f_hgate.py read`; `$PY v8f_hgate.py dk`; `$PY diag_gate.py`; `$PY build_result.py`.

Inputs reused read-only (not modified): `hblend/` (`v8f_hblend.py`, `fit_windows_M2b_s*.npz`, `weights_M2b_s*.json`, `box2/join_train.npz`, `inputs/banked/eval_M2b_s*-kind-own.{json,npz}`, `inputs/b2_p10_eval_pred6.npy`, `deps/`), `viz/tree` (code tip a99ca5d6), `viz/work` (checkpoints, eval context).
