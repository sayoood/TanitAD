# v8F-mini LONGITUDINAL distance keeping: the instrument is now reached, and what it reads (2026-10-22)

**Stamps**
* **Tier:** T1-class open loop, the stamp the source evals carry. The arm is conditioned on its own held upper-layer outputs, never on GT. The lead's logged future does not react to the arm, so "closes on the lead" means "closes on where the lead was".
* **Evidence class:** MEASURED (ours). Every number below re-derives from `raw/2026-10-22-dk/dk_read.json`, `raw/2026-10-22-dk/dk_perwindow.npz` and the four banked `raw/full/eval/eval_*.npz`.
* **Estimator:** `ci.episode_cluster_bootstrap` for means and `ci.paired_episode_cluster_bootstrap` for every delta, B = 2,000, resampling EPISODES. It answers "would another draw of episodes say this?" only. Training-seed variance enters only through F = |mean s0 - mean s1| per arm (two seeds). Inference variance is not in play: the eval path is `.eval()` plus `no_grad` with no sampling call (a grep of `v8f_mini_eval.py` and `forward_infer`; the checkpoints were not re-run twice).
* **n:** the grid is 2,399 windows in 141 episodes. 743 windows (31.0 %) in 77 episodes are LEAD windows, and every distance-keeping mean is over those. Per-metric n is in the tables. The bootstrap resamples the 77 episodes that carry data, not 141.
* **No bar is registered** for distance keeping on v8F-mini, so nothing here is PASS or FAIL. A positive delta means MORE gap, time or TTC than the comparator, not "better". The reading of better is the distance to the human and LEAD-VIOL.

## 1. What was wrong, and what was done

BACKLOG R1 (`D-REFAV1-LEAD-BLOCK`, 2026-09-02) already built the instrument for this exact 141-episode grid: `build_lead_block_b1.py` (obstacle.offline) -> the banked per-frame block -> `lead_metrics.py`. Only `refav1_arm.py` ever called it. v8F-mini's `families()` still says "the family's own lead-track input is not built (UNAVAILABLE)" (`stack/scripts/v8f_mini_eval.py:79`). This is the "built, tested, unreachable from its caller" class.

`code/v8f_dk_read.py` is the missing call site. It defines **no new metric**: every number is `lead_metrics.distance_keeping`, `distance_keeping_by_speed`, `paired_distance_keeping`, `lead_viol` or `paired_lead_viol`, on the arm's `pred` path against the same logged lead future as the human (GT) path.

Two additions that are readings of the same functions, not metrics:
* **Distance to the human:** `|arm - human|` per window, and the signed arm-minus-human, both on windows where both are finite.
* **Stride-2 TTC:** the same `distance_keeping` called on every other waypoint with `dt = 0.4`. It is a sensitivity check on path jitter.

## 2. Frames and units (established from source, then re-proved on the data)

| object | frame and units | where established |
|---|---|---|
| `pred` [N,30,2] | metres, window-origin ego frame at cache step t (x forward, y left), knots t+0.2 s ... t+6.0 s | scored against the GT by `np.linalg.norm(pred - gt, axis=-1)`, `stack/scripts/v8f_mini_eval.py:69`; `DT = 0.2` at `stack/tanitad/data/v8f_mini_data.py:49` |
| GT [N,30,2] | same frame: `fk.gt_waypoints(grid, ts, 1..30)` rotates by -yaw(t0) over the 5 Hz grid `poses[0::2]` | `v8f_mini_data.py:371-388`, `taniteval/taniteval/floors_kin.py:213-223` |
| window <-> raw frame | cache step t <-> raw 10 Hz frame 2t | `v8f_mini_data.py:385` (`k = 2 * ts`); `build_lead_block_b1.py:31` |
| lead block | lead CENTRE in the same window-origin frame, x forward, y left, metres; `ts_rel_s = 0.2 * (1..K)`; gap = along - size_x/2 (rig origin to the lead's rear face) | `build_lead_block_b1.py:42`, `lead_metrics.py:23-34` |
| ego speed | m/s; `v0` of the eval equals the block's `speeds` at the joined row | proved below |

**Proofs run on the real data** (each is a refusal in the reader):
* **GT rebuild:** the rebuilt GT reproduces every arm's saved `ade6`, max |diff| = 0.0 on all four arms. The P10-yaw-4512 predictions reproduce `floors_perwindow.npz` `P10_refit__ade6`, max |diff| = 0.0.
* **Join coverage:** all 2,399 windows have a row in the block, keyed by (clip, frame 2t).
* **Speed proof:** max |eval v0 - block speed| = 2.62e-05 m/s, against a 1e-3 tolerance (`refav1_arm.py:238`).
* **Kinematic contract:** the human path recomputed here reproduces the block's own `gt_*` columns. Median |d headway| is 0.019 m at 2 s and 0.022 m at 6 s. The check refuses above 0.25 m; a frame or unit error is metres.
* **Inputs:** the four local `eval_*.npz` and `floors_perwindow.npz` are byte-identical to the blobs at `f135c49` (`git hash-object` equality).

## 3. Horizons, and one artifact that is not yet in the repo

The banked block has K = 10 (**2 s**). v8F-mini scores ADE at 2 s and 6 s. I read 2 s from the banked block (sha256 `c0525943...`, unchanged). For **6 s** I rebuilt the block with the same builder at `--k 30` from the same inputs: clip index, egomotion tar, timestamps tar and 145 obstacle parquets are all sha-identical to the banked build's.
* **Control:** on the 6,579 rows that are LEAD in both blocks, the first 10 lead steps, `lead_lens` and `gap0_m` are identical bit for bit.
* **Where they differ:** the 6 s block labels more rows NO_LABEL because a 6 s horizon leaves the labelled span more often. On this grid the LEAD windows are the same 743.
* **Banked:** the K = 30 block is at `raw/2026-10-22-dk/b1_eval_lead_block_k30.npz` (4.0 MB, sha256 `7a879baa...`), with the builder's `.report.json`. It was re-checked independently before landing (section 9.1).

## 4. The numbers

**Window states** (identical for all arms; the block's labels): 2 s: LEAD 743 · NO_LEAD 561 · NOT_STRAIGHT 937 · NO_LABEL 158. 6 s: LEAD 743 · NO_LEAD 552 · NOT_STRAIGHT 931 · NO_LABEL 173.
* NO_LEAD is road clear within 80 m.
* NOT_STRAIGHT means the ego's own future leaves the straight corridor, so a bend hides the lead. It is its own state, never free flow.
* NO_LABEL means no obstacle.offline or a horizon outside the labelled span.
* None of the three enters a mean.

### 2 s horizon (banked block). Mean [95 % CI over episodes]

| run | n headway | headway (m) | n time-gap | time-gap (s) | n TTC (closing) | min-TTC (s) | LEAD-VIOL (at-risk) | progress |
|---|---|---|---|---|---|---|---|---|
| human (GT path) | 719 | 27.46 [23.45, 31.49] | 653 | 4.26 [3.32, 5.43] | 719 (365) | 22.75 [21.01, 24.36] | 0 by construction | 1 |
| M1_s0 | 713 | 27.31 [23.54, 31.48] | 661 | 4.26 [3.35, 5.37] | 713 (423) | 20.56 [18.59, 22.45] | 5/605 = 0.83 % | 0.979 ok |
| M1_s1 | 714 | 27.26 [23.46, 31.46] | 660 | 4.14 [3.27, 5.17] | 714 (383) | 21.18 [19.29, 23.01] | 4/605 = 0.66 % | 0.984 ok |
| M2b_s0 | 705 | 27.52 [23.73, 31.62] | 661 | 4.14 [3.32, 5.16] | 705 (413) | 20.61 [18.67, 22.44] | 5/605 = 0.83 % | 0.980 ok |
| M2b_s1 | 722 | 27.22 [23.44, 31.36] | 661 | 4.29 [3.35, 5.43] | 722 (396) | 21.36 [19.58, 23.07] | 5/605 = 0.83 % | 0.986 ok |
| P10 | 723 | 27.41 [23.66, 31.59] | 661 | 4.22 [3.33, 5.41] | 723 (354) | 22.55 [20.80, 24.22] | 5/605 = 0.83 % | 0.978 ok |

### 6 s horizon (rebuilt K = 30 block)

| run | n headway | headway (m) | n time-gap | time-gap (s) | n TTC (closing) | min-TTC (s) | LEAD-VIOL (at-risk) | progress |
|---|---|---|---|---|---|---|---|---|
| human (GT path) | 725 | 25.66 [21.63, 29.79] | 659 | 4.00 [3.10, 5.08] | 725 (447) | 19.79 [17.41, 21.94] | 0 by construction | 1 |
| M1_s0 | 728 | 25.75 [22.03, 29.95] | 664 | 4.04 [3.12, 5.16] | 728 (471) | 19.33 [17.28, 21.32] | 20/595 = 3.36 % | 0.840 **crawl** |
| M1_s1 | 730 | 25.90 [21.89, 30.01] | 666 | 4.00 [3.10, 5.09] | 730 (433) | 19.70 [17.53, 21.59] | 22/595 = 3.70 % | 0.903 ok |
| M2b_s0 | 722 | 25.62 [21.84, 29.74] | 665 | 4.02 [3.10, 5.27] | 722 (458) | 19.40 [17.31, 21.34] | 21/595 = 3.53 % | 0.863 **crawl** |
| M2b_s1 | 728 | 25.54 [21.63, 29.65] | 663 | 3.95 [3.06, 5.03] | 728 (445) | 19.52 [17.47, 21.32] | 36/595 = 6.05 % | 0.919 ok |
| P10 | 728 | 25.12 [21.21, 29.50] | 663 | 3.83 [2.96, 4.97] | 728 (411) | 19.95 [17.71, 22.02] | 48/595 = 8.07 % | 0.938 ok |

* **Definitions** (from `lead_metrics`, unchanged):
  * headway = min gap over the horizon, rig origin to the lead's rear face;
  * time-gap = headway / v0, undefined (NaN, counted) below 0.5 m/s;
  * min-TTC is capped at 30 s when never closing, so it is a censored mean and `n_closing` rides with it.
* **Why a time-gap n is below the headway n:** the 44 (2 s) and 57 (6 s) standstill windows of M2b_s0 have no time-gap; the other arms are similar. The reasons are in `no_value_reasons`.
* **Lead lost:** 13 to 38 LEAD windows per arm have a path that never lies inside the corridor behind the lead (`lead_window_but_path_never_in_corridor`). They are counted and have no value for that arm.
* **LEAD-VIOL** is the share of at-risk windows (lead window AND human time-gap >= 1.0 s: 605 at 2 s, 595 at 6 s) whose plan's min time-gap is < 1.0 s. "progress" is `lead_progress_ratio_clipped`, the control that stops LEAD-VIOL being met by crawling; below 0.90 the arm fails it.
* **The thresholds** (1.0 s, 0.90) are registered for refcv8 (X7). They are applied here descriptively.

### Paired deltas, a - b (delta [lo, hi]; sep = CI excludes 0)

Direction: positive = more gap, time or TTC than b. LEAD-VIOL rows are on the at-risk denominator, and positive = MORE violations.

| contrast | seed | 2 s headway (m) | 2 s min-TTC (s) | 6 s headway (m) | 6 s time-gap (s) | 6 s min-TTC (s) | 6 s LEAD-VIOL |
|---|---|---|---|---|---|---|---|
| M2b - P10 | 0 | -0.22 [-0.44, +0.01] ns | **-1.76 [-2.58, -1.02] sep** | **+0.53 [+0.08, +0.97] sep** | **+0.186 [+0.044, +0.360] sep** | -0.49 [-1.57, +0.65] ns | **-0.0454 [-0.0752, -0.0189] sep** |
| M2b - P10 | 1 | -0.11 [-0.28, +0.10] ns | **-1.19 [-1.80, -0.64] sep** | **+0.42 [+0.01, +0.85] sep** | +0.117 [-0.004, +0.264] ns | -0.43 [-1.27, +0.52] ns | -0.0202 [-0.0490, +0.0050] ns |
| M1 - P10 | 0 | -0.15 [-0.33, +0.03] ns | **-1.89 [-2.80, -1.09] sep** | **+0.59 [+0.13, +1.08] sep** | **+0.207 [+0.078, +0.378] sep** | -0.62 [-1.80, +0.58] ns | **-0.0471 [-0.0766, -0.0211] sep** |
| M1 - P10 | 1 | -0.11 [-0.31, +0.11] ns | **-1.28 [-2.11, -0.58] sep** | **+0.73 [+0.24, +1.30] sep** | **+0.175 [+0.044, +0.331] sep** | -0.26 [-1.18, +0.75] ns | **-0.0437 [-0.0754, -0.0164] sep** |
| M2b - M1 | 0 | -0.07 [-0.15, +0.01] ns | +0.16 [-0.21, +0.59] ns | -0.04 [-0.31, +0.21] ns | -0.015 [-0.100, +0.081] ns | +0.16 [-0.23, +0.63] ns | +0.0017 [-0.0070, +0.0101] ns |
| M2b - M1 | 1 | -0.01 [-0.10, +0.07] ns | +0.08 [-0.30, +0.51] ns | -0.26 [-0.65, +0.06] ns | -0.051 [-0.134, +0.015] ns | -0.17 [-0.56, +0.29] ns | **+0.0235 [+0.0084, +0.0412] sep** |

**Seed floors** (F = |s0 - s1|) and the programme's 2F rule, "separated AND |delta| > 2F at BOTH seeds":
* F_M2b: 2 s TTC 0.75 s; 6 s headway 0.078 m, time-gap 0.072 s, TTC 0.13 s, LEAD-VIOL 0.0252.
* F_M1: 2 s TTC 0.62 s; 6 s headway 0.149 m, time-gap 0.038 s, TTC 0.38 s, LEAD-VIOL 0.0034.
* **Passes 2F:**
  * 2 s: M1 - P10 TTC only.
  * 6 s: M2b - P10 headway; M1 - P10 headway and time-gap.
* **LEAD-VIOL at 6 s:** M1 - P10 clears 2F (0.044-0.047 > 0.0068); M2b - P10 does not (0.045 and 0.020 against 2F = 0.050).

### Distance to the human (signed arm - human; the human's own value is the reference)

| run | 2 s headway (m) | 2 s min-TTC (s) | 6 s headway (m) | 6 s time-gap (s) |
|---|---|---|---|---|
| P10 | -0.15 [-0.42, +0.08] ns | -0.15 [-0.49, +0.18] ns | -0.61 [-1.30, +0.02] ns | **-0.195 [-0.413, -0.006] sep** |
| M2b_s0 | **-0.43 [-0.68, -0.22] sep** | **-1.90 [-2.76, -1.08] sep** | -0.17 [-0.74, +0.39] ns | +0.000 [-0.153, +0.226] ns |
| M2b_s1 | **-0.27 [-0.49, -0.06] sep** | **-1.31 [-1.94, -0.69] sep** | -0.17 [-0.75, +0.41] ns | -0.068 [-0.199, +0.080] ns |

Absolute error vs the human, M2b - P10 (positive = farther from the human):
* 2 s TTC: +1.60 [+0.89, +2.35] and +0.92 [+0.40, +1.49], both sep.
* 6 s headway: -0.41 [-0.78, -0.10] and -0.37 [-0.74, -0.08], both sep (closer to the human).
* 6 s time-gap: -0.12 ns at seed 0 and -0.14 sep at seed 1.

## 5. What the numbers say about M2b versus P10

1. **6 s: M2b keeps slightly MORE headway than P10 and sits closer to the human's.**
   * +0.53 and +0.42 m (about 2 % of a 25 m gap), separated at both seeds and above 2F (0.16 m).
   * P10 reads 0.61 m under the human's headway (ns, the CI touches 0) and 0.195 s under its time-gap (separated). M2b is statistically on the human's value.
   * This fits the lon-rough note (P10 overshoots progress by 10-11 m on stops and hard braking), but the mechanism here is **UNVERIFIED**.
2. **6 s violations: not established across seeds.**
   * P10 violates the 1.0 s time-gap on 8.07 % of at-risk windows (48/595). M2b_s0 is at 3.53 % and M2b_s1 at 6.05 %.
   * The delta is separated at seed 0 only. F_M2b = 0.025 makes 2F = 0.050, above both deltas.
   * M2b_s0's apparent gain sits under a **failed progress control** (0.863 < 0.90): a plan that covers less of the human's distance violates less.
   * M2b_s1 passes progress (0.919) and its violation gain over P10 is not separated.
3. **2 s: M2b closes on the lead FASTER than P10 and than the human.**
   * Min-TTC is lower by 1.76 s and 1.19 s, separated at both seeds. M2b sits 1.9 s and 1.3 s under the human's own value, and P10 matches the human.
   * More windows close: about 400 of 710, against 354 for P10 and 365 for the human.
   * The 2F rule is NOT met at seed 1 (1.19 < 1.50), so the size is seed-dependent while the sign is not.
   * **Jitter is not the cause.** The learned arms' paths are about twice as rough as the human's (`2026-10-20-lon-rough/NOTE.md`), and min-TTC is a min over per-step closing rates. The stride-2 check (0.4 s spacing) keeps the deficit: M2b - P10 reads -1.58 [-2.36, -0.83] and -1.12 [-1.61, -0.62], so roughness explains at most about 10 %. The mechanism is **UNVERIFIED**.
   * At 6 s the TTC difference is gone (ns at both seeds).
4. **2 s LEAD-VIOL cannot discriminate.** 5 violations in 605 at-risk windows for P10 and for M2b at both seeds; 4 or 5 for M1. Nothing to read.
5. **M2b - M1 (the world-model lever on distance keeping): nothing consistent.**
   * Every delta in the table is ns at both seeds and both horizons, except 6 s LEAD-VIOL at seed 1: M2b is worse by +0.0235 [+0.0084, +0.0412], against +0.0017 ns at seed 0. That is exactly the seed disagreement F_M2b (0.025) already shows.
   * The abs-error-to-human time-gap contrast is also separated at seed 1 only (+0.022 s at 2 s, +0.057 s at 6 s).
6. **M1 is the cleaner distance-keeping arm against P10.**
   * It clears 2F at 6 s on headway, time-gap and LEAD-VIOL.
   * But only M1_s1 passes the progress control (0.903); M1_s0 fails it (0.840).

**Net:**
* Distance keeping gives no evidence that M2b is worse than P10 at 6 s. It is modestly better on headway, replicated.
* The 2 s closing is a real, unexplained cost. It is not mirrored by LEAD-VIOL, because the 1.0 s threshold is far from the 2 s gaps (27 m, 4.2 s).
* The violation story at 6 s is seed-dependent and partly crawl-confounded. A third seed, or a progress-matched comparison, is the discriminating next experiment.

## 6. Coverage: what this read does NOT score

* **NOT_STRAIGHT: 937 windows (39 %).** These include 315 windows where the v9 labels say a lead exists (`lead_valid`, 1,061 windows in all; 665 are also B1 LEAD, 396 are v9-only, 78 are B1-only). Distance keeping on a bend is not scored. The instrument withholds it on purpose, because a straight corridor cannot describe a curved lane, and scoring those windows as NO_LEAD would manufacture free flow.
* **Per-metric n:** only 31 % of windows carry a lead and 77 of 141 episodes contribute. Speed-band strata (`by_speed` in the JSON) are all above `MIN_STRATUM_N = 30` for M2b_s0, but are descriptive only.
* **Open loop:** the lead does not react (see the stamp).
* **Existing eval column:** the eval's `viol_LEAD` (the v9 CONSISTENCY LEAD constraint, `v8f_consistency.py`) is a label-constraint check, not headway, time-gap or TTC. It is not replaced by this.

## 7. Controls and tests

* **Tests:** `stack/tests/test_v8f_dk.py`, 25 tests, CPU, 10 s (`raw/2026-10-22-dk/test_v8f_dk.out`), covering:
  * a synthetic straight road with analytic answers (headway = g0 - 0.4 K, time-gap = headway / v0, min-TTC = (g0 - 0.4 (K-1)) / 2);
  * an EXACT 0.0 [0.0, 0.0] delta for identical arms, on every metric, on the abs-error contrast and on LEAD-VIOL;
  * a deliberate-regression plan at 13 m/s that closes on the lead, which reads WORSE on headway (-6.0 m exactly), TTC, time-gap, LEAD-VIOL and distance-to-human, all separated;
  * a crawling plan that fails the progress control;
  * NO_LABEL, NOT_STRAIGHT and NO_LEAD windows carrying a lead track that are still not scored;
  * standstill time-gaps that are counted;
  * a lane-leaving plan whose lost lead is counted;
  * key-not-position joins;
  * 14 refusal tests that exit 2 with a message and write nothing.
* **Mutation check:** `raw/2026-10-22-dk/mutate_dk.py` -> `raw/2026-10-22-dk/mutation_dk.json`. **14 of 14 mutants killed** (re-run before landing, section 9.1).
  * Deleting the gap line, `gap = along - float(lead_len) / 2.0` at `lead_metrics.py:126`, turns 24 of 25 tests red. The `gap = along` variant turns 2 red.
  * The other mutants are: headway max instead of min, lead assignment deleted, speed proof removed, contract check removed, NO_LABEL scored as LEAD, join by position, human reference taken from the arm, paired sign flipped, dt doubled, identity control removed, LEAD-VIOL scored against the arm itself, and the ade guard removed.
  * The first pass left one survivor (the identity-control guard). I added a monkeypatched-noisy-instrument test and it now dies.
* **Reader guards (rc 2):** missing eval, block or P10 file; positional block; horizon-grid mismatch; K above the block's; a grid window with no row; speed proof; kinematic contract; pred in the wrong units (first-step displacement ratio outside [0.2, 5]); ade6 mismatch; eid mismatch; P10 not reproducing the floors; identity control not exactly +0 [0, 0]; an import from the dead G: tree.
* **Instrument provenance:** the JSON records the `lead_metrics.py` and `ci.py` paths imported (the export of `f135c49`, not the venv's stale editable install).

## 8. UNVERIFIED, and integration needed

* **UNVERIFIED:**
  * (a) The mechanism of the 2 s TTC deficit. Jitter is mostly excluded; the early-horizon speed/accel profile of the learned arms is the untested candidate.
  * (b) Everything is two training seeds. F is a 2-point range, and the 2F rule inherits that.
  * (c) The K = 30 block is a rebuild on the dev box, sha-identical in its inputs and bit-identical on the first 10 steps where both blocks say LEAD, but new (re-checked, section 9.1).
  * (d) Not re-run on a checkpoint: the read is on the banked `eval_*.npz` only.
* **Integration: done in the land that carries this file** (the list as the agent escalated it):
  1. Land `code/v8f_dk_read.py` at `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/code/`.
  2. Land `tests/test_v8f_dk.py` at `stack/tests/`.
  3. Land the K = 30 block and the results at `.../2026-10-17-v8f-mini/raw/2026-10-22-dk/`: `b1_eval_lead_block_k30.npz` (+ `.report.json`), `dk_read.json`, `dk_perwindow.npz`, `mutation_dk.json`. The reader's default picks up that K = 30 block path automatically.
  4. Update the "distance keeping: UNAVAILABLE / not instrumented" strings. They are stale for v8F-mini: `v8f_mini_eval.py:76-79` (docstring), `v8f_mini_full_report.py:193` and `:257-258` (the table row and the footnote built from `families.longitudinal.distance_keeping.status`), and `RESULT_FULL.md` lines 44, 63 and 116. Also GOALS_AND_CLAIMS wherever it records v8F-mini's LONGITUDINAL family as incomplete (not searched here).
  5. If wanted: pin "v8F-mini distance keeping is reached" with a test that greps the eval caller for the reader, so this class cannot recur silently.

## 9. Re-verification before landing, and what the 2 s closing comes from (Master Mind, 2026-10-22)

### 9.1 Re-verified, not inherited

* **The read.** Re-run from a clean export of the tip `abccc45`, on the repo's own banked inputs:
  * the inputs: `raw/full/eval/eval_M*_s*.npz`, `floors_perwindow.npz`, the banked B1 block and the banked P10 predictions;
  * the result: every numeric field of the agent's first read, and `dk_perwindow.npz` byte for byte. Only the recorded import paths differ.
  * The banked `raw/2026-10-22-dk/dk_read.json`, `dk_read.out` and `dk_perwindow.npz` are this re-run.
* **Tests and mutants.** 25 / 25 pass and **14 / 14 mutants are killed**, re-run against the same export (`mutation_dk.json`, `mutation_dk.out`). The script is `mutate_dk.py`, run from a copy that points at the export and deletes nothing.
* **The K = 30 block, independently** (`k30_control.py`, `.out`):
  * the same 29,556 (clip, frame) keys as the banked K = 10 block;
  * on the 6,579 rows that are LEAD in both, the first 10 lead steps and `lead_lens` are bit-identical, and `speeds` is identical on every row;
  * the only state changes are rows becoming NO_LABEL (LEAD 1,762, NO_LEAD 1,506, NOT_STRAIGHT 2,382 of 29,556), never the reverse;
  * the builder (`taniteval/tools/build_lead_block_b1.py`) and the four input sha256s are those of the banked build.

### 9.2 The 2 s min-TTC deficit, split by closing class (`ttc_decomp.py`, zero GPU)

min-TTC is capped at 30 s on a path that never closes, so a mean delta mixes classes.

| M2b − P10, 2 s | mean delta | windows only M2b closes on | windows both close on | windows only P10 closes on |
|---|---|---|---|---|
| seed 0 (701 windows) | −1.76 s | −1.09 s (69 windows) | −0.83 s (259) | +0.16 s (6) |
| seed 1 (719 windows) | −1.19 s | −0.73 s (55) | −0.58 s (261) | +0.13 s (4) |

About 60 % is windows where the plan closes and P10's does not; the rest is faster closing where both close.

### 9.3 The low-TTC tail is real (`ttc_tail_ci.py`; paired episode-cluster bootstrap, B 2,000)

Share of LEAD windows whose min-TTC is < 5 s, at 2 s:

| run | share |
|---|---|
| human | 9.6 % |
| P10 | 12.2 % |
| M2b_s0 / s1 | 16.9 % / 14.7 % |
| M1_s0 / s1 | 17.0 % / 14.0 % |

* M2b − P10 is **+4.4 pp [+1.2, +8.2] and +2.5 pp [+0.4, +5.1], separated at both seeds**. The seed floor is F = 2.2 pp, so 2F is met at seed 0 only.
* M2b − human: +7.2 / +4.9 pp, separated.
* **M2b − M1 is flat at both seeds** (−0.3 / +0.8 pp, ns). The early closing belongs to the shared S→T→O stack, not to the world-model lever.
* At 6 s it vanishes against P10 (−0.8 to −1.7 pp, ns). M2b and P10 alike have a larger < 5 s tail than the human (+4.6 to +5.8 pp, separated).

### 9.4 T's code or O? The GT-code oracle (`dk_oracle_read.py`; rule written before the run; EXPLORATORY)

Arms: `eval_<c>-tacorc` (O driven by the whole GT code) and `-tacorc-all-stoplead` (GT except stoplead), all banked. Both read GT, so they are **T0**.

**Verdict by the committed rule: M2b `UPPER`** (M1 reported: `MIXED`). With the GT code, M2b's 2 s tail excess over P10 is no longer separated:

| M2b, 2 s, < 5 s share | seed 0 | seed 1 |
|---|---|---|
| own code − P10 | +4.4 pp, sep | +2.5 pp, sep |
| GT code − P10 | +1.8 pp [−0.3, +4.4], ns | +0.6 pp [−1.6, +3.2], ns |
| GT code − own code | −2.6 pp [−5.1, −0.4], sep | −2.0 pp [−3.9, −0.2], sep |

* **Mean min-TTC is only half fixed:** the GT code closes about half of it (−1.76 → −0.82 s, −1.19 → −0.78 s). O still closes faster than P10 with the GT code, separated at both seeds. So part of the mean deficit is O's own.
* **My stoplead hypothesis is refuted.** GT everywhere except stoplead does as well as the full oracle: −0.3 / +0.7 pp, ns.

### 9.5 Which group carries the fix: leave-one-out (`dk_oracle_loo.py`; rule written before the run; EXPLORATORY)

The loss of group g is S5(GT except g) − S5(whole GT code): the tail that comes back when T's own g is put back. **Verdict by the committed rule: `latcons`, T's LATERAL constraint group, carries the fix. It is the only group that carries it at both seeds, for M2b and for M1 alike:**

| run | latcons: < 5 s share | latcons: mean min-TTC |
|---|---|---|
| M2b_s0 | +3.7 pp [+1.3, +6.5] | −1.51 s |
| M2b_s1 | +1.8 pp [+0.3, +3.5] | −1.07 s |
| M1_s0 | +3.5 pp [+1.3, +6.1] | −1.73 s |
| M1_s1 | +2.7 pp [+0.4, +5.3] | −1.39 s |

All separated.
* lonspeed is sign-inconsistent: −2.3 pp separated at M2b_s0, +2.4 pp separated at M2b_s1.
* goal, latdec, londec and stoplead are flat.

### 9.6 How a lateral error causes longitudinal closing (`latcons_mech.py`, LEAD windows, paired, EXPLORATORY)

The arms: T's own latcons against the GT latcons, with GT everywhere else.
* **O's plan is further along at 2 s:** +0.27, +0.17, +0.27 and +0.26 m (M2b s0 / s1, M1 s0 / s1), separated at all four. Its along-track error against the human rises by the same amount.
* **Its speed at 2 s is lower** (−0.17 to −0.25 m/s, separated): the plan front-loads its progress.
* **Lateral position, lateral error and heading are flat** (heading separated at M1_s1 only). So it is not the instrument's corridor rotating with a heading error. O drives the first two seconds faster when T's lateral constraints are wrong.
* ⇒ **A cross-family coupling inside O: T's lateral-constraint error leaks into O's longitudinal profile.** On straight lead windows it shows up as early closing on the lead.
  * What exactly in latcons does this is not tested; the group holds turn and lane-change timing and geometry (`v8f_mini_data.py` `LAT_CONSTRAINTS`).

### 9.7 What it means for the next lever

* Two levers act on this, and neither is run:
  * **(a) a more accurate T:** the `pool8x20` input arm. latcons is T's second-worst group, at 0.686 SD present-slot RMSE at M2b_s0 (`RESULT_V8F_WC.md`).
  * **(b) an O trained against T's own code**, so a lateral error stops moving its speed. That is the committed next lever of `PREREG_V8F_WC`. This read gives it a measured, O-side defect to remove, which the WC result alone did not.
* Either arm should carry the 2 s < 5 s TTC share and the latcons leave-one-out as pre-registered secondary reads.
* **Caveats:**
  * The oracle reads are T0, exploratory and post hoc, on two training seeds. They guide the next arm and decide nothing.
  * The 2F rule is met at seed 0 only for the M2b − P10 tail.
