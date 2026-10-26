# PREREG_V8F_T1_HEADS: the STRATEGIC family and the T head's own decisions at T1-kin, from per-tick head dumps

**Written 2026-10-26, before any head output of any T1 run exists.** Claim row `H-V8F-T1-HEADS-1`. Zero training. One new rollout set (§3).

The facts it rests on come from a source fact sheet: `raw/2026-10-26-t1heads/FACTS.md`, with `file:line` for every item.

## Why

`RESULT_V8F_T1KIN_HACC.md` row 27 still reads "STRATEGIC and the model's own T code: NOT COMPUTED".
* The trajectory-derived tactical family and distance keeping are read (`RESULT_V8F_T1_FAMILIES`). They say what the car DID.
* This read asks what the hierarchy DECIDED while driving on its own motion.
* `v8f_t1_eval --dump-ticks --dump-heads` (`81aa3e36`) records the S and T head outputs at every model tick, from the same forward call that made the plan.

## 1. Scoring at a tick: match the route, not the clock

At a T1 tick the executed car is not where the human was, so the human's labels at the same ROW describe a different place. Each head output is
therefore scored against the human's labels at the **nearest logged point of the route**:

* r′_S = the logged row of the same clip, at or after t0, whose logged pose is nearest in world xy to the pose S was computed from (`s_pose`).
* r′_T = the same for `t_pose`.
* If the nearest distance exceeds **3.0 m**, the output is **OFF-ROUTE**: counted per head and per tick, never scored.
* At tick 0 the executed pose is the logged one, so r′ = t0. This reproduces the T0 scoring (control K3).

The GT is v9 eval139:
* S: `rc_A30/A50/A80_{x,y,psi,valid}` at r′_S, transformed from the logged frame at r′_S into the window-origin frame;
* T: `lat_cls` / `lon_cls` / goal at r′_T.

The vocabularies are lat 5, lon 7 and goals 22 (`v8f_mini_data`). The S output is compared in the same frame (`s_ckpt_w0`).

## 2. Rows

**STRATEGIC**, decisive configuration = HACC with the causal speed bucket, m = 10. Per window, the mean over its scored ticks ≥ 1 (own state).

| row | definition | comparator | worse if |
|---|---|---|---|
| **S-POS@30 / @50 / @80** (m) | \|S checkpoint − GT checkpoint at r′_S\| | **CTRV**: constant speed and yaw rate from the same pose; the curvature is the executed track's mean over the 1 s before that row (logged poses where that second precedes t0); points at arc 30 / 50 / 80 m | S − CTRV > 0 |

Beside, never decisive:
* the heading error S-HEAD@30 / 50 / 80;
* S − STRAIGHT (straight ahead along the pose heading);
* the tick-0 rows;
* the degradation (ticks ≥ 1) − (tick 0).

**TACTICAL DECISION** (the T head's argmax), same configuration and ticks.

| row | definition | worse if |
|---|---|---|
| **T-LAT-F1 / T-LON-F1** | macro-F1 of the argmax against the label at r′_T: (ticks ≥ 1) − (tick 0), on the same windows | see the outcomes |

* **Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0, over the 141 episodes. For F1 it uses the T0 eval's own F1
  reducer (`v8f_mini_eval._f1_reducer`): both sides are recomputed in every resample.
* **Seed floor F** (MEASURED on the banked T0 "fresh" evals, `raw/2026-10-25-sfresh/eval/eval_M2b_s{0,1}-sfresh-fresh.json`): lat 0.0124,
  lon 0.0073. ⇒ 2F = 0.0248 / 0.0146.
* **Beside:**
  * goal macro-F1;
  * accuracy;
  * the majority baseline (lat 0.3170, lon 0.0707);
  * the per-class confusion;
  * the **crawl stratum**: ticks whose own speed is below the GT speed at r′_T − 2 m/s (`R76`'s launch crawl).
* **Tier:** T1 (T1-kin: a self-action OPEN loop, never "closed loop"). The nav input is dead-reckoned. S and T are sample-and-hold at the
  tool's ages.

## 3. The run

`v8f_t1_eval.py run` from a pinned export of `81aa3e36` (stack + taniteval), sha256-pinned before and after every run. It uses the inputs, flags
and checkpoints of the banked causal run (`raw/2026-10-26-t1kin-causalbucket`), plus `--dump-ticks --dump-heads`.

| configuration | seeds | role |
|---|---|---|
| `--limit causal` | 0, 1 | **decisive** |
| `--limit logged` | 0, 1 | control K4 and beside |

## 4. Controls (each must pass, or the read is REFUSED)

| id | control | must read |
|---|---|---|
| K1 | identity | every row's paired delta exactly +0 [0, 0] (S vs itself; T vs itself) |
| K2 | frames | `s_ckpt_w0` equals `s_ckpt` transformed by `s_pose` / `pose_w0` to ≤ 1e-4 m on every row |
| K3 | the tick-0 match | r′ = t0 at tick 0 for every window, matching distance ≤ 1e-6 m |
| K4 | tick 0 is T0 | logged-bucket run, tick 0: S pos_err@30/50/80 and T lat/lon macro-F1 equal the banked T0 fresh values to 1e-3 — s0 0.977 / 2.296 / 5.887 m, F1 0.8352 / 0.5062; s1 0.929 / 2.204 / 5.896 m, F1 0.8476 / 0.5135 (the fact sheet's §5). A failure is REFUSED **and diagnosed** (e.g. the dead-reckoned nav at tick 0, or the window set), never loosened |
| K5 | strategic deliberate regression | S checkpoints mirrored (y → −y in S's own tick frame) vs S itself: S-POS@30, @50 and @80 each separated-worse at s0 |
| K6 | tactical deliberate regression | T outputs of a speed-matched CROSS-EPISODE donor window (the same tick index; the nearest v0, ties to the lowest index) scored against this window's labels, vs the arm itself: T-LAT-F1 and T-LON-F1 separated-worse at s0 |
| K7 | the run is the banked run | the causal run's `cl`, `fl` and `g` dumps equal the banked causal dumps bit for bit |

Both regression arms are compared with the arm they degrade, not with a third arm (the lesson of `R75`).

## 5. Outcomes, committed now, per family (never pooled)

**STRATEGIC:**

| outcome | condition | committed next |
|---|---|---|
| **S ADDS ROUTE** | S-POS separated-better than CTRV at all three arcs, at both seeds | the S level uses the route while the model drives itself: support for the hierarchy claim at T1 |
| **NO WORSE** | no arc separated-worse at either seed, but not ADDS | diagnose what S lacks at T1: the dead-reckoned nav's drift against the logged nav, then S's staleness at T1 |
| **WORSE** | any arc separated-worse at ≥ 1 seed | the same diagnosis; S is reported as worse than kinematics at T1 |

**TACTICAL DECISION:**

| outcome | condition | committed next |
|---|---|---|
| **HOLDS** | the lower bound of Δ ≥ −2F, for lat and lon, at both seeds | support: T's decision quality survives its own motion |
| **DEGRADES** | the upper bound of Δ < 0 and the point Δ < −2F, for lat or lon, at ≥ 1 seed | T is not robust to its own state. T supervision is folded into O-RECOVER (`PREREG_V8F_HACC_MC` §7; box 2, needs PI sign-off) |
| **INCONCLUSIVE** | otherwise | n and the CI width are reported; next is m = 5 (more ticks per window), under its own read |

**REFUSED:** any control fails.

## 6. Not covered

* The S and T outputs of the beside configurations (strict, logged) beyond K4.
* Any training arm.

## Amendment 1 (2026-10-26): the reader's interpretations adopted, before any decisive head output was read

`raw/2026-10-26-t1hdread/NOTES.md` items 1–20 are adopted as written. The four that resolve the text, each consistent with the sections above:
1. **CTRV's curvature** is the mean of `_pose_kappa0` over the 10 raw frames (1 s) before S's row, on the rebuilt executed track (logged
   frames before t0). A second curvature rule, CTRV2, is reported beside only and decides nothing.
2. **"The same windows" for the T rows** are the windows with a scored tick-0 cell and at least one scored cell at ticks 1–2.
3. **K6** is a level comparison: F1(donor cells) − F1(own cells) on the same cells.
4. **The F1 reducer is unchanged.** It is wrapped with an integer cast, because the bootstrap passes float arrays and numpy 2.5 rejects float
   indices.

**What existed when this was written:**
* The decisive causal s1 rollout had finished. Only its K2, K3 and executed-track-rebuild checks were computed, as a scale test. No S or T row
  of any real run was computed.
* Both deliberate-regression effects had been computed on the banked T0 evals before any T1 data (the `R75` lesson): K5 mirror − S is +3.5 /
  +8.6 / +17.7 m and K6 donor − self is −0.48 / −0.30 F1, all separated.

Nothing else moves.
