# RESULT_V8F_T1_HEADS: on its own motion, the S head still sets the route (S ADDS ROUTE), but the T head's decisions DEGRADE

`Project Steering/PREREG_V8F_T1_HEADS.md` (`ae659d49`), with Amendment 1 (`33085ce8`, adopted before any decisive head output was read). Claim row
`H-V8F-T1-HEADS-1`.

* **Read:** `raw/2026-10-26-t1heads-read/T1HD_READ.json` (sha256 `53225b30…`), by `raw/2026-10-26-t1hdread/t1hd_read.py`, applied verbatim.
* **Runs:** four rollouts of `v8f_t1_eval --dump-ticks --dump-heads` (`81aa3e36`): HACC m = 10, causal bucket (decisive) and logged bucket (K4),
  seeds 0 and 1, on eval-141 (2,399 windows, 141 episodes).
* **Tier:** T1 (T1-kin: a self-action OPEN loop, never "closed loop").
  * The cameras are the LOGGED ones at the clock row, and the nav is dead-reckoned.
  * Each head output is scored against the human's labels at the nearest logged point of the route (§1 of the PREREG). Ticks 1–2 are the
    model's own state; tick 0 is the logged state.
* **Estimator:** paired episode-cluster bootstrap, B = 2,000; F1 through the T0 eval's own reducer.

## Controls: all PASS

| control | read |
|---|---|
| K1 identity | exactly 0 on every row |
| K2 frames | the tool's own float32 transform replayed: max 0.0 m |
| K3 the tick-0 match | r′ = t0 for every window |
| **K4 tick 0 is T0** | logged run, s0: S pos_err 0.97736 / 2.29654 / 5.88728 m against the banked T0 fresh 0.9774 / 2.2965 / 5.8873 (Δ ≤ 4.4e-5); lat / lon F1 0.83517 / 0.50620, **Δ = 0.0** |
| K5 strategic regression (mirrored S − S) | separated-worse at all three arcs (+1.0 / +3.7 / +9.4 m at s0) |
| K6 tactical regression (cross-episode donor − self) | separated-worse: lat F1 0.355 vs 0.614, lon 0.177 vs 0.336 at s0 |
| K7 the run is the banked run | `cl`, `fl` and `g` are bit-identical to the banked causal dumps |

## STRATEGIC: **S ADDS ROUTE** (verbatim)

S − CTRV checkpoint position error (m), ticks ≥ 1, decisive configuration:

| arc | s0 | s1 | S / CTRV level, s0 (m) |
|---|---|---|---|
| 30 m | **−1.18 [−1.58, −0.84]** | **−1.26 [−1.64, −0.91]** | 4.90 / 6.08 |
| 50 m | **−3.27 [−4.12, −2.51]** | **−3.47 [−4.48, −2.61]** | 8.88 / 12.16 |
| 80 m | **−7.79 [−9.73, −6.07]** | **−8.81 [−10.87, −6.95]** | 16.07 / 23.86 |

All three arcs are separated-better at both seeds. Beside, never decisive:
* **Against STRAIGHT-ahead,** S is level at 30 m (4.90 vs 4.89 m at s0) and better at 50 m (8.9 vs 9.8) and at 80 m (16.1 vs 19.7). The CTRV
  extrapolation of the executed track's curvature is the weaker kinematic baseline near the car.
* ⚠️ **S's own error grows about 5× from tick 0 to the own-state ticks** (0.98 → 4.90 m at 30 m). Much of that is shared with any baseline,
  because the scored point moves with the car (CTRV 2.47 → 6.08 m). It is reported, and it is not yet attributed.
* Off-route outputs (more than 3 m from the logged route) are never scored. There are none at ticks 0–1, and 240 / 284 windows at tick 2.

## TACTICAL DECISION: **DEGRADES** (verbatim)

| row | s0 | s1 | F1 at tick 0 → ticks ≥ 1, s0 / s1 |
|---|---|---|---|
| T-LAT-F1 Δ (2F = 0.0248) | **−0.235 [−0.305, −0.174]** | **−0.227 [−0.297, −0.169]** | 0.848 → 0.614 / 0.858 → 0.632 |
| T-LON-F1 Δ (2F = 0.0146) | **−0.125 [−0.185, −0.060]** | **−0.131 [−0.182, −0.034]** | 0.461 → 0.336 / 0.487 → 0.357 |

Beside:
* **The loss is in the turns:** per-class lateral F1 for TURN_L / TURN_R falls from 0.81 / 0.75 to 0.38 / 0.50 at s0, while LANE_KEEP holds
  (0.98 → 0.96).
* **Goal macro-F1:** 0.266 → 0.214 / 0.259 → 0.186.
* **The crawl stratum of `R76`:** lower F1 there (lat 0.42 vs 0.63 at s0), but it holds only 4.5 % / 2.0 % of the cells, so the drop is not a
  crawl artefact.

**Is it the T head, or the replayed cameras?** This was checked descriptively, after the outcome was read, and it decides nothing
(`PROBE_LAG.json`, `probe_lag.py`, plain macro-F1).
* T1-kin shows the model the logged cameras of the clock row. A car that lags therefore sees a view from further along the route than its own
  position.
* But **at zero row offset, where the label row is the very row whose cameras T saw**, lateral F1 is still 0.606 / 0.647 against 0.828 / 0.857
  at tick 0. Longitudinal F1 is 0.350 / 0.364 against 0.461 / 0.467, and it falls further as the offset grows.
* ⇒ Most of the drop is the T head reacting to its OWN ego state (the executed speed, yaw rate and history it is fed), not to the camera
  mismatch.

## What this settles

* ⇒ **The strategic level works on the model's own motion:** S's checkpoints track the human's route better than a kinematic arc, at both
  seeds. This is the first strategic-family read at T1.
* ⇒ **The tactical level does not:** given its own executed state, T's turn decisions lose about half their F1. Together with
  `RESULT_V8F_FLAT` (O gains nothing over flat from S's and T's own outputs), this localises the hierarchy's defect at the T level and its
  interface to O, not at S.
* **Committed next (outcome row):** T supervision is folded into **O-RECOVER**, an ego-state-perturbation fine-tune (`PREREG_V8F_HACC_MC` §7).
  * It trains on perturbed ego states toward the human's labels and future, so T and O learn to decide from states off the logged trajectory.
  * It needs box-2 time and the PI's sign-off. Box 2 is running OACC now, and MIXO (the FLAT read's committed lever) is next in that queue.
