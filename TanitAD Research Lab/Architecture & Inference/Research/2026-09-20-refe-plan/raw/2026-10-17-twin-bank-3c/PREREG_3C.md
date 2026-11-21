# PRE-REGISTRATION — P6 lane-rank twin bank, option 3-C (commit filter). Written 2026-10-17 by the Master Mind, BEFORE any production number.

**Decision.** PI, 2026-10-17, choosing among 3-C / 3-S / 3-T after `raw/2026-10-16-twin-reroll/RESULT.md` (landed a1a83d2):
*"3-C commit filter (Recommended)"*.

**Input (MEASURED 2026-10-17 on the pod).** `/workspace/data/refe_navtrain/train_grow_cmd/targets_rank1.jsonl`: 86,851 rows;
`aug.kind = lane_rank` K=1 **32,929**, K=2 **8,090**, K=3 **3,390** (= 44,409); `goal_horizon_s` **42,442**. `targets_rank0.jsonl` is not touched.

## Procedure (per lane-rank twin; every other row passes through BYTE-IDENTICAL)
1. Teacher on CPU, `REFE_SIM_HZ=10`, rank-K route **held** for the whole rollout (`REFE_ROUTE_HOLD=1`, `route_lane_rank_patch.held_route`),
   **goal horizon 6 s** (`route_goal_horizon_s = 6.0`; `route_goal_min_speed_mps` left at the config default), rolled **10 s** from the twin's frame.
2. f(t) = d_logged / (d_logged + d_twin) at 1..10 s against the twin's and the logged PDM centrelines (`mc_final` parts, exactly as
   `probe/long_rollout.py`). Missing parts or a scenario too short for 10 s ⇒ the twin is DROPPED and counted by reason.
3. **KEEP iff f(10 s) >= 0.75.** A kept row = the input row with `traj`/`traj_world` = the FIRST 4 s of that rollout, `goal` = the goal the
   teacher received, and `aug` += {`route_hold: step0`, `goal_horizon_s: 6.0`, `commit_f10`, `reroll: 3c`}. A dropped twin is absent from the output.

## Identity controls (on the 74-twin D7 mini bank, BEFORE the full run; any failure stops the build)
- **I1** the per-twin f(t) and commit set reproduce the probe's `held_h6` rows (`raw/2026-10-16-twin-reroll/results/long_lever/dedup/held_h6_s*.jsonl`) bit-for-bit.
- **I2** each kept row's 4 s `traj` equals the probe's 4 s re-roll `rr_h6` trajectory bit-for-bit (the first 4 s of a 10 s rollout IS a 4 s rollout).
- **I3** every non-lane-rank row is byte-identical to the input; the output's line count = input − dropped.
- **I4 (mutation)** with `REFE_ROUTE_HOLD=0` the commit set must DIFFER from I1's (the check can see the hold).

## Acceptance (c'), replacing D7 (c) for this bank (per rank, judged where n >= 5)
- **(c'1)** every kept twin has f(10 s) >= 0.75 (asserted on the audit file; true by construction).
- **(c'2) no slowdown:** a random sample of up to 300 kept twins per rank (seed 20261017), paired with a route-held 12 s 4 s re-roll of the
  same twin: path-length ratio median >= **0.95** AND end-speed change median >= **−0.3 m/s**.
- **(c'3)** D7 `diag_d7_twin_route.py verify` checks (a) and (b) still pass on the new bank (instrument unchanged; (c) REPORTED, not judged —
  it is unsatisfiable for this teacher, addendum 2 of the re-roll prereg).
- **Yield guard:** expected keep share K1 0.37 / K2 0.29 / K3 0.22 (n 46/14/9 on the mini bank, no interval). **If the K=1 keep share is
  < 0.20 the build STOPS and is reported** (the mini bank was unrepresentative) — no bank is handed to P6.

## Committed outcomes
- **3C-PASS**: I1-I4 pass, (c'1)-(c'3) pass, yield guard holds ⇒ the bank goes to the D7 cache step (`build_metric_cache.py build --lane-rank K`
  for any kept twin without parts) and P6 re-gates on it (new data-manifest sha ⇒ new launch token).
- **3C-SLOW**: (c'2) fails ⇒ the h6 filter does not remove the slowdown at scale; report per rank; next lever: h8 + commit filter.
- **3C-YIELD**: the yield guard trips ⇒ report the per-rank keep shares and the drop reasons; PI decision.
- **3C-INVALID**: any identity control fails ⇒ no number from the full run is read.

**Cost (ESTIMATED from the probe: ~20 s per 10 s rollout per process):** 44,409 × 20 s / 90 processes ≈ 2.7 h of pod CPU. It runs BEFORE the
safety-objective A40 arm launches (never add load to a training pod).
**Not changed:** rank-0 rows, goal-horizon twins, the PDM-route-rank fix, D7's instrument.
