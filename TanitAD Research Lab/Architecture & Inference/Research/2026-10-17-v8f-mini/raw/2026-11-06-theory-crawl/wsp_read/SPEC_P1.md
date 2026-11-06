# SPEC_P1: the return-map read, fixed before any aggregate of the data was computed

Author: WSP-READ agent, 2026-11-06. Source claim: `THEORY_CRAWL.md` (sha256 `6e83451c...`) P1, read as written. Evidence class of anything this spec leads to: MEASURED (ours, T2 re-perception rollouts, dev-box CPU).
What was looked at before this file: ONE record's field layout (scene 132ae6cb, C0C_s0) to learn the schema. No aggregate, no bin, no per-arm number.

## Definitions
* Unit of data: a tick. A record (start) has 3 ticks, k = 0, 1, 2, 2.0 s apart (rig index 2*step_k on the 10 Hz track, step_k = extra[k].step). Tick 0 has the human's history, ticks 1 and 2 have the arm's own executed history (the 1.8 s window is the arm's own from tick 1).
* v_k = `extra[k].v0_ms` (equals `track[2*step_k][3]`, checked).
* g_plan(v_k) = the plan's speed at +2.0 s = |plan[11] - plan[7]| / 0.8 on `plans[k]` (plan[j] is at (j+1)*0.2 s; a 0.8 s secant centred on 2.0 s). `plans[k]` is the plan the executor was given (equals `extra[k].plan_s`).
* g_exec(v_k) = the executed speed at the next tick = `track[2*step_k + 20][3]` (for k = 2 this is the last track row).
* Human: matched logged speed at the same scene and time = `ego_from_rig` v (backward difference, as `t2v8f_lead.ego_from_rig`) at rig index f0 + 20k; human next = f0 + 20(k+1).
* Primary map = g_plan (the task: "g(v) is the plan's next speed against the executed one"). g_exec is the second map (the closed iteration). Both are reported with identical statistics.
* Bins: 0.5 m/s over 0 to 12 (24 bins), by v_k. Band = v_k in [3, 8). Windows with v_k >= 12 are counted, not binned.
* Population: T2 / objects / deciding cells, DONE-marked, excluding the 14 R-SCN scenes of `stage4/EXCLUSIONS4.json`. Arms: C0C_s0, C0C_s1 (stage 4, 62 scenes), P10 (stages 2 and 3), plus C0_s0, C0_s1 (stages 2 and 3, as extra arms). A C0C-vs-P10 contrast is on the intersection of (scene, f0) keys only.
* Interval: scene-cluster bootstrap, B = 2000, `numpy.random.default_rng(0)`, resampling scenes with replacement; a ratio of sums within the resample. Paired contrasts resample the same scenes for both arms.
* Fit: linear g = a + b*v over v_k in [0, 12) (all windows, sufficient statistics per scene). v* = a / (1 - b) if b < 1 (stable fixed point), else "none". Slope near v*: the same fit restricted to |v_k - v*_hat| <= 1.5 m/s, window fixed from the point estimate.
* Bin-crossing v*: the smallest bin centre where the bin mean of (g - v) changes from > 0 to <= 0 going up in v, among bins with n >= 10; reported beside the fit, never instead of it.
* Human controls (added to separate the planner from regression to the mean in v_k): the human's own return map g_h(v_k) = logged speed at +2.0 s against logged speed now, on the same windows; and the human's median speed over the matched windows (all ticks; and over windows whose logged speed is in the band).

## Verdict rule (committed)
* Leg A, contraction. Per C0C seed, B = window-weighted mean of (g - v) over the band. HOLDS if B < 0 with a 95 % CI upper bound < 0 for both seeds. FAILS if the band-mean B >= 0 for both seeds (this is P1's stated fail condition, g >= v over the band). Otherwise INCONCLUSIVE.
* Leg B, fixed point below the human. HOLDS if v* exists, lies in [3, 8), and (v* - human median, matched windows) has a 95 % CI upper bound < 0 for both seeds.
* Leg C, P10 nearer the identity. HOLDS if (B_C0C - B_P10) on the paired windows has a 95 % CI upper bound < 0 for both seeds.
* P1 verdict: HOLDS if A, B and C all hold; FAILS if A fails; otherwise PARTIAL with each leg named. The verdict is first stated for g_plan. If g_exec disagrees, both are reported and neither is promoted.
* Nothing here is changed after the data is read. If the data show a defect in this spec, it is reported as a defect with a new, dated spec beside it.
