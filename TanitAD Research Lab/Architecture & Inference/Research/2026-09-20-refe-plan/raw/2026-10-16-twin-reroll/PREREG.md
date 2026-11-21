# PRE-REGISTRATION — twin re-roll probe (written 2026-10-16 ~20:30Z, BEFORE the first rollout)

**Question.** Is the per-step lane re-ranking (`feature_builder.py:398-418` rebuilds the route from the CURRENT ego every step, and
`route_lane_rank_patch` re-applies "rank K" against that moving anchor) the cause of D7 check (c) failing, and does holding the
step-0 rank-K lane chain fixed for the rollout fix it?

**Twins.** The same 74 D7 mini-bank twins on which (c) failed (`/workspace/ops/d7fix/bank_test` K=1: 51, K=2: 14; `bank_k3` K=3: 9).
Their twin and logged PDM parts in `/workspace/ops/d7fix/mc_final` are reused unchanged: a part depends on (log, token, K) only,
never on the teacher trajectory.

**Arms** (same harness `build_teacher_rollouts.make_row`, CPU teacher, 10 Hz, the pod's frozen D7 tree + the two patched files):
* `legacy` = `REFE_ROUTE_HOLD=0` — the unchanged per-step re-ranking. CONTROL.
* `held`   = `REFE_ROUTE_HOLD=1` — the step-0 lane chain held for the whole rollout. FIX.

**Instrument.** `diag_d7_twin_route.py verify`, UNCHANGED (frozen pod copy), `--twin-cache = --logged-cache = mc_final`, on a bank
holding the re-rolled rows, with `aug.divergence_m` recomputed against the frame's rank-0 row by `augment_search.divergence_m`.

**Primary criterion (bar unchanged from D7):** (c) `traj_closer_to_twin_div_ge_0_5` >= **0.75** per rank. Judged only where that
rank's n (div >= 0.5) >= 5; a rank below n = 5 is reported, not judged.

**Committed outcomes.**
* ROOT CAUSE VERIFIED + FIX WORKS iff `held` clears 0.75 at every judged rank AND `legacy` stays below 0.75 at K=1 (reproduces the
  failure). `legacy` must also reproduce the banked trajectories (median max-displacement vs bank < 0.10 m); if it does not, the
  harness is not the bank's and NOTHING is concluded.
* HOLD NOT SUFFICIENT iff `held` misses 0.75 at a judged rank: then the next lever is diagnosed in the same run (route-hold misses,
  lateral offset reached at 4 s vs the lane-change duration, start-roadblock mismatch, dead-end lanes) — not reported as a finding.

**Secondary (reported, not judged):** mean distance of the trajectory to the teacher's own step-0 rank-K route vs the rank-0 route;
(c) on the FIXED subset the bank's own `divergence_m >= 0.5` selected (paired comparison legacy vs held); per-rollout hold
hits/misses; the goal at step 0 must be unchanged (|Δ| <= 1e-3 m) in both arms; rollout seconds.

## ADDENDUM 1 — written 2026-10-16 20:12Z, AFTER probe 1 and BEFORE the lever run (disclosed)
Probe 1 outcome: **HOLD NOT SUFFICIENT.** `held` (c) = 0.0 / 0.0 / 0.0 (n 30/6/3) vs `legacy` 0.031 / 0.0 / 0.0 (n 32/7/4); `legacy`
reproduces the bank (median 0.000 m). Per the committed outcome the next lever is run in the same session:
* **Lever arm:** route HELD + the released goal-horizon knob shortened to **6 s** and **8 s** (base 12 s) on the same 74 twins, so the
  teacher's two goal points sit nearer and the lateral demand toward the rank-K lane is larger. The row's `goal` becomes the one the
  teacher received (`aug.goal_horizon_s`).
* **Same instrument, same bar:** unchanged verify, (c) >= 0.75 per rank, judged where n (div >= 0.5) >= 5.
* **Committed:** if an arm clears every judged rank, it is the candidate fix (an augmentation-semantics change ⇒ flagged for the PI,
  because it moves the twin's goal and so its goal-rule command). If neither clears, the twins' 4 s horizon cannot reach the rank-K lane
  under this teacher at these knobs, and the remaining levers (per-twin adherence mask; longer-horizon teacher) are named with costs.

## ADDENDUM 2 — written 2026-10-16 20:22Z, AFTER the lever run and BEFORE the long-rollout diagnostic (disclosed)
Lever outcome: **goal horizon 6 s / 8 s do NOT clear (c)** — h6 (c) 0.044 / 0.0 / 0.0 (n 45/13/9), h8 0.0 / 0.0 / 0.0 (n 36/13/8) — although
h6 moves the twin further: median lateral fraction toward the twin lane at 4 s **0.435** (K=1) vs 0.156 held-12 s.
**Derived (analytic, not a measurement):** for a lane change of duration T starting at t = 0 with any profile symmetric about its
midpoint (e.g. min-jerk), the mean lateral fraction over a 4 s window is 1 − T/8 (T <= 4 s) and (c) needs it > 0.5 ⇒ **(c) can pass only
if the whole lane change completes in under 4 s, starting immediately** (with a reaction delay δ: T < 4 − 2δ).
**Discriminating diagnostic (no bank row written):** roll the SAME 74 twins **10 s** (route HELD vs LEGACY, base 12 s horizon) and read
f(t) at 1..10 s against the same PDM centrelines.
* **A — lane-change ONSET:** held median f(8 s) >= 0.75 at K=1 ⇒ the twins ARE rank-K route intents whose 4 s window is the start of a
  lane change; (c) is unattainable for this teacher by construction and the bar, not the bank, is what fails.
* **B — NO COMMITMENT:** held median f(10 s) < 0.5 at K=1 ⇒ the teacher does not follow the rank-K route; the bank needs another lever.
* in between ⇒ reported as such, per rank.
* **Hold relevance (secondary):** legacy vs held f at 6–10 s; the hold should matter only after the ego passes the lane midpoint.

## ADDENDUM 3 — written 2026-10-16 20:31Z, AFTER the 10 s diagnostic and BEFORE the 10 s lever run (disclosed)
10 s outcome: **B — NO COMMITMENT.** Held, base 12 s horizon: median f(t) at K=1 stays 0.045 → 0.238 over 1..10 s (n 46); f(10 s) >= 0.5
on 19/46; K=2 / K=3 medians stay <= 0.04. Legacy is indistinguishable (0.244 at 10 s). Hold misses 0 (the hold engaged on every build).
⇒ the route was never "lost": the teacher simply does not drive onto the rank-K lane. Holding the route is NOT the lever.
**Last cheap lever (same 10 s instrument):** route held + goal horizon **6 s** and **4 s**. Committed reading: an arm whose K=1 median
f(10 s) >= 0.75 makes the teacher COMMIT and is a candidate generator (still not a (c) pass, which is bounded by the lane-change
duration, see addendum 2); an arm below 0.5 means no knob of the released goal pipeline makes this teacher follow the rank-K route.

## ADDENDUM 4 — written 2026-10-16 ~20:58Z (Master Mind), AFTER the 10 s lever run and BEFORE the h3/h2 run (disclosed)
**10 s lever outcome (addendum 3's reading, as written): BETWEEN for both arms — neither a candidate generator nor "no knob".**
Route held, K=1 (n 46, 16 logs), median f(10 s): base 12 s **0.238**, **h6 0.500**, **h4 0.500**; share f(10 s) >= 0.75: 0.174 / **0.370** /
**0.391**; f(4 s) median 0.086 / 0.409 / 0.532 (`results/long_lever/summary.json`, de-duplicated: the run was launched twice and every row was
appended twice; **148/148 duplicate pairs are bit-identical** excl. wall time ⇒ the CPU teacher rollout is DETERMINISTIC, no inference-seed variance
in this instrument; raw appended files kept in `results/long_lever/appended_raw/`). The medians sit at exactly 0.5 because the distribution
is BIMODAL: a twin either commits (f -> 1) or stays (f -> 0).
**Next cheap lever (same instrument, same bars as addendum 3):** route held + goal horizon **3 s** and **2 s**. Committed: K=1 median
f(10 s) >= 0.75 ⇒ candidate generator; < 0.5 ⇒ the horizon knob is exhausted; between ⇒ reported, and the per-twin COMMIT question
(which twins commit, and is a non-committing twin an infeasible intent: dead end, wrong direction, no connected continuation) is the next
diagnostic, read on the rows already banked. A horizon arm is ranked by its share f(10 s) >= 0.75 at K=1, then K=2, then K=3.

## ADDENDUM 5 — written 2026-10-16 ~21:15Z (Master Mind), AFTER the h3/h2 runs and BEFORE any goal-min-speed number (disclosed)
**h3/h2 outcome.** 10 s instrument, route held, K=1 (n 46): median f(10 s) **h3 0.610 (BETWEEN), h2 0.877 (COMMITS: candidate generator per
addendum 3)**; commit share 0.435 / 0.543; K=2 0.286 / 0.357, K=3 0.111 / 0.111. D7 verify, UNCHANGED, (c) at K=1: held 0.000, h6 0.044,
**h3 0.320, h2 0.340** (bar 0.75; FAIL at every horizon, as addendum 2's analytic bound predicts).
**Side effect found (`code/progress_check.py`, 4 s re-rolled trajectories paired with the route-held 12 s arm, n 72):** a shorter horizon
SLOWS the teacher — path-length ratio median h8 0.996, h6 0.988, **h3 0.893, h2 0.810**; share of twins below 0.9: 0.056 / 0.167 / 0.514 /
**0.681**; end-speed change median +0.00 / -0.02 / -0.66 / **-1.32 m/s**. ⇒ an h2 bank would teach "twin = slower": a CONFOUND, refused.
**Mechanism (source, DriveRL `datatypes/.../route_goal_positions` "goals ending at speed * horizon_s ahead", floor `min_speed_mps` default
5.0):** goal distance = max(|v|, min_speed) x horizon, so a slowing ego pulls its own goal closer — a feedback loop.
**Lever (opt-in, `probe/patch_minspeed.py`; unset = byte-identical):** `--goal-min-speed v0max` floors the goal distance at
max(frame speed, 5.0) x horizon for the whole rollout. Arms: route held, h2 and h3, v0max; the 10 s instrument AND the 4 s re-roll +
D7 verify + progress check.
**Committed (K=1):** (i) COMMIT = median f(10 s) >= 0.75; (ii) NO SLOWDOWN = path-ratio median >= 0.95 AND end-speed change median >= -0.3 m/s.
* **GENERATOR** iff (i) and (ii): the arm is the candidate twin generator, proposed to the PI with a commit filter and a commit check (c').
* **COMMIT-WITH-SLOWDOWN** iff (i) and not (ii): the released goal pipeline cannot separate lateral demand from speed; the clean option is
  the per-twin LONGEST committing horizon >= 6 s (no slowdown measured at h6), with its yield.
* **NO-COMMIT** iff not (i): the floor removes the commitment too (the commitment WAS the slowdown).

## ADDENDUM 5 OUTCOME — read 2026-10-16 21:15Z (Master Mind)
As written: **NO-COMMIT for both floor arms** — K=1 median f(10 s) h2+v0max **0.749** (bar 0.75), h3+v0max 0.599. ⚠ The committed gloss
("the floor removes the commitment too; the commitment WAS the slowdown") does NOT describe the data: the floor changed NEITHER much —
commit share 0.543 -> 0.500 (h2), path ratio 0.810 -> 0.829, end speed -1.32 -> -1.27 m/s. ⇒ the feedback-loop mechanism of addendum 5 is
REFUTED: the slowdown is the teacher's response to a short-range lateral demand, not its goal creeping closer. The outcome stands as committed;
the gloss is noted, no goalpost moves. The released goal pipeline's three knobs (horizon, min speed, pair mode) are exhausted for
"commit without slowdown" at the population level.
