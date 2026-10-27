# NAV_INPROGRESS: why window 2158 turns left against the road, and what that failure class costs DAG (2026-10-27)

**EXPLORATORY. It decides nothing and moves no bar.** It answers the PI's question on video v5, chapter 4 (window #2158, clip `ed87040c`,
t = 11.8 s): *"Why in this case the trajectory is not following the nav command?"*
* **Evidence class:** MEASURED (ours). Script `b1_inprogress_cost.py`, output `b1_inprogress_cost.json`.
* **Inputs (all banked):** DAG's T1-kin dumps (`v8f-dagger-t1/dump_DAG_s{0,1}_m10_causal_heads/`: `cl` = executed path, `fl` = the floor
  P10, `g` = GT; overall ADE6 2.278 / 2.456 m and P10 3.443 m, so DAG − P10 = −1.165 / −0.987 m, which reproduces Read A); the nav table of
  `raw/2026-10-25-navturn/repro_eval141.npz`; the join of `raw/2026-10-25-t-attrib/raw_in/join_eval141.npz`.
* **Tier:** T1-kin, a self-action OPEN loop (⛔ not closed loop), m = 10, causal bucket.
* **Estimator:** episode-cluster bootstrap, B = 2,000, 141 episodes. It answers *another draw of EPISODES* only; one training seed per arm.

## 1. Window 2158: the model does follow the nav. The nav is what is wrong

The rows around NOW (join rows of clip 126, 0.2 s apart; NOW = row 59):

| row | nav token | d_next (m) | in progress | turn end (m) | announced angle | label-side event | v (m/s) |
|---|---|---|---|---|---|---|---|
| 55 | TURN_L | 0.7 | no | 14.3 | +20° | −40° bend, vmin 8.8 m/s, `is_turn` 0 | 7.7 |
| 58 | TURN_L | 0.0 | yes | 9.6 | +20° | same | 7.8 |
| 61 | TURN_L | 0.0 | yes | 4.8 | +20° | same | 8.1 |
| 64 → 88 | FOLLOW | — | — | — | — | — | 8.5 → 17.2 |

* **At NOW the nav says "TURN_L, in progress"** for a small +20° left turn that ends about 8 m ahead.
* **The model receives none of that extent.** Its nav input is 5 numbers: the FOLLOW / TURN_L / TURN_R one-hot, d_next / 100 and a valid
  flag. At d_next = 0 that is just "turning left now", with no angle and no end point.
* **The −40° right bend that follows is never announced.** The builder announces a turn only if `is_turn` holds, which needs vmin ≤ 8 m/s.
  This bend's vmin is 8.8 m/s, so from +1 s the nav reads FOLLOW. That is the speed-gate artefact the 2026-10-25 audit found on 74 % of its
  turn-cell mismatches (`NAVTURN_AUDIT.md`, headline 2).
* **So the hierarchy acts on "TURN_L" for the whole horizon.** T predicts TURN_L against its own LANE_KEEP label; S places its checkpoints
  to the left; O executes a 6 s left arc while the road bends right. ADE6 at T1-kin: DAG 19.32 / 17.84 m, P10 13.42 m. P10 extrapolates the
  current yaw and does not read the nav, so it is the least wrong here.
  * **Nearly the same answer in every arm read.** T's lateral head at NOW (tick 0, softmax of `t_lat_logits` in the T1-kin tick files)
    puts TURN_L on top in 7 of 8 arms: DAG 0.624 / 0.778, DAG-0 0.635 / 0.890, ORC 0.478 / 0.665, ORC-0 s1 0.713. The exception is ORC-0
    s0 (LANE_KEEP 0.408 against TURN_L 0.385).

## 2. What the class costs, on all 2,399 windows (DAG at T1-kin)

| class at NOW | n (episodes) | DAG ADE6 | P10 ADE6 | DAG − P10 [95 %] | share of DAG's mean ADE6 |
|---|---|---|---|---|---|
| FOLLOW | 2,004 (138) | 2.079 / 2.293 | 3.035 | −0.956 [−1.147, −0.775] / −0.743 [−0.951, −0.550] | 1.737 / 1.915 m |
| TURN ahead (d_next > 0) | 278 (35) | 2.878 / 2.828 | 5.187 | −2.310 / −2.359 | 0.334 / 0.328 m |
| TURN in progress (d_next = 0) | 117 (24) | 4.248 / 4.382 | 6.286 | −2.038 [−2.859, −1.204] / −1.904 [−2.585, −1.171] | 0.207 / 0.214 m |
| ⤷ of which "stale": the GT ends heading against the token, or within ±10° of straight | 25 (15) | 5.627 / 5.622 | 7.473 | −1.846 [−3.752, +0.585] / −1.851 [−3.420, +0.045] | **0.059 / 0.059 m** |

* **Windows inside an announced turn are DAG's hardest class:** about 2× the FOLLOW error. DAG still beats the floor there by about 2 m.
* **The screenshot's subclass is small:** 25 windows, 1.0 % of the eval. It carries 0.059 m of DAG's 2.278 m mean ADE6, and DAG is not
  separated from P10 on it.
  * ⇒ Even a perfect fix of this subclass moves the mean ADE6 by at most 0.059 m. That is about F/2 and far below 2F = 0.2415 m.
  * Rule Zero's lever ranking therefore keeps the crawl rows (DAGGER2) and the longitudinal rows (DAG-OACC) ahead of it.
* ⚠ The "stale" rule (GT heading over the last 0.6 s of the horizon) is a post-hoc probe written for this question, not a registered
  definition.

## 3. The lever it points to (BACKLOG, not pre-registered)

* **The information exists and is not fed to the model.** The v9 release already carries `nav_d_end_m` (the distance to the end of the turn
  in progress) and `nav_dyaw_next_deg` (the announced turn angle). The model's nav input has neither.
* **Candidate arm:** add the remaining turn extent and the announced angle to the nav input.
  * It has the same supplier as d_next (the ego's own future path on PhysicalAI), so it is no more optimistic than the route input already in
    use.
  * It carries nothing from the situation classifier, so it passes the goal-input rule (2026-08-03).
* **Not proposed:** lifting the builder's vmin ≤ 8 m/s gate so that fast bends are announced. The 2026-10-25 audit flagged that this would
  turn the nav token into the situation label (the goal-input rule).
* **Ceiling:** about 0.06–0.2 m of mean ADE6 (this table). It matters for visible failures like window 2158 more than for the dominance
  read.
