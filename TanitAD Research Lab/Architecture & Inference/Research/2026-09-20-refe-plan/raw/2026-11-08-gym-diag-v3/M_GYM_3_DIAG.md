# M-GYM-3 DIAGNOSIS (2026-11-08) - scorer-only RL simulator
## ANSWERS FOR THE MASTER MIND (all MEASURED, artifacts in this folder; n stated per line)
1. **q_stall excuse precision per cause** (12 states: pilot 0,12,14,15 + stall 1,2,4,8,14,16,18,19; 1,224 cycle records, 719 satisfy the stall precondition, 653 excused, 66 fired; `q2_agg.out`). Truth = strict 1.15 m ego-width corridor, non-receding lead, ahead conflict entering the corridor within 4 s, on-route red connector:
   * LEAD valid 343/427 = 80.3 % (not-receding 335/427 = 78.5 %); CONFLICT valid 265/306 = 86.6 % (73.9 % at 3 s; 29/306 = 9.5 % have only objects behind ego); RED valid 460/460 = 100 %.
   * Sole-cause: red 90/90, lead 140/145 = 96.6 %, conflict 26/31 = 83.9 %.
   * Because causes overlap, only **10 of 653 excused cycles (1.5 %) have no valid excuse**, all 10 in stop-policy scenes (10/342 = 2.9 %); 0/311 in the 24-arm-set members. Weighted by the stored cause split (lead 1,212, conflict 1,977, red 715 across main+stall; ESTIMATED weighting): about 87 %.
   * Verdict: the excuse is mostly sound; its weak spots are lead (barrier or edge outside the strict corridor, 20 %) and conflict (cross traffic that never enters the corridor, 13 %). Tightening is cheap but not blocking.
2. **Constant-speed ladder on free road** (`q3_*.json`, 4 states, one policy per target, open loop; free-road, hard-event-free states are pilot 5 and stall 9 only; pilot 6 is hard from 0.7x; pilot 4 hard for every target):
   * **No slow target matches 0.9-1.0 x limit.** Pilot 5: 0.5x 6.22, 0.7x 8.63 vs 0.9x 12.18, 1.0x 13.36. Stall 9: 0.5x 6.33, 0.7x 9.00 vs 0.9x 11.34, 1.0x 12.10. Crawl 1.5 m/s: 0.27-0.37 in all four states, against PDM best arm 6.16-11.52 (pilot 4 9.06, 5 11.52, 6 6.16, stall 9 8.53), i.e. 17-40 times less.
   * Only exception, pilot 6: 0.5x 6.446 vs 0.9x 6.365 / 1.0x 6.214; cause is an `off_drivable` hard event at cycles 11-13 for 0.7x and faster, not a reward term (open-loop constant speed leaves the road; the hard event zeroes later reward).
   * **Which term carries it:** mostly progress (mean_prog 0.07-0.09 at crawl vs 0.42-0.81 at 1.0x on the free states), then q_free (floor 0.3; crawl loss_free 0.14-0.82, 1.0x 0.15-0.35).
   * **A crawl cannot fire q_stall** (rule: all of the last 20 ticks below 0.5 m/s; `q_stall_fired` = 0 in all 24 ladder runs). **What prices a crawl is progress x q_free**; q_free applies only with no lead/conflict and vstar >= 1 and bottoms at 0.3. When a lead or conflict is present (pilot 5 crawl: q_free applicable in 7/16 cycles) nothing prices slowness.
3. **Stall-10 oracle miss** (24 arms x 3 ttc variants, `cf_stall10.json`): with the ttc predicate OFF the miss disappears: 16/24 arms clean, oracle arm 6 return 1.070 with no hard event, but **8/24 arms end in real collisions** (a genuine hazard for a third of the arms). **Braking-aware variants (3 and 6 m/s^2) do NOT fix it**: still 24/24 ttc at cycle 6, oracle 0.580. So the defect is the predicate's lateral/corridor test on an oncoming car, not the missing braking model. On the other ttc states (`cf_stall11.json`, `cf_stall15.json`, n=24 each): baseline 5/24 and 13/24 arms with ttc; with ttc off or either braking variant 0/24, no collision, oracle unchanged (2.305, 8.934), i.e. pure false alarms. Stall 7 (`cf_stall7.json`): off gives 16 collisions, brake3 14 ttc + 2 collisions, brake6 13 + 3; oracle arm 23 (6.689) still hard and best clean 4.356 unchanged, so the oracle definition is the cause there, not the predicate.
4. **Is the environment valid enough to size a B1 teacher?** Not yet; two fixes come first: (a) oracle definition (lexicographic: fewer hard events, then return; stall 7), (b) ttc predicate restricted to objects whose predicted path enters the ego corridor (stall 10, 11, 15; braking-awareness alone is insufficient). Then (c) map pre-screen (main 10) and (d) all-tie cells (main 19), which are cheap. With those, the 4 label-free cells of 80 (5 %) and the 1 oracle-misdefined cell of 6 mixed cells would resolve. The q_stall excuse and the reward's speed-holding are fit for purpose (items 1 and 2). **Every fix must be validated on a fresh, seeded set of 20 states (new seed, navtest excluded, both samplers); the main and stall sets are used.**

---
# Full diagnosis (details behind the answers above)
Evidence class: MEASURED (ours, artifact path given) unless marked. Nothing in `pilot_v2/` was changed. Tooling lives in this folder (`trace_event.py`, `q1_*.py`, `q2_probe.py`, `q3_*.py`, `cf_ttc.py`, `gym_env_cf.py` = diag copy of gym_env with one switch). Final raw outputs are copied into this folder (`q2_*.jsonl`, `cf_*.json`, `q3_*.json`, `q2_agg.out`).
Tier: environment diagnostics of the gym itself; no model capability claim.

## Q1. Oracle misses
Method: replayed each state with a per-tick tracer wrapped around `RewardEvaluator.cycle` (`trace_event.py`). The replay reproduces the stored returns exactly (main 10 arm 1 = 2.7170, arm 8 = 2.1368, main 19 arm 0 = 0.0 against `pilot_v2/run_main/states.jsonl.gz`).

### Stall state 7 (log, collision at cycle 16 of the oracle arm): AVOIDABLE; oracle/return-definition defect on top of a log-replay artefact
* All 24 arms (MEASURED, `tr_stall7.jsonl`, stored `pilot_v2/run_stall/states.jsonl.gz`): **6 of 24 arms (pdm0, pdm1, pdm5, pdm6, sweep0.5, sweep0.6) finish all 17 cycles with no hard event; 16 arms end in ttc at cycles 13-14; 2 arms (sweep1.2, sweep1.3) end in collision at cycle 15 step 3.** Oracle = sweep1.3, return 6.689. Best clean arm pdm0, return 4.356 (gap 2.333). idm mode: 0 of 24 arms hard.
* Mechanism: vehicle `...5a528b` (log replay) changes lane across the lane the logged human drove in. In the log it is 21 m ahead of the human at 9.5 m/s and moves from ly -2.7 m to +4.6 m over 10 s (`q1h_stall7.out`). The sim ego runs 2-3 m/s faster than the human (13.1 vs 10.0 m/s) and arrives beside the car mid-manoeuvre: cycle 13 lx +2.5 m, ly -3.1 m, box gap 0.87 m, closing 1.97 m/s, object velocity pointing into the lane. Contact at cycle 15 step 3 is `ACTIVE_LATERAL` (enum 4) with ego `MULTIPLE_LANES=True`, so it is at-fault under the PDM rule coded at `gym_env.py:476-478`.
* Why 6 arms escape: the first action sets a ~2.4 m longitudinal phase (arm 0 at y=3997485.5 at cycle 12, arm 23 at 3997483.1) and the PDM continuation brakes in time only when it sees the cut-in from farther back (arm 0 speed at cycles 12-16: 12.9, 12.0, 10.7, 9.3, 7.9 m/s). The outcome bifurcates on about 2.5 m of phase.
* **Why the oracle still picks a colliding arm: return arithmetic.** A hard event only zeroes the remaining cycles' reward and carries no terminal cost. A collision at cycle 15 of 17 forfeits about two cycles (~0.9) while the faster arm banked +2.3 more progress earlier. `episode_return` (`gym_env.py:747`) is a plain sum; the PREREG oracle is "argmax of the full-continuation return" and "KNOWN iff no hard event". These two statements disagree whenever a late-event arm out-earns every clean arm. It happens in 1 of the 6 mixed cells (table below); in the other 5 the argmax arm is clean.
* Classification: not a map defect. Partly a simulator artefact (non-reactive log agent cuts in on the logged speed; idm has no event) and a real, avoidable planning outcome. The recorded miss is an oracle/return-definition failure, not a one-step planning failure.

### Stall state 10 (log, ttc at cycle 7, 24 of 24 arms): unavoidable by any first action; over-sharp predicate on an oncoming pass
* All 24 arms: ttc at cycle 6 (step 0-4), returns 0.12-0.58 (`tr_stall10.jsonl` arms 9, 0, 20; stored table). Oracle arm pdm9: ttc at cycle 6 step 1 at ego v=2.6 m/s.
* Mechanism (`tr_stall10.jsonl`, arm 9): oncoming vehicle `...735619` at 9.0 m/s, 16.9 m ahead, ly -1.8 m, closing speed 11.3 m/s, box gap 10.5 m. The predicate (`gym_env.py:469-490`) shifts the ego footprint straight ahead at constant speed and the object along its velocity for dt = 0.2..1.0 s and fires on any footprint overlap. At dt=1.0 the relative advance 11.3 m exceeds the 10.5 m gap and the 1.8 m centre-to-centre lateral offset is below 2.15 m (half-widths 1.15 + 1.0), so it fires on a car still 10 m away. The human passes the same vehicle at 1.77 m clearance (`q1h_stall10.out`, t=4.0 s) with a centre offset of 3.9 m, i.e. ordinary narrow-road passing; the sim ego is about 1.8 m closer to the oncoming car's line at that instant (cause of that offset UNVERIFIED, not traced).
* The predicate ignores ego braking, so no braking can avoid it (the ego at 2.6 m/s stops in under 1.2 m at 3 m/s^2). Counterfactual (`cf_stall10.json`, n=24 arms per variant): predicate off, 8/24 arms collide for real, 16 clean, oracle clean at 1.070; braking-aware 3 and 6 m/s^2, still 24/24 ttc at cycle 6 (oracle 0.580). So the miss vanishes without the predicate (partly a real collision for 8 arms), and braking-awareness alone does not help.
* Classification: simulator defect (constant-velocity ttc predicate), unavoidable for all 24 arms as defined.

### Main state 10 (log, off_drivable cycle 8-9, 24 of 24): MAP defect on the route centreline
* `q1_cl_main10.out`: a Pacifica footprint placed **exactly on the PDM route centreline** (heading = tangent) leaves the strict drivable union by up to **0.254 m** between s = 18.5 and 23 m (0.149 m at s=19.0, 0.254 m at s=21.0). The start pose is clean (slack 0.0). The sim ego reaches s=21.7 at v=2.1 m/s and overshoots 0.059 m on the first tick of cycle 9 and 0.213 m on the last (`tr_main10.jsonl`). The logged human drove about 2.5 m to the left of that centreline at that arc length and stayed inside (overshoot 0.000 over the horizon, `q1_main10_map.out`). The union is one connected polygon; the stretch is the seam between drivable polygons 177 and 6.
* All 24 arms follow the same centreline once PDM takes over, so no first action escapes. idm mode shows no event (mechanism UNVERIFIED). Unavoidable; cause is map geometry along the route centreline.

### Main state 19 (known 1 cm overshoot)
* Slack in force 0.1587 m; sim footprint overshoot per tick of cycle 0: 0.168, 0.170, 0.165, 0.152, 0.132 m (`tr_main19.jsonl`, `pilot_v2/analysis_v2_state19_diag3.out`). All 24 arms in both modes end in cycle 0 with return exactly 0.000. The state carries no label; 1.1 cm decides a whole state.

### Is hard termination too sharp in general? (MEASURED, 80 state x mode cells, 3,840 arm-episodes, `run_main` and `run_stall` `states.jsonl.gz`)
| quantity | main (20 states) | stall (20 states) |
|---|---|---|
| arm-episodes with a hard event, log / idm (of 480 each) | 54 (11.25 %) / 28 (5.8 %) | 60 (12.5 %) / 0 |
| by type | off_drivable 72 (states 10, 19 only), wrong_way 8 (state 4 only), collision 2 (state 8) | ttc 58 (states 7, 10, 11, 15), collision 2 (state 7) |
| cells with ALL 24 arms hard | 3 (10 log, 19 log, 19 idm) | 1 (10 log) |
| cells with SOME arms hard | 3 (4 log, 4 idm, 8 log) | 3 (7, 11, 15 log) |
| mixed cells where the oracle arm is hard while a clean arm exists | 0 | 1 (state 7) |
* 4 of 80 cells (5 %) are label-free because every arm dies. In all four the trigger is a small margin or an extrapolation: state 19 1.1 cm over the slack, main 10 6-25 cm on a map seam, stall 10 a constant-velocity prediction at 10 m range. Sharp for label generation: one threshold deletes the state.
* In the 6 mixed cells the events fall at cycles 2-3, 2, 5, 13, 13-15 and discriminate between arms; those are informative, not too sharp.
* Counterfactual ttc variants (`cf_stall{7,10,11,15}.json`, n=24 arms each, log mode; arms with a hard event, baseline 18 / 24 / 5 / 13):
| state | off | brake 3 | brake 6 | oracle arm clean? |
|---|---|---|---|---|
| 7 | 16 collisions | 14 ttc + 2 coll | 13 ttc + 3 coll | no, in all (6.689 vs best clean 4.356) |
| 10 | 8 collisions | 24 ttc | 24 ttc | off: yes (1.070); braking: no (0.580) |
| 11 | 0 | 0 | 0 | yes (2.305) |
| 15 | 0 | 0 | 0 | yes (8.934) |
Stall 11 and 15 are pure predicate false alarms (18 arm-episodes, no collision when it is off or brake-aware). Stall 7 is a real hazard and the oracle defect, not the predicate. Stall 10 mixes both.

## Q2. q_stall excusal: precision per cause (MEASURED, `q2_agg.out`, `q2_*.jsonl`)
Probe `q2_probe.py` replays 12 states (pilot 0,12,14,15; stall 1,2,4,8,14,16,18,19) in log and idm modes under the stop policy and the 24-arm members (pdm_best, sweep0.5): 1,224 cycle records, 719 with the stall precondition, 653 excused, 66 fired. Truth definitions in `q2_agg.py`.
| cause | all excused cycles | stop-policy scenes | arm-set members (pdm_best, sweep0.5) |
|---|---|---|---|
| LEAD (<= 10 m) enters strict corridor | 343/427 = 80.3 % | 208/248 = 83.9 % | 135/179 = 75.4 % |
| LEAD and not receding (vlon < 2) | 335/427 = 78.5 % | 202/248 = 81.5 % | 133/179 = 74.3 % |
| CONFLICT ahead enters strict corridor <= 4 s | 265/306 = 86.6 % | 125/146 = 85.6 % | 140/160 = 87.5 % |
| CONFLICT <= 3 s | 226/306 = 73.9 % | 104/146 = 71.2 % | 122/160 = 76.2 % |
| CONFLICT only objects behind ego | 29/306 = 9.5 % | 17/146 = 11.6 % | 12/160 = 7.5 % |
| RED on route and in corridor | 460/460 = 100 % | 204/204 = 100 % | 256/256 = 100 % |
| sole LEAD valid | 140/145 = 96.6 % | 93/98 = 94.9 % | 47/47 |
| sole CONFLICT valid | 26/31 = 83.9 % | 24/29 = 82.8 % | 2/2 |
| sole RED valid | 90/90 | 30/30 | 60/60 |
| excused with >= 1 valid excuse | 643/653 = 98.5 % | 332/342 = 97.1 % | 311/311 = 100 % |
* Falsely excused (q_stall should have fired): 10 of 653 = 1.5 %, all in stop-policy scenes (log mode 9, idm 1). Weighting the per-cause precision by the stored cause split (main lead 555, conflict 1,131, red 115; stall lead 657, conflict 846, red 600) gives about 87 % (ESTIMATED, computed from the table above).
* Cause combinations (all, n=653): LCR 153, L 145, LR 112, CR 105, R 90, C 31, LC 17. Overlap is why single-cause imprecision rarely becomes a false excuse.
* A free-road crawl can never fire q_stall: precondition is `stalled = all(v < 0.5 for the last 20 ticks)` (`gym_env.py:594`). It is priced by `q_free` (`gym_env.py:566-569`: floor 0.3, applicable only with no lead, no conflict, vstar >= 1) and by progress. The constant-speed ladder confirms `q_stall_fired` = 0 in all 24 runs (Q3).
* q_stall fires when the ego truly stops: stop policy 67/169 (main) and 37/220 (stall) precondition cycles (stored `reg['stop:log']`); this probe: 66 fired in 719 precondition cycles.
* `stalled_by_red` (`gym_env.py:566`) is read and never assigned (dead branch).
* Pilot state 0 example (`q2_pilot_0.jsonl`): lead excuse is a barrier whose edge is 1.49 m from the centreline (outside the 1.15 m half-width); conflict objects are cross traffic 15-17 m ahead; a route red connector (id 53453) is 2.7 m ahead; only the red is valid.
* Caveat: truth is my strict-corridor definition, not a human label; 12 states, 8 of them stall-sampler.

## Q3. Does the reward pay for holding speed? Yes; no slow target matches 0.9-1.0 x limit
Stored sweep (first action only; `q3_stored.out`, log mode; free-road states = at least 90 % of cycle records with q_free applicable and v0 >= 1): main 4, 5, 6, 9, 19 and stall 9.
* Hold (x1.0) beats slow (best of x0.5-0.7) in 3 of 4 informative states: main 4 (9.09 vs 7.89), main 5 (11.01 vs 10.44), stall 9 (7.86 vs 7.60). Reversed in main 6 (hold 4.98, slow 6.11). Ties in main 9 (limit 4.5 m/s is below v0, so 8 of 9 targets collapse to 2 distinct speeds) and main 19 (hard).
* The main 6 reversal is not a speed effect. After 0.5 s the PDM continuation re-plans; arm 20 (x1.0) collects `q_comf` 0.7 (cycles 11-12), `q_osc` 0.8 (13-15) and `q_cons` 0.9 (qprod 0.63, 0.63, 0.50) from a brake-accelerate flip; the x0.7 arm does not. Loss by term (sum progress x qprod/q x (1-q)): hold q_free 0.34, q_cons 0.28, q_comf 0.28, q_osc 0.25; slow q_free 0.46, q_comf 0.43, q_cons 0.19, q_osc 0.17. Responsible terms: q_comf, q_osc, q_cons (continuation noise), not q_free.
* The first-action sweep is a weak probe: 0.5 s of action, then PDM resumes; mean speed over the last 8 cycles differs by 0.1-0.6 m/s between x0.5 and x1.0 (main 5: 10.14 vs 10.59).
* The always-slowest regression policy (persistent, 2.2-3.0 m/s) returns 0.62, 0.76, 1.44, 0.78 against PDM 9.06, 11.52, 6.16, 8.53 on main 4, 5, 6 and stall 9: a persistent crawl is paid 7-15 times less. Analytic steady state on straight free road, f = v/limit: reward per cycle = f x q_free(f) with q_free = clip(1 - 0.7 (0.9 - f)/0.4, 0.3, 1): f=1.0 -> 1.00, 0.9 -> 0.90, 0.7 -> 0.46, 0.5 -> 0.15, 0.1 -> 0.03. Monotone; no slow policy matches the hold.
* Weak spots that do exist (MEASURED in logs): (i) the episode-start brake baseline pays nothing for the first 1-3 s for ANY arm (main 6: reward 0.0 in cycles 0-1 for every arm, v0 6.4 m/s, stop time 3.2 s); (ii) vstar = min(limit, curve_speed) lets q_free reach 1.0 at 7.6 m/s on a 15 m/s road (main 6, cycles >= 4); (iii) with a lead, q_free is off and q_gap pays 1.0 for gap time 1.5-2.5 s and 0.5 beyond 3.5 s; with a conflict present no term prices slowness and q_yield penalises going.
* Constant-speed ladder (MEASURED, `q3_pilot4/5/6.json`, `q3_stall9.json`, full constant-speed rollout, 1 run per target, 6 targets, `q3_agg.py`). Return (H = hard event):
| state | crawl 1.5 | 0.3x | 0.5x | 0.7x | 0.9x | 1.0x | PDM best arm |
|---|---|---|---|---|---|---|---|
| pilot 4 (limit 11.2, all H off_drivable) | 0.325 H | 0.844 H | 1.069 H | 1.681 H | 1.367 H | 1.385 H | 9.06 |
| pilot 5 (11.2) | 0.370 H coll | 1.449 | 6.219 | 8.627 | 12.175 | 13.359 | 11.52 |
| pilot 6 (15.0) | 0.353 | 3.488 | 6.446 | 6.378 H | 6.365 H | 6.214 H | 6.16 |
| stall 9 (15.0) | 0.266 H coll | 1.589 H coll | 6.327 | 9.001 | 11.337 | 12.096 | 8.53 |
* Clean monotone ladder in pilot 5 and stall 9: 0.9x and 1.0x beat 0.7x by 3.5 and 4.7 (pilot 5) and 2.3 and 3.1 (stall 9). **No slow target matches the 0.9-1.0 target.** Pilot 6 is the only near-tie (0.5x 6.446 vs 1.0x 6.214) and it is a hard `off_drivable` at cycles 11-13 for 0.7x and faster, not a reward term; pilot 4 is off_drivable for every target by cycle 5-11 (constant-speed open-loop leaves the road), uninformative.
* Term attribution (`loss_free` vs `loss_other`): the crawl deficit is mostly progress, mean_prog 0.07-0.09 vs 0.42-0.81 at 1.0x on pilot 5 / stall 9 (q_free also floors at 0.3: crawl loss_free 0.14-0.82). At 0.9-1.0x loss_free is 0.15-0.35 (mean q_free 0.88-0.92). Where a lead or conflict is present (pilot 5 crawl: q_free applicable in 7/16 cycles) no term prices slowness.
* q_stall fired 0 of 24 runs (crawl mean speed 1.9-2.0 m/s is above the 0.5 m/s stall threshold, by the code rule). What prices a crawl is progress times q_free, nothing else.
