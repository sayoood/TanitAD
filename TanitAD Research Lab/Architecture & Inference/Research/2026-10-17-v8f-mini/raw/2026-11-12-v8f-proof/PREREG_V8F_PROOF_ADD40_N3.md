# PREREG_V8F_PROOF_ADD40_N3 : IMAGPLAN, control p5b not met on LFT_s0; the candidate family is changed to the one the imagination was read for

Note to card `PREREG_V8F_PROOF_ADD40.md` and notes N1, N2. Written after the second plumbing sequence on LFT_s0 (changed tool of note N2; pod sequence `lft0b`, `imagplan/wait_seq_lft0b.out`, `imagplan/read_plumb/PLUMB2_LFT_s0.json`). **Seen when this note was written:** the controls p1 (first part), p1b, p3, p4b, p5b, p7 of LFT_s0 and a breakdown of the response of p5b by candidate, by progress difference, by speed and by tick (from the log of the run IPib_LFT_s0). **Not seen:** no unsafe rate, contact rate, position error, family value or chosen candidate of any run of the tool; no log of LA_s0; no value of an arm of note ADD37_N3. Statistics, names and bars of the card stay as written. Tier T1-kin. STRATEGIC: PAUSED (PI item 25).

## 1. What was measured (LFT_s0, changed tool)

* Met: p1 first part and p1b (141 of 141 files equal `off` bit for bit), p3 (141 of 141 equal), p4b (yaw correlation 0.9959 over all steps, acceleration 0.9952 over steps 2 to 10 and 0.946 at the first step, median difference 0.034 m/s2), p7 (2,399 windows, 3 ticks each, 3,012 active ticks, the same set for `cand-cv` and `cand-imag`). Beside: 2 active ticks with a plan above 60 m/s.
* **p5b NOT met:** LFT is `IMAG-DEAF` in the reader of card ADD36 (slope -0.41 at this base), and the tool reads a slope of +0.46 [0.30, 0.66] over the candidates 1.0 to 4.0 m behind the plan at 2.0 s (bar for a deaf arm: below 0.25).
* Breakdown (same log): per candidate the slope at 2.0 s is -0.06 to +0.03 for every c; by progress difference +0.97 (0.25 to 1 m), +0.56 (1 to 2 m), +0.45 (2 to 4 m), +0.25 (4 to 8 m), -0.04 (above 8 m); by speed at the tick +0.39 (below 2 m/s), +0.67 (2 to 6), +0.33 (6 to 12), +0.18 (above 12 m/s); by tick +0.43, +0.49, +0.45. The yaw bound of note N2 acts on 65 percent of the active ticks (58,097 action steps).
* Reading: the candidates of the card differ from the plan in two ways for which the imagination was never read. A speed-scaled form changes its speed in a step, at the braking limit (first actions at -6 m/s2). And its yaw channel is rebuilt from its own track, so it differs from the plan's, most at low speed where the rebuilt heading is noisy. Cards ADD36 and ADD37 read the response to a constant acceleration offset with the yaw channel untouched, and the consistency term trains exactly that.

## 2. Amendment (before any unsafe rate, chosen candidate or family value of any run is read)

**Candidates are the plan under a constant deceleration.** For d in {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} m/s2: step speed v_d(i) = max(0, v_P(i) - d * 0.2 s * i) with v_P the step speeds of the plan, progress a_d(i) = the cumulative sum of v_d * 0.2 s, knots = the plan's polyline sampled at a_d(i). d = 0 is the plan itself, bit for bit. No candidate is ahead of the plan and none brakes harder than B_MAX beyond what the plan does.
* Actions of a candidate: the plan's own action sequence (definition of note N1 A3, yaw bound of note N2) with the acceleration channel replaced by the forward difference of the candidate's speed at the cache steps (v(j) measured, then v_d); **the yaw channel is the plan's, unchanged.**
* Rule: the executed plan is the safe candidate with the smallest d; if none is safe, d = 6.0. Gate, check horizons, safety test, sources and constants as in the card and note N1. `cand-cv` and `cand-label` use the same candidates, so the modes still differ in the forecast only.
* Controls: p1 first part is read with the candidate set {0} and no fallback; p1b, p3, p6, p7 unchanged; p4b unchanged (the plan's actions). **p5b unchanged in statistic and bars**, over the candidates whose progress at 2.0 s lies 1.0 to 4.0 m behind the plan's (d = 0.5 to 2.0 give 1 to 4 m when the candidate does not stop). It must hold on LFT (deaf: below 0.25) and on LA (responds: at least 0.5) before the tool is run on a card arm.
* The speed-scaled family stays in the tool as an option; a run with it is a tool check and names nothing.

## 3. If p5b is not met with this family

No name of the card is given with the tool. The difference between the tool's imagined gap and the reader's is then searched in the state the rollout starts from (the tool recomputes the tick state from the tokens of one frame; the reader uses stored states of the clip), by running both on the same frames, and reported.

## 4. Runs

The runs on LA_s0 that are in progress with the speed-scaled family (pod sequence `la0`) finish as tool checks; their logs are read for p5b only (one more reading of the same control on an arm that responds), no unsafe rate of them is read. After the build of the amendment and its tests: the plumbing sequence with the new family on LFT_s0 and LA_s0 under new run names.
