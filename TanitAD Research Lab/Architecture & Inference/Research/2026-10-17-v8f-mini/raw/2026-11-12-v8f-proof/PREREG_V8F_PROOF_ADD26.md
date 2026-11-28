# PREREG_V8F_PROOF_ADD26: DISTRULE. A distance rule in the safety check of the rollout: the plan is slowed so that the gap to an imagined constant-speed lead keeps a bound. First with the label gap (an upper bound of what a perfect lead state gives, privileged), then with the gap and closing speed of the supervised head of card ADD24 (vision only).

Frozen before any number of this card is computed. Addendum to `PREREG_V8F_PROOF.md` and the committed next step of `PREREG_V8F_PROOF_ADD25.md` (its section 5: a model-induced unsafe rate above 5 % of lead windows). Tier T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class of every result: MEASURED once run. STRATEGIC: PAUSED (PI item 25).

## 1. What is known before it (MEASURED, card ADD25, `headway/out_q12/VERDICT_HEADWAY.md`, controls all PASS, 642 lead windows of 56 episodes)

* Model-induced unsafe rate U (a step with gap below 2.0 m or time gap below 1.0 s in a window where the logged driver has none): HIER 0.1275 [0.0970, 0.1581], FLAT 0.1332, G 0.1296 (six seeds each). Contact rate C against the non-reacting lead: 0.0460 / 0.0439 / 0.0452. Share of windows where the model falls more than 10 m behind the driver at the last step, F: 0.2349 / 0.1812 / 0.2160.
* `DISTANCE-EQUAL` (HIER against FLAT) and `GRID-DISTANCE-EQUAL` (G against HIER): no arm keeps distance better than another. The logged driver's own unsafe-step share by the same rule is 0.1246 over the 642 lead windows, contact 0.
* The label rule R2 of card ADD20 predicts the gap 2.0 s ahead with skill +0.750 over the copy when it is given the label gap and closing speed: a constant-speed lead is a strong model at this horizon.

## 2. The rule (fixed here; constants are not tuned on any read of this card)

At a planning tick at window step j the rule sees: the rolled ego speed v0; a lead source giving presence, gap g_src and closing speed c_src at the tick frame, from frames at or before the tick only; the logged ego speed v_l(j).

* Gap of the rolled ego: g0 = g_src + a_l(j) - a_m(j), with a the progress along the logged path of the logged ego (a_l) and of the rolled ego (a_m), as in card ADD25. In T1-kin the camera frames are the logged ones, so a source reads the gap of the logged ego; this term moves it to the rolled ego and stands in for re-perception.
* Lead speed: v_L = max(0, v_l(j) + c_src). Imagined lead progress over the plan steps i = 1..n at 0.2 s: s_L(i) = g0 + v_L * 0.2 * i (a constant-speed lead: the fixed transition of the object slot of card ADD20).
* Constants: D = 3.0 m, TAU = 1.2 s, B_MAX = 6.0 m/s2 (the unsafe rule of the metric is 2.0 m and 1.0 s).
* Projection of the plan, forward over i, with a_P(i) the arc length of the plan along its own path, a'(0) = 0, v'(0) = v0:
  * wanted: a_w(i) = a'(i-1) + (a_P(i) - a_P(i-1));
  * cap: a_c(i) = s_L(i) - max(D, TAU * v'(i-1));
  * floor from the braking limit and no reversing: a_f(i) = a'(i-1) + max(0, v'(i-1) - B_MAX * 0.2) * 0.2;
  * a'(i) = max(a_f(i), min(a_w(i), a_c(i))); v'(i) = (a'(i) - a'(i-1)) / 0.2. A step where the floor decides against the cap is a saturated step (counted).
* The new plan is the same path polyline sampled at the arc lengths a'(i); the lateral geometry is unchanged. The rule is inactive (plan unchanged, bit for bit) when the source says no lead or g_src is above 60 m.
* Sources:
  * `label` (privileged, an upper bound, not deployable): presence and gap from the label join of the evaluation episodes at the tick frame under the continuity rule of card ADD20; c_src = least-squares slope of the label gap over the trailing run of at most 5 frames ending at the tick frame, at least 3 frames, else 0.
  * `head` (vision only): the supervised head of an arm of card ADD24 applied to the level-1 grid states of the tick frame and the four frames before it, computed from the logged camera tokens by the arm's own model; presence = probability at or above 0.5; gap and closing speed = its outputs. No label enters.
* The model, its weights and its inputs are unchanged; `v8f_t1_eval.py` stays pinned and byte-identical. The rule lives in a wrapper around its planner.

## 3. Arms (evaluation only, banked checkpoints, pod GPU; schedule, replan steps and modes of the banked dumps)

* A, oracle source: rule `label` on HIER_s0 to s5 and G_s0 to s5 (12 rollouts), each against the banked no-rule dump of the same checkpoint.
* B, head source: rule `head` and rule `label` on LS_s0/1, LSH_s0/1, LSD_s0/1 (12 rollouts), each against the no-rule dump of the same checkpoint. Run when those arms are done; A does not wait for B.

## 4. Reads

* Distance keeping: `headway/headway_read.py` (card ADD25 cells: U primary, C, F, gap bias and error, time-gap error) for rule against no rule of the same checkpoint, paired on windows, two-level bootstrap (seeds, then episodes), B 2,000, rng 0. Inference is deterministic and the two rollouts share the checkpoint, so the difference is the rule alone; the same-sign and interval conditions of card ADD11 apply, the spread condition applies to the cost cells only.
* Cost: `spatial/q3/q3_read.py` four families on all 2,399 windows, rule against no rule, spread of the baseline arm over six seeds. LONGITUDINAL (speed, along-track error, and the lead block above), LATERAL, TACTICAL reported; STRATEGIC PAUSED.
* Reported, naming nothing: the share of ticks where the rule is active, the share of saturated steps, U and ADE on the lead windows split by whether the logged driver itself has an unsafe step.

## 5. Controls (all must pass before any outcome is named)

* r1 rule off through the wrapper equals the banked dump of the same checkpoint bit for bit (one arm, all windows).
* r2 never-binding rule (D = -1e9, TAU = 0) equals rule off bit for bit.
* r3 deliberate regression: the cap replaced by a push (the plan advanced by 0.5 m/s) must be named worse on U by the reader.
* r4 no future input: with every label frame after the tick replaced by NaN (label source) and every camera token after the tick replaced by noise (head source) no output changes.
* r5 unit checks of the projection on hand-built plans: a stopped lead 10 m ahead at 10 m/s gives the braking-limited profile and counts the saturated steps; a lead faster than the plan leaves the plan unchanged; progress never decreases.
* r6 the head source equals `lead_dump.py` of the same checkpoint on the same frames within 1e-4 (m and m/s) on at least 64 ticks.
* r7 the controls of `headway_read.py` (h1 to h6) and of `q3_read.py` on the new dumps.

## 6. Outcomes (both committed)

* A: `RULE-CUTS-UNSAFE` if U falls under the rule in all six seeds with an interval that excludes 0 (per arm type HIER, G); else `RULE-FLAT`. `RULE-CUTS-CONTACT` likewise on C. Cost: `RULE-NO-COST` if no cell of the four families is WORSE under the rule of card ADD11; else `RULE-COSTS` with the family named.
* B: `HEAD-RULE-CUTS-UNSAFE` if U falls under the head-source rule in both seeds with an interval that excludes 0 (per arm type LS, LSH, LSD); else `HEAD-RULE-FLAT`. `HEAD-RULE-REACHES-ORACLE` if the head-source fall of U is at least half of the label-source fall on the same checkpoint in both seeds.

## 7. Committed next steps

* `RULE-CUTS-UNSAFE` with `RULE-NO-COST`: the distance rule becomes a standing part of the safety check in oracle form, and the gap between head source and label source is the measure of the lead state.
* `HEAD-RULE-CUTS-UNSAFE`: the first vision-only distance keeping result of the programme; next, training with the rule in the loop (the model learns from its own vetoed plans), own card.
* `RULE-FLAT`: a constant-speed lead at this replanning interval does not reach the unsafe steps; the rule is read at one replanning step per tick, own card.
* `RULE-COSTS`: reported as written; the constants are not tuned on this evaluation set, a tuning split is made first.
* `HEAD-RULE-FLAT` with `RULE-CUTS-UNSAFE`: the head is the limit; its gap error on the lead windows is reported and the head is trained further (card ADD24 next steps).

## 8. Cost and limits

* 24 rollouts of about 8 GPU-minutes each on the pod after the arms of card ADD24, at most three at a time; no training.
* The lead does not react; the label source is privileged and is an upper bound, never a deployable result; the head source reads logged frames and uses the rolled progress as a stand-in for re-perception.
* 642 lead windows of 56 episodes; one evaluation set; the rule constants are fixed by this card and their margin over the metric thresholds is a choice, not a fit.
