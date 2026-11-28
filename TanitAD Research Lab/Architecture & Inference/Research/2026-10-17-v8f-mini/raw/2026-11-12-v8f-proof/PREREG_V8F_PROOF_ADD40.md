# PREREG_V8F_PROOF_ADD40 : IMAGPLAN, planning by imagination at T1-kin (the distance rule applied to the imagined lead gap of candidate plans)

Frozen before the tool of this card exists and before any value of an arm of note `PREREG_V8F_PROOF_ADD37_N3.md` (RLFT, RLA, RLAm, RLFTb) has been read: at the time of the freeze chain `actcons_c0` trains its first arm and no dump of these arms has been pulled. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames, the lead does not react; a diagnostic, never driving performance). STRATEGIC: PAUSED (PI item 25). The card is the step that section 6 of card ADD36, section 6 of card ADD37 and section 5 of note ADD37_N3 commit.

## 1. Why (MEASURED, with the artifact)

* Row H-V8F-PROOF-54 (card ADD26): the distance rule on the plan cuts the model-induced unsafe steps by about 60 percent when it is told the label gap, and by 13 to 32 percent when it is fed by the model's own lead head with a constant-velocity forecast of the lead. Row H-V8F-PROOF-58 (card ADD30): a margin on the head's gap read reaches the label-fed rule at a price in position error over 6 s of +0.86 to +0.95 m (label-fed rule +0.18 m).
* Row H-V8F-PROOF-68 (card ADD36): the lead head reads the gap 2.0 s ahead better from the world model's rolled-out states than from the present states (H1 +0.074 for the fine-tune control LFT), and the imagined gap did not respond to the ego's actions (`IMAG-DEAF`).
* Row H-V8F-PROOF-76 (card ADD37): a consistency term makes the imagined gap respond to the ego's actions (slope 0.855 [0.694, 0.958] against -0.452) with no cost on the families; the fine-tunes on the base without the rollout loss lose the level of the imagined gap. The anchored arms of note N3 are in training.
* Rows H-V8F-PROOF-74 and 75: a lead input to the plan heads is not used, and at the window start the lead does not explain the plan's speed error. Distance keeping rests on what is told the model's own gap. An imagination that responds and keeps its level is the first model-internal source of that: the gap that each candidate plan would produce.

## 2. The tool (built after this freeze; anything the build forces to change goes into a note before any number of a card arm)

A wrapper around the pinned T1-kin tool in the form of `distrule/v8f_t1_eval_rule.py` (the pinned file is not edited; rule off = the pinned tool bit for bit). Constants of card ADD26 unchanged: D = 3.0 m, TAU = 1.2 s, B_MAX = 6.0 m/s2; own progress against the logged path as in DISTRULE_DEFS E2 (g0 = g_src + a_l(k) - a_m(k)); replan schedule of the banked dumps (30 knots, the first 10 executed).

* Candidates at a tick: the model's plan P and speed-scaled forms of it on the same path. For c in {1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0}: progress a_c(i) = max(c * a_P(i), a_B(i)), where a_B(i) is the progress under braking at B_MAX from the measured speed v0 (so every candidate is reachable and c = 0.0 is full braking); the knots are the plan's polyline sampled at a_c(i); v_c(i) = (a_c(i) - a_c(i-1)) / 0.2 s.
* Gate (the same for every vision mode): active iff the lead head's present read says present (probability at least 0.5) with a gap of at most 60 m and all inputs finite (DISTRULE_DEFS E3, E4). An inactive tick returns the plan untouched.
* Check horizons: h = 1.0 s and 2.0 s (knots 5 and 10; the two horizons at which cards ADD36 and ADD37 read the imagination).
* Candidate c is safe iff at both horizons gap_c(h) >= max(D, TAU * v_c(h)). The executed plan is the safe candidate with the largest c; if none is safe, c = 0.0.
* Sources of gap_c(h):
  * `cand-imag` (the treatment): the world model rolls the tick state out under the candidate's own actions (level-1 rollout of the stack, 10 steps, the tool of `imaghead/imag_dump.py`); the lead head reads the rolled-out states at 1.0 s (steps 1 to 5) and 2.0 s (steps 6 to 10); gap_c(h) = imagined gap + a_l(k) - a_m(k). The actions of a candidate are computed from its knots by the same definition that gives the logged actions of the corpus (acceleration and yaw-rate channels in the trainer's scales).
  * `cand-cv` (the comparison, equal in everything but the forecast): gap_c(h) = g_head + v_L * h - a_c(h) + a_l(k) - a_m(k), with g_head and the closing speed from the head's present read and v_L as in DISTRULE_DEFS E2.
  * `cand-label` (privileged, the reach of the search form): the same as `cand-cv` with the label gap and the label closing speed.
* Beside, unchanged from card ADD26 and run on the same checkpoints: `off`, `head` and `label` in the projection form and with the options of the banked runs of row H-V8F-PROOF-54.

## 3. Arms

* The imagination arm X: the arm that note ADD37_N3 promotes (RLA at both bases, if it reads `ACT-CONS-RESPONDS-RL` with `ACT-CONS-KEEPS-IMAGINATION-RL` and `NO-COST`). If the note reads otherwise, no name of this card is given until an arm that responds and keeps the level exists (the level term of note N3 section 5 or the confirmed low weight); the tool and its controls are built and checked regardless.
* The control arm: RLFT at both bases (the same fine-tune without the term), every mode.
* Tool checks on arms that exist now (they name nothing): LFT (level kept, no response) and LA (response, level lost).
* No arm is trained by this card. Budget (ESTIMATED): six modes on two arms at two bases plus the controls, about 3 to 4 hours of one pod slot (L40).

## 4. Reads and names (each needs both bases; U = model-induced unsafe rate of the headway read; every difference is paired on the same windows of the same checkpoint; two-level bootstrap over bases then episodes, B 2,000)

* `IMAG-PLAN-CUTS-UNSAFE`: U(`cand-imag`) minus U(`off`) below 0 at both bases with the interval excluding 0. Else `IMAG-PLAN-NO-CUT`.
* `IMAG-PLAN-BEATS-EXTRAPOLATION`: U(`cand-imag`) minus U(`cand-cv`) below 0 at both bases, the interval excluding 0 and the pooled difference at least 0.010. `IMAG-PLAN-WORSE-THAN-EXTRAPOLATION`: above 0 at both bases with the interval excluding 0. Else `IMAG-PLAN-EQUALS-EXTRAPOLATION`.
* `IMAG-PLAN-DOMINATES`: `IMAG-PLAN-BEATS-EXTRAPOLATION` and the position error over 6 s of `cand-imag` minus that of `cand-cv` does not lie above 0.
* `IMAG-PLAN-REACHES-ORACLE`: U(`cand-imag`) minus U(`cand-label`) at most +0.010 at both bases and the interval does not lie above 0.
* Price: families of `cand-imag` against `off` by the rule of ADD12 Q3 with the same-sign rule of ADD13 (`IMAG-PLAN-NO-COST`, `IMAG-PLAN-COSTS-<family>`); the position error over 6 s against `off`, against `cand-cv` and against `label` is reported in metres.
* Beside (names nothing): contact rate; gap bias at 2.0 s; the share of active ticks and the histogram of the chosen c per mode; the same differences on the control arm RLFT and the difference of the two arms (what the response adds), read against the difference RLFTb minus RLFT under the same mode as the floor of two checkpoints; `head` against `cand-cv` (projection form against search form at one source).

## 5. Controls (all must pass; else nothing is named)

* p1 (identity): with the candidate set {1.0} alone, and with a source that returns a gap of 100 m everywhere, the dumps equal `off` bit for bit.
* p2 (known positive): `cand-label` cuts U against `off` at both bases with the interval excluding 0.
* p3 (no future frame): with the tokens after the tick replaced by noise the dumps of `cand-imag` are unchanged bit for bit.
* p4 (actions): with the logged future path given as the plan and c = 1.0, the candidate's action sequence equals the logged actions of the corpus for the same windows (correlation at least 0.99 on each channel; the median absolute difference of the acceleration channel at most 0.1 m/s2); measured on real windows before any run.
* p5 (rollout): at the first tick, under the logged actions, the tool's imagined gap equals the banked read of `imag_dump.py` for the same windows and checkpoint (in one environment to 1e-4 m; else within the difference of a twice-run compare, stated).
* p6 (planted level error): `cand-imag` with 20 m subtracted from every imagined gap raises the position error over 6 s against `cand-imag` (interval excluding 0): the tool shows a level error as a price.
* p7 (counts): 2,399 windows, 642 lead windows of 56 episodes; the active ticks of `cand-cv` and `cand-imag` are the same set; every window is rolled once per mode.
* p8: the controls h1 to h5 of the headway read and q1 to q5 of the family read.

## 6. What follows (committed before the data)

* `IMAG-PLAN-CUTS-UNSAFE` with `IMAG-PLAN-BEATS-EXTRAPOLATION` (or `-DOMINATES`): the imagination-fed planner is the distance-keeping line of the model. Next, own cards: the term with the rollout loss from the start on new seeds (the replicate that a claim about a lever needs); the imagined gap read at every step of the rollout; and the imagined outcome used at training (the plan head trained on the plans that the imagination scores safe), which is self-improvement driven by imagination.
* `IMAG-PLAN-CUTS-UNSAFE` with `IMAG-PLAN-EQUALS-EXTRAPOLATION`: at the precision of the present lead read (5 to 6 m of gap error) the imagination does not add to the constant-velocity forecast in the rule. The next lever inside the present encoder is the trained low-quantile gap head (card ADD31 section 8) read through the same tool; the finer lead read stays the open decision of the PI.
* `IMAG-PLAN-NO-CUT` or `IMAG-PLAN-WORSE-THAN-EXTRAPOLATION`: reported as written; p6 and the reads on LFT and LA say whether the level or the response is the cause; the next lever is a calibration of the imagined gap on training clips or the level term (own card).
* Any `IMAG-PLAN-COSTS-<family>`: reported beside the cut with the price of the margin rule of row H-V8F-PROOF-58 as the reference.

## 7. Limits

Logged frames: the own gap comes from the rolled progress against the logged path as a stand-in for re-perception (privileged geometry, as in card ADD26); the lead does not react; two bases of fine-tunes; one corpus; a discrete candidate set on the plan's own path (no lateral candidates); imagined reads at two horizons only; one replan per 2.0 s. A re-perception tier is not provisioned.
