# VERDICT: STOPBAL second dose = BAL-FP-COSTS for both levers, as predicted (Master Mind, 2026-11-12 ~00:40)

Four runs (BAL-G2 at goal W 20.81 and BAL-L2 at lon W 22.08, seeds 0 and 1) trained on box 2 at code of record `715ef8fc` (AMENDMENT_3_DOSE2.md). BAL-L2_s1 was banked by the adopt script after the dev-box power-off at 23:10 (LOOP_STATE). It used the chain's own post-run steps, verified by sha256.

The read ran 00:22-00:24 on the first dose's package with only the two new tags (`RESULT_DOSE2.json`, 153.4 s, **0 control problems**). The banked NULL arms were re-asserted by checkpoint sha256. Tier: **T1-kin** (self-action open loop, m = 10, causal speed bucket, dead-reckoned nav, logged scene; deterministic planner), never driving performance. Estimator: paired episode-cluster bootstrap, B 2,000, seed 0 ("another draw of episodes"). The floors come from BAL-NULL_rep against BAL-NULL_s0 ("another training run"). STRATEGIC: **PAUSED** (PI item 25).

**The prediction was written before any number** (AMENDMENT_3_DOSE2.md, HYPOTHESIS): BAL-FP-COSTS for both levers, i.e. hit ≥ 0.30 at both seeds and false-positive rate > 0.05 at one or more. **It held.**

## Mechanism gate M (hit ≥ 0.30 AND false-positive rate ≤ 0.05, at both seeds)
| lever | seed | hit NULL → arm [CI] | fpr NULL → arm [CI] | AUROC NULL / arm | M |
|---|---|---|---|---|---|
| BAL-G2 (goal W 20.81) | s0 | 0.060 → **0.453** [0.317, 0.600] | 0.0054 → **0.120** [0.083, 0.159] | 0.819 / 0.813 | fpr fails |
| BAL-G2 | s1 | 0.120 → **0.530** [0.368, 0.687] | 0.0085 → **0.126** [0.089, 0.164] | 0.850 / 0.849 | fpr fails |
| BAL-L2 (lon W 22.08) | s0 | 0.060 → **0.547** [0.420, 0.678] | 0.0112 → **0.133** [0.098, 0.169] | 0.816 / 0.836 | fpr fails |
| BAL-L2 | s1 | 0.145 → **0.615** [0.468, 0.764] | 0.0134 → **0.145** [0.105, 0.185] | 0.851 / 0.854 | fpr fails |

Floors: hit 0.068 / 0.137, fpr 0.010 / 0.011, AUROC 0.038 / 0.036 (goal / lon). **AUROC moves by −0.006 to +0.020, every value inside its floor.** Full balance lifts recall 4-9× and multiplies false stops 9-22×, along the same ROC.

## P and NR (tactical ADE, knots 11-30; delta = arm − BAL-NULL of the same seed, m)
| lever | seed | P: 117 stop-point windows, floor 0.703 | NR: 2,399 windows, floor 0.183 |
|---|---|---|---|
| BAL-G2 | s0 | −0.218 [−0.420, −0.034] (separated, below the floor) | +0.040 [+0.003, +0.075] (separated, below the floor) |
| BAL-G2 | s1 | +0.096 [−0.124, +0.309] | +0.036 [−0.008, +0.079] |
| BAL-L2 | s0 | +0.112 [−0.193, +0.452] | +0.006 [−0.046, +0.056] |
| BAL-L2 | s1 | −0.017 [−0.377, +0.365] | −0.014 [−0.072, +0.044] |

* P is not a gain for either lever at either seed. NR is not a cost anywhere.
* Beside (descriptive): about 50 % of stop-point windows now read p ≥ 0.5, yet P stays near 0. Stage 0 of STOPPERC (`theory/stopperc/VERDICT_S0.md`) forced the GT stop as a HARD code on 50 % of these windows and read about −0.6 m. A soft p just over 0.5 does not carry the plan the way the hard code does. STOPHARD and STOPTAU closed hardening as a lever, because hardening at this false-positive rate puts false stops into plans.

## Four families (48 cells per lever and seed; named costs and gains are separated, on their side and above the floor)
* **Costs, each at ONE seed only:**
  * BAL-G2 s1, stop-point tactical cross-track +0.205 m [0.075, 0.364] (floor 0.204);
  * BAL-G2 s1, all-window operative headway error +0.020 m [0.000, 0.048] (floor 0.001);
  * BAL-L2 s0, stop-point tactical speed +0.470 m/s [0.019, 1.174] (floor 0.200);
  * BAL-L2 s0, all-window operative headway error +0.011 m [0.000, 0.022] (floor 0.001).
* **Gains, each at ONE seed only:**
  * BAL-G2 s0, stop-point operative time-gap error −0.006 s;
  * BAL-G2 s1 and BAL-L2 s1, all-window operative cross-track −0.001 / −0.002 m.
* LONGITUDINAL and LATERAL are read per cell in `RESULT_DOSE2.json`. TACTICAL is the M gate above (hit, false-positive rate, AUROC) plus the tactical accuracies in the result file. STRATEGIC: PAUSED.

## Outcome, as committed
* **BAL-FP-COSTS for BAL-G2 and for BAL-L2**: hit ≥ 0.30 at both seeds, false-positive rate > 0.05 at both. No other predicate holds. BAL-REGRESSES does not hold.
* **The dose path has no M-passing point** (AMENDMENT_3_DOSE2's committed reading). The first dose (BAL-RECALL-SHORT) and the second (BAL-FP-COSTS) sit on one ROC, and weighting only chooses the point on it. **Weighting is closed as a lever for the stop decision.**
* **Next lever, as committed: discrimination, the perception card** `theory/stopperc/PREREG_V8F_STOPPERC.md` (PI-approved 2026-11-11). Stage 0 has read CEILING-NEAR: a lever must lift T's hard stop coverage to about 60-70 % with false stops held. S0-P's probes run on box 2 after SLOWWM. Stage 1 (A first) goes to the PI with all of stage 0.
