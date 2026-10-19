# PLAN_REFCV9 — fix what refcv8 measured, in the model, and prove it against refcv8 + its shipped stage

*Master Mind, 2026-10-19 ~19:30 Berlin. Status: **DRAFT v0** (PI: "start the refcv9 design while waiting").
Nothing here is authorised compute. Every number carries its evidence class and the register row it comes from
(`Project Steering/GOALS_AND_CLAIMS.md`). Planner numbers: T1 open loop on held-out eval139, refcv8 EMA-30k,
ceiling-compliant windows (S2 4,159 / S6 3,251, 138 episodes), paired episode-cluster bootstrap over EPISODES
(one training seed), unless stated otherwise.*

## 0. The bar refcv9 has to beat

refcv9 is not compared with plain refcv8. It is compared with **refcv8-final + the shipped re-timing stage**
(D-R8-SHIP-RETIME, if SHIP-R8 passes at 48,852), because that is what would be deployed. A refcv9 lever that only
recovers what the frozen stage already gives is worth nothing. Hence:

* **B9-0 (no regression):** no family (ADE, LONGITUDINAL, LATERAL, TACTICAL; STRATEGIC if the layer is on) separated
  worse than refcv8-final + stage, at both inference seeds, on the same windows.
* **B9-1 (the gain must be in the model):** refcv9 WITHOUT any post-hoc stage passes E1 / E3 against `ha0_ext_pose`
  (D-R8-E1-REF) at its final checkpoint.
* **B9-2 (lever claims need a replicate):** any claim that a refcv9 lever moved a metric carries a replicate arm
  (H-ESTIM-SEED-1); a separated episode-bootstrap CI alone is necessary, not sufficient. The planner samples, so the
  inference-seed floor is reported beside every effect.

## 1. What refcv8 measured — the defects, ranked by measured effect (Rule Zero §5)

| # | defect | MEASURED size | register row |
|---|---|---|---|
| D1 | **A noisy along-track speed choice** (CORRECTED 2026-10-19 ~21:00, R55). The path shape is right; the plan's progress residual over `ha0_ext_pose` is about the human's SIZE (ρ 0.84 [0.77, 0.92]) and unbiased in mean (+0.142 vs +0.143 m at 2 s), but poorly ALIGNED with the human's (r 0.38 [0.34, 0.43]). The best flat gain g\* = r/ρ = 0.46 is the least-squares shrink of that noise, NOT a 2x over-scale. It does not fade with training (0.50 at 5k → 0.46 at 30k, Δ −0.048 [−0.089, −0.010] / −0.058), is worse on eval than train (0.46 vs 0.69), and is inherited unchanged from refcv7-50,400 (r 0.39). The GT-nearest refined candidate in the fan has r 0.76 / g\* 0.91 (a T0 ceiling; whether a RANDOM fan member beats the emitted plan is being measured). Source: `refcv9/L1_DESIGN_ANNEX.md` §1. | E1 fails without a stage: at EMA-30k os − pose **−0.0136 [−0.0286, +0.0018]** / −0.0127, not separated (E1REF series: EMA passes at 5k/10.5k/20k; at 15k and 30k the CI's upper end sits at +0.0002 / +0.0018; raw fails throughout). Frozen train-gain re-timing: E1 vs pose **−0.0318 [−0.0433, −0.0201] / −0.0313**; S2 ADE vs refcv7-50,400 −0.023 SEP; S6 vs os −0.19 SEP. | H-R8-RS-1, H-R8-RETIME-TRAINFIT-1, H-R8-GAINTRAJ-1, H-R8-E1REF-SERIES |
| D2 | **False braking in 0–6 s cruise.** GT-cruise windows carry 127 % [77, 390] of refcv8's S6 ADE regression vs refcv7 (+0.166 [+0.027, +0.315] m); along-track at 6 s −2.21 m vs refcv7 +0.06; false deceleration **0.274 vs 0.096**. The trade is real: refcv8 misses fewer real decelerations (0.345 vs 0.548). | as left | H-R8-S6-LON-1 |
| D3 | **Near-term over-run in slow / slowing windows** (~72 % of E1's losses, +0.0189 [+0.0127, +0.0260]). Forcing the tactical class does NOT fix it (oracle class closes +0.172, n.s.): the defect is the decoder's speed profile given the class, i.e. D1. | as left | H-R8-E1-LOSS-1, H-R8-E1-TIMING-1 |
| D4 | **Inference variance.** The planner samples; averaging 4 draws improves E1 by −0.0225 / −0.0210 (and stacks with re-timing: −0.036). Today that costs 4x inference. | as left | H-R8-KDRAW-1, H-R8-COMBO-1 |
| D5 | **Sharp-turn under-rotation.** At 15k: turn slope −0.049 [−0.087, −0.018] vs refcv7 on the 8 % sharp windows; refinement recovers 57 % of the anchors' under-rotation vs refcv7's 74 %. At 30k on the matched instrument: −0.026 [−0.055, +0.001], NOT separated (refcv7's own milestone swing is larger). **Undecided until refcv8's final.** | as left | H-R8-TURN-UNDERCORRECT-1, H-R8-TURN-PHASE-1 |
| D6 | **The speed ceiling label is wrong on 12.52 % of eval windows** (the human exceeds the fed ceiling; the 30 km/h class in 23 % of its windows), so refcv8's bars are decided on compliant windows only. PI: relabelling is refcv9. | 12.52 % of windows outside the bars | D-R8-CLAMP-COMPLIANT |
| D7 | **The tactical layer barely trains the trunk** (trunk-update share 0.0042 < 0.01) and its lateral head collapsed to one class at the ladder rung; no refcv8 arm moved turn DIRECTION beyond noise. | as left | H-R8-BUDGET-1, H-R8-RECIPE-1 |

## 2. Levers, ranked by measured effect

| lever | fixes | mechanism (in the model) | evidence for the size | status |
|---|---|---|---|---|
| **L1 progress head (annex mechanism c4)** | D1, D3, part of D2 | Decouple "how fast" from "which path": a deterministic PROGRESS HEAD (`z_tac` + v0 / a0 / yaw rate at t0, L1 loss on the human's arc-length residual over the prior, ~68k params) drives an in-forward re-timing layer on the SELECTED path; lateral untouched by construction. The same layer with the frozen table is (a) = refcv8 + stage at initialisation, with a fitted gain (b). Penalty / reweighting variants (c1, c2) act on the wrong object. **Mandatory control:** an ego-scalar ridge (v0, a0, yaw rate, fed ceiling; fit on TRAIN) passes E1 WITHOUT VISION (−0.0212 [−0.0299, −0.0130] along the prior's path) and TIES the shipped stage along refcv8's path (−0.0328 vs −0.0318) ⇒ L1 shows vision only if it beats this scalar floor, separated. | MEASURED post-hoc: stage E1 −0.032; scalar floor −0.033; T0 ceiling (GT-nearest candidate's progress on refcv8's path) −0.081 | annex done (`refcv9/L1_DESIGN_ANNEX.md`); experiment: bank `z_tac` (~4 GPU-h dev box, ESTIMATED) then CPU fits |
| **L2 cruise false-brake** | D2 | First measure how much of D2 the re-timing already absorbs (it pulls the plan toward a constant-acceleration prior, which is exactly "keep cruising"). Whatever remains is a longitudinal tactical-head bias (BRAKE_TO windows carry 109 % of the delta, association only): calibrate the lon head on TRAIN, or fix its label timing (R8-1 2–8 s labels). Must keep missed decelerations ≤ refcv8's 0.345. | MEASURED (`…/2026-10-15-refcv8-milestone-battery/raw/refcv9_headroom/L2_HEADROOM.json`, prereg f779c3a9 before data): **PARTIAL** — the shipped re-timing halves the cruise ADE excess (compliant S6 GT-cruise 1,654 w / 119 ep: os +0.420 [+0.21, +0.65] → DEPLOY8 +0.199 [+0.03, +0.36] vs refcv7, still SEP worse) but barely moves false braking (0.266 → 0.253, n.s.; refcv7 0.091) and keeps the missed-deceleration advantage (0.387 → 0.358; refcv7 0.559). ⇒ L2 must target the braking DECISION, not ADE. **Locus (H-R9-BRAKE-LOCUS-1, association):** head-to-plan COUPLING — refcv8's plan obeys a decel call 0.67 vs refcv7 0.12, while its head calls decel on GT-cruise LESS often (0.26 vs 0.36); the brake is already in the selected unrefined anchor in 86–89 % of false-brake windows. Candidates (HYPOTHESIS): confidence-gated conditioning instead of argmax; train the conditioning on PREDICTED / noised classes; a TRAIN-fitted decel calibration | measured; a CPU re-roll (~5.5 h) settles SELECTION vs CANDIDATE and runs the F2 / S0 interventions |
| **L3 K-draw distillation** | D4 | Train a deterministic plan head toward the K-draw mean / medoid (or a multi-sample consistency loss), so the averaging gain comes at 1x inference. | MEASURED target: −0.0225 E1 (ENS-4) | HYPOTHESIS |
| **L4 ceiling relabel** | D6 | Relabel the fed speed ceiling so the human is compliant (PI-assigned); bars then cover 100 % of windows. | MEASURED coverage gain: 12.52 % of windows | PI-assigned, design pending |
| **L5 refinement heading loss** | D5 | Near-horizon heading/curvature loss on the refined candidates, weighted by GT \|yaw\|. **Only if** refcv8's final EMA confirms the turn gap. | MEASURED at 15k only; n.s. at 30k | gated on 48,852 |
| **L6 tactical trunk budget** | D7 | Rebalance the tactical loss so its trunk share clears 0.01 without the 100x weight's measured costs (speed MAE +0.03, constraint MAEs worse). | premise unmet so far | HYPOTHESIS, risky |

## 3. Expected improvements (what refcv9 should deliver, and how sure we are)

| area | refcv8 today | refcv9 target | basis |
|---|---|---|---|
| E1 (S2 vs `ha0_ext_pose`) without any post-hoc stage | fails: −0.0136 [−0.0286, +0.0018] at EMA-30k, not separated | passes AND beats the ego-scalar floor (−0.033) separated; T0 ceiling −0.081 | MEASURED floor and ceiling; in-model is HYPOTHESIS |
| S2 ADE vs refcv7-50,400 | −0.023 only WITH the stage | ≤ −0.023 without it | MEASURED post-hoc |
| S6 0–6 s in GT cruise vs refcv7 (compliant cruise windows) | +0.42 m worse; +0.20 with the shipped stage | ≤ 0 | MEASURED residual after the stage (L2) |
| false braking in cruise | 0.266 (refcv7 0.091); 0.253 with the stage | ≤ 0.10, missed decel no worse than refcv8 + stage (0.358) | MEASURED gap; fix is HYPOTHESIS |
| inference cost of the averaging gain | 4x | 1x | HYPOTHESIS (L3) |
| windows the bars are decided on | 87.5 % (compliant only) | 100 % | MEASURED coverage (L4) |
| sharp-turn refinement recovery | 57 % (refcv7 74 %) at 15k | ≥ 74 % | gated: decided at refcv8's final |
| lever claims | one training seed | replicate per claim (B9-2) | rule |

## 4. Running now (0 GPU)

1. ~~L2 headroom~~ DONE 20:14: PARTIAL (see §2 L2). The residual is the false-brake decision itself.
2. ~~L1 design annex~~ DONE ~20:45 (`refcv9/L1_DESIGN_ANNEX.md`, probes in `refcv9/l1_probe/`): mechanism corrected (D1), recommendation c4, scalar-floor control, experiment in annex §4. Open: the random-fan-candidate control for the SELECTION reading (running with the braking-locus read).
3. The step-40,000 battery (running) re-reads D1, D2 and D5 at a new checkpoint; refcv8's final (48,852, ~10-20
   evening) decides D5 and SHIP-R8.

## 5. Open PI decisions (none blocks the design work)

* Compute and timing: refcv9 training cannot start before refcv8 finishes (~10-20 evening) on Thor.
* Warm start: from refcv8-final (keeps its gains; inherits D1–D7) vs from refcv7-50,400.
* Strategic layer: refcv8 runs `--no-strategic` (v7F R6). Turning it on is the hierarchy thesis but has no measured
  lever behind it yet.
* Training seeds: B9-2 needs a replicate; two seeds double the GPU time.
