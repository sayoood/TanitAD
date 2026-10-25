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
| **L2 cruise false-brake** | D2 | First measure how much of D2 the re-timing already absorbs (it pulls the plan toward a constant-acceleration prior, which is exactly "keep cruising"). Whatever remains is a longitudinal tactical-head bias (BRAKE_TO windows carry 109 % of the delta, association only): calibrate the lon head on TRAIN, or fix its label timing (R8-1 2–8 s labels). Must keep missed decelerations ≤ refcv8's 0.345. | MEASURED (`…/2026-10-15-refcv8-milestone-battery/raw/refcv9_headroom/L2_HEADROOM.json`, prereg f779c3a9 before data): **PARTIAL** — the shipped re-timing halves the cruise ADE excess (compliant S6 GT-cruise 1,654 w / 119 ep: os +0.420 [+0.21, +0.65] → DEPLOY8 +0.199 [+0.03, +0.36] vs refcv7, still SEP worse) but barely moves false braking (0.266 → 0.253, n.s.; refcv7 0.091) and keeps the missed-deceleration advantage (0.387 → 0.358; refcv7 0.559). ⇒ L2 must target the braking DECISION, not ADE. **Locus (H-R9-BRAKE-LOCUS-1, association):** head-to-plan COUPLING — refcv8's plan obeys a decel call 0.67 vs refcv7 0.12, while its head calls decel on GT-cruise LESS often (0.26 vs 0.36); the brake is already in the selected unrefined anchor in 86–89 % of false-brake windows. Candidates (HYPOTHESIS): confidence-gated conditioning instead of argmax; train the conditioning on PREDICTED / noised classes; a TRAIN-fitted decel calibration. **Mechanism found in source (H-R9-ALLOC-ORACLE-1):** training ALWAYS forces the GT tactical hypothesis into ≥ 4 of the 32 allocated candidates (`refc.py:2512-2513`, ungated by the teacher coin); inference allocates by the head's posterior with no gating, and allocated candidates are emittable. ⇒ lead levers: (1) anneal / remove the GT allocation floor in training; (2) a GT-free "keep speed" floor or flatter allocation at inference — (2) is under test on refcv8's own checkpoint (HEDGE arms, re-roll Amendment 3) | measured; a CPU re-roll (~5.5 h) settles SELECTION vs CANDIDATE and runs the F2 / S0 interventions |
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
| false braking in cruise | ~~0.266 (refcv7 0.091); 0.253 with the stage~~ **at per-frame nav (SHIP-R8 EMA-48,852): 0.137 with the stage, missed decel 0.470 (refcv7 0.091 / 0.559)** -- an operating point, not lost discrimination (braking AUC 0.744 vs 0.727, O-SAME, H-R9-L2-OPPOINT-1) | braking AUC above refcv7's, separated; FD ≤ 0.10 at MD ≤ refcv7's 0.559 | MEASURED; the old target was set on clip-fed EMA-30k numbers |
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
## 6. v1 update -- the PI question of 2026-10-21 (Master Mind, 19:30 Berlin)

Full synthesis: `TanitAD Research Lab/Architecture & Inference/Research/2026-10-21-refcv8-why-poor/RESULT.md` (GOALS_AND_CLAIMS R8-WHYPOOR). New defects measured on refcv8 EMA-48,852 (per-frame feed), added to §1:

| # | defect | MEASURED size | register row |
|---|---|---|---|
| D8 | **Late decoder drift carries the NavSim loss** | decoder-only blend +12.97 [+10.55, +15.68] navtrain; the decoder moved only 3.7 % | R8-WISEFT-MOD2 |
| D9 | **Lateral head echoes the nav token** | adds +0.0226 over a token lookup; vision-only 0.8873 < constant 0.9039 | H-R8-LAT-ECHO-1 |
| D10 | **Route direction below bar** | 0.870 [0.794, 0.925] vs bar 0.95 (proxy) | R8-ROUTE-PROXY-1 |
| D11 | **Planner starved of trunk gradient** | trajectory 0.016 % of the trunk update (refcv7 P-GRAD) | WPD-PGRAD |
| D12 | **No safety terms in selection; box3d not wired** | REFe scorer-only +3.79; box3d read only by its loss | R8-DD-CONFORM-1, R8-PERC-WIRING-1 |

New levers, ranked (all HYPOTHESIS for refc unless marked):

- **L0 decoder protection** -- anchor or freeze the decoder late in training (MEASURED post-hoc +12.97).
- **L7 DDv2-style safety selector** -- NC / DAC / TTC / EP / C sub-scores (MEASURED +3.79 on REFe).
- **L8 nav-blind lateral loss.**
- **L9 a planner share of the trunk update well above 0.016 %.**
- **L10 perception set-prediction fixes**, gated on a causal test.

⭐ **The decision experiment that fixes refcv9's architecture** (needs PI compute):
- Fine-tune θ_DEC, front camera only, on NAVSIM navtrain.
- **≥ ~85** ⇒ data was the gap; refcv9 keeps the refc trunk.
- **≤ ~75 while REFe is at 89** ⇒ sensors / backbone; refcv9 adopts REFe's DINOv3 ViT-L and multi-camera input.
- Both outcomes are to be registered before any run.

## 7. Perception track P0–P4 (Master Mind, 2026-10-25 ~01:10 Berlin; PI in chat: *"follow your plan as porposed"*)

L10 (§6) was gated on a causal test. That test has run, and so has an eval-grade perception read. MEASURED (perc8: T0 perception read; perc2plan: T1 open loop, os-only),
refcv8 EMA-48,852 vs refcv7@50,400, one training seed per arm, paired episode-cluster bootstrap over the 139 eval
episodes (the CI answers *another draw of episodes* only — `H-ESTIM-SEED-1`):

* **The planner reads its BEV tokens.** Zeroing them worsens S2 ADE +0.206 / +0.202 m (inference seeds s0 / s1) and
  S6 +1.15 / +1.13 m, separated worse in every reported family (ADE, longitudinal, lateral, tactical); inference-seed floor 0.0018 m (S2) / 0.0237 m (S6). Agent slots:
  S2 +0.015 / +0.016, S6 +0.114 / +0.094. The box3d head's output is not read by the planner (D12).
  Source: `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-21-refcv8-perc2plan/RESULT.md`.
* **Box head:** AP@2 m 0.328 [0.301, 0.354] vs 0.249 [0.227, 0.270] (paired +0.079 [0.071, 0.087]); AP@0.5 m 0.065;
  AP@2 m at 40–60 m 0.092; **2.14 boxes per detected object** (67.8 % of objects carry ≥ 2); the declared 0.5 gate
  finds 1.3 % of objects (recall 0.013).
* **Map head:** drivable IoU 0.49–0.61, sidewalk 0.46–0.53, but **lane 0.199 (0–20 m) / 0.060 / 0.026 / ≈ 0**,
  crosswalk 0.149 (0–20 m) / ≈ 0 beyond, arrow / non-drivable edge / hatched ≈ 0 in every band. Deltas vs refcv7
  ≤ +0.013 per map class — refcv8's levers did not touch perception.
  Source (both bullets): `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-21-refcv8-perc8/RESULT.md` (e0fba03; the
  58 MB `perc_det.pkl` and the window lists stay on the dev box, see that file's manifest).

| # | lever | what | measured basis / bar | status |
|---|---|---|---|---|
| **P0** | ship refcv8's box post-processing | NMS 2.5 m + presence gate 0.2165, fitted on TRAIN-DIAG (1,112 windows / 139 train episodes), frozen, re-scored on EVAL | MEASURED: AP@2 m 0.328 → **0.447** (within-arm +0.119 [0.096, 0.139]), F1 0.457, 1.088 boxes / object | **LANDED** with R8-P2-MAPTOL: `stack/tanitad/configs/refcv8_det_{nms,presence_gates}_train.json`, pinned by `stack/tests/test_refcv8_det_shipped_pin.py`; an opt-in readout config (`--det-nms` / `--det-presence-gates`, post-hoc box reads) -- no planner or loss consumer |
| **P1** | box set prediction in the model (= L10) | one-to-one (Hungarian) matching loss + a calibrated objectness loss; box3d read from the 0.25 m grid instead of the pooled 0.5 m BEV | bar: AP@2 m > 0.447 **without** NMS, AP@0.5 m > 0.065, ≤ 1.1 boxes / object; a replicate arm per claim (B9-2) | HYPOTHESIS; ranked by P4 |
| **P2** | map tolerance re-read | P / R / F / IoU_k at k = 0, 1, 2, 5, 10 cells (0–1.0 m) per class × 20 m band, paired vs refcv7, GT line-width census; per thin class a pre-registered MISPLACED / ABSENT / MIXED verdict | separates *slightly off* from *not there*, which exact 10 cm IoU on a 10–15 cm line cannot (a 10 cm offset halves it) | **DONE 2026-10-25 (R8-P2-MAPTOL)**: lane / crosswalk **MIXED at 0–20 m** (P_5 0.934 / 0.745, R_5 0.384 / 0.321: under-predicted, not misplaced), **ABSENT at 20–40 m** (R_5 0.120 / 0.006); refcv7 reads the same verdict in all four cells |
| **P3** | thin classes + far range | P2 = MISPLACED ⇒ a thin-structure loss (Dice / Lovász-style) with uncapped thin-class weights and the 10 cm head read from finer features; P2 = ABSENT ⇒ temporal BEV fusion of past frames for > 20 m | MEASURED imbalance at 0–20 m: lane pixels 1/22 of drivable, arrow 1/507; the median-frequency class weights are clipped at 25 (refcv7 map design, INHERITED — re-verify in the P3 spec) | HYPOTHESIS; P2 chose BOTH branches: 0–20 m is a RECALL lever (thin-class loss / weights), 20–40 m an INPUT lever (temporal fusion / resolution) |
| **P4** | true-map planner oracle | the planner fed the TRUE map vs its OWN map through the identical path (an adapter's capacity cancels), a shuffled-GT arm that must not help, an identity control that must equal BASE bit-for-bit; both inference seeds, all four families | decides the rank of P1–P3 against L1 / L2 (Rule Zero §5): NO-HEADROOM ⇒ perception levers wait behind the planner; HEADROOM ⇒ its size becomes P1–P3's target | **RUNNING** on the dev-box GPU since 2026-10-25 04:41 CET (prereg sha256 `d9fc8245…`): a no-training δ-oracle on `map_hires_bev` that makes the frozen 10 cm decoder read the target map, TRUE − OWNMAP through the identical procedure, SHUF + identity controls |

A true-map input is privileged: P4 is an oracle bound, never a deployable arm (vision-only inference rule).

## 8. L2 re-read at the SHIP-R8 checkpoint (Master Mind, 2026-10-25 ~11:55 Berlin)

* **The D2 premise was stale.** §1 D2 and §2 L2 quote EMA-30k numbers read with the CLIP-level nav feed. On the banked EMA-48,852 per-frame battery (compliant S6) refcv8 + stage reads false-decel 0.137 / missed-decel 0.470 vs refcv7 0.091 / 0.559.
* **It is an operating point, not a defect in discrimination.** Pre-registered (`…/2026-10-25-refcv9-l2/roc/PREREG_L2_ROC.md`, sealed before data): braking AUC refcv8 + stage 0.7445 / 0.7361 vs refcv7 0.7266 / 0.7182 at inference seeds 0 / 1, Δ +0.018 [−0.008, +0.046] → **O-SAME** (one training seed). refcv8 trades false decels for caught decels on the same frontier.
* **Consequence for the levers.** L2 as a threshold / lon-head calibration can only move along that frontier (its GPU arms run, queued on the dev-box lock; the 0-GPU bound already says the head alone cannot reach FD 0.10). The lever that would beat BOTH models is higher braking discrimination (AUC ~0.73 today): the speed-choice defect D1, i.e. **L1**. L1's arms are to be read on this AUC as well as on E1.

## 9. L1 result (Master Mind, 2026-10-25 ~13:20 Berlin) -- O2 on train-fitted, but the information is there

* **O2, FAIL as registered.** A ridge progress head on refcv8's z_tac fitted on train139 (refcv8's own training clips) is separated WORSE than the shipped stage and than the ego-scalar floor (E1 +0.0264 vs H0 −0.0343). Its braking AUC also drops (0.667).
* **Why:** the z_tac → progress relation is memorised on the trunk's training clips (held-out-episode R² inside train 0.54 / 0.25 / 0.27 / 0.33 → eval 0.41 / −0.27 / −0.49 / −0.55), while z_tac's distribution barely shifts (domain AUC 0.547).
* **The lever survives.** Cross-fitted on clips refcv8 never trained on, z_tac beats the scalar floor, separated at both seeds (S2 −0.0044, S6 −0.129 m). The exploratory size of the prize vs the shipped stage: −0.0105 on S2, −0.20 m on S6 (not a claim).
* **Consequence for refcv9's design:** the progress head must be fitted on a CLEAN CALIBRATION SPLIT (clips the trunk never trains on). Co-training it on the trunk's own clips would learn the memorised relation that failed here. Options and the PI decision: PI_DECISION in chat 2026-10-25. Package: `TanitAD Research Lab/Architecture & Inference/Research/2026-10-25-refcv9-l1/`.
