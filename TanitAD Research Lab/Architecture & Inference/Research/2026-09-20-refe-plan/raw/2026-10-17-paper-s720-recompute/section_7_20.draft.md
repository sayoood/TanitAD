### 7.20 REFe on NAVSIM: a published-row comparison at last, a gap that is a zero tail, a zero tail that is a selection problem, and one correction of our own (2026-09-24 → 10-17)

Until this round the paper held no number comparable to a published one: §12.4 records that the programme
contained no NAVSIM implementation, and names the order that follows — benchmark, then reward, then RL. This
section is the benchmark step. **REFe** is a reference arm in the sense of §3.8: a reproduction attempt of
DriveZero's camera planner (arXiv:2609.06055, banked as library key `2609.06055`) on its own data and its own
benchmark, built to calibrate our instrument and to say *where* the gap to the published state of the art sits. It
is not the 4B architecture and nothing here is a claim about it. **In one sentence: the best REFe checkpoint
scores 88.25 PDMS against a committed bar of 94.55 — the bar is NOT cleared; 65.9 % of the gap is a tail of
zero-scored tokens; that tail is a *selection* failure whose mechanism is now located; an execution-side lever
aimed at it was closed by our own measurement, and the next (P6) has not yet trained.** Notation: `RP` = `TanitAD
Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`; `REG` = `Project
Steering/MODEL_REGISTRY.md`; `GAC` = `Project Steering/GOALS_AND_CLAIMS.md`. Every number below is **T1** unless
it says otherwise.

**(a) The instrument.** *(MEASURED unless marked; `REG` §14.1–14.2, `RP/REFE_MODEL.md`.)* REFe is a frozen DINOv3
ViT-L/16 with LoRA adapters, four cameras at 512 × 960 (H × W), 64 trajectory proposals (winner-takes-all) and a
six-component PDM scorer: **322,070,854 parameters, 18,991,426 trainable** (the paper's: 338.46 M / 18.58 M,
PUBLISHED, Table A12 — the DriveVFM ViT-L configuration). It trains on navtrain (103,288 tokens, 1,192 logs) under
a DriveRL teacher for 10,075 optimiser steps. Its inputs are the benchmark's — cameras, seven ego-kinematic
scalars and a route-derived goal — and **no HD-map geometry is an input to the planner** (PI, 2026-09-28:
navigation commands are admissible, map geometry for planning is not; maps appear as training labels, PI,
2026-10-11); the one place a map graph touches an input is A9 below. The ego-status shortcut of §7.2 therefore
applies to this interface, and the anti-echo controls are in (f). It departs from the paper in seven declared ways
(bf16 autocast, a growing data bank, 10 Hz teacher rollouts, 4 × 64 gradient accumulation, a scorer supervised on
a fixed candidate set until step 3,707 and on its own proposals after, and NAVSIM's own drivable-area and comfort
labels), so it is a *reproduction attempt*, not a replication. **Scoring** *(PUBLISHED, arXiv:2406.15349 Eq. 1,
library key `2406.15349`)*: PDMS = NC × DAC × (5·EP + 5·TTC + 2·C)/12 per token — the plan is queried once,
executed by an LQR controller and a kinematic bicycle for 4 s against logged agents that do not react, and a
collision (NC) or a drivable-area violation (DAC) zeroes the token. **Tier T1** in the sense of §5.4 (the
planner's own plan is executed, perception is fixed at t₀, agents are replayed) and ⛔ *not* closed-loop driving;
T2 is not provisioned, and §12.4's warning stands — PDMS buys comparability, not truth.

**Estimator, and the variance it answers.** Paired log-cluster bootstrap over the 136 navtest logs, 10,000
resamples, seed 20260927 (`RP/raw/2026-10-17-zero-tail-mechanism/code/zero_tail.py`, class `Boot`) — the NAVSIM
analogue of §5.3's episode-cluster bootstrap. Three different questions ride on any interval in this programme:
*would another draw of logs say this? would another training run say this? would another inference run say this?*
(H-ESTIM-SEED-1 in `GAC`; §5 does not yet state it and this round owes it). This interval answers **only the
first**. Inference is deterministic (P2's re-derived picks are pose-bit-identical and PDMS-exact on 509/509
tokens, `RP/raw/2026-10-15-p2-zero-tail/p2_own_table/P2_MEASURED.json` `same_pick_proof`), so the third is zero by
construction. The second is unmeasured for every arm but one pair: P2 seed 1 minus P2 seed 0 reads **+0.33 [+0.04,
+0.64] PDMS** (`RP/eval/RESULT_P2REP.md`, table RR2) — one draw of a seed difference, never a variance, and the
only floor measured at this tier. Every separated interval below is therefore necessary and not sufficient for a
lever claim. **Harness controls.** Our harness reproduces the published constant-velocity baseline (20.65 vs 20.6,
PUBLISHED, NAVSIM Table 1) and scores the logged human at 94.55 [93.78, 95.23] against a published 94.8
(`RP/eval/SPEC_NAVTEST.md` line 12; `RP/eval/RESULT_P4_TEACHER_NAVTEST.md` table C2). The do-nothing floor is
**61.82** — EP is 1 by rule when compliant progress is under 5 m — so standing still beats constant velocity by 41
points and the scale is compressed from below; the earlier programme model refcv4b A1 reads 58.97, *below* it
(`RP/eval/RESULT_NAVTEST_FINAL.md` headline table). **Three declared test-time changes ride on every REFe row
below** (`REG` §14.1): **A7** executes the chosen plan with its t = 4.0 s heading replaced by its t = 3.8 s
heading (60.76 → 76.96, +16.20 [+12.83, +19.52], on 923 tokens / 43 logs disjoint from the earlier 200-token
subset, at the epoch-15 snapshot; registered before its confirmation data existed); **A9** repairs the benchmark's
own route with NAVSIM's roadblock graph — only the navigation goal moves (+1.22 [+0.59, +2.07] on the full
navtest; adopted on the PI's ruling, 2026-10-04; the lane-graph goal sanitisation A8 is a new map use and is
**not** deployed); and NAVSIM v1's own selection formula (for u400, multiplied by a learned lane output p(lane) —
a scorer output trained on map-derived labels, not a map read at inference).

**(b) The ladder, and the gap to the published row.** *(full navtest, 12,146 tokens, 136 logs; PDMS × 100.)*

| row | PDMS [95 % CI] | evidence · artifact |
|---|---:|---|
| constant velocity | 20.65 | MEASURED · `RP/eval/RESULT_NAVTEST_FINAL.md` (published 20.6) |
| do-nothing (STOP) | 61.82 | MEASURED · same file; no interval banked |
| REFe `model_final` | 83.24 [81.77, 84.63] | MEASURED · `REG` §14.1 Status row; recomputed identically here |
| + A9 route fix | 84.46 [83.18, 85.77] | MEASURED · `REG` §14.1 (recompute here [83.16, 85.69]; endpoints differ ≤ 0.08) |
| **SFT-4 B u400** (scorer-only fine-tune; best REFe checkpoint) | **88.25 [87.05, 89.36]** | MEASURED · `REG` §14.2; interval recomputed for this section¹ |
| P2, seed 0 (joint PDM fine-tune) | 88.82 [87.86, 89.77] | MEASURED · `GAC` D-REFE-P2-R2; interval recomputed¹ |
| P2, seed 1 | 89.15 (interval not computed) | MEASURED · `RP/eval/RESULT_P2REP.md` |
| teacher (DriveRL `checkpoint_2400`, privileged) | 89.79 [88.55, 90.97] | MEASURED · `REG` §14.2 (point); interval `RP/eval/RESULT_P4_TEACHER_NAVTEST.md` line 58, recomputed identically¹ |
| human (logged future) | 94.55 [93.78, 95.23] | MEASURED · `REG` §14.2 (point); interval `RP/eval/RESULT_P4_TEACHER_NAVTEST.md` table C2, recomputed identically¹ |
| **matched published row** (DriveZero, DINOv3 ViT-L, navtrain only, no SimScale) | **94.55** | PUBLISHED · arXiv:2609.06055 Table A13, verified in the banked PDF |

¹ Same estimator, rerun from the harness CSVs; the controls `model_final`, teacher and human reproduce
their banked values to two decimals (`RP/raw/2026-10-17-paper-s720-recompute/marginal_ci.json`, to be landed).
⚠️ **Two different quantities carry the value 94.55** — the published *student* row and our harness's
*human* row (the published human is 94.8). Neither is a typo; the equality is a coincidence of two separate measurements.

*Paired steps:* SFT-4 over `model_final` + route fix **+3.79 [+3.05, +4.54]** (`REG` §14.2; 0 of 21
longitudinal/lateral family components adverse); P2-s0 over u400 **+0.57 [+0.02, +1.18]** (the GO of `GAC`
D-REFE-P2-R2); P2-s1 over u400 +0.90 [+0.36, +1.59] **but TTC −0.49 [−0.97, −0.07], adverse**, so the committed
rule reads **NOT REPLICATED** (`RP/eval/RESULT_P2REP.md`) — the size of the PDMS gain replicated, the safety
clause did not, and P2's +0.57 is of the order of one seed difference (+0.33).

**The gap decomposes as a zero tail first.** *(MEASURED, exact and sequential — gate first, then EP, TTC, C;
paired log-cluster bootstrap, same draws for every term; intervals recomputed for this section, control =
`RP/raw/2026-10-11-ep-gap-decomposition/ep_lowtail.out` reproduced to two decimals.)*

| human − u400 = **6.30 [5.39, 7.23]** | PDMS points [95 % CI] | share |
|---|---:|---:|
| tokens zeroed by NC × DAC (529 of 12,146 = 4.36 %; human 0) | **4.15 [3.31, 5.07]** | 65.9 % |
| TTC (TTC = 0 on 4.37 % of u400 tokens; human 0 %) | 1.31 [1.09, 1.53] | 20.7 % |
| EP, on the tokens that are not zeroed | 0.86 [0.41, 1.28] | 13.6 % |
| comfort | −0.01 [−0.03, +0.00] | −0.2 % |

(`RP/raw/2026-10-17-paper-s720-recompute/gap_decomp_ci.json`, to be landed; registry statement: `REG` §14.2 "Gap
decomposition", `GAC` D-REFE-GAP-1.) The split is order-dependent — gate first is a choice, not a law — and it is
a decomposition, not a lever claim. Against the published *student* profile (PUBLISHED, Table A13) the picture is
the same from the other side: NC 98.94 vs 98.80 and TTC 95.63 vs 95.74 already match; **DAC 96.58 vs 99.25 and EP
81.20 vs 92.72** do not. The published planner reaches 94.55 by being *faster* than the human (EP 92.72 against
the human's 87.5); the human reaches it by never being zeroed. This corrects an earlier internal reading that "EP
is the largest gap", which was true of the published profile and not of the bar's decomposition (`GAC`
D-REFE-GAP-1, second row of that id).

**(c) The zero tail is selection, not proposal.** *(MEASURED; pre-registered before any count; **T1**; estimator
as above — answers another draw of logs only; both arms are one training seed.)* A zero-PDMS token is **SEL** if
some proposal among the 64 scores above zero (a non-zero alternative existed and the scorer did not pick it) and
**PROP** if all 64 score zero. On u400's own 64-proposal table: of its 529 zeros, **455 are SEL and 74 PROP — SEL
share 86.0 % [80.1, 91.7]**, and handing every SEL zero its best proposal would add **+3.62 [+2.85, +4.55]** PDMS
(`RP/raw/2026-10-15-p2-zero-tail/RESULT.json`, `Q1a_u400_own`). On P2's own table
(`…/p2_own_table/P2_MEASURED.json`; an earlier bridged estimate of 439/70 was INFERRED and is replaced): of 509
zeros, **445 SEL and 64 PROP, 87.4 % [81.0, 93.2], ceiling +3.54 [+2.90, +4.23]**. ⚠️ Both ceilings are *ceilings*
— the gain of a perfect selector — and not the expected gain of any scorer (this distinction is the subject of
(e)). P2's zeros split **DAC-only 376 (73.9 % [66.7, 80.3]), NC-only 129, both 4**; NC-type zeros are 130/133 SEL,
and the DAC-only SEL tokens alone carry +2.50 [+1.88, +3.20] of the 3.54. The PROP share is mostly off-road fans
(59 of 64 have all 64 proposals leaving the drivable area), capped at ≈ 0.49 PDMS (ESTIMATED). The SEL fan is not
a needle in a haystack (median 19 non-zero proposals of 64, POST-HOC). Turning is over-represented: among u400's
DAC-only SEL zeros, the enrichment over a command's share of navtest tokens is ×1.59 [1.12, 2.18] for left turns
and ×1.89 [1.30, 2.70] for right turns, against ×0.65 [0.47, 0.81] for straight
(`RP/raw/2026-10-17-zero-tail-mechanism/RESULT.json`, `arms.A.Q2`). The TTC term has the same shape: of P2's 394
non-zeroed tokens with TTC = 0, 96.7 % have a TTC-ok proposal in the fan (INFERRED through the bridge; ceiling
+1.44 [+1.17, +1.72]).

**(d) The DAC zero tail is near-twin grazes the scorer cannot see.** *(MEASURED,
`RP/raw/2026-10-17-zero-tail-mechanism/`, `PREREG_ZT3.md` hashed before any number, two amendments before any
target value; **T1**; gates: DAC re-simulated on 2,996/2,996 trajectories, simulated corners equal to the banked
tracked dump (max 0.0 m), the human scores DAC = 1 on 1,224/1,224.)* The target set is u400's **343 DAC-only SEL
zeros**. **60.1 % [51.4, 68.0] are ON_PATH near-twin grazes** — the pick matches the best-ranked non-zero proposal
to a median 0.29 m lateral offset, a length ratio of 1.01 and a heading difference of 0.04°; the route differs on
only 1.7 % [0.3, 3.6] and the manoeuvre class equals the human's on 95.3 % — carrying **1.55 [1.06, 2.15] of the
2.55 [1.83, 3.41] PDMS at stake** (P2 replicates: top class equal, 63.2 %). They are shallow, late and far: 74.1 %
[66.8, 80.5] leave the drivable area by under 0.5 m (median 0.23 m), at a median 3.1 s and 17.9 m out (38.5 %
[28.2, 48.6] inside 15 m). **The scorer's DAC head is blind exactly there:** it rates the pick at least as
drivable as the alternative on 72.0 % [63.9, 78.5], with a within-token AUC of 0.46 [0.36, 0.57] against 0.92
[0.90, 0.94] on random tokens (near-twin pair accuracy 0.39 [0.27, 0.57] against 0.96 [0.94, 0.98],
`ZT4_BASELINE.json`; both POST-HOC and conditioned on failure, so not a causal scene effect). Of the five
pre-committed readings, R1 read GEOMETRY (not route), R2 EXECUTION_ARM, R3 HEAD_WRONG, R5 NEAR_BLIND — and **R4
read FAR** (61.5 % beyond 15 m), whose committed meaning is an *anticipation* problem rather than a near-field
one. Finally **42.0 % [33.9, 49.4] (144 of 343) exist only in NAVSIM's simulated tracking**: the plan's own
footprint stays on the road, the tracked car leaves it. The lever ranking printed with this package listed *reach*
as 58.0 % / 1.48 [1.04, 1.95] PDMS for the P6 trunk DAC gradient, 42.0 % / 1.07 [0.71, 1.55] for an execution-side
fix, 25.7 % / 0.65 [0.43, 0.89] for map supervision (P7.1) and 2.9 % / 0.06 [0.01, 0.13] for lane-rank twins.
**Those numbers are gross. See (e).**

**(e) The execution side — and a correction of our own (R47).** *(MEASURED,
`RP/raw/2026-10-17-tracking-execution/`, `PREREG_ZT5.md` hashed before any number, three amendments before their
data; **T1**.)* The mechanism is real and it is the **heading channel**: on the 144 tracker-made navtest picks (44
logs; 529 post-stratified random controls; gates 673/673), replacing the plan's heading with its own path tangent
rescues **56.9 % [45.1, 66.0]** of them, actuator lag 47.2 % [38.5, 57.3], steering 18.1 % [12.0, 26.3], speed 7.6
% [3.3, 13.2]; the heading edit breaks 1.7 % [0.2, 3.8] of controls. The tracked car under-turns and runs wide
late. **The fix is not found.** On a development set of 2,466 held-out navtrain sets (13 navtest-disjoint logs;
baseline PDMS 89.28 [85.53, 91.61]) no admissible fix clears the pre-registered bar (lower 95 % bound > 0 with a
point ≥ +0.05, positive on two further scorer seeds, no sub-score adverse): a heading-consistent plan (F_c; it
generalises A7 to every knot) **−0.30 [−0.72, +0.04]** (19 zeros rescued, 32 broken), track-consistent selection
among near-ties (F_b) −0.03 [−0.15, +0.07], the two combined (F_bc) −0.19 [−0.68, +0.27], a fan-consensus margin
selection added by amendment 3 (F_m) +0.00 [−0.06, +0.05], and even the *inadmissible* pre-compensation ceiling
(F_a) −0.19 [−0.76, +0.63]. No navtest confirmation was run (`PREREG_ZT5` §6). One development set failed its
strict gate (EP only, 0.0286; weight 1/2,466).

⚠️ **The correction.** The package of (d) overstated what it had found. `GAC` D-REFE-ZEROTAIL-3 and our report to
the PI the same day quoted the tracking-made 42 % as an execution-side *reach* of **1.07 PDMS** — the
second-largest lever. **1.07 is the gross reach: the PDMS carried by zeros that exist only in simulated tracking.
It is not what a fix returns,** because at the pick the tracker also *saves* plans whose own footprint fails DAC
and whose simulated execution passes. Measured post-hoc, on the development set the tracker saves 27 picks and
makes 20 (difference +0.28 pp [−0.88, +1.24]); on navtest the saved count is ESTIMATED at ≈ 132 [44, 237] against
144 made. The interval spans zero, so the claim is not "equal" but "not separable" — which is why the closure
rests on the pre-registered fixes failing, not on this count. The net reach of the fixes tried is **about 0**
(`RETRACTION_LOG.md` R47; `GAC` D-REFE-ZEROTAIL-4). It was caught only because ZT5's fixes kept breaking as many
zeros as they rescued, which made the reverse flow visible. **The class, stated as method:** *a reach computed
from one side of a two-sided flow is a ceiling, not a lever* — the same family as quoting an oracle-on-SEL gain
(c) as an expected gain — so **before quoting any lever's reach, count both directions at the decision point.**
The P6 figure of 1.48 is gross for the same reason and is labelled so until (g) nets it. The honest scope of the
closure: 13 development logs, wide intervals, and the specific fixes tried — it closes *those* fixes, not every
conceivable execution-side one.

**(f) The four metric families, as they apply to NAVSIM** *(§5.5; u400 on the full navtest, 12,146 tokens / 136
logs; MEASURED, `RP/eval/RESULT_SFT4.md` "Four families"; point estimates, intervals in the JSON named there;
tactical rows carry none and are reported, not tested)*:

| family | measured | not measured, and why |
|---|---|---|
| **LONGITUDINAL** | EP 81.20, TTC 95.63; speed MAE 0.740 m/s, along-track MAE 1.035 m, acceleration MAE 0.519 m/s², progress ratio 1.003 | distance-keeping (headway / TTC *to the lead agent*): **UNAVAILABLE**, no lead-agent track in the seam — a work item, not an excuse; the benchmark's TTC is the nearest proxy, not the instrument |
| **LATERAL** | DAC 96.58 (DDC 97.66 is reported; weight 0 in v1.1); heading MAE 2.532°, cross-track MAE 0.378 m, curvature MAE 0.0199 1/m, yaw-rate MAE 2.983 °/s | — |
| **TACTICAL** | lateral decision accuracy / κ 0.925 / 0.863; longitudinal 0.679 / 0.516; 5-way 0.783 / 0.718 | intervals; and the zero tail is *invisible* here — the manoeuvre is right on 95.3 % of near-twin picks (d) |
| **STRATEGIC** | — | **PAUSED** (PI ruling, 2026-09-19, `PI_DECISION_QUEUE` item 25: the strategic layer is switched off until tactical and operative drive well) and independently UNAVAILABLE in NAVSIM (no strategic decision is scored). Reported as PAUSED — not a pass, not omitted |

NC (98.94) and C (99.97) are joint gates and reported beside the families, not assigned to one. Two further
controls guard the ego-status interface: u400 beats the hold-v0 control and the copy detector is clean (echo index
0.0013 against the logged human's 0.0338). **The ADE-only failure of §5.5 has a NAVSIM form:** PDMS is one scalar,
and the family table shows what it hides — SFT-4's PDMS gain comes with every MAE improving, whereas P2-s0 over
u400 carries 3 of 21 longitudinal components adverse (all speed tracking, `GAC` D-REFE-P2-R2) and P2-s1 over u400
none (`RP/eval/RESULT_P2REP.md`): a family-level verdict that differs between two seeds of one recipe.

**(g) What is next, and why.** The remaining levers are ranked by measured effect, not by interest. **P6** — the
paper-faithful full REFe run (`RP/SPEC_P6_PAPER_FAITHFUL.md` §0 and design table: un-detached scorer context, PDM
targets online on the model's own proposals, ≈ 7 A40-days, ESTIMATED) — is the only lever with a non-zero share on
the near-twin class (d), because it puts the DAC gradient into the trunk. The frozen-feature version of the idea
(S1, nine arms) was not promoted: head DAC AUC 0.772 / 0.765 / 0.781 against the scorer's 0.787, and navtest P1
IoU on [0, 40) m 0.496 / 0.521 / 0.486 against a 0.80 bar (`RP/raw/2026-10-15-s1-read/S1_READ.json`, `GAC`
D-REFE-S1-READ) — which is why P6 un-detaches the trunk instead. Its gross reach is 58 % / 1.48; **ZT4 measures
its net reach**: the package of (d), unchanged, run on P6's own 64-proposal fan, comparing P6's DAC-only SEL zero
count with u400's 343 on paired per-token indicators (so zeros removed and zeros created net out). Both outcomes
are committed in advance (`RESULT.json` `next_experiment_ZT4`): **TRUNK-REACHES** iff zeros ≤ 240 (−30 %) with a
paired CI excluding 0 *and* near-twin accuracy ≥ 0.70 (control 0.39) → map supervision (P7.1) goes on top of P6;
**TRUNK-MISSES** iff the reduction is < 15 % or its CI includes 0, or accuracy ≤ 0.50 → the camera trunk does not
learn sub-0.5 m late boundaries from PDM labels (anything between is PARTIAL, reported per class), and the next
arm targets the tracker-made 42 % by training-side margin targets (P5-C), which needs the PI's GPU go. Cost ≤ 1
dev-box GPU-hour. **Blocked on one thing: P6's checkpoint**, which waits on its commit-filtered twin bank (3-C, PI
2026-10-17); the control arm is already measured. A P6 PDMS claim additionally needs a replicate, with the P2
pair's +0.33 as the floor it must clear.

**What §7.20 does not establish.** (i) The bar is not cleared: 88.25 [87.05, 89.36] against 94.55, one training
seed per arm except the P2 pair. (ii) Nothing here is closed-loop driving. (iii) The zero-tail mechanism is
measured on u400 and P2, two members of one lineage; it is not shown for the published model. (iv) The
scorer-blindness statistics are conditioned on failure. (v) The execution-side closure is bounded by 13
development logs and by the fixes tried; the gross-versus-net count is post-hoc. (vi) The oracle-on-SEL ceilings
and every "reach" are ceilings. (vii) STRATEGIC is paused, not passed, and distance-keeping is unmeasured. (viii)
The recomputed intervals (footnote ¹ and the decomposition) are this section's own and are not in the registry
until landed.
