# Auxiliary drivable-area head: pre-registration (DRAFT)

> ⛔ **DRAFT. It is not in force until all three hold:**
> 1. the PI has decided **D-AUX-1** (inside P6, or a separate arm) and **D-AUX-2** (`SPEC_AUX_DRIVABLE_HEAD.md` §0); **MET 2026-10-12: D-AUX-1 = (b) separate arm, D-AUX-2 = yes (PI in chat; SPEC §0 "PI DECISIONS").**
> 2. the label store is built and its **S0 gate passes** (`SPEC_AUX_DRIVABLE_HEAD.md` §7: alignment, no-leakage, exact-0.0 controls, mutation arms RED);
> 3. this file is finalised, its blanks (last section) are filled, and it is **landed in git before the first training step of the first arm**.
>
> After landing nothing below changes without a dated amendment written before the read.
> Design: `SPEC_AUX_DRIVABLE_HEAD.md`. Host recipes: `SPEC_P6_PAPER_FAITHFUL.md`, `eval/PREREG_P6.md`, `eval/RESULT_SFT4.md`.
> Written 2026-10-11. Evidence classes as in the SPEC.

## The treatment (identical in every host)

The head of `SPEC_AUX_DRIVABLE_HEAD.md` §2: 2 × `CrossBlock(256, 8)` over 40 × 32 BEV queries reading `visual_ctx`; output a mask logit and a signed
distance per 2 m cell; loss `BCE + Huber`; `w_aux` fixed by the gradient-norm balance of §2.3 before launch and then frozen. Labels: the int8 SDF store of §3.
Two variants, never mixed in one arm:

- **H1** representation only (`--aux-drv-feed off`).
- **H2** H1 plus the three detached per-proposal scalars into the scorer (`--aux-drv-feed on`).

## Hosts, arms and what each can show

| host | trunk | arms | comparator | what it can show |
|---|---|---|---|---|
| **S1** SFT-4 recipe, dev box | frozen | **H2**; **C-shuffle** (H2 with a derangement of the label keys); **C-seed** (no head, another seed, the seed-floor arm); ≥ 3 seeds each | banked u400 (88.25, `RESULT_SFT4.md`) and C-seed | the perception proof, and H2's effect with attribution. **H1 cannot act here by construction** (no path from the head to the planner), so S1 tests H2 only |
| **S2** graft, pod, P2 scale | trainable (P2's recipe) | **H1**, **H2**, no-head control from the **same init and steps**; 2 seeds each | the no-head control | the representation lever and H2 with attribution; a **weaker** version of the lever (late, small LR) |
| **S3** inside a full run | trainable from step 0 | head inside P6 (option (a)) or a second full run (option (b-i)) | P6 / C1 for (a) (a 7-lever bundle, **no attribution**), the head-less run for (b-i) | only a bundled claim for (a); a clean contrast for (b-i) |

⚠️ UNVERIFIED: that the SFT-4 trainer (`refe/scorer_finetune.py`) can read `visual_ctx` on the dev box at a cost the S1 budget allows. If it cannot,
S1 is replaced by S2 and the perception proof is read from a head-only fit on cached `visual_ctx` of the C3 tokens.

## Estimator, tier, and what the interval answers

- **Tier:** NAVSIM v1 PDMS, an open-loop plan pseudo-simulated against logged agents. **Not closed loop.**
- **Population:** the full navtest, **12,146 tokens over 136 logs**, NAVSIM v1.1. Harness and seam as for `RESULT_SFT4.md` (route fix ON, A7 repair ON,
  `navsim_v1` selection), on the dev box, never on a training pod. System against system: each arm runs under its own registered selection rule.
- **Estimator:** paired log-cluster bootstrap, **10,000 resamples, seed 20260927**, for every difference and level.
- **It answers one question:** *would another draw of logs say this?* It is blind to a second training run (H-ESTIM-SEED-1, measured: a zero-lever
  replicate produced "separated" cells at 9.5 %). The seed floor is therefore read from replicates (§Replicates).
- **Inference variance:** REFe inference is deterministic, **asserted**: a second full evaluation of the same checkpoint must be bit-identical in
  per-token PDMS. A mismatch is a defect, investigated before any read.
- **Minimum detectable effect (ESTIMATED):** ≈ **0.75 PDMS**, from the half-width of SFT-4's paired interval ([+3.05, +4.54], `RESULT_SFT4.md`). The expected
  effect (+1.0 to +1.6, SPEC §1) sits just above it, so **NOT PROVEN is a likely outcome and is written down as such.**

## Endpoints (paired, same tokens)

| id | endpoint | role |
|---|---|---|
| **E1** | **PDMS** (arm − comparator) | the decision endpoint |
| **E2** | **DAC sub-score** and **DAC-zero share** (fraction of tokens with DAC = 0; u400 reads 3.42 % = 415 tokens) | the **mechanism**: a PDMS gain that does not come with fewer DAC zeros is not credited to the head |
| **E3** | **zeroed-token share** (NC × DAC = 0; u400 reads 529 tokens = 4.36 %) | the zero tail |

All six sub-scores are reported with their paired CI. Nothing is pooled across families.

## Rule (committed now)

**ADOPT** (per host) iff **all** of:
1. E1's lower bound > 0;
2. E2's DAC-zero-share upper bound < 0 (the mechanism moved in the stated direction);
3. no longitudinal or lateral family component separates adversely against the comparator;
4. where the C-shuffle arm was run: its E1 Δ is below **half** of the arm's Δ **and** its interval contains 0;
5. E1's lower bound exceeds the host's **measured seed floor** (§Replicates). If the floor is unmeasured at read time the verdict is
   **"separated, single seed, not yet a lever claim"**, never ADOPT.

**REFUTED** iff E1's upper bound < 0, **or** E2's DAC-zero-share lower bound > 0.

**NOT PROVEN** otherwise.

**Size label (reported with every ADOPT):** **LARGE** iff E1 Δ ≥ **+1.0** and DAC-zero share ≤ **2.39 %** (a ≥ 30 % cut, ≤ 290 tokens); else **SMALL**.
These are the scales the SPEC's arithmetic gives (+0.98 PDMS at a 30 % cut). They are labels, not bars: the verdict is the rule above.

## Perception proof (reads on the head itself; gates H2, not an endpoint)

Computed on navtest labels (build path A, `SPEC` §3.2) for the arm's own head.

| id | read | committed consequence |
|---|---|---|
| **P1** | held-out mask IoU in the bands [0, 20), [20, 40), [40, 72) m, split by frames that have ≥ 1 u400 DAC-zero vs none | IoU in [0, 40) < **0.80** ⇒ **K2** |
| **P2** | SDF MAE within ±3 m of the true boundary | reported |
| **P3** | **AUC of the minimum predicted SDF along each proposal's footprint corners** (0.1 s grid) for NAVSIM DAC = 0 vs 1, over the 64 proposals × the 1,058 C3 tokens (`raw/2026-10-11-c3-selection-headroom/tokens.json`); against the **current scorer's DAC AUC on the same set, computed, not the inherited 0.75–0.81** | AUC ≥ max(0.90, scorer AUC + 0.05) ⇒ H2 is evaluated; AUC < scorer AUC ⇒ **H2 is killed**, representation only continues |

⚠️ The C3 population is selected on u400's failures (529 zeroed + 529 random). P3 is a **diagnostic on a failure-enriched set**; it may not be quoted as a rate.

## Kill conditions (each stops the arm and names the next one)

| id | condition | then |
|---|---|---|
| **K1** | the label gate fails: alignment < 95 % on ≥ 64 real frames, or any mutation arm is not RED | no launch; fix the label store |
| **K2** | P1 IoU in [0, 40) m < 0.80 after the head's own fit | the road is not readable from these features: try the `scene` tap or a graft with an unfrozen trunk; H2 not built |
| **K3** | measured step time rises > **3 %** or peak memory > **1 GB** at micro-batch 4 | drop to 1 block / 4 m cells and re-gate |
| **K4** | at any full-navtest read, E1's upper bound < 0 | REFUTED; stop; §Next arm |
| **K5** | the C-shuffle arm gains ≥ half of the arm's Δ, **or** the zero-weight / detached exact controls are not bit-identical to the comparator | the gain is capacity or a leaked path, not map information: the head is **not credited**; investigate the path first |
| **K6** | `aux_bce` ≥ 0.9 × its step-0 value after 10 % of the arm's steps | stop; check labels and tap |
| **K7** | DAC-zero share falls but EP's upper bound < −1.0 and E1 Δ ≤ 0 | the head converted DAC failures into slow plans: stop; route to the selection/progress calibration (0 GPU) |

## Deliberate-regression controls

| control | form | expected |
|---|---|---|
| **label shuffled across samples** | an arm at S1 (cheap); optional at S2; **not run at S3** (no budget, stated). A derangement of the label keys, seed **20261011**, 0 fixed points, same multiset of labels | **no gain**: E1 Δ inside the noise and E2 not moving. A gain here means the lever is capacity or regularisation |
| **head detached** | an **exact test**, not an arm: with the head fed `ctx.detach()` and the feed off, the trunk after N steps is **bit-identical** to the head-less run from the same seed (the head draws from its own `Generator`) | exactly equal. Anything else is a leaked path |
| **zero weight** | exact test: `w_aux_drv = 0` ⇒ every gradient exactly 0.0, and the first N optimiser steps bit-identical to the comparator | exactly equal |
| **feed off ⇒ planner unchanged** | exact test (G-EVAL): trajectories and scores bit-identical with the head weights removed | exactly equal |
| **label misalignment** | mutation of the alignment gate: y-flip, or a one-frame offset | the gate must read RED |

`SPEC_AUX_DRIVABLE_HEAD.md` §7 gives each check's gate entry; `raw/2026-10-11-aux-drivable/head_proto.py` already runs the gradient controls (MEASURED GREEN) and
the detach mutation (MEASURED: trunk gradient exactly 0.0).

## Replicates and the seed floor

- **S1:** ≥ 3 seeds for the arm, for C-shuffle and for C-seed. The floor is the largest |Δ| between two C-seed runs; the published NAVSIM figure (std 0.56 over 3
  training seeds of TransFuser, `2406.15349`) is **context only**, a different model.
- **S2:** 2 seeds per arm; the floor is the control's seed pair.
- **S3 (a):** **no replicate is in budget.** The verdict is capped at "separated, single seed, not yet a lever claim", and a seed-1 replicate is requested only if
  the arm is ADOPT and Δ is within 2× the floor read at S1/S2.

## Four families (binding; per family, never pooled)

Instrument: the one that produced `sft4_B_u400_families.json`, on the identical tokens, each tested component with a paired CI.

| family | components | directional prediction (registered before the read) |
|---|---|---|
| **LONGITUDINAL** | speed MAE and bias; along-track MAE and bias; EP and TTC sub-scores; headway/time-gap where a lead exists (n and the reason where none) | EP is the **adverse risk** (a road-aware scorer may prefer shorter plans); TTC unchanged |
| **LATERAL** | heading MAE; cross-track MAE; curvature MAE; yaw-rate MAE; **DAC sub-score**; last-pose heading error reported separately (D5) | DAC up; cross-track MAE not worse |
| **TACTICAL** | lateral / longitudinal / 5-way decision accuracy and κ; command adherence of the selected proposal; **selection quality as random / actual / best-of-64 on the same tokens, the random control mandatory** | H2 raises (actual − random); H1 does not |
| **STRATEGIC** | **PAUSED** (PI 2026-09-19, `PI_DECISION_QUEUE.md` item 25). Reported as PAUSED with that ruling: not a pass, not omitted, not extended to the other three | – |

## Next arm, by outcome (Rule Zero; committed now)

| outcome | next lever (executed in the same turn that reads the verdict) |
|---|---|
| **ADOPT, LARGE** | (1) the seed replicate at the host; (2) the next rung of the ladder (`SPEC` §5); (3) the field-loss arm (`SPEC` §8) on the same labels; (4) the scorer ensemble (P7.2) |
| **ADOPT, SMALL** | the replicate first, then the field-loss arm; do not spend 7 days on a lever whose lower bound sits at the floor |
| **P3 passes, E2 does not move** | the scorer's *use* of the feature is the bottleneck: the field loss on proposals (`SPEC` §8), or the feed with gradient |
| **P1/P3 fail (K2)** | the frozen features do not carry the road at this precision: the `scene` tap, a graft with an unfrozen trunk, or a 1 m grid |
| **REFUTED (K4)** | bisect in this order: C-shuffle (capacity?), H1 vs H2, `w_aux` ÷ 4, then drop the head and run the field-loss arm alone |
| **NOT PROVEN** | the replicate (cheapest at S1), then a larger-n read is **not** the answer: more logs narrow an interval that already answers the wrong question |
| **K7 (EP trade)** | the selection / progress calibration on the same proposals, 0 GPU, dev box |

## Blanks to fill at landing

| blank | value |
|---|---|
| label store: path, frame count (expect 103,288), manifest sha256 | _ |
| navtest label store: path, frame count (expect 12,146), sha256 | _ |
| head config (cell, blocks, tap, feed) and `w_aux` after the balance, with the 8-micro-batch ratio it came from | _ |
| comparator checkpoints and their sha256 (u400: `eval/RESULT_SFT4.md`) | _ |
| seeds for every arm; the derangement seed (20261011) | _ |
| harness commit and the seam's flags | _ |
| gate PASS token id (commit, argv sha256, data-manifest sha256s) | _ |
