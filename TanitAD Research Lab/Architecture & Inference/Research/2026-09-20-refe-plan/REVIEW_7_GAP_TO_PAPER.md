<title>REVIEW 7 — Why REFe is ~11 PDMS below DriveZero, and how to close it</title>

# REVIEW 7: the gap to the DriveZero paper, an independent review and closure plan

**Reviewer:** independent review agent, 2026-10-04. I did not build REFe, and I have tried to be adversarial
toward our own documents, recent ones included.
**Method:** I worked from the paper (all 32 pages, banked PDF), our code, and raw artifacts: the official
`run_pdm_score` CSV, the per-token hooks, the 64-proposal table with stored logits, and the per-proposal NAVSIM
census. All analysis ran on CPU (peak RAM < 0.5 GB). Scripts and JSON: `raw/2026-10-04-gap-review/`.
**Evidence classes:** MEASURED (ours, artifact named) · PUBLISHED (paper, page/table) · INHERITED (our docs,
not re-verified) · ESTIMATED · HYPOTHESIS.
**Tier:** every PDMS here is NAVSIM v1.1 PDMS on navtest, i.e. a non-reactive pseudo-simulation of a 4 s open-loop
plan. **Intervals:** log-cluster bootstrap over 136 logs (10,000 resamples, seed 20260927). They answer "another
draw of LOGS" only: one checkpoint, deterministic inference.
⚠️ **Everything computed on navtest in this review is EXPLORATORY.** navtest has been read many times, and none of
it is a confirmation.

---

## 0 · Headline answer

1. **The comparison is apples-to-apples to within ~0.3 PDMS, but the brief uses the wrong anchor.**
   * Our harness scores the logged **human** at **94.55 [93.80, 95.24]** on all 12,146 navtest tokens. The
     published value is 94.8, and the CI contains it. PDM-Closed's NC/DAC/TTC/C reproduce NAVSIM's published
     values to 0.1. (MEASURED, `gap_decompose.json` C_census)
   * DriveZero's DINOv3 row closest to our setup (ViT-L, 4 cameras, navtrain only, rank-32 Q/V LoRA) is
     **Table A13 DINOv3 ViT-L = 94.55**, not Table 7's ViT-S 93.88 (PUBLISHED, p. 31).
   * **The gap is 11.31 PDMS** (83.24 vs 94.55). The registry's own decision row D-REFE-ARM-DECISION-1 already
     names 94.55 as the anchor.

2. **The gap splits roughly in half** (ESTIMATED, `gap_decompose.json` B):
   * **≈5.5–6.0 points are multiplicative zeros.** 8.09 % of our picks score 0, against ≈1.9 % implied by the
     paper's NC/DAC. Drivable area dominates (≈5.0 points); collisions add ≈1.0.
   * **≈5.0–5.5 points are ego progress on the surviving tokens.** EP is 82.2 there, against ≈94.5 implied for
     the paper. TTC costs ≈0.2 and comfort ≈0.04.

3. **The proposals already contain a better-than-paper answer. Selection is the dominant measured bottleneck.**
   (MEASURED, `gap_selection.json`, `gap_heads.json`)
   * The best of our 64 proposals averages **97.07**, the mean of 64 is 72.19, and our pick is 83.24.
   * **With perfect knowledge of only the NC, DAC and TTC heads, the shipped scorer's selection would read 95.16**,
     above the paper's 94.55.
   * Single-head oracles: DAC **+5.23**, TTC **+3.98**, NC **+1.42**, comfort +0.07.

4. **Low EP is a symptom, not a separate cause.** (MEASURED)
   * The pick sits at **median rank 49 of 64 by path length**, and **47.5 %** of picks come from the shortest
     length quintile. Yet the fan's safe proposals average human-level progress (EP 87.5 vs the human's 87.0).
   * **Giving the selector NAVSIM's true EP alone makes it worse, −1.18** (leak-free oracle). Faster plans then fail
     the safety and TTC terms, which the scorer cannot judge.
   * The predicted TTC head is badly calibrated (mean 0.56 vs true 0.84) and over-penalises length (within-set
     r −0.62 vs true −0.42).
   * ⇒ **EP recovers only when safety and TTC discrimination improve**, or when the fan gets cleaner.

5. **The scorer is not trained the way the paper trains it. This is the root cause with the strongest evidence,
   and it contradicts the team's current reading of "paper-pure".**

   | | the paper (PUBLISHED, p. 6 Fig. 2; p. 8) | REFe final |
   |---|---|---|
   | what the scorer is trained against | *"trained against **PDM targets**"*; the six components *"are supervised by their corresponding PDM targets [17]"*. [17] is the PDM paper (Dauner et al., CoRL 2023), the scorer NAVSIM is built on. | **teacher-reward calculators** for NC, TTC and DDC; **EP = min(1, advance/teacher advance)** (`refe/onpolicy_label.py:125-129`); NAVSIM's own label only for DAC and comfort, and only from step 3,708 |
   | which proposals | the student's own, every sample | on-policy sets from epoch 12 onward, covering **0.8 → 37 %** of samples (`train.log`; `summary.json scorer_cov 0.375`), and labels **≈3,400 optimiser steps stale** by the end (`metrics.jsonl lag_steps`) |
   | labelled plan | the executed plan | the **raw** plan, with the corrupted last heading (repaired-plan labels V3r measured **+2.00 [+0.53, +3.57]** and were never deployed: D-REFE-V4R-1) |
   | gradient into the encoder | only the *trajectories* are detached | trajectories **and** visual context detached (`detach_scorer_context=True`, `model.py:152`), so the PDM loss never shapes the LoRA features |

   ⇒ **SFT-1/2/3 all train on teacher-simulator labels and call that "paper-pure". By the paper's own text it is
   not.** I measured a cross-fitted re-ranker that gets NAVSIM labels but no images. It recovers only
   **+1.12 [+0.50, +1.76]**, lifting within-set DAC AUC from 0.75 to 0.81 (`gap_rerank_crossfit.json`). So the
   missing information is visual. The two levers are therefore faithful labels **and** letting the score loss
   adapt the encoder.

6. **A dirty fan explains most of the zeros.** (MEASURED, `gap_fan_quality.json`, `gap_dirty_fans.json`)
   * On **15.2 %** of tokens fewer than half the proposals are NAVSIM-safe. There the pick scores 54.2 with 38 %
     zeros, which accounts for **72 % of all zero picks**.
   * On clean fans (≥ 95 % safe, 44 % of tokens) the pick scores 90.9 with 0.4 % zeros.
   * Dirty fans concentrate in turns (22.6 % dirty at far-goal bearing ≥ 60°, 10.3 % straight) and off-route goals
     (40.6 %).
   * The teacher's own targets leave NAVSIM's drivable area on **4.97 % [1.30, 10.31]** of held-out samples
     (5.4 % even on un-augmented rank 0). The human does 0.00 %, and the paper's DriveRL scores DAC 99.9 on navtest.
     ⇒ the fan inherits teacher failures (60 % of proposals fail DAC on those samples, against 24 % overall: INHERITED,
     PREREG_SFT2).

7. **Smaller, real contributors.**
   * **Train/test goal construction.** Every training goal used DriveRL's *corrected* route
     (`driverl/nuplan/feature_builder.py:407-413`, "Public DriveRL inference always consumes corrected route IDs").
     The student's test-time goal used the raw scenario route until A9. A9 measured +1.05.
   * **Training schedule.** The growing bank ran the first 4 epochs (the high-LR part of the cosine) on 7.5K → 72K
     scenes with no twins and no scorer targets, so only ≈21 epochs saw the full data. Both curves were still
     improving at LR = 0.
   * **Heading.** The last-pose heading defect is repaired at test time, but the raw heading still enters the scorer
     query and the labels.

8. **Ruled out as material causes:**
   * metric version (v1.1 = their navtest), the split, 4 cameras, 960×512, 64 proposals, 16 compressing registers,
     the 4-layer decoder, lr/wd/schedule/clip, effective batch 256;
   * total sample-passes (2.58 M ≈ 25 × 103K, about the paper's budget), bf16 (gradient cosine 0.99978, INHERITED);
   * 10 Hz teacher stepping (ADE 0.23 m, INHERITED), the 20 → 8 pose interpolation (correct, `eval/refe_navtest_seam.py:58-60`).

9. **What to do** (§4). There is no free lunch at test time: every CPU-only selection rule I tested moved PDMS by
   +0.1 to +0.2 cross-fitted. The plan, in order:
   * **(P1)** relabel the 190,696 banked on-policy sets and the 3,137 held-out sets with NAVSIM's own PDM scorer, on
     the executed plans, and fine-tune the scorer: the paper's actual supervision.
   * **(P2)** a joint fine-tune with the scorer context un-detached (the paper's detach), so the PDM loss reaches the LoRA.
   * **(P3)** test-time route parity with DriveRL's own `correct_route_roadblock_ids`.
   * **(P4)** measure our teacher on navtest against the paper's DriveRL row (95.8), the one experiment that says
     whether the teacher caps the student.
   * **(P5)** clean or re-query the ~5 % failing teacher targets.
   * **(P6)** a next full run done the paper's way, plus an auxiliary map-supervised drivable-area head (labels only,
     vision-only inference) aimed **above 94.55**.

**Decisions the PI needs to make are listed at the end of §4.**

---

## 1 · Is the comparison apples-to-apples?

### 1.1 What the paper's numbers are (PUBLISHED, read from the banked PDF)

| row | where | backbone | data | PDMS | NC | DAC | EP | TTC | C |
|---|---|---|---|---|---|---|---|---|---|
| DINOv3 ViT-S | Table 7, p. 13 (backbone block) | DINOv3 ViT-S | navtrain, no SimScale | **93.88** | 98.93 | 99.01 | 91.31 | 95.77 | 99.97 |
| **DINOv3 ViT-L** | **Table A13, p. 31** | **DINOv3 ViT-L, rank-32 Q/V LoRA** | **navtrain, no SimScale** | **94.55** | 98.80 | 99.25 | 92.72 | 95.74 | 99.94 |
| DriveZero (DriveVFM ViT-L) | Table 4, p. 11 (A13: 94.83) | DriveVFM ViT-L | navtrain | 94.8 | 99.0 | 99.2 | 93.1 | 96.0 | 100 |
| Supervision block: human / DriveRL / DriveRL + goal aug | Table 7 | **DriveVFM ViT-S**, not DINOv3 | navtrain | 93.92 / 93.61 / 94.41 | | | | | |
| DriveRL teacher, GT symbolic inputs | Table 4 | (privileged) | navtest, 4 s rollouts | 95.8 | 99.8 | 99.9 | 91.5 | 99.1 | 99.0 |
| Human | Table 4 | | | 94.8 | 100 | 100 | 87.5 | 100 | 99.9 |

Corrections to the brief:
* (a) The supervision-block rows (93.92 / 93.61 / 94.41) are **DriveVFM ViT-S**: "DriveRL + goal aug" = 94.41 equals
  the DriveVFM ViT-S backbone row. They are not DINOv3 rows.
* (b) 93.88 is ViT-S. Our configuration matches **94.55** (ViT-L) in every stated variable except the backbone
  weights (frozen DINOv3 in both).
* (c) DriveZero-Scale (95.3) uses SimScale and is out of scope.

What the 93.88/94.55 rows include (PUBLISHED, §3.2 and Table A12):
* inputs: four cameras CAM_F0/B0/L0/R0 at 960×512, current frame only (no history), ego kinematics and a
  navigation command;
* planner: 64 proposals, 256-d, a 4-layer proposal decoder, 16 registers per camera;
* training: frozen backbone plus rank-32 Q/V LoRA; 25 epochs, batch 256, AdamW lr 2e-4 cosine to 0 with no warm-up,
  wd 0.01, clip 1.0, FP32; teacher-rollout WTA with a separate PDM-target scorer; selection is the argmax of the
  aggregated predicted score;
* **no test-time search, ensembling or TTA for the student** (TTS exists only for DriveRL on nuPlan);
* **no extra data or pseudo-labels** beyond the teacher rollouts and goal augmentation. The teacher is the
  single-ego DriveRL at 2,400 updates. The released `checkpoint_2400.pt` matches the update count and the 5.7 M
  parameters (INHERITED, RELEASED_ARTIFACT_TEARDOWN R1).

### 1.2 The metric (MEASURED, NAVSIM v1.1 source + our CSV)

* **PDMS formula.** Our CSV reproduces `NC × DAC × (5 EP + 5 TTC + 2 C)/12` exactly (max |Δ| 0.0 over 12,146 rows).
  It is off by up to 1.0 if DDC is multiplied in. DDC has weight 0 (`pdm_scorer.py:42`), as in the paper, whose
  Table A13 lists DDC (≈95.9) separately while PDMS stays 94.5–94.8. **DDC is not in the gap; ours is higher, 96.93.**
* **EP normalisation.** NAVSIM normalises the agent's centerline progress by `max(agent, PDM-Closed)` after
  multiplying by NC·DAC, and forces EP to 1 when the max is ≤ 5 m (`pdm_scorer.py:163-173`). This is identical in
  their evaluation and ours.
* **Harness check, the decisive one.** Our harness scores the logged human 94.55 [93.80, 95.24] (EP 86.96) against
  the published 94.8 (EP 87.5), and PDM-Closed NC 94.66 / DAC 99.83 / TTC 86.97 / C 99.91 against the published
  94.6 / 99.8 / 86.9 / 99.9.
  ⇒ **The measurement difference is ≤ ~0.25 PDMS (≈0.5 EP), inside the CI.** If our harness is that much stricter,
  REFe would read ≈83.5 on theirs. That is not a model difference.
* **Split:** navtest, 12,146 tokens, 136 logs (MEASURED, manifest `C3_missing 0`). The frame-control failure on 36
  tokens changes nothing (83.24 without them).

### 1.3 Configuration differences that are real (MEASURED from `config.json` / code, PUBLISHED paper side)

| item | paper | REFe final | effect on the gap |
|---|---|---|---|
| backbone | DINOv3 ViT-L (A13 row) | DINOv3 ViT-L/16, frozen, rank-32 Q/V LoRA | none (matched) |
| student nav input | **navigation command** derived from the teacher's goal point [56 = OpenScene]; at test, NAVSIM's `driving_command` (inferred from *"follows the standard sensor-input protocol of each benchmark"*, p. 8) | **two goal points** (Fourier-encoded) from the route polyline | changes robustness to route errors (§3.5) |
| scorer supervision | **PDM targets**, every sample, own proposals, trajectories detached | teacher calculators (NC/EP/TTC/DDC), NAVSIM DAC/C late, 0–37 % coverage, raw plans, context detached | **dominant** (§3.3) |
| data schedule | 25 epochs over the full set | growing bank: epochs 0–3 on 7.5K–72K scenes; twins from epoch 4; scorer targets from epoch 5 (fixed candidates) and 12 (on-policy) | moderate, ESTIMATED (§3.6) |
| goal augmentation | "alternative route intents" (method unstated) | lane rank / goal horizon, tau 0.3 m, 71 % of scenes get a twin | small (paper's whole aug effect is +0.80) |
| precision | FP32 | bf16 autocast | negligible (INHERITED) |
| yaw in WTA | L1 over points (paper says nothing about wrapping) | position L1 + 0.1 × **wrapped** yaw on the winner (`model.py:782-792`) | source of the last-pose heading defect (§3.7) |

---

## 2 · Decomposing the gap by sub-score

### 2.1 Our sub-scores (MEASURED, CSV)

| | PDMS | NC | DAC | EP | TTC | C |
|---|---|---|---|---|---|---|
| REFe, all 12,146 | **83.24** | 97.64 | 93.72 | 75.53 | 92.99 | 99.32 |
| REFe, the 11,163 tokens with NC·DAC > 0 | 90.57 | | | **82.19** | 95.37 | 99.72 |
| paper DINOv3 ViT-L | **94.55** | 98.80 | 99.25 | 92.72 | 95.74 | 99.94 |
| paper, implied on surviving tokens (ESTIMATED: EP = 0 on M = 0 tokens by NAVSIM construction) | ≈96.4 | | | **≈94.5** | | |

Zero picks: **983** (8.09 %): DAC only 721, NC only 220, both 42. NC = 0.5 on 49 tokens.

### 2.2 Points per sub-score (ESTIMATED; `gap_decompose.json` B; method: PDMS = mean(NC·DAC) × the NC·DAC-weighted mean of W, split by log-ratio, the weighted half split linearly)

| source | points of the 11.31 gap (vs 94.55) | basis |
|---|---|---|
| **multiplicative zeros** | **5.8** (log-ratio split); **≈5.5** by direct count | 983 zero picks × 90.57 (surviving mean) = 7.33 points, minus ≈1.8 for the paper's own ≈1.85 % zeros |
| … of which DAC | **≈5.0** | DAC fail 6.28 % vs 0.75 % |
| … of which NC | **≈1.0** | NC deficit 2.36 vs 1.20 |
| **EP on surviving tokens** | **≈5.3** | 82.19 vs ≈94.5 × 5/12 |
| TTC on surviving tokens | ≈0.2 | 95.37 vs ≈95.9 |
| comfort | ≈0.04 | |
| **sum (cross terms aside)** | ≈11 | |

### 2.3 What drives the low EP (MEASURED, `gap_decompose.json` D, `gap_selection.json`, `gap_heads.json`)

* **94.7 % of the EP loss is longitudinal.** The executed path is shorter than PDM-Closed's progress, and the path
  length ≈ the projected progress. Only 0.8 % is geometric (off-centreline or wrong way), and 1 token is a stop.
* **The fan is not slow; the pick is.**
  * Safe proposals' mean EP is 87.5 (the human: 87.0), and the fastest safe proposal per token averages 97.4.
  * The pick's EP is 82.2. Pick path / fan mean path = 0.89; the pick is shorter than the fan median on **75 %**
    of tokens, at median length rank **49/64**.
  * Picks by length quintile, shortest first: **47.5 % / 19.9 / 16.0 / 9.6 / 7.0 %**.
* **The EP label is teacher-relative and saturates at teacher speed** (`onpolicy_label.py:125-129`). The predicted
  EP head correlates **0.90** with path length within a set but only **0.29** with NAVSIM's EP.
* **But the EP head is not where the points are.**
  * Replacing p_EP by NAVSIM's true EP on safe proposals reads **−1.18 [−2.07, −0.39]**.
  * On clean fans only it reads +0.16 [+0.03, +0.29].
  * The 2-fold cross-fitted fan-conditional progress tilt reads +0.18 [+0.04, +0.30].
  * The predicted multiplicative and TTC heads penalise length more than reality: within-set r with length is
    pred TTC −0.62 vs true −0.42, pred NC·DAC −0.25 vs true safe −0.30. They are also under-confident: mean
    predicted TTC 0.56 vs true 0.84, DAC 0.71 vs 0.87, C 0.60 vs 0.99.
  * Faster safe plans exist, but the selector cannot tell which fast plans are safe, so preferring them costs more
    than it gains.

⇒ **EP is set by the scorer's safety and TTC discrimination and by the fan's cleanliness.** It is not an independent
lever. This overturns the intuitive reading of "EP 75.5 vs 91.3 ⇒ train for speed".

### 2.4 Proposal quality vs selection quality (MEASURED, `gap_selection.json`, `gap_fan_quality.json`)

| selector (same 64 proposals) | PDMS | Δ vs pick [95 % CI] |
|---|---|---|
| mean of 64 (random) | 72.19 | |
| **shipped pick** (navsim_v1 aggregate; control reproduces 12,146/12,146 picks) | **83.24** | 0 |
| oracle NC only | 84.66 | +1.42 [+1.09, +1.78] |
| oracle TTC only | 87.22 | +3.98 [+3.39, +4.59] |
| oracle DAC only | 88.47 | +5.23 [+4.23, +6.33] |
| oracle NC + DAC | 89.68 | +6.44 [+5.47, +7.46] |
| oracle EP, leak-free (true EP only where NC·DAC = 1) | 82.05 | −1.18 [−2.07, −0.39] |
| oracle NC + DAC + TTC | **95.16** | **+11.92 [+11.02, +12.87]** |
| oracle all (best of 64) | 97.07 | +13.83 [+12.81, +14.93] |

* Selection skill (pick − random)/(best − random) = **0.44**.
* **793 of the 983 zero picks had a safe proposal in their fan.** 191 tokens have no safe proposal at all; those
  failures belong to the proposals.
* ⚠️ The brief's "98.33 vs 84.46" is the 8,760-token *unchanged-by-route-fix* subset (`lane_result.json`). On all
  12,146 tokens it is **97.07 vs 83.24**.
* Within-set AUC of the shipped heads (mixed-label sets): NC 0.855, DAC 0.750, TTC 0.825, DDC 0.616. The brief's
  "NC 0.88" does not reproduce under this definition (n 4,645 sets).

---

## 3 · Root causes, each with its evidence

Ranked by how many points the evidence can put on them.

### 3.1 ⭐ Selection: the scorer cannot judge safety or TTC. MEASURED. ≈6–12 points of headroom

* **Evidence:** the table in §2.4. Perfect NC/DAC/TTC alone reaches 95.16, above the paper.
* **Why this matters more than the fan:** the fan's best is 97.07, and **14 of 15 zero picks on W3's 200 tokens
  had a hypothesis scoring ≥ 80** (INHERITED, `RESULT_NAVTEST_FINAL.md`; consistent with the 793/983 measured here).
* **Challenge to our own docs:** RESULT_NAVTEST_FINAL calls the zeros "selection failures, not proposal failures".
  That is **mostly right but incomplete.** 72 % of zero picks sit on dirty fans (§3.4), where a better scorer has a
  much smaller target to hit. Better selection and a cleaner fan multiply.

### 3.2 ⭐ The scorer's labels are not the paper's. PUBLISHED vs MEASURED. Explains why 3.1 exists

| component | paper | ours (final run) | evidence |
|---|---|---|---|
| NC | PDM at-fault collision | DriveRL `NuPlanCollision.info` | `train.py:339` |
| DAC | PDM | NAVSIM's own, **from step 3,708** (fixed candidates with the teacher's curb label before) | `train.py:340`, registry §14.1 (6) |
| EP | PDM progress vs PDM-Closed | **advance / teacher advance, clipped at 1** | `onpolicy_label.py:125-129`, `train.py:342` |
| TTC | PDM binary | DriveRL `NuPlanTTC.ttc_reward`, continuous | `train.py:343`; calibration 0.56 vs 0.84 (MEASURED) |
| C | PDM | NAVSIM's own (label v3, from epoch 14) | registry §14.1 (7) |
| DDC | PDM | DriveRL wrong-way category: recall 31 % / precision 46 % of NAVSIM's verdict | `teacher_vs_navsim_ddc_heldout.json` |

* **The paper's words:** Fig. 2: *"the proposal-scoring branch is trained against PDM targets"*. p. 8: *"These
  components are supervised by their corresponding PDM targets [17]"*, and the same augmented route *"is used to
  evaluate student proposals"*. EP has no counterpart in DriveRL's released reward set (PAPER_CONFORMANCE_REVIEW D4a,
  verified by grep). The only reading under which all six components exist is NAVSIM/PDM's own scorer.
* ⛔ **The 2026-10-04 "paper-pure" decision (PREREG_SFT2 status, PREREG_SFT3) rests on a misreading.** It treats
  teacher-simulator labels as the paper's. The paper's text supports the opposite.

### 3.3 ⭐ The scorer had far less, staler and less faithful supervision than the paper's. MEASURED (train log)

* **Timeline from the run's own log:**
  * epochs 0–4: no scorer loss;
  * epochs 5–11: fixed candidates attached to the nearest proposal, a mechanism not in the paper, during which
    selection skill *fell* 0.46 → 0.01 (D-REFE-SEL-1);
  * epoch 12 onward: on-policy sets, starting at **1,618** and reaching **127,385** at epoch 25.
* **Coverage:** per-step coverage 0.28–0.37 in the last 13 epochs (`gap_training_curve.json`), ≈5× less labelled
  supervision than "every sample, every epoch" (ESTIMATED, ≈0.5 M vs ≈2.5 M labelled sample-visits).
* **Staleness:** `lag_steps` rose to **3,451** at the last step.
* **Labels on the wrong plan:** labels were computed on the **raw** plan, while execution uses the heading-repaired
  plan. Relabelling on executed plans (V3r) measured **+2.00 [+0.53, +3.57]** on 923 fresh tokens in a dev-box
  scorer fine-tune and was never deployed (D-REFE-V4R-1, INHERITED). The comfort head's calibration (0.60 vs 0.99
  true) is the fingerprint of that mismatch.
* **Context detached:** `detach_scorer_context=True` (config.json `args`). The PDM loss reaches only the scorer
  modules. The paper detaches only the trajectories.
* **Evidence that the visual input is binding:**
  * the cross-fitted NAVSIM-label re-ranker on logits + geometry + v0 recovers only +1.12 (DAC AUC 0.75 → 0.81);
  * SFT-1 (scorer-only, existing labels) moved +0.02/−0.10/−0.22 at updates 400/800/1,200 (INHERITED);
  * SFT-3's own fall-back says the same ("the binding constraint is then the scorer's INPUT").

### 3.4 ⭐ The fan is dirty in hard scenes, partly inherited from the teacher. MEASURED. ≈5.6 points vs the clean-fan level

* **Dirty fans** (< 50 % safe) cover 15.2 % of tokens: pick 54.2, zero rate 38 %, best of 64 87.6.
* **Clean fans** (≥ 95 %) cover 43.8 %: pick 90.9, zero rate 0.4 %.
* Correlation between a token's fan safe share and its pick PDMS: r = 0.51.
* **Dirty rate by scene type:** off-route 40.6 %, far-goal bearing ≥ 60° 22.6 %, < 10° 10.3 %, v0 < 1 m/s 5.8 %.
* **Teacher contribution:**
  * our teacher fails NAVSIM DAC on 4.97 % [1.30, 10.31] of held-out samples (rank 0: 5.40 %; 1–5 m/s: 7.64 %);
  * the paper's DriveRL scores DAC **99.9** on navtest (PUBLISHED, Table 4).
* **Why this is not yet attributable:**
  * we have **never** measured our teacher protocol's PDMS on navtest;
  * the released checkpoint's evaluation status is `artifact_bundled_pending_public_eval`
    (`release/driverl_checkpoints.yaml`);
  * DZ-11, the headline reproduction against the paper's nuPlan numbers, stalled at the download on 2026-09-20.
* **Rollout fidelity is not the issue:** we roll the released planner and controller exactly as their runner does,
  5 Hz policy in a 10 Hz sim, with their accel/steering-rate overrides (`refe/build_teacher_rollouts.py:126-197`).
  ⇒ **HYPOTHESIS:** either the released checkpoint is weaker on navtrain-style frames than the paper's teacher, or
  the paper's navtest teacher row was produced in their own simulator rather than through the nuPlan adapter.
  P4 decides this.

### 3.5 Train/test inconsistency in the goal input. MEASURED effect +1.05 (A9). PUBLISHED/code mechanism

* **Training:** every goal is the one the teacher received (`build_teacher_rollouts.py:272-280`), built by DriveRL's
  feature builder, which **always** corrects the route (`feature_builder.py:407-413`).
* **Test:** our planner built the goal on the raw scenario route (`planner.py:625-639`, `goal_fix` off until A9).
* **Consequence:** on the 3.9 % of tokens where the raw route misses the ego, the goal lands 349–480 m ahead, an
  input the student never saw. Those tokens score **55.45** vs 84.37 on-route.
* **Reframing for the PI:** A9's correction is **restoring train/test parity, not adding a map input.** The exact
  parity fix is DriveRL's own `correct_route_roadblock_ids`, the function that produced the training goals, rather
  than PDM's.
* **Secondary point (HYPOTHESIS):** the paper's student takes a *discrete command* (NAVSIM `driving_command`), which
  is immune to route errors by construction. Ours takes continuous goal points, which are richer but brittle.

### 3.6 Training schedule. MEASURED schedule, ESTIMATED effect ≈0.3–1.0

* `--grow` started training while data prep ran (D-REFE-GROW-1): epoch 0 had 7,536 scenes (13.7 passes over them
  in 403 steps at peak LR), epoch 3 had 72,289, and twins appeared from epoch 4.
* Only ≈21 of the nominal 25 epochs saw the full 189,888-tuple bank.
* At LR = 0 the trajectory L1 was still falling (0.1448 → 0.1410 over the last 3 epochs) and train-side scorer
  skill was still rising (0.631 → 0.637). That is under-training, not convergence.

### 3.7 Heading head. MEASURED defect, repaired at test time

* The yaw loss wraps the difference (`model.py:788`), so the raw last-step heading is free modulo 2π. Its median
  error is 1.1 rad, with > π on ~60 % of proposals (D-REFE-LASTYAW-1).
* The repair fixes execution (+16.20, A7), but the **scorer's query MLP still reads the raw trajectory**
  (`model.py:746`), and the on-policy labels were computed on raw plans (§3.3).
* The paper's plain L1 over points has no wrap. Ego-frame yaw over 4 s stays inside (−π, π] except for U-turns.

### 3.8 Explanations in our docs that the evidence does not support

* "EP is low because the policy is slow / the teacher is slow": the fan is human-speed (§2.3), and the teacher-slow
  claim was already retracted (FINDING_TEACHER_STALL §0).
* "A better within-scene ranking objective (ListNet / expected reward) will fix selection" (SFT-1 B, SFT-3): the
  objective is not the binding constraint while labels are unfaithful and the encoder is detached. SFT-1 B read
  −0.73/−0.92/−0.58 (INHERITED).
* "Test-time rule changes": my cross-fitted rules read +0.03 to +0.18. The confidence-gated STOP read +0.07
  (INHERITED, `RESULT_NAVTEST_FINAL.md`).

---

## 4 · Proposals to close the gap and to beat the paper

**Ranking:** (expected gain × confidence) / cost. All gains are ESTIMATED, with their basis stated. "A40-h" means
one A40. Constraints respected throughout: no HD-map or lane geometry as a model input; nav goal and command
allowed; privileged signals only as labels or teacher; inference is vision + ego + nav.

### P3 · Test-time route parity (zero training)

* **Mechanism:** build the student's goal with DriveRL's `correct_route_roadblock_ids`, exactly as every training
  goal was built.
* **Evidence:** §3.5; A9 `pdm_route` +4.17 [+1.98, +7.21] on 3,045 confirmation tokens, +1.05 on the full mean.
* **Gain:** **+1.0 to +1.3.** A9 is the lower bound, since the exact function should track the training goals at
  least as well.
* **Cost:** CPU plus one navtest inference pass (dev box, as A9). **Risk:** low.
* **Test:**
  * first, CPU on navtrain frames: our planner's goal with DriveRL correction must equal the teacher's recorded
    `goal` within 0.1 m on ≥ 99 % of rank-0 rows (parity gate);
  * then navtest: arm `driverl_route` vs OFF on A9's confirmation tokens, paired log-cluster bootstrap;
  * ADOPT iff lower bound > 0 and no family component separates adversely.
* **Needs:** the PI's admissibility ruling (pending since A9).

### P1 · Relabel with NAVSIM's PDM scorer on executed plans, then a scorer fine-tune (the paper's supervision)

* **Mechanism:** score every banked on-policy proposal with NAVSIM v1.1's `PDMScorer`, on the heading-repaired
  executed plan, against the sample's own route (augmented route for twins). That covers the 190,696 training sets
  (`onpolicy_sets_logs_full.tar.zst`, D:) and the 3,137 held-out sets. All six components then match the benchmark
  definition: at-fault NC, EP vs PDM-Closed, binary TTC, DDC. Fine-tune the scorer modules on them.
* **Evidence:**
  * label mismatch, §3.2;
  * V3r (executed-plan labels only) **+2.00** on 923 fresh tokens;
  * partial oracles DAC +5.2 / TTC +4.0;
  * TTC calibration 0.56 vs 0.84;
  * DDC label recall 31 %;
  * the team already has NAVSIM-faithful DAC and lane checks with 100 % agreement (`navsim_dac.py`, `navsim_lane.py`).
* **Gain:** **+2 to +4** scorer-only. Basis: V3r +2.0 from fixing one of the label defects, plus the TTC and EP
  label fixes. Capped well below the +11.9 oracle by the detached visual input (+1.1 diagnostic).
* **Cost:**
  * CPU: building navtrain metric caches. NAVSIM's metric caching needs navtrain `navsim_logs` + maps, not sensors.
    Then scoring runs at ≤ ≈1.6 s per 66-trajectory set per worker. MEASURED from the lane census: 0.36–0.43 s/token
    aggregate over 4 workers, and that census also computed lane geometry, so this is an upper bound. ≈190K sets ≈
    85 worker-hours, ≈5–6 h on 16 cores.
  * GPU: ≈10–20 A40-h for the fine-tune, since the backbone forward is needed for the context.
* **Risks:**
  * lane-rank twins need a metric cache on the augmented route; label goal-horizon twins and rank 0 first;
  * downloading the navtrain logs needs the PI.
* **Test:**
  * reuse PREREG_SFT1's two-stage design with the labels swapped, and add an arm that only relabels (BCE);
  * held-out: the 3,137 sets, now with NAVSIM labels; PASS iff the lower bound of pick(arm) − pick(base) > 0;
  * navtest: Amendment 5's 923 tokens; ADOPT iff the lower bound > 0 and no adverse family.

### P2 · Joint fine-tune with the scorer context un-detached (the paper's detach), on P1's labels

* **Mechanism:** `detach_scorer_context=False`, so the PDM loss trains LoRA, `scene_proj` and the registers
  alongside the WTA loss. The trajectories stay detached, as in the paper. The DAC failure needs road-boundary
  perception that the trajectory loss alone does not force into the features.
* **Evidence:**
  * paper p. 8 detaches only the candidates;
  * DAC AUC 0.75 and only 0.81 with faithful labels but no visual change (§3.3);
  * 72 % of zeros are DAC.
* **Gain:** **+2 to +5 on top of P1** (overlapping). Basis: the remaining oracle headroom of NC+DAC+TTC (+11.9)
  after P1. Confidence low-medium, since nothing in our data isolates it yet.
* **Cost:** ≈2–4 A40-days (3–6 epochs at ≈0.22 s/sample). **Risk:** proposals change, and the fan could degrade.
* **Test:**
  * pre-register best-of-64 and mean-of-64 non-inferiority (lower bound > −0.5) as a co-primary;
  * the deliberate-regression arm, detach kept with everything else identical, must read the P1 number;
  * the launch gate must confirm a non-zero LoRA gradient from the score loss and exactly 0 when detached.

### P4 · Measure our teacher on navtest: does the teacher cap the student? (measurement)

* **Mechanism:** roll our exact teacher protocol on navtest tokens and score the 4 s trajectories with the
  unchanged harness, to compare with the paper's DriveRL row (95.8 / DAC 99.9 / EP 91.5).
* **Evidence for need:** held-out teacher DAC fails 4.97 %; DZ-11 never ran; the release is "pending public eval".
* **Gain:** 0 directly. It **decides** whether P5 and a teacher fix are worth GPU-days.
* **Cost:** CPU hours (the teacher ran on CPU on the pod). It needs teacher `ScenarioData` for navtest logs, either
  from navtest nuPlan DBs (download) or by building them from the OpenScene navtest pickles already used for the
  metric cache (REFE_PLAN §Stage 2 says every input exists in NAVSIM's `Scene`).
* **Test, both outcomes registered:**
  * teacher PDMS ≥ 94.5 and DAC ≥ 99.5 ⇒ the teacher is not the cap, and the fan defects are student-side;
  * teacher DAC < 98 ⇒ the student inherits teacher failures, so prioritise P5 and the teacher protocol, and report
    the teacher's own sub-scores.

### P5 · Clean the trajectory targets, then a trajectory-head fine-tune

* **Mechanism:** score all 189,888 teacher targets with `navsim_dac.py` (and NC once P1's metric caches exist).
  Re-query each failing target with the teacher's own **test-time action search** (released `tts.py`: value-guided,
  N = 16–64, the paper's +0.56 nuPlan lever), or drop it. Then fine-tune the proposal head (WTA) with LoRA.
* **Evidence:** the fan fails DAC 60 % on teacher-fail samples vs 24 % overall (INHERITED); dirty fans cost ≈5.6
  vs the clean level (§3.4).
* **Gain:** **+0.5 to +2.** It acts on the 191 proposal-bound tokens and the dirty-fan tokens; the bound is the
  teacher failure rate.
* **Cost:** CPU labelling (hours); teacher TTS re-query on ≈10K samples (CPU/GPU-light); 1–2 A40-days fine-tune.
* **Risk:** using the critic for targets departs from the paper. Dropping is the paper-neutral arm.
* **Test:** held-out fan safe share and best-of-64 (co-primary, lower bound > 0), then navtest 923-token PDMS.

### P6 · A next full run done the paper's way, aimed ≥ 94.55

* **Recipe:**
  * the full bank (targets + twins) precomputed before step 0; it is already banked
    (`Sayood/tanitad-refe-navtrain-banks`);
  * **online NAVSIM PDM labels on every sample's own proposals from step 0**, from a CPU labeller pool. At
    ≤ 1.6 s per 64-set per worker, ≈8 workers keep up with 256 samples per 57 s step, and 16 leave headroom
    (ESTIMATED);
  * context un-detached; executed-plan labels; the plain (unwrapped) yaw L1 or positions-only plus a derived
    heading; the DriveRL route correction.
* **Gain:** **91–95** (ESTIMATED: 83.24 + P3 + P1 + P2 + P5 at their central values, with overlap discounted).
* **Cost:** ≈7 A40-days, as before, plus a ~16-core labeller.
* **Test:** register 94.55 as the bar on the full navtest, with the four families and the 923-token confirmation first.

### P7 · Beyond the paper (> 94.55, toward > 95)

1. **Auxiliary map-supervised drivable-area and agent-occupancy head (labels only).** A small BEV head on the
   register or visual tokens, supervised by nuPlan map polygons and agent boxes at training time. The scorer reads
   its pooled features along each proposal's footprint. Inference stays vision-only: the map is a label, as the PI
   ruled.
   * It targets the dominant failure (DAC 721/983 zeros) directly.
   * TransFuser-family NAVSIM models use exactly such a BEV-semantic auxiliary (banked: Hydra-MDP 2406.06978,
     DiffusionDrive 2411.15139; I did not re-read them for this review, so the claim stays PUBLISHED-SECONDARY).
   * **Gain:** if DAC fail drops from 6.3 % to ≈1.5 %, **+4**. HYPOTHESIS. **Cost:** part of P6.
2. **Scorer ensemble or multi-head averaging** (2–3 scorer heads, mean probabilities). Selection is
   variance-sensitive at AUC ≈ 0.8. ESTIMATED +0.3–0.8; cost: cheap inside P2/P6.
3. **Data and backbone:** SimScale's published navtest effect is +0.5 (Table 4) and DriveVFM over DINOv3-L +0.28
   (A13). Both need data or weights we do not hold.

### Ranked proposal table

| rank | proposal | ESTIMATED gain (PDMS) | confidence | cost | needs PI? |
|---|---|---|---|---|---|
| 1 | **P3** route parity (DriveRL's own correction) | +1.0 to +1.3 | high (A9 measured +1.05) | CPU + 1 navtest pass | **yes**: admissibility |
| 2 | **P1** NAVSIM-PDM relabel on executed plans + scorer fine-tune | +2 to +4 | medium (V3r +2.0 measured; label defects measured) | ≤ ~85 CPU worker-h + 10–20 A40-h | **yes**: navtrain logs download; "paper-pure" reading |
| 3 | **P4** teacher-on-navtest measurement | 0 (decides P5) | n/a | CPU hours | maybe (data) |
| 4 | **P2** un-detached joint fine-tune | +2 to +5 (overlaps P1) | low–medium | 2–4 A40-days | **yes**: model change |
| 5 | **P5** clean / TTS-requery teacher targets + traj fine-tune | +0.5 to +2 | low–medium | CPU + 1–2 A40-days | yes (GPU) |
| 6 | **P6** paper-faithful full rerun (+P7.1 aux BEV head) | to 91–95 | medium | ~7 A40-days + labeller | **yes** |
| 7 | **P7.2** scorer ensemble | +0.3 to +0.8 | low | marginal | no |
| — | CPU test-time rules (EP tilt, gates, re-weights) | +0.03 to +0.18 (MEASURED, cross-fitted) | — | done | eliminated |

### Decisions the PI needs to make

1. **"Paper-pure" scorer labels:** confirm the paper's own reading, PDM targets (NAVSIM's PDM scorer) for all six
   components, as P1 assumes. This reverses the 2026-10-04 choice of teacher-simulator labels for SFT-2/3.
2. **Route parity (P3):** admit DriveRL's own route correction at test time, the computation that produced every
   training goal. It is pending since A9.
3. **Download/compute for P1:** the navtrain `navsim_logs` and maps (no sensors) for metric caching, plus a CPU
   labelling budget, on a pod or the dev box.
4. **Model change for P2/P6:** un-detach the scorer's visual context, which is the paper's detach.
5. **P4 data:** whatever the navtest teacher rollouts need. Check first whether the OpenScene navtest pickles
   suffice.
6. **Whether to commit ~7 A40-days to P6** after P1–P4 read out, rather than more fine-tunes.

---

## 5 · Open questions and unverified items

* **The paper's teacher on navtest vs ours (P4)** is the largest unmeasured term. Our teacher's EP relative to the
  human on navtest is unknown. The paper's teacher EP (91.5) exceeds the human's (87.5); ours is untested.
* **The paper's labelling details are unstated:** online vs precomputed PDM targets, and how lane/route
  augmentations enter the PDM route. My reading, the NAVSIM PDM scorer, follows from the six named components and
  ref. [17]. It is not a quoted implementation.
* **How goal-augmented targets enter an epoch in the paper** is unstated (D-REFE-EPOCH-1 records ours as a decision).
* **P2's gain cannot be isolated from banked data.** It needs the fine-tune.
* **The SFT-1 interim reads** (+0.02/−0.10/−0.22; −0.73/−0.92/−0.58) are INHERITED from PREREG_SFT2/3 prose. I
  found no banked raw readout.
* **V3r's +2.00** is a dev-box fine-tune × 3 seeds from one checkpoint, not a pod retrain (INHERITED).
* **The paper-side split in §2.2 is ESTIMATED.** It assumes EP = 0 on M = 0 tokens (true by NAVSIM construction) and
  borrows our NC = 0.5 share.
* **The cross-fitted re-ranker (+1.12) is diagnostic only.** It is trained on navtest labels (cross-fitted by log),
  so it is never a reportable score.
* **Not done (no GPU, per brief):** any forward pass, including a navtest-half scorer fine-tune that would bound P1+P2.
* **Not re-read:** Hydra-MDP, DiffusionDrive, GTRS and DriveSuprim (all banked) were not re-read for this review.
  The P7.1 analogy is PUBLISHED-SECONDARY.

### Register row proposed (escalated, not written: `GOALS_AND_CLAIMS.md` is a shared file under concurrent edit)

> **D-REFE-GAP-1** — REFe 83.24 vs DriveZero DINOv3 ViT-L 94.55 (Table A13; not 93.88), harness-matched (human
> 94.55 vs 94.8). Gap ≈ 5.5–6 points multiplicative (DAC ≈ 5) + ≈ 5.3 points EP. Selection-bound: best of 64
> 97.07; the oracle on NC + DAC + TTC alone reads 95.16. The leak-free EP oracle reads −1.18, so EP follows safety
> discrimination. The scorer's labels are teacher-calculator (NC/EP/TTC/DDC), not the paper's "PDM targets [17]".
> The scorer's context is detached, and its labels sit on raw plans at 0–37 % coverage. MEASURED/PUBLISHED;
> `REVIEW_7_GAP_TO_PAPER.md`, `raw/2026-10-04-gap-review/`.

---

## 6 · Artifacts

**Primary sources, all already banked in `TanitAD Research Lab/Library/papers/`; nothing new was downloaded:**
* DriveZero `2609.06055` (read in full);
* PDM, "Parting with Misconceptions" `2306.07962`: DriveZero's ref. [17] for the scorer targets;
* NAVSIM `2406.15349`;
* the NAVSIM v1.1 scorer source (`pdm_scorer.py`, sha `3e8291bf`);
* the released DriveRL repo (`C:/Users/Admin/dz/DriveZero/DriveRL`).

| artifact | what it is |
|---|---|
| `raw/2026-10-04-gap-review/gap_decompose.py` → `gap_decompose.json` | formula check, sub-score summary, paper-side estimates and gap split, harness validation (human / PDM-Closed), census means, EP mechanics on the pick |
| `gap_selection.py` → `gap_selection.json` | partial oracles per head, descriptive selection bias, cross-fitted deployable rules |
| `gap_heads.py` → `gap_heads.json` | leak-free EP oracle, per-head within-set AUC, length correlations, calibration, length-quintile table |
| `gap_rerank_crossfit.py` → `gap_rerank_crossfit.json` | DIAGNOSTIC: NAVSIM-label re-ranker on outputs + geometry, cross-fitted by log |
| `gap_fan_quality.py` → `gap_fan_quality.json` | pick outcome by fan safe share; clean vs dirty fan weighted-part breakdown |
| `gap_conditional.py` → `gap_conditional.json` | EP oracle on clean fans; fan-conditional progress tilt (cross-fitted) |
| `gap_dirty_fans.py` → `gap_dirty_fans.json` | dirty-fan rate by route distance, turn bearing, v0 |
| `gap_training_curve.py` → `gap_training_curve.json` | per-epoch training curve (traj L1, score loss, coverage, on-policy lag) |
