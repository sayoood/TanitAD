# DRAFT PREREG L2-OFFPROP: can REFe's EP head learn to order its OWN off-proposal perturbations, so that L1's search admission re-passes? (Master Mind, 2026-10-19)

**Status: DRAFT, nothing run, no L2 number exists.** It is the committed next arm of L1 Step A's outcome **O2 NOT ADMITTED** (`PREREG_L1_SEARCH_ADMISSION.md` §4: "use [Step A's labelled off-proposal pairs] as the training data of L2 … search is retried only after L2's scorer re-passes this Step A"). It is sealed by the Master Mind after the build agent's code and planted controls exist and BEFORE any L2 arm is trained.

## 0. Why, with the evidence class of every number
- **L1 Step A (MEASURED, SEEN-SCENE, 24 logs, 400 tokens):** EP concordance off-proposal **A1 = 0.685 [0.614, 0.761]** against the in-fan reference 0.887. The aggregate is NOT exploitable: A4 0.015, A5 0.008. TTC concordance A2 0.843, aggregate A3 0.766.
  - ⇒ The search fails on EP ordering, not on safety exploits.
- **HEADROW-PROBE (`D-REFE-HEADROW-PROBE-1`):** the EP row is about 98 % slot prior and INCONCLUSIVE, NOT feature-limited. The safety rows ARE feature-limited.
  - The EP row never saw within-slot variation. A perturbation keeps the slot and changes the trajectory, which is exactly what Step A's pairs supply.
  - ⇒ The ten-arm head-level closure (`MM_RULING_EPRELIAB.md` §3) does NOT cover this arm: those arms trained on in-fan data only.
- **Where it fails (L1 RESULT §5, POST-HOC, descriptive):** EP ordering is a coin flip on SPEED-UPS (0.559) and better on slow-downs (0.769). The weakest cells are speed-ups from the scorer's own top-8 anchors (0.495 / 0.523), which is exactly the direction an EP-led search climbs. ⇒ The A1-OOF table is reported per speed scale and anchor type, and the speed-up cells are the ones the arm must move.
- **MM interpretation of "re-passes this Step A" (L1 prereg §4 O2):** the Step A PROTOCOL, read OUT-OF-FOLD by log on these 24 logs (§1). Before any search it is re-run on a FRESH carve of navtrain held-out logs (§4, L2-ADMIT). Re-scoring the same 400 tokens in-sample is never a re-pass.
- **EP-RELIAB (`D-REFE-EP-RELIAB-1`):** the EP row over-predicts its own gain at EP-led picks (+0.124 predicted, −0.011 realised). Any retrained EP head must also pass the EP-ADMISSION gate before a navtest look.

## 1. Data and folds (fixed now)
- Step A's banked arrays (`D:/Projects/TanitAD-artifacts/l1_stepA_run/`): the bundle (400 tokens, 24 logs), `perts.npz`, `labels.npz` (true PDM sub-scores per perturbation), `scores.npz`.
- **Folds: 5, by LOG** (log rank mod 5 over the 24 logs, sorted by log name). Every number below is OUT-OF-FOLD: a token is never in its own fold's training set.
- ⛔ The 24 logs are SEEN scenes (A1.3) and P2 is IN-SAMPLE on them. Every L2 number is stamped SEEN-SCENE, SELECTION tier.

## 2. Features
- For every (token, anchor, perturbation) and every in-fan proposal: the **EP head's INPUT token** as P2 computes it with the bundle's cached ctx (`model.score_trajectories` path; the same call Step A's step 6 made). It is exported once.
- **C-FEAT gate:** P2's own EP head on the exported tokens reproduces Step A's banked EP scores. Max abs diff ≤ 1e-4 on ≥ 99 % of rows, else STOP.
- CPU is admissible (step 6 took 72 min); GPU only under the dev-box lock.

## 3. Arms (each trained per seed in {0, 1}, per fold)
- **(H) EP head retrained.** P2's EP-head architecture, initialised from P2's EP head. Loss: pairwise logistic on Step A pairs whose true EP differs by ≥ 0.05 (the A1 definition), plus a pointwise BCE on in-fan EP labels (weight fixed by the agent BEFORE sealing). Frozen everything else.
- **(H-null) noise-trained null (R52).** Same recipe; the perturbations' true EP is PERMUTED within each (token, anchor) group, so the pair ORDER is noise and the label marginals are intact.
  - 5 replicates (seeds 100–104). A1-OOF of H-null replicates is the empirical null.
- **(P2) reference:** P2's EP head untouched (A1 = Step A's numbers, re-computed on the same OOF pair set).

## 4. Primary read and committed outcomes
**A1-OOF** = Step A's A1 (EP concordance on off-proposal longitudinal pairs, true-EP difference ≥ 0.05), computed with the arm's OOF EP scores. Estimator: paired log-cluster bootstrap over the 24 logs, 10,000, seed 20260927. A4 / A5 are recomputed with the retrained EP in the V1_W = (5, 5, 2) aggregate (`refe/planner.py:537`).

| outcome | rule (fixed now) | next arm, committed now |
|---|---|---|
| **L2-ADMIT** | on BOTH seeds: A1-OOF point ≥ 0.80 AND lower CI > the H-null replicates' max point; AND A4 ≤ 0.10, A5 ≤ 0.05 (points, as L1); AND in-fan EP concordance OOF ≥ P2's on the same pairs − 0.02 (point) | **L1 Step B with the L2 EP head**, on a FRESH carve of navtrain held-out logs (not these 24), Step A re-run there first. Then the EP-ADMISSION gate (`--rules r1`), then the shared navtest look |
| **L2-PARTIAL** | A1-OOF lower CI > H-null max on both seeds, but point < 0.80 or a non-degradation term fails | **L2-BRANCH:** the same pairs, but the last scorer-branch block is trainable as well (cached ctx; dev-box GPU; shares SH5's lockstep trainer). Own amendment before any run |
| **L2-FLAT** | A1-OOF lower CI ≤ H-null max on either seed | the EP head token does not carry within-slot EP ordering ⇒ **L2-BRANCH**, and if that is also flat, the trunk (P6) |
| **VOID** | C-FEAT fails, or ≥ 2 of 5 H-null replicates exceed P2's A1-OOF by > 0.05 (the machinery manufactures concordance) | fix the instrument; no call |

## 5. Controls (each must be able to fail)
- **C-ID:** H with zero optimizer steps reproduces P2's EP scores bit for bit; then A1-OOF equals Step A's A1 on the same pairs.
- **C-REACH (planted):** a planted feature equal to the true EP difference's sign, appended to the token, gives A1-OOF ≥ 0.98. It proves the training and read path can learn ordering at all.
- **C-LEAK:** no feature of a held-out token enters training. It is tested by a RED mutation that puts one held-out fold into training, which must raise A1-OOF on that fold.
- **G-LIVE:** every loss term finite; the EP head's parameters receive a non-zero gradient; every frozen parameter's gradient is exactly 0.0.
- **Four families:**
  - LONGITUDINAL: A1, A2, and speed-scale-wise A1.
  - LATERAL: the ±0.3 m pairs' EP and DAC concordance.
  - TACTICAL: the command-class change share, unchanged by a head retrain, reported.
  - STRATEGIC: PAUSED (PI ruling 2026-09-19).

## 6. Not licensed by this arm
- No capability claim, and no navtest look. A1-OOF on 24 SEEN logs licenses only the fresh-carve Step A re-run.
- Inference stays vision-only: the EP head reads the same token at inference. The privileged PDM labels are TRAINING targets only.
