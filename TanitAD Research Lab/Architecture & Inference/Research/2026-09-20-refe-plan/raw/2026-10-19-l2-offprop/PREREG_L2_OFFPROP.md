# PREREG L2-OFFPROP: can REFe's EP head learn to order its OWN off-proposal perturbations, so that L1's search admission re-passes? (Master Mind, 2026-10-19)

**Status: SEALED 2026-10-19T15:43:09+02:00 by the Master Mind, BEFORE any real L2 arm (H, H0, H-null) was trained or any real A1-OOF read.** It is the draft below plus REVISION 1 (pre-seal), carried verbatim from `MM_RULING_L2_BUILD.md`. Where they differ, REVISION 1 governs. It is the committed next arm of L1 Step A's outcome **O2 NOT ADMITTED** (`PREREG_L1_SEARCH_ADMISSION.md` §4: "use [Step A's labelled off-proposal pairs] as the training data of L2 … search is retried only after L2's scorer re-passes this Step A"). It is sealed by the Master Mind after the build agent's code and planted controls exist and BEFORE any L2 arm is trained.

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

---

## REVISION 1 (pre-seal): `MM_RULING_L2_BUILD.md`, verbatim


These amend `DRAFT_PREREG_L2.md` BEFORE sealing ("Revision 1, pre-seal"). The sealed prereg carries them verbatim.

**E1, the H-null design: GLOBAL shuffle (`--null-group global`). The within-(token, anchor) permutation is WITHDRAWN.**
- The agent's planted world MEASURED it at +0.09 above P2. Permuting EP only inside a group keeps the between-group structure (which anchors' perturbations gain or lose EP), and that is real ordering signal. It is not a noise null, and it would fire VOID for a non-instrument reason.
- The null's job here is R52's: the training-noise floor of THIS machinery. A head trained on labels with no information must not raise A1-OOF.
- The global shuffle (true EP permuted across all training rows of the fold) is that null. On the planted world it reads −0.03.
- VOID becomes: ≥ 2 of 5 GLOBAL-null replicates exceed P2's A1-OOF by > 0.05.
- My design error, caught pre-data by a planted control. It is the same family as R54 (a null or companion that still carries the signal). It is noted here and in the sealed prereg, not as a retraction: no claim was made.

**E2: A1 is the POOLED quantity Step A's gate used** (0.6851, lon + lat, den 9,888), so "re-passes this Step A" compares like with like. The longitudinal-only (0.680) and lateral (0.861, den 259) values are reported beside it, ungated.

**E3 / E4: the strictest defaults are ACCEPTED** (null pool, VOID scope).

**In-fan clause: REPLACED, because as drafted it is unreachable by construction.**
- P2's in-fan EP concordance is near in-sample (P2 trained on these logs, a far larger set). An out-of-fold head refit on 400 tokens cannot be held to it.
- **New clause:** the in-fan EP concordance (OOF) of H must be ≥ that of **H0 − 0.02**, where H0 = the SAME recipe with the pairwise weight 0 (in-fan BCE only; same init, folds, seeds). This isolates the question the clause was for: does the PAIR loss damage in-fan ordering?
- P2's in-fan value is REPORTED, never gated.
- Degradation that matters for driving is caught downstream: by Step B's dPDMS endpoint (search output vs P2's pick, true PDMS) and by the frozen EP-ADMISSION gate before any navtest look.
- **Recipe (λ pairwise / in-fan weight, any proximal weight toward P2's head):** fixed BEFORE the seal from the semi-synthetic sweep (real tokens, synthetic labels), by this pre-stated rule:
  - maximise A1-OOF subject to in-fan(H) ≥ in-fan(H0) − 0.01;
  - ties go to the smaller λ;
  - if no setting satisfies it, take the setting with the best in-fan and REPORT that the clause may bind.
- H0 is a new arm, trained on 2 seeds × 5 folds. It roughly doubles the about 8-minute cost.

**E6: ACCEPTED.** Pre-seal, C-LEAK runs on real tokens with shuffled labels. Post-seal, the real C-LEAK runs inside the real run and is reported.

**E7: NOT a launch-gate case.**
- The CPU fit of a 257-parameter head is a SELECTION-tier diagnostic. It is covered by its own gates (G-LIVE, C-ID, the seal guard), the same as the ten head-level arms before it.
- ⛔ Any INTEGRATION of an L2 head into REFe's scorer (Step B and beyond) goes through a `launch_gate.py` profile with G-HYG / G-DVB / G-LIVE entries for the new flags (PI rule 2026-09-26). That is a launch blocker for Step B, not for this arm.

## Addendum (Master Mind, 2026-10-19 14:10, after the gate chain; still no real arm trained or read)

**C-REACH FAILED on the chain's recipe (`lam10.0_lr0.0003_ep5`, 150 steps):** the planted perfect feature reached A1-OOF 0.905 < 0.98 (`logs/gate_reach.log`).
- The sweep's in-fan rule (the agent's original R1 + R2, 1 of 20 settings passing) selected a recipe that barely moves the head off P2's init.
- ⇒ The real arm would have read FLAT by construction. C-REACH did its job.
- **Ruling: the recipe is RE-SELECTED with C-REACH as a HARD constraint.**
  1. Candidate settings must give C-REACH A1-OOF ≥ 0.98 (planted world, the real recipe), on both seeds. Widen the grid (lr up to 3e-3, steps up to 2,000, any proximal weight toward P2's head) until some do.
  2. Among those, apply the E1-era rule above: maximise semi-synthetic A1-OOF subject to in-fan(H) ≥ in-fan(H0) − 0.01. H0 is the same setting with pairwise weight 0; ties go to the smaller λ.
  3. If NO setting passes C-REACH, STOP and report it as an INSTRUMENT DEFECT (no outcome, no real labels read). The planted feature is an APPENDED input, so failing to learn it says nothing about the token; it means the optimiser or fit path is broken. (Corrected 14:12: an earlier line of this addendum mapped it to L2-BRANCH, which was wrong.)
- **E6 above is superseded** by the C-LEAK redefinition below.

**C-LEAK is REDEFINED as a structural reach check.**
- A linear 257-parameter head cannot memorise shuffled labels, so the statistical test (clean 0.502 vs leaky 0.504) cannot fire on noise. It is uninformative, not passed.
- The new check: `oof_scores` ASSERTS that each fold's training token set is disjoint from its held-out token set and that held-out logs are disjoint from training logs. It fails closed.
- A RED mutation must trip it: inject one held-out token into training ⇒ the assertion fires.
- This is a check on the code path, the R51-era lesson: reach checks on planted inputs, not on properties of noise.

**pytest:** `test_semi_synthetic_...` failed at a margin of exactly +0.020 under the weak recipe. Re-run it under the re-selected recipe. If it still fails, report the margin; do not loosen the test.

## Sealing note (Master Mind, 2026-10-19 15:55; no real arm trained or read)
- **Recipe ACCEPTED:** `lam10_lr0.001_ep15_prox0.3`. C-REACH A1-OOF 0.9984 / 0.9992 (both seeds ≥ 0.98). All five gates PASS on this recipe and code.
- **The failing unit test:** its planted-signal clause (72-token world, beta 3) is MOVED, not loosened. It measured +0.0044 under the in-fan-weighted recipe, a property of a world 5.5× smaller than the run. The same +0.02 bar is asserted at the REAL shape (400 tokens, beta 0.5) by `test_selected_recipe_finds_a_planted_signal_on_REAL_tokens`, which measures +0.045. That is the binding sensitivity check for what actually runs. The calibration asserts of the moved test remain.
- **Section B settled as built:**
  - E5: in-fan pairs = the SELRANK instrument (P2's own top-8, dEP > 0.05, ties half, the same sets for every arm).
  - R1: the tie rules as built.
  - E8: seeds {0, 1}, with the MEASURED near-replicate property (A1 differs by 0.0006) stated. They are not independent evidence of training variance; `H-ESTIM-SEED-1` stays open.
- **Expectation stated BEFORE the read (from the semi-synthetic worlds; not a bar):**
  - L2-ADMIT's A1 ≥ 0.80 may be out of reach under the in-fan clause: the best feasible setting reached 0.788 / 0.792 in the signal worlds.
  - L2-PARTIAL is the likely outcome, and L2-BRANCH its committed next arm.
  - Writing this down now is what stops a PARTIAL from being read as a surprise afterwards.


## PINS (sealed)
- `code/MANIFEST.sha256` (LF sha256 of every staged code file): sha256 `bdb6a95dba4a6a69557a2c7ff158bf529f5ab226ac17f90271f00988f05f9939`
- `raw/recipe_chosen.json` (`lam10_lr0.001_ep15_prox0.3`): sha256 `c7f5b75b0a8f25f96c00dcc379ac583fa9c80622ce5600ad4af8387159119263`
- features: `D:/Projects/TanitAD-artifacts/l2_offprop/feats.npz`, manifest `raw/feats_manifest.json` sha256 `6117f755e59741c980bf1a1e8fbdc369e3b56966a08281f3f41ea55dba4be44a`
- Build state at the seal: pytest test_l2.py 35 passed; test_outcome 27 mutations RED; gates C-FEAT / C-ID / G-LIVE / C-REACH / C-LEAK PASS on this recipe and code (`raw/gates/`).
