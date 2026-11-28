# MM ruling on the L2-OFFPROP build escalations (Master Mind, 2026-10-19; before any real L2 arm is trained or read)

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
