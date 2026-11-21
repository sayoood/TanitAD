# DRAFT (not sealed, not a PREREG_ file) -- next arm after HSGATE8 FAIL (G2)

Author: refcv9-hsgate agent, 2026-10-26. For the Master Mind to seal, amend or reject. Nothing here has been run on eval139.

## Why the committed s5 lever cannot run as written

PREREG_HSGATE s5 commits, on G2 FAIL: "the same gate with its threshold chosen on FIT rows under an FD cap (FD_FIT(gate) <= FD_FIT(H2cn-OOF))".

MEASURED, FIT side only. Sources: `next/FDCAP_FEASIBILITY_FIT.json` and `next/FDCAP_FRONTIER_FIT.json`. Exact per-row prefix scan over all 7,147 distinct thresholds:

| reading | thresholds meeting the cap | best MD under the cap | F-rate there |
|---|---|---|---|
| registered gate, held-out logits (5-fold, fold seed 1, lambda 0.1) | 1 of 7,147 | 0.4624 (= H2cn-OOF, gain 0) | 0.014 % |
| frozen gate, in-sample | 17 of 7,147 | 0.4624 (gain 0) | <= 0.24 % |

The FIT baseline is H2cn-OOF: FD 0.0815 on 3,704 GT-cruise rows, MD 0.4624 on 1,503 GT-decel rows.

The gate's highest-scoring windows add false decels before they fix any missed decel. Under the cap, the committed arm therefore degenerates to H2CN_DEPLOY8 (F-rate ~0) and could not move G1. Running it on eval139 would re-score H2CN.

## Measured headroom (oracle, FIT rows)

- 265 rows where gating to FACTORED fixes a missed decel.
- 403 rows where it adds a false decel.
- 108 rows where it removes a false decel.
- Gating exactly the 265 fix rows would give MD 0.2861 at no FD cost, so the headroom is real.

## Candidate replacement: HSGATE8-MDFIX (same 15 inference features, new label)

**Label.** Labels may use privileged GT; inference stays the same 15 features.

`y = 1 iff GT-decel AND plan_lon_class(P_F) == decel AND plan_lon_class(P_H-OOF) != decel` (base rate 0.0371 on FIT).

**Descriptive held-out readings.** These use the registered lambda 0.1, which was not re-selected, and are FIT rows only:

| arm (held-out) | FD cost cap | F-rate | FD | MD | staged ADE_S6 |
|---|---|---|---|---|---|
| H2cn-OOF | -- | 0 | 0.0815 | 0.4624 | 2.3860 |
| registered gate, tau 0 | (none) | 10.0 % | 0.0894 (+0.0079) | 0.4358 (-0.0266) | 2.3785 |
| registered gate | <= +0.002 | 4.9 % | 0.0834 | 0.4544 (-0.0080) | 2.3824 |
| registered gate | <= +0.005 | 7.6 % | 0.0861 | 0.4491 (-0.0133) | 2.3786 |
| **MDFIX** | <= +0.000 | 0.34 % | 0.0815 | 0.4584 (-0.0040) | 2.3795 |
| **MDFIX** | <= +0.002 | 1.5 % | 0.0834 | 0.4405 (-0.0219) | 2.3727 |
| **MDFIX** | <= +0.005 | 2.2 % | 0.0861 | 0.4351 (-0.0273) | 2.3681 |

A second label, LAB_NOFD (= the ADE label with FD-adding rows set to 0), did not help: under the cap the best MD was 0.4624, and at an FD cost of +0.002 the MD was 0.4564.

At equal FD cost, MDFIX buys about 2.7x the registered gate's MD gain (+0.002 cap) and about 2x (+0.005 cap). It reaches the registered gate's tau-0 MD gain at about 60 % of its FD cost and with better staged ADE.

## Proposed spec (open choices marked OPEN)

1. **Features, fit and folds.** The same 15 features, L2 logistic regression with the D1 objective, the same OOF H2cn, the same FIT rows, and CV fold seed 1.
2. **Lambda.** CV over the same grid. OPEN: the criterion is either held-out BCE, or held-out MD subject to the FD cap.
3. **Threshold.** Minimise held-out FIT MD subject to `FD_FIT <= FD_FIT(H2cn-OOF) + c`.
   - OPEN: the value of c. The draft suggests 0.002. The eval FD CI half-width is ~0.01, and the A4d fit-seed floor reached +0.0092 without separating.
   - OPEN: whether the final all-rows gate inherits tau as a logit value, or as the held-out F-rate quantile.
4. **Scoring.** Unchanged from PREREG_HSGATE sections 2-3: G1-G4 at both inference seeds, SHIP-R8 via copies, the same controls, and the beside rows including the A4d floor.
5. **Committed outcomes.**
   - PASS: joins the shipped-stage choice.
   - G2 FAIL again: the 15 features cannot buy MD without FD. The next lever is then richer gate features, which is an architecture decision for the PI/MM.
   - G1 FAIL: the MD gain does not transfer at this F-rate. H2CN_DEPLOY8 stands.

## Disclosures (must travel with any sealed version)

- The MDFIX label was chosen AFTER the HSGATE8 phase-2 eval read (G2 FAIL). It was chosen on FIT-row arithmetic only, but eval139 has now been read once for this gate family: a forking-paths exposure at the programme level.
- All numbers above come from 5-fold episode CV on the 7,147 FIT rows. There is no interval, one training seed and one fit. They are descriptive, not bars.
