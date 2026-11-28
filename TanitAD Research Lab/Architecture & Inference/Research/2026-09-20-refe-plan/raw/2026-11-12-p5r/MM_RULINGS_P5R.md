# MM rulings on P5-R, written 2026-11-12 ~08:30Z, BEFORE `raw/verdict.json` exists

The card is `PREREG_P5R.md`, sha256 `cfd23e0651d66e497238a0708c717ba00acbcb2189295efcfb15dad657922a42`, frozen 2026-11-12T08:03:43Z. Its hash was re-verified by the MM. These rulings do not edit the card.

## R1: the penalty target (noted, accepted)
- The P5 card writes the regulariser as λ‖Δ²r20‖². PREREG_P5R commits instead to Δ² of the correction (r20 − traj), on x, y and yaw (yaw scaled by L = 2.588 m/rad).
- Reason given: a penalty on r20 itself straightens the teacher's own path, and λ → ∞ could never return `traj`, which control (g) requires.
- This was decided and frozen before any P5-R data. The MM accepts it as the faithful reading of "regularised inversion" and not a goalpost move.
- Every P5-R quote names it as "Δ² of the correction".

## R2: the PI's tracker-aware-label ruling (2026-11-11) and P5-R
- P5-R labels are tracker-aware labels of the same class as Tc20s: a pre-distortion of the teacher's trajectory chosen so that NAVSIM's LQR tracker executes it closer to the intent.
- The ruling's two conditions apply to P5-R unchanged:
  1. every number LQR-tracked AND tracker-transparent;
  2. the claim worded "the student's plans survive the benchmark's tracker", never "drives better".
- Swapping Tc20s for P5-R inside P5-C's approved arm form (C1 vs C0, 2,000 steps) is an MM-level label substitution. It is nevertheless **stated to the PI explicitly before any P5-C launch**, together with the C0r question and the 57–58 L40-h cost already queued for the PI.
- It is not launched on this ruling alone.

## R3: TB-0 for P5-R labels (cross-environment identity)
- P5-C's TB-0 demands max abs diff 0.0 between pod and dev-box labels. P5-R's solver goes through LAPACK, so bit-identity across machines is not guaranteed.
- If P5-R replaces Tc20s, PREREG_P5C gets a dated amendment, **committed before any pod P5-R label exists**, with this TB-0:
  - **max abs diff ≤ 1e-6 m (yaw ≤ 1e-6 rad) on every common row, AND identical sim8 pass/fail verdicts on every common row.**
- TB-0 for Tc20s stays 0.0.

## R4: which verdict governs
- P5-R's own verdict (EVAL split, rank 0 decides) is read as pre-registered.
- **Which label P5-C uses** is decided by the order the MM committed in `p5c/gates/preview_63k/PREVIEW_TB6.md`:
  - the pod's full-table TB-6 on Tc20s passes → P5-C runs on Tc20s, and P5-R stays banked;
  - it fails at rank 0 and P5-R passes → P5-R is the label, after R2 and R3;
  - both fail → the committed next lever is P5-T, the selective label (PREREG_P5R).
