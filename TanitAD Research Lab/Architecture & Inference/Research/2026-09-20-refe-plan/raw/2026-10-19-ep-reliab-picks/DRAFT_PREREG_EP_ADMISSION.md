# DRAFT PREREG: EP-ADMISSION, an EP-reliability admission gate and a pessimistic-EP pick rule (Master Mind, 2026-10-19)

**Status: DRAFT, not sealed, nothing run.** It is sealed against a specific CANDIDATE (a P6 checkpoint, R0e-pick, or any future EP-led head or RL change) BEFORE that candidate's selection-tier bank is read. It is the committed next arm of EP-RELIAB-PICKS (`D-REFE-EP-RELIAB-1`, `MM_RULING_EPRELIAB.md` §4).

## Why
EP-R1's composite predicted an EP gain of +0.124 at its picks and realised -0.011, and it paid in DAC and NC to get it. Every EP-led change so far traded real safety for an EP gain that was not real, and that was only found after the fact. The gate makes the check mandatory BEFORE any navtest look is spent.

## (A) Admission gate: the primary purpose
For candidate C against P2, on a selection-tier bank built from C's own forward (C's rows, C's picks; P2's picks for the pair), with the same estimator as EP-RELIAB-PICKS (paired log-cluster bootstrap 10,000, seed 20260927, rank-free mass form when C's EP is off the label scale):
- **A1, EP reliability:** D_M = predicted − realised EP mass at C's picks against P2's. **ADMIT-A1 iff its upper CI ≤ +0.02** (EP mass units).
- **A2, safety reliability at C's picks:** observed / predicted unsafe mass for NC, DAC and TTC. **ADMIT-A2 iff every CI overlaps [0.8, 1.25].**
- **A3, realised gain:** realised dPDMS x100 vs P2 on the bank. Its CI is REPORTED, not gated; the navtest look owns the decision.
- A candidate failing A1 or A2 does NOT get a navtest look. Its failure is the result, with the attribution of (D) from EP-RELIAB-PICKS.
- Controls: the same C-ID / C-REACH / empirical-null set as EP-RELIAB-PICKS. Every control's false-fire rate is simulated on noise-trained replicates first (R51 / R52); reach checks use planted inputs.

## (B) Pessimistic-EP pick rule: secondary
`EP_used = EP_slot + lam * (EP_row - EP_slot)`.
- `EP_slot` is the feature-free slot-prior EP: per proposal index, fitted on training folds.
- `lam` is the OOF within-set calibration slope of the EP row against the EP label (EP-R1: 0.31). It shrinks the EP row's deviations to their measured reliability.
- Read: dPDMS x100 and the unsafe-pick rates vs P2. **RECOVERS iff the dPDMS lower CI ≥ 0 AND the NC / DAC unsafe-rate differences have upper CI ≤ +0.001**, both seeds.
- Prior (ESTIMATED): it reverts toward P2, so the value is low. It is registered so that an over-dispersed EP row can never again be used raw.

## Committed next arms (to be confirmed at sealing)
- A1 and A2 pass → the candidate's shared navtest look (`MM_RULINGS_HEADROWS.md` R1-R9 rules).
- A1 fails → the candidate's EP gain is not real on the selection tier: no look. Attribute the cause and return the lever to the trunk.
- A2 fails → the safety rows are mis-calibrated at the candidate's picks: register a calibration arm for THAT candidate, with reliability at the picks pre-registered.

## REVISION 1 (Master Mind, 2026-10-19, pre-seal; no candidate read)
P6-READ-PREP measured that the drafted A1 bar sits INSIDE the estimator's own noise: EP-R1's D_M half-width is 0.026 against the +0.02 bar. A perfectly calibrated candidate would be ADMITTED only about 32 % of the time (ESTIMATED). An admission gate that refuses most good candidates is mis-specified: the R51 class again, a bar that does not account for its own estimator's false-refusal rate. A2 as drafted refused only significant miscalibration beyond 25 %. Both are revised before any sealing:
- **A1 (revised): FAIL-A1 iff, on ANY seed, D_M has a lower CI > 0 (significant over-prediction) OR a point estimate > +0.05 (material over-prediction).** Otherwise ADMIT-A1.
  - False refusal of a calibrated candidate: about 5 % across 2 seeds (ESTIMATED).
  - EP-R1-sized over-prediction (+0.135, lower bound +0.111) is refused with power about 1.
  - The sealed version must SIMULATE both rates on the real estimator (calibrated and planted-over-prediction replicates) and report them.
- **A2 (revised): FAIL-A2 iff, for any of NC / DAC / TTC, the observed / predicted unsafe mass at the candidate's picks has a lower CI > 1.0 on any seed** (the rows significantly UNDER-predict risk at the picks, i.e. the pick exploits over-confident safety rows). Ratios below 1 are conservative and are not refused.
- **B slot source: `row`** (EP_slot = the EP ROW's own per-slot mean on training folds, so EP_used stays on the row's scale). Measured on EP-R1's smoke: `label` gives dPDMS -0.725 and `row` gives -0.211 x100; `label` mixes scales.
- **Seeds: 2** (as every arm in this line). **In-sample:** S0's tokens are navtrain fit tokens, so P6, like P2, is IN-SAMPLE there. Its selection-tier read is biased toward OVER (conservative for A1), and every report states it.
- `ep_admission.py` implements the DRAFT rules. It must gain the revised rules (keeping the draft rules reproducible) with RED mutations BEFORE sealing (BACKLOG `EP-ADMISSION`).
