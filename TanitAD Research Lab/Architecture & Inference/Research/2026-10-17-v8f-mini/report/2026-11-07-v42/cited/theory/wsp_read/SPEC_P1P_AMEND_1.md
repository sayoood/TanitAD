# SPEC_P1P_AMEND_1 (2026-11-07 local, written after the first full run of `p1p_read.py` and after `P1P_RESULT.json` existed; `SPEC_P1P.md` is not edited)

A defect in SPEC_P1P.md section 5, reported as a defect. No criterion, threshold, window rule, group definition, bootstrap setting or sensitivity changes.

## The defect
* SPEC_P1P.md section 5 words controls N2 (identity) and N3 (planted mechanism) as verdict strings: N2 "the verdict is FAILS", N3 "must return HOLDS".
* The verdict function applies the criterion's group-size rule first (NOT EVALUABLE if a group holds fewer than 30 windows). On the primary window set the CONFIDENT group holds 24 windows (section 3 of `P1P_RESULT.md`), so the verdict function returns NOT EVALUABLE for any ρ whatsoever, including ρ ≡ 1 and ρ = posterior. A verdict-string wording of N2 and N3 therefore cannot be met by any pipeline on this window set. `P1P_RESULT.json` as run carries `N2.pass = false` and `N3.pass = false` for that reason alone; the statistic values beside them are the ones the controls were written to test.
* The defect is in how the controls were worded, not in the pipeline: the controls are meant to test the statistic (group medians, Δ, interval, separated, positive), and the group-size rule is a separate gate that the primary verdict applies.

## The reading applied from here
* N2 is read on the statistic: every ρ equals 1.0, both group medians equal 1.0, Δ equals 0.0 exactly, the interval is [0.0, 0.0], `separated` and `positive` are false, at both seeds.
* N3 is read on the statistic: Δ > 0 and `separated` true at both seeds, with ρ set to the window's own posterior.
* The amended flags are computed by `p1p_finalize.py` from the numbers already stored in `P1P_RESULT.json` (no new data is read) and written beside the original flags under `controls_amended`; the original flags stay in the file as run.
* The primary verdict, the group counts and every other control are unchanged by this amendment.
