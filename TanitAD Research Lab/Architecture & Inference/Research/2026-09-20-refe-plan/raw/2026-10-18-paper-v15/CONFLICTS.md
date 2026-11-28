# CONFLICTS: where a package's prose disagrees with its own raw JSON, or the brief's wording needs a scope note

Paper v1.5 §7.21, drafted 2026-10-18. Rule applied throughout: **the raw JSON wins; the text of §7.21 follows the JSON and says so where it matters.**
No separation call (an interval's side of zero, or a bar crossed) changes between prose and JSON in any entry. The packages are read at mirror tip `5fb509a`.

## A. Prose versus raw JSON

**C1. The non-zero-pick term of the oracle gap: 5.88 (prose) against 5.875 (JSON).** `p2-full-table/RESULT.md` (anatomy table and Reading 1, 2),
`selrule-dac/RESULT_CONFIRM.md` (Reading 3) and the brief print **5.88 [5.54, 6.21]**. `p2-full-table/anatomy.json`,
`oracle_gap_by_cause_at_pick_pdms_pts/pick_nonzero` holds **5.874989315858168 [5.5348189393942935, 6.212801783721596]**: at two decimals that is 5.87 [5.53, 6.21];
the prose rounded the three-decimal print 5.875 upward. Effect: none on a separation call. §7.21 prints **5.875 [5.535, 6.213]** and derives the 62.4 % / 37.6 % split from
the JSON point (5.874989 / 9.412685 = 0.62416; from the prose 5.88 / 9.41 one would get 62.5 %). The four cause terms (5.875 + 2.502 + 1.012 + 0.024) add to 9.413, the JSON's oracle − pick.

**C2. "R0B removes the slow bias; the intervals do not overlap" (prose) against overlapping endpoints (JSON).** `free-point-navtest/RESULT.md` §3 ("intervals non-overlapping"),
the register row and the brief ("slow bias 'removed'"). `raw/result/families/fp_BASE_families.json` and `fp_R0B_families.json`,
`families/refcv6/longitudinal/ci/components/along_final_bias_m`: BASE **−0.8263 [−1.1897, −0.4778]**, R0B **−0.1950 [−0.5492, +0.1529]**. R0B's lower end (−0.5492) is
*below* BASE's upper end (−0.4778): the two intervals overlap on [−0.5492, −0.4778]. `speed_bias_mps` likewise overlaps (BASE [−0.3258, −0.1369], R0B [−0.1561, +0.0282]).
§7.21(b) therefore reports a point-estimate movement toward zero and says in so many words that it is **not** a separated change; no paired interval exists for the family metrics.
The "0 of 21 adverse" count is unaffected (it is the JSON's own `families_adverse_vs_BASE/R0B/adverse`, an empty list).

**C3. "TACTICAL rows carry no interval" (prose) against intervals present (JSON).** `free-point-navtest/RESULT.md` §3 ("rows without an interval, reported, not tested").
Both `fp_*_families.json` carry `families/refcv6/tactical/{lateral_decision,longitudinal_decision,maneuver_5way_collapsed}/ci/{accuracy,kappa}` as {mean, lo, hi}.
§7.21(b) prints them (12 claims t_*), labelled single-arm / 2,000 resamples, and reported, not tested.

**C4. SO-CARVE A2: "A2-ALIGNED(T* = 0.02)" (RESULT headline, register row) against `outcome: INVALID` (JSON).** `so-carve/tempsweep_a2.json`, key `outcome` = **INVALID**
(the pre-registered control 3 failed: T = 1e-4 must reproduce the pick to 1e-6; the differences read −0.015, +0.026, −0.017 with 0 picks differing from the argmax).
The RESULT body (§8) says the same ("By the prereg's own rule A2 is INVALID as written") while its headline and the draft register row say "tracks the pick". §7.21(c) follows the JSON:
the T = 0.02 reading is **PROVISIONAL, and by the registered rule invalid as written**; the rule is not relaxed. The post-hoc diagnosis (near-tie mass; exactly 0.0 at T = 1e-6 and 1e-10)
is from `logs/a2_control3_diag.log` and is marked post hoc.

**C5. "1.4 to 4.5 times smaller" (RESULT §0 and §8, the register row) against a ratio whose denominators span zero.** The ratios reproduce from `fullval_a1.json`
(`states/r0_u*/P1_TTC/L_va_full` over `states/r0_u*/P2_TTC/L_va_full`): 1.2333 / 0.8506 = 1.450, 0.6333 / 0.2297 = 2.757, 1.0000 / 0.2229 = 4.486. But the dual's quantity has
`ci_L_va_full` = [−0.1033, +0.5851] at u500 and [−0.1453, +0.6202] at u1000, so a ratio with that denominator is not interpretable (the rule that withdrew the lean-gap ratio in §7.19b).
§7.21(c) declines the ratio and reports the **paired difference** (quantity − pick), which is separated at one of three states (u1000 −0.777 [−1.320, −0.251]). The numbers are the same; the statistic is the admissible one.

**C6. Sign convention of the in-sample wedge (RZ-1).** `so-carve/RESULT.md` §5 and the register row give the R0d in-sample pick gap as **−0.297** (B minus A, ×100) and g as **−0.034**;
the post-hoc rows as pick −0.265 (se 0.089) and g −0.066 (se 0.032). §7.21(c) writes both as losses (positive = B below A), so g reads −0.034 and the pick **+0.297**; −0.265 becomes **+0.265**.
The magnitudes are the JSON's (`r0d-outcome/raw/sft.jsonl`, `con_g_ttc`, `con_p1_gap_ttc`; the post-hoc rows from `logs/r0d_insample_posthoc.log`); only the sign convention is changed, and it is stated in the text.

**C7. "166 logs" is in prose only.** `so-carve/RESULT.md` §2 and §11 ("16,000 sets in 166 logs, counted from the extracted `rlfan_sets_train` jsonl"); no raw JSON in the package holds it.
§7.21 does not print 166. The count that is printed, 16,000 RL-train sets with none in a held-out log, is R0d's own `data` event (`r0d-outcome/raw/sft.jsonl`, claim r_train_sets).

**C13 (minor). BASE against the banked u400, prose rounding.** `free-point-navtest/RESULT.md` §1 prints "−0.01 [−0.02, +0.00]"; `freepoint_result.json`
`identity/IC3_IC4/IC4/IC4b_paired_diff_BASEpod_minus_banked` holds −0.008178 [−0.024887, +0.000490]. §7.21 prints −0.0082 [−0.0249, +0.0005] (the upper end is positive, which "+0.00" hides).

## B. Class or scope notes (no number changes)

**C8. What 94.55 is.** `p2-full-table/RESULT.md` calls 94.55 "the human reference in D-REFE-GAP-CI … INHERITED". Paper §7.20 (base lines 3160-3165) distinguishes **two quantities that carry the value 94.55**:
the published matched student row (PUBLISHED, arXiv:2609.06055 Table A13, verified in the banked PDF) and our harness's logged human (94.55 [93.78, 95.23], MEASURED). §7.21(a) uses the **matched published row**,
class PUBLISHED, as §7.20 does; the 5.73-point / 60.9 % arithmetic (94.55 − 88.82; ÷ 9.41) is point arithmetic with no interval.

**C9. R0 against R0d in the proxy-wedge numbers.** The brief and the register row state "+0.22 vs +1.00 [+0.30, +1.73]" under the SAFEOBJ constraint. Those numbers are measured on the
**unconstrained R0's** states on the full RL-val (`fullval_a1.json`); R0d's own val-side E_π[TTC] was **not computed** (its score dumps are not on the dev box; `so-carve/RESULT.md` §9).
§7.21(c) says so (the wedge is measured at the R0 states and in-sample on R0d, not on R0d's val side) and keeps R0d's own pick loss (+0.767 [+0.065, +1.487]) beside R0's (+1.000 [+0.300, +1.735]) with the note that two runs without a replicate give neither an effect nor a null.
The upper end prints as +1.735 here; the prose's +1.73 is the same value (1.7348890) at two decimals.

**C10. A correction to my own draft (not a source conflict).** A first draft of §7.21(b) described C0A as "the control branch of the same pass". It is not: C0A is arm A of the
separate control run C0 (`rls1-c0`, state key A, update 1000), R0B is arm B of run R0 (`rls1-r0`, key B, update 1000) (`free-point-navtest/PLAN.md` §4; the "one pass" wording of `eval/PREREG_RL_S1.md` line 95 describes the two arms *within* one run).
Corrected before the final assembly; §7.21(b) now says C0A and R0B come from two runs.

**C11. A source-internal inconsistency in the literature row.** CLOVER (arXiv:2605.15120), Stage 2: the selected PDMS prints 0.9448 in Table 5 and 0.9443 in Table 23 (banked PDF).
§7.21(a) uses only the printed Gap, 0.0495 (×100 = 4.95), which is what the paper prints, not a difference recomputed from either selected value.

**C12. Line endings of the local extracts.** 9 package JSON/JSONL files had CRLF added in the first local working copy. For those files the tip's blobs carry no CR. All values in SOURCES.md were read from the parse of the files (CR-insensitive),
and `files_read.json` records that every local copy is byte-equal to the tip after stripping CR; the sha256 digests in SOURCES.md are of the tip's bytes as committed and match those the R0d package quotes for itself.

## C. Things the brief states that the raw files confirm without change

* "9.41 splits as 5.88 / 2.50 / 1.01": 2.50 and 1.01 are 2.502 and 1.012 in the JSON; 5.88 is C1. The both-zero term (0.024, 4 tokens) completes the sum, which the brief does not list.
* "Only 64 of 12,146 tokens have no non-zero proposal": `anatomy.json` `n_best_of_64_zero` = 64.
* "R0B vs the deployed scorer: PDMS +0.55 [+0.24, +0.84], NC −0.25, TTC −0.40 (both separated adverse), EP +1.81": confirmed from `freepoint_result.json` (claims b22, b26, b28, b29).
* "SELRULE-DAC +0.0235 PDMS = 0.7 % of the DAC oracle": +0.0235 / 3.184 = 0.74 %; the lower bound is +0.0002, which §7.21 prints.
