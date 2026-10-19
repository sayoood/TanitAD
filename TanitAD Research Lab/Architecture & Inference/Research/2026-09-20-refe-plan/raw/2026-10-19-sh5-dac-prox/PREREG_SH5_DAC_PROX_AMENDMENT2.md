# PREREG SH5-DAC-PROX: AMENDMENT 2 (2026-10-19). The CTX-REACHABLE bar becomes "lower CI > 0 AND point >= +0.02" on both seeds, because the sealed bar (lower CI > +0.02) cannot be cleared by the +0.03 effect SH5 was sized for; made from power arithmetic only, before any SH5 number exists

Amends `PREREG_SH5_DAC_PROX.md` (sealed 2026-10-19T13:39:48+02:00, sha256 `687e08c8468c16b6904d968a2c0316c76066841244718789f62c3f927d0a0839`) after its Amendment 1 (code only, sealed 2026-10-19T14:25:19+02:00, sha256 of the text `c0af04201cc439f8ce0298b763eece4f415264648da9c4c396cb3d4532ef9034`). Sealed at 2026-10-19T14:43:27+02:00. **This amendment changes ONE registered decision rule (CTX-REACHABLE), the S-shuf read that is defined by reference to it, and the code that implements them.** No data, arm, recipe, control, conversion threshold, NOT-REACHABLE rule, C-POWER band or committed next arm changes. Ordered by the Master Mind (`MM_RULING_SH5.md`, sha256 `77fbcffc1ca4811ac593526a16e54b5a51b69cae4083312dd1fe15e709455a25`, section 1).

## 1. The ruling (Master Mind, 2026-10-19 14:30, verbatim from `MM_RULING_SH5.md` section 1)

> **1. AMENDMENT 2: the CTX-REACHABLE bar.** The sealed bar (lower CI > +0.02 on both seeds) has MDE +0.050 on LOCAL and +0.035 on FULL, both ABOVE the +0.03 effect the Master Mind sized SH5 for (`MM_RULING_PRIVGEOM.md` §3). A bar the sized effect cannot clear is mis-specified, the same class as EP-ADMISSION's drafted A1 (`D-REFE-EP-ADMISSION-R1`).
> - The change is made BEFORE any SH5 number exists, from power arithmetic only.
> - **CTX-REACHABLE iff, on BOTH seeds, Delta_F lower CI > 0 AND Delta_F point ≥ +0.02.** The lower-CI conjunct gives significance; the point conjunct keeps materiality.
> - **NOT-REACHABLE unchanged:** upper CI < +0.01 on both seeds.
> - **INCONCLUSIVE:** everything else, including seeds that disagree.
> - The other calls (PICK-GAIN, GAIN-NOT-FROM-CTX, UNDERPOWERED, VOID) keep their sealed rules. GAIN-NOT-FROM-CTX applies the NEW CTX-REACHABLE rule to S-shuf.
> - Before Amendment 2 is sealed, its power is COMPUTED with `sh5_design_calc.py` on the same inputs that gave the sealed MDEs: P(CTX-REACHABLE | true +0.03) per seed and for both seeds, LOCAL and FULL. It is reported in the amendment.

(Section 2 of the ruling accepts S-shuf with GAIN-NOT-FROM-CTX, C-NULL-LABEL, the C-POWER conversion, A-TEST and the pre-flights; sections 3-4 accept Amendment 1 and put FULL to the PI. None of that changes.)

## 2. What changes in the prereg text (the rest stands verbatim)

| where | sealed text | after Amendment 2 |
|---|---|---|
| section 6, first bullet | CTX-REACHABLE iff `Delta_F` lower CI > **+0.02** on BOTH seeds | CTX-REACHABLE iff, on BOTH seeds, `Delta_F` lower CI > **0** AND `Delta_F` point >= **+0.02** (the point is the full-set estimate `AUC_F(L+S) - AUC_F(L)`, the same number the interval is built around) |
| section 6, NOT-REACHABLE / INCONCLUSIVE | upper CI < +0.01 on both seeds; otherwise INCONCLUSIVE | **unchanged** (a seed whose point is >= +0.02 does not sit under an upper CI < +0.01 unless the interval excludes its own point, and `seed_call` tests REACH first) |
| section 6, conversion 5 (GAIN-NOT-FROM-CTX) | S-shuf "ALSO passes the CTX-REACHABLE rule on both seeds" | the same sentence, now read with the **new** rule (code: `L.seed_call` is the one definition used for the raw call and for S-shuf) |
| section 7, C-NULL-LABEL (VOID) | a replicate FIRES iff `Delta_F` lower CI > +0.02 "(the CTX-REACHABLE rule)" | **the numeric rule is unchanged: lower CI > +0.02** (the ruling: VOID keeps its sealed rule). The parenthesis "(the CTX-REACHABLE rule)" is therefore no longer true; see section 5 |
| section 7, false-fire simulation, first row | read with lower CI > +0.02: 0 / 16 | re-read under the new rule from the banked rows: still **0 / 16** (`raw/amend2_power.json`, `falsefire_sim_banked_rows_reread_under_new_rule`; 0 / 16 replicates even have a lower CI above 0, their largest point is +0.0008) and 4 / 4 for the branch that reads the ctx |
| section 8 (power), the table and the "bar sits ABOVE" paragraph | MDE +0.035 FULL / +0.050 LOCAL; P(CTX-REACHABLE per seed \| +0.03) 0.26 / 0.09 | superseded for the CTX-REACHABLE rule by section 3 below; the NOT-REACHABLE / UNDERPOWERED paragraph and the statement that LOCAL cannot return a final NOT-REACHABLE **stand** |
| section 11 (forecast) | registered under the sealed rule | **not revised** (a prediction made before any number; the probabilities of an outcome shift with the rule, section 3 gives them at +0.03) |
| section 15, second bullet | "the committed CTX-REACHABLE bar is above the +0.03 the ruling sizes against: a PI / MM decision" | **resolved by this amendment** |
| everything else (data, sets, arms, recipe, folds, estimator, controls, C-POWER band, PICK-GAIN, committed next arms, limits) | | **unchanged** |

## 3. Power of the new rule (computed BEFORE sealing, `raw/amend2_power.json`, sha256 `4962c3af5cbdf177410a303c9c6af9184fbaeed769d4a791ba1d6be9a402ed7f`; script `code/sh5_amend2_power.py`; EVIDENCE CLASS: MEASURED inputs, ESTIMATED probabilities)

**Inputs: the same as the sealed MDEs.** PRIVGEOM-SAFE's banked sign-clean arm (`drop0`, +0.0276 [+0.0130, +0.0430] on all 4,941 sets) against its `L` row on FULL (41 logs) and on the LOCAL ctx subset (38 logs, 1,444 sets). The script re-derives `se = (hi - lo) / (2 x 1.96)` from the arrays and **asserts it equals `raw/mde.json`'s** (FULL 0.0076529, LOCAL 0.0154329), and records that the sealed rule's MDE and probabilities reproduce those of `raw/mde.json` to 1e-9 (FULL 0.0350 / 0.2568 / 0.0659, LOCAL 0.0502 / 0.0948 / 0.0090; `reproduces_sealed_MDE_inputs`). A point estimate `p` passes the sealed rule iff `p - 1.96 se > +0.02`, the new rule iff `p - 1.96 se > 0 AND p >= +0.02`, i.e. iff `p >= max(1.96 se, 0.02)`: **FULL: the point conjunct binds (1.96 se = 0.0150 < 0.02), a seed needs point >= +0.0200; LOCAL: the lower-CI conjunct binds (1.96 se = 0.0302 > 0.02), a seed needs point >= +0.0302** (sealed rule: +0.0350 / +0.0502).

**Between-seed correlation: STATED, NOT MEASURED** (there is no banked pair of seeds of the sign-clean arm). The sealed table's convention: each seed's point has marginal sd = se; the two seeds share the logs and the bootstrap draws, so the both-seed probability lies between rho = 1 (the two seeds identical: both-seed = per-seed) and rho = 0 (independent: the sealed table's "if independent"); rho 0.8 and 0.5 are intermediate rows. A second, harsher reading (**sensitivity**, `H-ESTIM-SEED-1`): training variance EQUAL to the sampling variance on top of it (point sd = sqrt(2) se, rho = 0.5, interval still se wide because the bootstrap holds the models fixed).

**P(CTX-REACHABLE | true Delta_F = +0.03)**

| set | rule | per seed | both seeds, rho = 1 | rho = 0.8 | rho = 0.5 | rho = 0 | nested bootstrap, per seed |
|---|---|---|---|---|---|---|---|
| FULL (se 0.00765) | sealed (lower CI > +0.02) | **0.257** | 0.257 | 0.175 | 0.125 | 0.066 | 0.230 |
| FULL | **new** (lower CI > 0 and point >= +0.02) | **0.904** | 0.904 | 0.862 | 0.839 | 0.818 | 0.903 |
| LOCAL (se 0.01543) | sealed | **0.095** | 0.095 | 0.053 | 0.030 | 0.009 | 0.124 |
| LOCAL | **new** | **0.494** | 0.494 | 0.391 | 0.327 | 0.244 | 0.623 |

*Nested bootstrap* (an outer bootstrap over the analysis set's logs, the registered percentile bootstrap inside each outer draw, the banked arm's interval relocated to +0.03; R = 3,000 x B = 2,000): a cross-check that does not assume a normal, symmetric interval. FULL agrees with the normal arithmetic; **LOCAL is higher than the normal figure (0.62 against 0.49) because the percentile interval of 38 logs is asymmetric** (lower distance 0.027, upper 0.034 at the banked arm). I quote the normal figure as the headline because it is the sealed convention, and the nested one as the other end of the range.

**Sensitivity (training variance = sampling variance, rho = 0.5), true +0.03, both seeds:** FULL new 0.718 (sealed 0.174); LOCAL new 0.329 (sealed 0.073).

**What the table says.** The sealed bar gave a true +0.03 effect a 26 % (FULL) and 9 % (LOCAL) chance per seed and 7 % / 1 % for two independent seeds, i.e. INCONCLUSIVE by construction. The new bar gives 90 % / 49 % per seed, and 82-90 % / 24-49 % for both seeds. **FULL is now powered for +0.03 (the rule's 80 %-power effect is +0.0264 per seed, sealed +0.0414). LOCAL is improved five-fold but is still a coin flip per seed (80 % power needs +0.0432, sealed +0.0632), and NOT-REACHABLE is still not a final call on LOCAL (section 8 stands).**

**P(false CTX-REACHABLE | true Delta_F = 0), the price of the change.** With the banked arm's se as the point's sd (CONSERVATIVE: an interval around a true zero is far narrower, `PRIVGEOM` RESULT 5.3 ~0.001 wide, and the 16 noise-trained CPU replicates scattered with sd 0.0006), per seed: FULL sealed 2.4e-6, **new 0.0045**; LOCAL sealed 5.7e-4, **new 0.0250** (the lower-CI conjunct IS a one-sided 2.5 % significance test, so on LOCAL a useless branch whose point scatters at the interval's own scale clears the rule 2.5 % of the time per seed); both seeds: FULL new 2e-5 (rho 0) / 4.3e-4 (rho 0.5) / 4.5e-3 (rho 1), LOCAL new 6.2e-4 / 4.6e-3 / 2.5e-2. Nested bootstrap at a relocated zero: FULL 0.0023, LOCAL 0.029 per seed. **Sensitivity (training variance = sampling variance):** per seed FULL new 0.032 (sealed 6.1e-4), LOCAL new 0.083 (sealed 0.011); both seeds at rho 0.5: FULL new 0.0066, LOCAL new 0.0248. The false-fire controls that guard it are C-NULL-LABEL (VOID, section 5), S-shuf (GAIN-NOT-FROM-CTX) and C-REACH; none of them is an estimate of the real rig's training variance, which is unmeasured (`H-ESTIM-SEED-1`).

## 4. The code change and its tests (`code/sh5_lib.py`, `code/test_sh5_lib.py`; `sh5_train_eval.py` is untouched)

- `sh5_lib.py`: two new constants `REACH_LO_MIN, REACH_POINT_MIN = 0.0, 0.02`; `seed_call` returns REACH iff `ci["lo"] > REACH_LO_MIN and ci["point"] >= REACH_POINT_MIN` (before: `ci["lo"] > INFO_MARGIN`), NOT iff `ci["hi"] < NOTREACH_MARGIN` (unchanged), else INC; `INFO_MARGIN` (0.02) stays and is now used only by `null_label_fires` (VOID's sealed rule); docstrings of `seed_call`, `null_label_fires`, `power_ok` updated; the table `branch_table_failures` grows from 39 to **57** checks. `analyze` already routes the raw call (`r["call"]`) and the S-shuf call (`r["call_shuf"]`) through `L.seed_call`, so **GAIN-NOT-FROM-CTX reads the new rule with no change to `sh5_train_eval.py`**; a test asserts the four lines of that wiring by source so it cannot be disconnected silently, and the existing `TestAnalyzeAllArms` runs both calls on the real `analyze` path (the CI dicts it builds carry `point`).
- **Each conjunct has its boundary, a planted case and a RED mutation.** Lower CI: `0` and `-1e-9` are not REACH, `1e-9` is. Point: `+0.0199` is not REACH, `+0.0200` is. **Planted cases:** "+0.019, significant" (lower CI +0.006) and "+0.005, significant" (lower CI +0.001, upper +0.012 => INC; upper +0.009 => NOT) are rejected by the point conjunct, so the mutation that DROPS the point conjunct passes them (test `test_red_dropping_the_point_conjunct_passes_planted_immaterial_cases`); "+0.025, lower CI -0.003" and "+0.040, lower CI exactly 0" are rejected by the lower-CI conjunct, so the mutation that DROPS the lower-CI conjunct passes them (`test_red_dropping_the_lower_ci_conjunct_passes_a_planted_insignificant_case`). (The ruling's parenthesis pairs the planted cases with the opposite conjuncts; the logical pairing above is what is tested, both ways, and both mutations are RED.) Also RED: each comparator flipped, `and` -> `or`, each threshold moved, **the pre-amendment rule `ci["lo"] > INFO_MARGIN` restored (the table must fail)**, and the VOID rule swapped to the new rule (VOID must keep its sealed rule). The mutation list grows from 13 to **23**.
- Tests: `test_sh5_lib.py` 33 -> **43** (`TestAmendment2Rule` 7, `TestAmend2PowerArithmetic` 3); `test_sh5_branch.py` 17, `test_sh5_ctx.py` 16, `test_sh5_guard.py` 4 and `test_sh5_pipeline.py` 15 are unchanged files. Totals and the full run are in `raw/tests_summary_amend2.json` (the evidence behind the `red_table` gate).

## 5. Disclosed and put to the Master Mind: VOID keeps its sealed numeric rule, so it is now LESS sensitive than the call it guards

The ruling keeps VOID's sealed rule. The sealed text defined the C-NULL-LABEL fire by pointing at the CTX-REACHABLE rule ("lower CI > +0.02 (the CTX-REACHABLE rule)"); after this amendment the two are different rules, and the new CTX-REACHABLE rule is the looser one: a noise-trained replicate with point >= +0.02 and 0 < lower CI <= +0.02 would be a false CTX-REACHABLE that does **not** void the run. I implemented the ruling literally (`null_label_fires` unchanged, pinned by a table check and a mutation) and did not choose the alternative. The numbers for the choice (`raw/amend2_power.json`, `void_sensitivity`; three replicates on one seed's interval each, which share the logs, so the truth lies between the bounds):

| set | P(any of 3 noise-trained replicates voids), SEALED rule | P(any of 3 would satisfy the NEW rule), i.e. the false-alarm cost if VOID followed it | same, training variance = sampling variance |
|---|---|---|---|
| FULL | 2.4e-6 .. 7.2e-6 | 0.0045 .. 0.0134 | sealed 0.0006 .. 0.0018; new rule 0.032 .. 0.094 |
| LOCAL | 5.7e-4 .. 1.7e-3 | 0.0250 .. 0.0731 | sealed 0.011 .. 0.032; new rule 0.083 .. 0.229 |

So VOID following the new rule would catch a leak that the sealed numeric VOID cannot, at the price of voiding a CLEAN machinery 0.5-1.3 % (FULL) or 2.5-7.3 % (LOCAL) of the time, and more if the real training variance is large. That is a Master Mind decision; if it is taken, it is a one-line change in `null_label_fires` plus the table check, i.e. an Amendment 3 before any number exists. Until then VOID keeps the sealed rule.

## 6. What I did not do, and what is still UNVERIFIED

- No SH5 number exists: no branch has been trained, no `Delta_F`, OOF logit or call has been computed on real data. At the time of writing the LOCAL ctx build (launched by the Master Mind at 14:27 from the sealed-state code) holds the GPU lock and had written 46 of 816 tokens (`run/ctx_local.log`: the last lines, which carry counts and 'bad 0' only, read at 14:38); **I did not read `fid.jsonl`, the ctx, or any other output of that build**, and the build's C-FID numbers (a property of the ctx) cannot inform a bar on `Delta_F`. The power arithmetic reads only PRIVGEOM-SAFE's banked arrays and the banked CPU false-fire rows, both disclosed in the sealed prereg (section 0.2).
- The between-seed correlation is an assumption, bracketed (rho 0 to 1) and not measured. The false-fire probabilities rest on the banked arm's se as the point's spread, which is itself conservative for a useless branch and not a measurement of the real rig's training variance.
- The nested bootstrap relocates the banked arm's interval; it is a check on the interval's shape, not an independent estimate of power.
- This amendment is sealed in a working copy (`par18/sh5_a2`); the Master Mind installs it into the live package after the ctx build (`SWAP.md`). Until it is installed, the live guard still accepts the Amendment-1 code and the sealed bar: **no stage may produce an SH5 number from the live package before the swap.**

## 7. Root-cause class (for `RETRACTION_LOG.md`, which this agent does not edit)

*A committed bar was not checked against the effect the programme sized for.* The prereg computed the MDE of its own bar (section 8) and flagged the mismatch (section 15) but, correctly, did not move a goalpost the Master Mind had fixed; the bar itself was drafted before the sizing ruling and never power-checked against +0.03. Same class as EP-ADMISSION's drafted A1 (`D-REFE-EP-ADMISSION-R1`). The rule adopted: **before sealing, compute P(each decisive call | the effect the programme is sized for) per seed with the sealed estimator's own se; a call the sized effect cannot reach at >= 50 % per seed on the headline set is a mis-specified bar, not a conservative one.** (On LOCAL even the amended bar is at 49 %: it is at the line, and FULL is the variant that is powered.)

## 8. Re-pinned code (the guard accepts exactly these new pins, from exactly these old pins) and the checkers

| file | pin before | pin after |
|---|---|---|
| `code/sh5_lib.py` | `08781b86b7a21d016f29dd47ee82d029a7f50ec3d4070d10f33c5610abdada9e` | `f7901f2a1b53926f34a9414f39576300208af0aaf7f6a139c314f93c3b7d88fa` |
| `code/test_sh5_lib.py` | `f0ebf31f19d056a38a4788b34319c2514715b79c9817dd5b82b5650cd3778f7f` | `2566dcc5cde60f6db8488d7d6c780c09ccb1ca8f14dbefc3d1b77cea1e38747a` |

The other fourteen pinned files keep their sealed or Amendment-1 pins. The guard (`sh5_guard.py`) already walks `PREREG_SH5_DAC_PROX_AMENDMENT<k>.md` + `.sha256` + `CODE_PINS_AMENDMENT<k>.json` in order, so it needed no change: **the pre-amendment `sh5_lib.py` / `test_sh5_lib.py` (pins `08781b86...` / `f0ebf31f...`) now refuse; the amended files accept; an Amendment-2 text or pin file edited or missing refuses** (`raw/amend2_guard_proof.json`, executed against copies, at the guard function and at the real `analyze` / `fit` entry points). The pre-amendment copies are kept in `sealed_state_amend1/` (`ARTIFACTS_D.json`).

| checker (not pinned) | sha256 now |
|---|---|
| `code/sh5_guard.py` | `ac7b35e788ffe228ee1e580517e28aa083b7a30bd546a48cc93550787431286f` |
| `code/seal.py` | `bf08e3ca19cf7aefa9b96477652f0aa57bc85e6cdb860bc976eb6423e26118a8` |
| `code/seal_amendment.py` | `cc221c30c8676cae90fb46936f5f33d9087e45206db837062c563585d7d5d4ff` |
| `code/test_sh5_guard.py` | `ef2d66166e77f61c99c732b8e9e853ff240d90d32850ddf76bfa3fbe9999b511` |

Evidence and tooling not pinned: `code/sh5_amend2_power.py` (its sha256 is recorded inside `raw/amend2_power.json`), `code/amend2_guard_proof.py`, `code/build_package.py` (its `ROOT` is now derived from its own location: it was hardcoded to `par18/sh5` and removed `ROOT/out` on every run, so running a copy would have deleted the original staging), `raw/amend2_power.json`, `raw/amend2_guard_proof.json`, `raw/tests_summary_amend2.json`, `raw/logs/run_tests_amend2.log`.
