<title>PREREG TF-N7 (draft)</title>

# PREREG TF-N7 (DRAFT, not frozen): a wider teacher candidate generator on the residual DAC-0 tokens, teacher only, no human fallback

**Status: DRAFT written 2026-11-13 by the engineering agent for the REFe Master Mind. The MM freezes it (sha256 into the chain's `--card-sha256`, ledger entry `TF-N7:<sha256>:E3` n = 50) before any navtest read.** Nothing here has been built, scored or read on navtest. The generator and every parameter were developed on navtrain only (`DEV_N7.md`, `raw/`). Chain code for N7 (an `n7_nt_build.py` adapting the TF-N6 chain to the cascade in `n7_select.py`) is NOT written: that is the integration step, listed under "Next engineering".

## 1. Question
TF-N6 (`../n6_tol/run/RESULT_N6.md`, MEASURED, T1): paper level (PDMS >= 95.8 AND DAC >= 99.9) was reached only with a human-log fallback on 16 of 12,110 tokens. **The teacher alone (N6-T) is PDMS 96.39, DAC 99.87 with 16 zero-DAC tokens; paper level allows <= 12, so the teacher needs >= 4 more tokens at DAC = 1.** On 15 of the 20 N5-A residual tokens no REPAIR-D or L1 iterate (12 candidates) reaches DAC = 1: the limit is the candidate generator, not the certificate.

**Question: does a wider teacher candidate generator (the navtrain-developed cascade of `DEV_N7.md`), under the unchanged TF-N6 tolerance certificate and with no human fallback, bring the teacher alone to paper level, without harming NC or DDC anywhere?**

## 2. Population, scorer, tier, estimator
Unchanged from TF-N6: navtest primary 12,110 tokens over 136 logs, the full 12,146-token seam scored; the pinned NAVSIM v1.1 harness (W3) with the LQR tracker, **tier T1**; MEASURED; paired log-cluster bootstrap over 136 logs, 10,000 draws, seed 20260927 (it answers "would another draw of logs say this?", not another training run: the arms are deterministic, C-DET); own CIs at K = 1 and Bonferroni at **K = 50**.

## 3. Arms
Start: the banked N5-A plans and seam, as in TF-N6. Residual set R = tokens where N5-A is builder-DAC 0 (20 tokens; C-DACMAP6, C-SET as in TF-N6). On each token of R run the **cascade** below; a token outside R is bit-identical (C-ID-C).

**Candidate generator (frozen; parameters are `n7_lib.LEVERS` and `n7_select.TIERS`, set on the 17 navtrain residual tokens, none from navtest):**
| tier | candidates |
|---|---|
| T0 | TF-N6 as is: REPAIR-D and L1 iterates, cap 5, margin 0.05 |
| T1 | REPAIR-D and L1 iterates to cap 20, margins 0.05, 0.10, 0.20 |
| T2 | time-stretch of the label (s = 0.95, 0.90, 0.85, 0.80) and of the target with the label rebuilt |
| T3 | lateral offsets of the label, dy = +-{0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0} m x ramp {0.2, 1.0, 2.0} s; and a stretch (0.9, 0.8) with the same offsets |
| T4 | deep stretch of the label, s = 0.7, 0.6, 0.5 (floor 0.5; 0.4 to 0.2 are NOT used) |

**Certificate (unchanged from TF-N6, builder tracked scorer, against the token's N5-A builder score):** DAC = 1; NC and DDC not worse; builder PDMS strictly better; TTC and C free. **Selection:** in the FIRST tier with a certified candidate, the maximum builder PDMS, ties to the earliest generation order. A token certified by T0 keeps the TF-N6 plan (no later tier can touch it).

| arm | tiers | human fallback | certificate |
|---|---|---|---|
| **N7-T** (PRIMARY) | T0 to T4 | **none** | as above |
| **N7-T3** (attribution) | T0 to T3 | none | as above |
| **N7-X** (deliberate regression) | T0 to T4 | none | inverted: DAC = 0 required; must change 0 tokens, bit-identical to N5-A (C-X0, C-UNCH), not scored, file identity read (C-REG), as N6-X |

**Naming rule:** *"RJ5 teacher + P5-R + REPAIR-D + L1 + wider generator (tiers T0 to T4) + tolerance (NC/DDC kept, TTC/C free), teacher only, N_teacher_tol of 20 residual tokens"*. N7-T3 says "tiers T0 to T3". There is **no human-log token in either arm**. The T4 tier edits the label speed by 40 to 50 %; every number for N7-T names T4, and N7-T3 is always reported beside it.

## 4. Bars
1. **PRIMARY = teacher-only paper level for N7-T** on the primary 12,110: **PDMS >= 95.8 AND DAC >= 99.9** (point estimates; **<= 12 zero-DAC tokens**), paired CIs vs N5-A and vs N6-T reported.
2. **No harm** (N7-T minus N5-A): NC and DDC deltas, point estimate >= 0, **and** no changed token with harness NC or DDC below N5-A's on that token. TTC and C deltas are reported (the declared tolerance), not bars. **Plus no regression against N6-T:** every token N6-T certified (4 of 4) is bit-identical in N7-T (the cascade guarantees it; C-N6SAME).
3. **Certificate integrity:** on every changed token the harness reads DAC = 1 where the builder certified (bar: every changed token; mismatches named).
4. **Four-family guard, N7-T vs N5-A** (executed_tracked, paired `fam_ci`): speed bias <= 0.25 m/s, speed MAE <= 0.15 m/s, cross-track MAE <= 0.05 m (differences). LONGITUDINAL, LATERAL, TACTICAL all reported; **STRATEGIC = PAUSED** (PI 2026-09-19), never a pass, never dropped. Added for this card: **the changed tokens' own longitudinal change** (speed MAE of the T4 tokens, reported per token), because T4 edits speed. A flag goes to the PI; it does not flip a bar.
5. **K:** ledger `TF-N7:<sha>:E3`, n = 50 (TF-N6 = 49).

**Outcome rule:** CONFIRMED iff bars 1, 2, 3 hold. Otherwise NOT CONFIRMED with the branch names, in card order: `paper_level_not_reached`, `harm`, `cert_integrity`. A missed committed bar FAILED and is reported so.
**Secondary (decides nothing):** N7-T3 against the same bar. If N7-T reaches paper level and N7-T3 does not, the result rests on the slow-down tier and is quoted so.

## 5. Both outcomes, and the next lever (Rule Zero)
- **REACHED (CONFIRMED):** the teacher alone reaches paper level on navtest, with the generator of `DEV_N7.md`. Report N_teacher_tol and the tier counts, and whether T4 carried it (N7-T vs N7-T3). Next: (a) apply the same cascade to the navtrain labels (the residual DAC-0 labels of the full 16,133 + the 877 rolled logs) so the P5-C successor trains on a no-zero-DAC teacher label set; (b) the tracker-aware check on T3 / T4 edited labels (large offsets, slow-downs) before any student sees them; (c) the PI decides whether a speed-edited label is teacher output.
- **NOT REACHED (`paper_level_not_reached`):** the build report names per remaining token the blocking class: *no candidate in any tier reaches DAC = 1* (stationary or boundary-hugging start, the navtrain `229a59137a705430` class), or *DAC-1 candidates exist but fail clause X* (NC: an agent in the way; DDC: base DDC 0.5 with laterally-shifted candidates at 0). Next lever by class: (i) **no DAC-1 candidate:** a map-feasible planner search (sample paths in the drivable corridor, score by the builder, not an iterate of the target); (ii) **NC-blocked:** a joint longitudinal + lateral search with the agent futures; (iii) **DDC-blocked:** a PI decision on the DDC clause (not changed here); (iv) widen T4's floor only with PI consent. Either way the turn is not finished at the verdict.
- **Harm (`harm`)**: name the tier of every harmed token, drop that tier, re-run the build only (the harness read of unchanged tokens is bit-identical).

## 6. Controls (all must pass; each with a deliberate-regression arm in the chain tests, as TF-N6)
Build: C-COVB, C-DACMAP6, C-DERIV, C-LABELEGO, C-SET, C-REPLAY, C-DET (the cascade run twice, bit-identical), C-SELECT (readback of the frozen `LEVERS` and `TIERS` equals this card; a flag with a disabling default is named in argv), C-CERT-B7 (the certificate readback equals CARD6 N6T), C-ORDER (tier order and tie rule), C-X0, C-UNCH, C-N6SAME (the 4 N6-T tokens identical), C-REACH (every tier executes on a real token; a mutated tier list drops its tokens). Read: C-COV, C-BUILD, C-UNCH, C-PIN, C-CERT, C-ID-C, C-EST, C-REG. Chain guards as TF-N6, with ledger n = 50.

## 7. Disclosure
- **The navtest residual is read for the FOURTH time.** The 20 residual tokens were read in the TF-N4 A1 anatomy, in TF-N5 A1, and in TF-N6 (where 15 of 20 were named as having no DAC = 1 iterate, 12 candidates each). This is a navtest-seen design: the generator was built on navtrain only, but the question is asked of tokens whose difficulty is already known. **A CONFIRMED here is a confirmation read, not an independent test.**
- No navtest token informed any parameter: the lever grids, cap 20, margins 0.05 / 0.10 / 0.20, the stretch grid, the offset grid, the T4 floor 0.5 and the tier order are all fixed from the 17 navtrain residual tokens (`DEV_N7.md`; n = 17 on 7 logs, small; the tier order and floor were chosen on these same 17: a navtrain-internal tuning, no held-out navtrain residual exists yet).
- Navtrain result the card rests on (MEASURED, builder scorer): TF-N6 generator 2 of 17 certified; cascade T0 to T3 13 of 17; with T4 16 of 17; 0 harmed of the 73 already repaired. The navtest residual is a different class mix (two-sided oscillation 16 of 20, guard-blocked 5 of 25; INHERITED from ANATOMY_25, not re-verified), so the navtest rescue rate is **ESTIMATED**, not inherited from 13 / 17.

## 8. Expected cost (ESTIMATED)
- Generator: the cascade costs about 1 CPU-min per residual token (navtrain: 17 tokens x all levers about 15 to 75 s each, `raw/`), so under 0.5 CPU-h for the 20 residual tokens.
- Chain: as TF-N6 (derive 0.38 CPU-h, build, harness scoring of the seam; N6 est. 2.5 to 3.5 h CPU), CPU only on the dev box, no GPU, no pod. Scoring only the changed tokens would be cheaper but the card keeps the full-seam read (C-ID-C needs it).
- Disk: as N6 (C: about 6 GB free; the run tree 129 MB).
- Student labels: after CONFIRMED, the cascade on the 877 other navtrain logs is a derive-only job (no GPU).

## 9. Next engineering (integration, for the MM)
1. Write `n7_nt_build.py` (derive, merge, the controls above) from `../n6_tol/chain/n6_nt_build.py`, replacing `capture` by the cascade (`n7_select.cascade`) and adding C-N6SAME and C-REACH; the levers are importable from this hold (`n7_lib`).
2. Pin the new files (`n6_make_pins.py` style), freeze this card, ledger n = 50, launch.
