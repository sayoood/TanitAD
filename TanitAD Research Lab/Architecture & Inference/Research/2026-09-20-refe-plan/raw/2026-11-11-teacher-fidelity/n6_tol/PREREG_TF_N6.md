<title>PREREG TF-N6</title>

# PREREG TF-N6: a no-harm tolerance on the residual DAC-0 tokens, with the human-log fallback, on top of TF-N5's N5-A

**Status: FROZEN by the REFe Master Mind 2026-11-13 (sha256 in `PREREG_TF_N6.sha256`; ledger entry `TF-N6:<sha256>:E3`, n = 49) before any N6 build or read on navtest. Drafted by an engineering agent; MM rulings in section 9.** Nothing in this card has been scored on navtest. Built and tested on navtrain and synthetic data only (`chain/CHAIN_N6.md`, `chain/test_run_n6_chain.log`, `chain/smoke/`).

## 1. Question (PI decisions, verbatim, 2026-11-13)
Teacher Fidelity target: **paper level on navtest, PDMS >= 95.8 AND DAC >= 99.9.** Current best, N5-A (`../n5_l1/a1_run/out/n5_result.json`): PDMS **96.367** [96.027, 96.680], DAC **99.835** [99.695, 99.941], **20 zero-DAC tokens** on the primary 12,110 (paper level needs <= 12). Evidence class MEASURED, tier T1.
The PI approved two levers:
- human-log fallback on residual DAC-0 tokens: *"yes you can replace if it will help the overall perfroamnce"*;
- a no-harm tolerance: *"ok accepüt slightly worse"*. MM definition: NC and DDC never worse; TTC and C may be worse; the token's builder PDMS must improve.

**Question: do the 20 residual tokens, repaired under that tolerance (teacher candidates first, the logged human future only where no teacher candidate passes), lift N5-A to paper level without harming NC or DDC anywhere?**

## 2. Population, scorer, tier, estimator (unchanged from TF-N5 A1)
- Navtest, primary 12,110 tokens over 136 logs; the full 12,146-token seam is scored (the 36 non-primary tokens are in the seam, not in the statistics).
- Scorer of record: the pinned NAVSIM v1.1 harness (W3, `score_p4_a1.py` -> `score_e.py`), plans executed with its LQR tracker. **Tier T1.** Evidence class MEASURED. Builder-side scores are the same scorer on the harness metric cache and are used only to certify.
- Estimator: paired log-cluster bootstrap over the 136 logs, 10,000 draws, seed 20260927, 95 % percentile (`Boot` of `tf_n1_analyze_a1.py`, checked against `eval/proxy_eval.boot` by C-EST), plus the Bonferroni interval at K = 49. It answers "would another draw of LOGS say this?". The arms are deterministic (C-DET, TF-N5 A1: 116/116 bit-identical), so it is silent on training and inference variance, which do not exist for these arms.

## 3. Arms (frozen by the MM; the numbers live in `chain/n6_nt_build.py`, `CARD6`, `CERT_BASE`)
Start: the banked N5-A plans and seam (`inputs/n5a/n5_plans_N5A.npz`, sha256 `9da77557…`; `refe_tfn5_n5a.npz`, `2a7b310a…`; harness CSV `refe_tfn5_n5a.csv`, `5e0820e0…`). **Residual set R = tokens where N5-A is DAC = 0 under the builder tracked scorer**, found by scoring the N5-A plan on all 12,146 tokens (C-DACMAP6). It must equal the banked TF-N5 A1 result's 20 primary zero-DAC tokens (C-SET: builder set == `n5_build_report.unrepaired_l1.N5A` == harness N5-A zero-DAC set).

On each token of R:
- **Teacher candidates:** every iterate that REPAIR-D (TF-N4 A1 settings: margin 0.05, cap 5, sign +1, full certificate) and L1 (TF-N5 frozen parameters: margin 0.05, cap 5, centre 0.05, fallback margin 0.005, damp 0.5, full certificate) score, in generation order (REPAIR-D k = 0..5, then L1 k = 0..5), de-duplicated by plan bytes (first kept; iterate 0 is the unrepaired N2-R plan = the N5-A plan). Both repairs are the banked functions run unchanged through a recording proxy of the scorer (`Rec`); nothing was re-implemented. They are deterministic (C-DET: every capture is run twice, bit-identical).
- **Tolerance certificate** (builder tracked scorer, against the N5-A builder score `ref` of the token): **DAC = 1; NC_cand >= NC_N5A; DDC_cand >= DDC_N5A; builder PDMS_cand > PDMS_N5A. TTC and C are free.** Among passing teacher candidates take the maximum builder PDMS; ties go to the earliest (REPAIR-D before L1, lower k).
- **Human fallback:** only where NO teacher candidate passes: the logged ego future of the nuPlan DB (the label-ego DB route: C-LABELEGO), 20 rear-axle poses at 0.2 s in the ego frame (`transp_score.human_rel`, sliced `[1::2][:20]` exactly as the harness human arm does), converted by the same seam function `ND.to_navsim` as every other plan, and it must pass the SAME certificate. Else the token keeps its N5-A plan.

| arm | order | certificate |
|---|---|---|
| **N6-H** (PRIMARY) | teacher, then human | DAC = 1, NC, DDC not worse, builder PDMS better, TTC/C free |
| **N6-T** (attribution) | teacher only | same |
| **N6-X** (deliberate regression) | teacher, then human | **inverted: DAC = 0** required, otherwise the same |

N6-X must change 0 tokens (a DAC-0 plan has builder PDMS 0, which cannot exceed the DAC-0 reference's 0), so its seam is bit-identical to N5-A (C-X0, C-UNCH). **It is therefore not scored**: its harness rows are the banked N5-A rows, and the read verifies the file identity (C-REG). Identity outside R for every arm: plans and seam rows bit-identical to N5-A (C-UNCH, C-ID-C).

**Naming rule (every quoted number carries it):** *"RJ5 teacher + P5-R + REPAIR-D + L1 + tolerance (NC/DDC kept, TTC/C free) + human-log fallback on N of 12,110 tokens"*, with **N_teacher_tol and N_human reported separately** (the read writes both). **The human-log replacement is not teacher output**; any number for N6-H includes it, N6-T is the teacher-only number.

## 4. Bars (MM-set; the MM freezes)
1. **PRIMARY = paper level for N6-H** on the primary 12,110: **PDMS >= 95.8 AND DAC >= 99.9** (point estimates; <= 12 zero-DAC tokens), with the paired log-cluster bootstrap CIs vs N5-A reported (own CIs at K = 1 and Bonferroni K = 49).
2. **No harm:** the paired delta (N6-H minus N5-A) of NC and of DDC, point estimate >= 0, **and** no changed token's harness NC or DDC below N5-A's on that token. TTC and C deltas are reported (the declared tolerance), not bars.
3. **Certificate integrity:** on every changed token the harness reads DAC = 1 where the builder certified; the agreement count is reported (bar: every changed token), mismatches named.
4. **Four-family guard, N6-H vs N5-A** (executed_tracked, paired `fam_ci`): speed bias <= 0.25 m/s, speed MAE <= 0.15 m/s, cross-track MAE <= 0.05 m (differences). All four families are reported (LONGITUDINAL, LATERAL, TACTICAL; **STRATEGIC = PAUSED**, PI 2026-09-19, never silently dropped, never a pass). A flag goes to the PI; it does not flip bars 1 to 3.
5. **K:** ledger entry `TF-N6:<sha>:E3`, n = 49 (TF-N5 A1 = 48). Bonferroni intervals at K = 49.

**Outcome rule:** CONFIRMED iff bars 1, 2 and 3 hold. Otherwise NOT CONFIRMED, with the branch names in card order: `paper_level_not_reached`, `harm`, `cert_integrity`. A committed bar that is missed FAILED and is reported so.

## 5. Both outcomes, and the next lever for each (Rule Zero)
- **REACHED (CONFIRMED):** the teacher-fidelity target is met **with a human-log fallback on N_human tokens**. Report it under the naming rule, never as "the teacher reaches paper level". Next: (a) quote N6-T separately: if N6-T alone is below the bar, the human replacements carry it and the label set is a teacher+human mix; the PI decides whether that is admissible as a teacher claim. (b) The human-sourced tokens go to the family guard and to a tracker-aware check (the human plan is a logged trajectory, not a plan the tracker was asked to follow). (c) Next lever for quality: a navtrain-wide replication of the tolerance (the rate of tolerance rescues outside navtest).
- **NOT REACHED:** the build report (`residual_table`) names, per remaining token, the blocking class: *no candidate (teacher or human) reaches DAC = 1* (the corridor is narrower than the footprint plus margin, or the human also leaves the drivable area), or *DAC-1 candidates exist but fail clause X* (NC / DDC / PDMS). Next lever by class: **(i)** NC/DDC-blocked with a DAC-1 candidate -> a bounded NC/DDC tolerance the PI must set (it trades a collision-class metric: PI decision, blocked on the PI); **(ii)** no DAC-1 candidate at all -> a wider candidate generator (REPAIR-D / L1 at cap 20 and margin 0.20 m were measured to rescue 4 and 8 of 25 on the navtest anatomy; navtest-seen, so it must be re-derived on navtrain first) or the human future with a shifted-lateral certificate search; **(iii)** the human future fails DAC itself -> those tokens are not repairable from logs and stay as the honest residual: report the exact count and the DAC ceiling it implies.
- Either way the turn is not finished at the verdict: the remaining tokens and their classes are the next arm's input.

## 6. Controls (all must pass; each has a deliberate-regression arm in `chain/test_n6_chain.py`)
Build: **C-COVB** (12,146 rows, no errors); **C-DACMAP6** (builder DAC of the N5-A plan == harness N5-A DAC on all tokens); **C-DERIV** (200 smallest-sha1 tokens: the label re-derived from the RJ5 target == the banked plan, max dev <= 1e-9 m); **C-LABELEGO** (every label build and every human future used the nuPlan-DB ego); **C-SET** (builder residual == banked 20 == harness zero-DAC set); **C-REPLAY** (the re-run REPAIR-D and L1 return no plan on every residual token, as banked; iterate 0 == the banked plan); **C-DET**; **C-SELECT** (the card selector re-run in the merge on the stored candidates picks exactly what the derive recorded); **C-CERT-B6** (every applied plan, re-scored by a separate call, reproduces its score and passes the arm's card certificate); **C-ORDER** (N6-H == N6-T wherever N6-T applies; human only where no teacher passes); **C-X0**; **C-SEAMFN** (`to_navsim` reproduces the banked N5-A seam on all 12,146 tokens); **C-UNCH** (files); **C-REACH** (REPAIR-D, L1 and the three selectors read back and behave as the card; two real navtrain fixtures).
Read: **C-COV, C-BUILD, C-UNCH, C-PIN** (+ the families TIP sha), **C-CERT, C-ID-C** (full set + 200 sentinels), **C-EST, C-REG**. If any fails the read writes no aggregate.
Chain guards: card sha, ledger (n = 49, TF-N5 A1 n = 48 present, nothing higher), pins, TIP `families_run.py` sha `44c62807…`, metric-cache listing, free disk, reach.

## 7. Disclosure (what was seen before this card)
- **The 20 residual tokens were read twice before:** in the TF-N4 A1 anatomy (`../n4_dac/a1_residual/ANATOMY_25.md`: 25 tokens; causes and which clause blocked the guard-blocked 5: C x3, TTC x1, NC + TTC x1) and in TF-N5 A1 (L1 status on them: CERTIFIED 5, CAP_EXHAUSTED 13, INFEASIBLE 5, CERT_BLOCKED 2 over the 25). The tolerance and the human fallback were chosen with that knowledge; the card is therefore a **navtest-seen** design and its result is a confirmation read, not an independent test.
- **Deliberately not looked at:** the banked human arm's DAC on these 20 tokens (`refe_p4_human_navtest.csv`). Whether the human future passes the certificate on navtest is UNVERIFIED. On the navtrain smoke the human candidate read DAC = 1 on 7 of 7 DAC-0 instances (`chain/smoke/`), which says nothing about navtest.
- **The human-log replacement is not teacher output** and is named as such in every quoted number.
- **Human plan format (MEASURED, navtrain smoke):** the seam function `to_navsim` linearly interpolates the 0.2 s poses to 0.5 s; against the logged 0.5 s poses the difference is <= 1.14 cm in position and <= 1.4e-3 rad in heading on 7 of 7 instances (`chain/smoke/hum_interp.json`).
- ESTIMATED from the anatomy (not a prediction to be used): the tolerance can only rescue tokens whose full-certificate block was TTC/C (4 of the 5 guard-blocked in TF-N4's anatomy, minus the one L1 certified); the other residual tokens are two-sided / infeasible corridors where no teacher iterate reaches DAC = 1, and only the human fallback can act there. Paper level needs 8 or more of the 20 resolved.
- Cost: ESTIMATED 2.5 to 3.5 h CPU (derive 0.7 to 1 h on all 12,146 tokens; two harness seams in parallel 1.2 to 1.7 h; families 40 to 50 min).

## 8. Files
Hold `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/n6_tol/`: this card, `chain/` (code, tests, pins template, launch template, `CHAIN_N6.md`), `HANDOFF.md`. Run tree for the launch: `C:/Users/Admin/n6nt/run` (or a copy on F: when it has room).

## 9. MM rulings at freeze (2026-11-13)
1. **Arms, bars and the tolerance are as the MM set them in the build brief** (PI decisions verbatim in section 1); the draft's text was checked against that brief and is unchanged except this section, the title and the status line.
2. **N6-X is not scored: accepted.** Its seam is identical to N5-A by construction (no DAC = 0 candidate can beat a DAC = 0 reference of builder PDMS 0); a structural zero is an identity, not an estimate, and C-X0 / C-UNCH / C-REG verify the file identity.
3. **TIP `families_run.py` taken from the hold `n3_dose/raw/instr/` (sha256 `44c62807...`): accepted.** The guard checks the sha, which equals the git tip blob's (TF-N5 A1 C-PIN).
4. **The human-log fallback counts toward the paper-level bar only under the naming rule**, with N_teacher_tol and N_human reported separately; N6-T is reported beside N6-H in every summary.
5. **Run tree on C:** `C:/Users/Admin/n6nt/run`, `--min-free-gb 5` (F: was below its 20 GB floor at freeze because of other sessions' data).
