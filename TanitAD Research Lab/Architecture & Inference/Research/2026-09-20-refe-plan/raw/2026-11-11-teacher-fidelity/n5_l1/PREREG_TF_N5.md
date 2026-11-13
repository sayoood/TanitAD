<title>PREREG TF-N5</title>

# PREREG TF-N5 (FROZEN 2026-11-13 by the Master Mind, before any N5 navtest number): L1 two-sided lateral repair on the zero-DAC tokens REPAIR-D leaves, one navtest read through the pinned harness

**Status: FROZEN by the Master Mind 2026-11-13 (sha256 in `PREREG_TF_N5.sha256`), before any N5 arm was built or scored on navtest. Drafted by the Research Lab agent; the MM's amendments at the freeze are listed in section 11. Numbers marked PROPOSED in the draft are ADOPTED as written unless section 11 says otherwise. Nothing here was set with a navtest number: L1's parameters are the pre-data defaults of `n4_repair_l1.py` (sha256 `7af0d3dd52becb63...`; full hash in the pins at freeze) and were not tuned on navtrain either (there was nothing to tune on: see section 1).** Format follows `PREREG_TF_N4_A1.md`. Everything not named here is unchanged from TF-N4 A1: population, scorer of record, estimator, base arm N2-R, the label pipeline (Tc20s + P5-R lambda 3.0 on the nuPlan-DB ego), the certificate.

## 0. Headline (decisions taken at the freeze)
1. **Navtrain gives parity, not superiority.** REPAIR-D (A1) and L1 repair every failing instance in every set (DEV30 30/30 both; RJ5 re-roll 1/1 both; three CONSTRUCTION stress sets 45/45, 30/30, 23/23 both), 0 certificate violations, the sign-flipped L1 arm repairs 0 of 129. No navtrain set in which REPAIR-D fails exists (`L1_DEV.md`). The ping-pong class is therefore a navtest-only observation (seen-on-navtest, descriptive). The N5 read would be the first discriminating test of L1. This is a named blocker, not a hidden one.
2. **Recommended arm: N5-A = N4-D plus L1 applied only to the tokens where REPAIR-D leaves tracked DAC = 0 (25 primary).** Not "L1 replaces REPAIR-D" (N5-R, builder-side count only). Why: (a) parity on navtrain gives no evidence for replacing; (b) the add-on keeps the 91 repaired plans bit-identical to banked N4-D, so the N5 minus N4-D contrast lives on at most 25 tokens and the no-harm control is an identity check; (c) L1 costs nothing on tokens REPAIR-D already repaired.
3. **PAPER LEVEL is reachable only if L1 repairs at least 13 of the 25** (DAC >= 99.9 allows at most 12 zero-DAC tokens), and 5 of the 25 are certificate-blocked (C x3, TTC, NC+TTC), out of scope for L1. So 13 of the 20 non-guard residuals are needed (65 %). I expect this to be hard (ESTIMATED, unproven). The five guard-blocked tokens are the PI-visible tolerance question (lever L3), not part of this card.
4. **Not taken (MM decision at the freeze):** a descriptive navtest pre-run of L1 on the 20 non-guard residuals. It would spend the out-of-sample read. L1's parameters are untuned pre-data defaults, so this card makes navtest L1's first test.

## 1. Question
TF-N4 A1 left "RJ5 teacher + P5-R + REPAIR-D" (N4-D) at PDMS 96.33 [95.98, 96.64], DAC 99.79 [99.64, 99.91], 25 zero-DAC primary tokens (`RESULT_N4_A1.md`, MEASURED, harness, navtest). The anatomy of those 25 (`ANATOMY_25.md`, descriptive, seen on navtest): 5 certificate-blocked; of the other 20, the return-vector sign flips >= 2 times in 16 while the executed path follows the shift in 15: a one-direction shift ping-pongs against a two-sided corridor. Does a two-sided lateral repair (L1) of the teacher target, applied to those 25, close the DAC gap to the paper (PDMS >= 95.8 AND DAC >= 99.9) without harming NC, DDC, TTC, C or the 91 tokens REPAIR-D already repaired?

## 2. Population
Navtest, the pinned W3 harness population: 12,146 tokens, 12,110 primary, 136 logs. Nothing is re-selected.

## 3. Inputs and the lever (frozen from navtrain)
- Base: banked N4-D plans and harness rows of TF-N4 A1 (`n4_plans_N4D.npz`, harness CSV of N4-D), and the N2-R plans. L1 runs only on tokens whose N4-D builder tracked DAC is 0 (the builder reports `unrepaired_dac0.N4D`, 25 tokens).
- **L1 (`n4_repair_l1.repair_l1`, imports the pinned `n4_repair` and `n4_geom`, copies nothing):** per iteration: label (Tc20s + P5-R lambda 3.0, DB ego) -> builder tracked score -> certificate `n4_repair.certified` (DAC = 1 and NC, DDC, TTC, C not worse than the N2-R label's own scores; reads logged agent futures, named as in TF-N4) -> else the per-step feasible lateral interval of the four footprint corners inside the drivable union eroded by the margin, window-intersected to the 20 poses, a minimum-total-variation ramp inside it, step along the executed-heading normal, halved on a sign flip of the dominant correction, at most 5 iterations; INFEASIBLE flag (and a recorded fallback margin 0.005 m) when a step has no feasible lateral offset.
- **Frozen parameters (read back from the built object before any build, `n4_repair_l1.readback`; a different value is REFUSED):** margin 0.05 m, cap 5, certificate `full`, sign +1, centre_m 0.05 m, fallback_margin 0.005 m, damp 0.5 non-cumulative, grid 0.01 m, dmax 1.5 m, eps 0.01. All were fixed before any failing set existed; none was tuned.
- Code pins at freeze: `n4_repair_l1.py`, `n5_env.py`, `test_n4_repair_l1.py`, the unchanged A1 run tree files (`n4_repair.py`, `n4_geom.py` sha256 `1966bb6d...`, `n4_nt_build.py` sha256 `e74ebf9f...`), `fam_ci.py`, and `families_run.py` at the TIP version (sha256 `44c62807f1254e2ca4a41e4094097071b96bccaba9e754703ddf4a689e4d73a6`, tip `f938dca6`, MEASURED: it writes `tokens` and `logs` under `--per-window`, lines 96 to 98). **A families stage that runs an older `families_run.py` produces no paired CI and must be refused by C-PIN** (the TF-N4 A1 families stage failed once on exactly this wiring).
- Builder integration (for the chain agent, an MM decision): the chain builder must call `make_l1(...)` only for the 25 builder-DAC-0 N4-D tokens and write the rest from N4-D. Reachability must be proven from the chain argv (trace the L1 parameters to the repair actually executed: `readback`), never from the library alone.

## 4. Arms
| arm | plan | role |
|---|---|---|
| **N5-A** | N4-D's plan; on the 25 tokens with N4-D builder tracked DAC 0, L1 (above); if L1 does not certify, N4-D's (= N2-R's) label is kept and counted unrepaired | **PRIMARY candidate** |
| **N4-D** | banked TF-N4 A1 arm (harness CSV, `harness_csv_n4.tar.xz`) | **comparator** |
| N5-X | the same with the L1 shift sign FLIPPED | **deliberate regression** (C-REG) |
| N5-R | L1 replaces REPAIR-D on all 116 N2-R zero-DAC tokens | secondary, **builder-side counts only**, decides nothing |
| N2-R | banked | descriptive anchor |
Naming: every number is "RJ5 teacher + P5-R + REPAIR-D + L1 on the REPAIR-D residual with a scorer no-harm guard", never "the teacher".

## 5. Scorer of record
The pinned W3 harness, TF-N4 A1's route. N5-A and N5-X are scored on the full 12,146 seam (the pinned chain's subset path stays UNVERIFIED), so C-UNCH and C-ID-C are full-set checks. ESTIMATED cost: about the TF-N4 A1 chain (build 1.3 h, two full seams about 1.7 h, families 1 to 1.5 h), CPU only, no pod, no box 2, at most 2 processes.

## 6. Bars and outcomes (committed before any navtest number of N5; PROPOSED, the MM may overrule)
**PRIMARY: L1 CONFIRMED** needs all four (N5-A minus N4-D, paired, 136 logs):
1. **Repair count:** at least **5 of the 20 non-guard residual tokens** (the `non_guard` list of `inputs/n5/residual25.json`, sha256 `fa2af718…`: the 25 builder-DAC-0 N4-D tokens minus the 5 ANATOMY_25 cause-(b) tokens) have N5-A harness DAC = 1. *Why 5 (PROPOSED):* the descriptive what-if of A1 with 4x the iterations (cap 20, margin 0.05) certified 4 of the 25 (`ANATOMY_25.md` sec. 6, seen on navtest); L1 has to beat what iteration count alone buys, or it is not a new lever.
2. **No harm vs N4-D:** paired dNC, dDDC, dTTC, dC, dEP each >= -0.10 (x100 means), and no token outside the L1-applied set differs from the banked N4-D row by over 1e-6 (the 91 REPAIR-D repairs and every unrepaired token stay bit-identical).
3. **Certificate integrity:** 0 certificate violations among L1-certified tokens at the builder (re-scored), and every builder-certified token has harness DAC = 1 (C-CERT, 0 exceptions).
4. **No new failure:** no token that N4-D had at DAC = 1 reads DAC = 0 in N5-A.
Otherwise **NOT CONFIRMED.** Bars 3 and 4 are also controls (C-CERT-B and C-CERT; C-ID-C and C-UNCH): a breach of any of them VOIDS the read (NO VERDICT, no aggregate is computed), as section 7 says; it is never reported as NOT CONFIRMED. dPDMS is reported with its paired interval but is NOT a bar: 20 tokens move PDMS by at most about +0.17 (ESTIMATED: each repaired token adds about 1/12,110 x 100 = 0.008 per unit of its PDMS), well inside the N4-D interval width, so a PDMS bar would only measure noise.

**PAPER-LEVEL read** (N5-A, point values with intervals): **PAPER LEVEL** PDMS >= 95.8 AND DAC >= 99.9 (at most 12 zero-DAC tokens, i.e. L1 repairs at least 13 of the 25) | **NEAR** PDMS >= 95.3 AND DAC >= 99.5 | **BELOW** otherwise. N4-D is NEAR (96.33 / 99.79).

**Four-family guard** (decides only whether N5-A may become a student label, never the verdict): `families_run.py --per-window` at the tip version for N4-D and N5-A, `fam_ci.py` paired; speed-bias diff <= +0.25 m/s, speed-MAE diff <= +0.15 m/s, lateral cross-track MAE diff <= +0.05 m, heading, curvature, yaw rate reported, longitudinal and tactical per family, never pooled. STRATEGIC: PAUSED (PI 2026-09-19), reported as paused, never as a pass. Expected: no separated change (<= 25 tokens of 12,110).

| outcome | next step (Rule Zero: no outcome ends at a verdict) |
|---|---|
| CONFIRMED, PAPER LEVEL | TF navtest goal closes under the RJ5 naming; PI rules on the guard flag before the label goes to the student (P5-C) |
| CONFIRMED, 5 to 12 repaired (NEAR) | anatomy of the remaining non-guard residuals by cause on navtrain-constructed analogues; then lever L2 (margin 0.20 / cap 10 schedule, parameters from navtrain) and lever L3 for the 5 guard-blocked tokens: the PI-visible NC/TTC/C tolerance decision, named as such |
| NOT CONFIRMED, repair count < 5 | the residual is not a lateral-shift problem: per-token trace by `n4_repair_trace.py` (INFEASIBLE flags = corridor narrower than footprint + margin; cap exhausted with a shrinking extent = more iterations; certificate-blocked = L3); next arm L2 and L3 together |
| NOT CONFIRMED, harm or C-CERT / no-new-failure breach | the offending tokens by metric; the certificate is extended with the metric that broke; re-develop on navtrain |
| any outcome | the 20 non-guard tokens now count as read; any L2 / L3 development uses a fresh navtrain set (e.g. the finished RJ5 re-roll rows) |

## 7. Controls (a failure voids the read; the analysis exits before any N5-A aggregate exists)
**Before the harness scores (build):**
- **C-PIN** code and input pins (including `families_run.py` tip sha256 and `n4_repair_l1.py`).
- **C-DERIV** the builder re-derives the N2-R labels of the 200 smallest-sha1 tokens from the nuPlan-DB ego to 1e-9 m (as A1; navtrain re-check MEASURED: dual labeller vs builder `labels()`, 98 rows, max 0.0).
- **C-LABELEGO** every label-building row (the 25 tokens x up to 5 iterations and the 200 C-DERIV tokens) records `label_ego: db`; a token needing a label with no DB is REFUSED.
- **C-DACMAP** builder tracked DAC of the unmodified N2-R plan equals the harness DAC on all 12,146 tokens, 0 mismatches.
- **C-UNCH** every token L1 was not run on has a plan bit-identical to banked N4-D's; the count of L1-run tokens equals 25.
- **C-DET** the repair run twice on every L1-repaired token and on 20 of the remaining gives bit-identical plans.
- **C-CERT-B** every L1-certified plan re-scored at the builder satisfies `n4_repair.certified` (0 violations).
- **C-REACH** the chain argv is traced to the built object: `readback(fn)` equals the card for margin, cap, sign, certificate, centre_m, fallback_margin, damp (REFUSE otherwise); the zero-correction identity (label returned equals base) and the sign-flip regression pass on the real failing fixture instances (`fixture_real_instances.json`).
- **C-TESTS** `test_n4_repair_l1.py` (8 tests, MEASURED 8 passed on 2026-11-13) and the existing N4 suites green at the pins.
- **C-DET (as built, stronger than the line above):** the repair is run twice on every one of the 116 N2-R zero-DAC tokens (bit-identical), and N5-A equals N5-R on the 25.
- **C-SET** the builder's DAC-0 set equals the banked 116 = 91 REPAIR-D-repaired + the 25 of `residual25.json`.
- **C-SEAMFN** `to_navsim` reproduces the banked N4-D seam bit for bit before any N5 seam is written.
**After the harness scores (read):** C-CERT (0 exceptions, N5-A and N5-X), C-ID-C (the 200 unchanged tokens equal banked N4-D within 1e-6 on all 7 metrics), C-COV (12,146 / 12,110 / 0 error lines), C-EST (draws reproduce `eval/proxy_eval.boot` to 1e-9), **C-REG** (N5-X repairs at most 10 % of the 25 and PDMS(N5-X) <= PDMS(N4-D) + 0.05).
**Mutation arms (each must go RED):** certificate off -> C-CERT-B or no-new-failure RED; L1 applied to ALL tokens -> C-UNCH RED; margin -0.05 or sign -1 -> refused at read-back / C-REG; a 2e-6 perturbation of one unchanged plan -> C-UNCH RED; a pinned file edited -> C-PIN RED; an older `families_run.py` -> families stage refused.

## 8. Estimator
Paired log-cluster bootstrap over 136 logs, 10,000 draws, seed 20260927, 95 % percentile (`eval/proxy_eval.boot`). It answers "would another draw of LOGS give the same result?". The label pipeline, the repair and the harness are deterministic (C-DET, C-ID-C), so no training or inference variance hides behind it. Bonferroni intervals beside every one with K = 47. For the repair COUNT (a bar on 20 tokens) the count and its exact binomial interval are reported; the log bootstrap is not the estimator of that bar.

## 9. K
Ledger max at the freeze = 46 (re-read 2026-11-13 from `raw/2026-11-09-scorer1p6-prereg/selectors/ledger_K.json`, sha256 `82a548a8…`, at tip `7f5e9425`). This read is **n = 47**, key `TF-N5:<sha256 of this file>:E3`, nominee `TF-N5`. The chain refuses unless the ledger holds exactly that key at n = 47, the TF-N4-A1 entry at n = 46, and nothing above 47.

## 10. Evidence and its limits
| claim | class |
|---|---|
| N4-D 96.33 / DAC 99.79, 25 zero-DAC (5 guard-blocked, 20 not) | MEASURED (`RESULT_N4_A1.md`, `ANATOMY_25.md`; navtest, harness / builder) |
| A1 and L1 both repair 30/30 DEV30, 1/1 RJ5 rows, 45/45, 30/30, 23/23 on constructions; 0 certificate violations; L1x 0/129 | MEASURED (navtrain, builder scorer, `L1_DEV.md`, `raw/`) |
| the stress sets are CONSTRUCTION (shifted targets, eroded maps); they test mechanism, not the navtest rate | CONSTRUCTION |
| L1 repairs the navtest ping-pong class | **HYPOTHESIS, untested**; N5 is its first test |
| 5 of 25 is a PROPOSED bar | ESTIMATED from a seen-on-navtest what-if |
| helps a student | not measured (P5-C's question) |

## 11. Freeze record (MM, 2026-11-13)
- **Chain:** `chain/` (`n5_nt_chain.py`, `n5_nt_build.py`, `n5_nt_analyze.py`, `n5_make_pins.py`, `test_n5_chain.py`, `CHAIN_N5.md`). The MM re-ran the tests at the freeze: **61 passed** (53 chain + 8 L1). Every section-7 mutation arm goes RED; every L1 parameter is reachable from the chain argv (perturbation refused, zero-correction control reads the base label exactly). The families stage ran end to end on 300 real banked A1 rows (tip `families_run.py` → `fam_ci` → famread), and the old A1-tree `families_run.py` is refused.
- **Pins:** run tree `F:/tanitad_scratch/n5nt/run`, `n5/N5_PINS.json` sha256 `0a465c1515c505a7ca183a916c971553291c07ef87831a4fbc5751da970c9347` (956 files, including `inputs/n5/residual25.json` and the tip `families_run.py` `44c62807…`). The card itself is checked by its own sha256, not by the pins.
- **Amendments at the freeze (wording, no number changed):**
  - the 20 non-guard tokens are named by file;
  - control breaches inside bars 3 and 4 are NO VERDICT;
  - C-DET as built;
  - C-SET and C-SEAMFN added;
  - the descriptive navtest pre-run is declined.
- **Disclosures:**
  - **L1's design was motivated by the seen-on-navtest anatomy** of these 25 tokens: its structure (two-sided interval, damping), not its parameters, which are untuned defaults. So N5 tests whether a lever designed for an observed failure class repairs it. It is not a test that L1 generalises to unseen failure classes.
  - **Bar 1 (>= 5 of 20) is anchored on the comparator's seen-on-navtest what-if** (REPAIR-D at cap 20 certified 4 of 25). It is set to beat what extra iterations alone buy, and no L1 number was seen.
  - **The navtrain A1 control reproduced DEV30 by count, not bit for bit** (max 8.2e-4 m, `L1_DEV.md`). On navtest, C-DERIV, C-UNCH, C-SEAMFN and C-ID-C bind bit-identity instead.
- **Cost (ESTIMATED):** 3.5 to 4.5 h wall, CPU only, at most 2 processes, no pod, no box 2. N5-X is scored on the full seam although its plans are expected to equal N4-D's; it runs in parallel with N5-A and is kept as the card's regression arm.

