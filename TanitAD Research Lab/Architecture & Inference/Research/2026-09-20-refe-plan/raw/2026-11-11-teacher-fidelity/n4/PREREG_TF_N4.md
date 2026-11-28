<title>PREREG TF-N4</title>

# PREREG TF-N4 (FROZEN 2026-11-12, not run): RJ5 + P5-R with a map-certified DAC repair (REPAIR-D), one navtest read through the pinned harness

**Status: FROZEN by the Master Mind 2026-11-12 (sha256 in `PREREG_TF_N4.sha256`), before any navtest number of any N4 arm.** Drafted by the development agent (`PREREG_TF_N4_DRAFT.md`, landed 13e30630, which records the pre-data MM amendments: full-seam scoring; admissibility corrected (the certificate's no-harm guard reads agent futures); secondary arm N4-DM; margin 0.05 m and cap 5 fixed before data). Freeze amendments, all from the chain build (`chain/CHAIN_N4.md` sec. 3b) and none touching a bar: sections 3, 5, 7, 10, 11 and 13 below. Format follows `PREREG_TF_N2_A1.md`.

## 1. Question
TF-N2-A1 left the privileged teacher "RJ5 teacher + P5-R" (N2-R) at PDMS 95.61 [95.08, 96.08] and **DAC 99.04**, against the paper's 95.8 / 99.9 (RESULT_N2_A1.md). Swapping the paper's DAC alone is worth +0.82 PDMS, EP is no longer short, and 116 primary tokens read DAC 0. Does re-aiming the teacher's own target by a map-certified margin on those tokens lift N2-R to paper level, without costing NC, DDC, TTC, C or progress?

## 2. Population
Navtest, the pinned W3 harness population of TF-N2-A1: 12,146 tokens, **12,110 primary**, 136 logs. Nothing is re-selected.

## 3. Inputs (banked; the only new computation is the repair)
- The N2-A1 RJ5 rollouts and the N2-R plans derived from them (`n2/navtest_a1/`, pins of TF-N2-A1).
- The harness's own stored metric cache for every navtest token (`metric_cache_navtest`; map, ego state). The builder reads the map from it via W-TRANSP's `--metric-cache-root` route, never from the nuPlan DB.
- The banked harness CSVs of TF-N2-A1 (N2-R, N2-C, N1-C, T, human). Comparator = the banked N2-R CSV.
- Code: the chain copies `n4/chain/n4_repair.py`, `n4_geom.py`, `n4_common.py` (landed with this card). They are the development files (`n4/`, 13e30630) with ADDITIVE edits only: `CAP = 5`, `certified_dac` + `CERTS = {full, dac}` (for N4-DM), `repair_target(..., iters=CAP, sign=1.0, cert=certified)`, `Ctx.from_parts` (the metric-cache route), env overrides for rows and P5-R. Defaults reproduce the development behaviour (navtrain smoke: REPAIR-D 5/5 as in development; INHERITED from the chain agent, closed on navtest by C-DERIV and C-DACMAP). Margin, cap, sign and certificate are read back from the built objects and compared with this card before any build (`n4_nt_build.py` read-back; a different margin is REFUSED, MEASURED in the smoke). The builder reads the map and agent tracks from the harness metric cache via the TIP `transp_score.py` (`--metric-cache-root`, sha256 `a6c727a1...`), which replaces TF-N2 A1's pinned copy (`e1dc987c...`, no cache route) in the run tree.

## 4. Arms
Plans are 8 poses at 0.5 s from the label (20 poses at 0.2 s), as in N2.

| arm | plan | role |
|---|---|---|
| **N4-D** | N2-R's label; on every token whose **builder** tracked DAC is 0, REPAIR-D (below); other tokens unchanged | **PRIMARY candidate** |
| **N2-R** | the banked TF-N2-A1 arm (harness CSV `inputs/n2r/refe_tfn2_n2r.csv`, sha256 `2a2d4cbcc6ee546336847b9b2a19c0590c56f073e3f341c35d38cdcb4009f709` (repo: `n2/navtest_a1/run/harness_csv_n2.tar.xz`)) | **comparator** |
| N4-X | the same repair with the shift sign FLIPPED (pushes the label outward) | **deliberate regression** (C-REG) |
| N4-P2 | CERT2's per-token choice (banked CSVs, TF-N2-A1 order) and, on its 79 none-pass tokens, REPAIR-D of N2-R | secondary, decides nothing |
| N4-DM | REPAIR-D with the certificate reduced to **tracked DAC = 1 only** (no NC / DDC / TTC / C comparison: map and ego state only, no agent futures); same margin, cap and order | secondary, decides nothing: N4-D minus N4-DM is what the future-aware no-harm guard is worth |
| CERT2, human | banked CSVs | descriptive |

**REPAIR-D (frozen, exactly `n4_repair.repair_target`).** For a token whose N2-R label has tracked DAC 0: shift the teacher's 20-pose target along the unit return vector of the worst-outside corner (computed by the scorer's own geometry: 4 corners against the union of ROADBLOCK / INTERSECTION / CARPARK polygons) by (excess + 0.05 m), ramped over [0, t_violation]; re-run Tc20s (`iterate20`, cubic) and P5-R (lambda 3.0, warm start Tc20s); repeat up to 5 times; **accept the first label with the certificate: tracked DAC = 1 and NC, DDC, TTC, C each not worse than the N2-R label's**. If none certifies, the N2-R label is kept (and counted as unrepaired). Margin 0.05 m, 5 iterations, the certificate: fixed here, no tuning on navtest.

**Admissibility (corrected by the MM, 2026-11-12).** The shift itself reads the map and the t0 ego state only. **The certificate reads more:** `n4_repair.certified` (`n4_repair.py:24-25`) compares NC, DDC, TTC and C from the builder's tracked scorer call, and NC and TTC there are computed against the **logged future motion of the other agents** in the metric cache. So the no-harm half of the certificate is future-aware, the way a scorer is and a planner is not. For a label generator this is admissible: labels may use anything (PI 2026-08-03), and nothing reaches a student input. For any claim about the teacher it must be named: quote **"RJ5 teacher + P5-R + map-certified DAC repair with a scorer no-harm guard"**, never "the teacher", and never compare it with the paper's teacher without that qualifier. The secondary arm N4-DM (DAC-only certificate, map and ego only) measures what the future-aware guard contributes. No harness score selects, certifies or tunes anything: the builder's own scorer call certifies (C-CERT checks it against the harness). It adds no goal / situation-classifier signal. The label warp stays refused.

## 5. Scorer of record
The pinned W3 harness (`score_p4_a1.py` -> `score_e.py`), TF-N2-A1's route. **Scoring unit (MM decision 2026-11-12): the FULL 12,146-token seam for N4-D, N4-X and N4-DM.** N4-X is kept although its flipped shift is expected never to certify (0 of 5 in the smoke), because it makes C-REG a real harness check. **N4-P2 is derived, not scored:** its harness row is CERT2's banked row where a candidate passes DAC, else the N4-D row (REPAIR-D of N2-R is exactly N4-D's plan on those tokens); labelled so in the read, exactly TF-N2-A1's route. No subset path is used, since the pinned chain's subset support is UNVERIFIED. Cost: about +1 to 2 h CPU per seam (ESTIMATED). C-UNCH still proves bit-identity of every unrepaired plan, and C-ID-C becomes a full-set check: every unrepaired token's harness row equals the banked N2-R CSV within 1e-6. The status line must read PASS with the full row count.

## 6. Bars and outcomes (committed before any navtest number of N4)
**PRIMARY.** N4-D minus N2-R, paired PDMS (x100) on the 12,110. **CONFIRMED** needs all four:
1. dPDMS >= **+0.50**.
2. 95 % CI lower bound > **0**.
3. **Repair rate:** of the tokens with N2-R harness DAC = 0 (116 primary), at least **75 %** have N4-D harness DAC = 1.
4. **No harm:** paired dNC, dDDC, dTTC, dC and dEP each >= **-0.10** (x100 means), and no unchanged token's harness row differs from N2-R's by over 1e-6.
Otherwise **NOT CONFIRMED.**

*Why these numbers (the MM may overrule before the freeze).* Navtrain read +1.07 with 5 of 5 repaired, but on 5 tokens in 2 of 13 logs: a mechanism result, not an effect estimate (DEV_N4_DAC.md sec. 0). Arithmetic: a repaired token moves PDMS by up to about 1/12,110 each, so repairing the 116 gives about +0.9; +0.50 needs about 55 to 60 tokens repaired at full value, bar 3 (87 of 116) sits above that and below the navtrain 100 %. The CI condition asks that 136 logs separate it from 0. No-harm is -0.10, not 0, only for rounding of per-token tolerances; the repair is certificate-gated so the expectation is exactly 0.

**PAPER-LEVEL read** (N4-D, point values with intervals, unchanged from N1/N2): **PAPER LEVEL** PDMS >= 95.8 AND DAC >= 99.9 (<= 12 zero-DAC tokens; needs >= 90 % repaired) | **NEAR** PDMS >= 95.3 AND DAC >= 99.5 | **BELOW** otherwise. The residual gap is decomposed by one-at-a-time sub-score swaps to the paper's NC 99.8, DAC 99.9, EP 91.5, TTC 99.1, C 99.0 (descriptive). Expectation if navtrain transfers whole (ESTIMATED, not a bar): PDMS about 96.5, DAC about 100.

**Four-family guard** (as TF-N2 section 10; decides only whether N4-D may become a student label, never the verdict): speed-bias diff <= +0.25 m/s and speed-MAE diff <= +0.15 m/s against N2-R, lateral cross-track MAE diff <= +0.05 m, heading and curvature reported. STRATEGIC: PAUSED (PI 2026-09-19).

**Secondary (decide nothing):** N4-P2 minus N2-R and minus CERT2 (is the repair needed beyond selection?), N4-D minus CERT2, **N4-D minus N4-DM and N4-DM's own paper-level read (the map-only form)**, human-anchored paper level.

| outcome | next step (Rule Zero: no outcome ends at a verdict) |
|---|---|
| CONFIRMED, PAPER LEVEL | the TF navtest goal closes under the RJ5 naming; the PI rules on the guard flag before the label goes to the student (P5-C) |
| CONFIRMED, NEAR / BELOW | anatomy of the residual zero-DAC tokens by cause (iteration cap, margin, NC / TTC trade blocking the certificate, teacher intent beyond reach); then raise the iteration cap / margin on navtrain first, and the TTC lever (+0.24 swap gain) |
| NOT CONFIRMED, repair rate < 75 % | first suspect is the builder-versus-harness DAC difference (C-CERT, C-DACMAP); then the 25 % residual by cause on navtrain; next arm is the wider-set CERT (navtrain +0.61) combined with repair |
| NOT CONFIRMED, harm | the offending tokens by metric; certificate extended with the metric that broke; re-develop on navtrain |
| NOT CONFIRMED, effect < +0.5 with repair rate met | the repaired tokens are valuable less than 0.95 each: anatomy of their EP / TTC on navtest; the lever stays, the EP lever is next |

## 7. Controls (a failure voids the read; the analysis exits before any N4-D aggregate exists)
**Before the harness scores:**
- **C-PIN.** Every code and input file matches the pins (the chain's guard, again in the read).
- **C-DERIV.** The builder re-derives the N2-R plans of the 200 smallest-sha1 tokens; they equal the banked N2-R plans to 1e-9 m (TF-N2's C-DERIV, navtest form).
- **C-DACMAP.** The builder's tracked DAC of the unmodified N2-R plan equals the banked N2-R harness DAC on **all 12,146 tokens, 0 mismatches**. It is the premise of the lever (DEV_N4_DAC.md sec. 1: measured on N1-T on navtest, inherited for the N2-R plans). FAIL stops the chain.
- **C-UNCH.** Every token the builder did not repair has a plan bit-identical to N2-R's; the count of repaired tokens is printed and must equal the count of builder DAC 0 tokens minus unrepaired ones.
- **C-DET.** The repair run twice on 50 repaired tokens gives bit-identical plans.
- **C-TESTS.** `test_n4_repair.py` (5 tests, incl. the sign-flip regression and the unchanged-label test) green at the pinned sha; the existing suites of the N2 pins unchanged.

**After the harness scores (in the read):**
- **C-CERT.** Every token the builder certified (DAC = 1) has harness DAC = 1: **0 exceptions**, checked on N4-D, N4-DM and N4-X (freeze amendment: all three arms; the draft named N4-D). FAIL voids the read: it means the builder's map is not the harness's. **C-ID-C** below is likewise applied to all three arms.
- **C-ID-C.** The 200 unchanged tokens re-scored equal the banked N2-R CSV within 1e-6 on all 7 metrics.
- **C-COV.** 12,146 tokens, 12,110 primary, 0 error lines.
- **C-EST.** The draws reproduce `eval/proxy_eval.boot` to <= 1e-9 (as N2).
- **C-REG.** N4-X repairs at most **10 %** of the N2-R zero-DAC tokens, and PDMS(N4-X) <= PDMS(N2-R) + 0.05 (a flipped repair must not help).
- **Diagnostic, never voids:** none. (The draft's human CSV re-read is dropped at freeze: N4 re-scores no human seam, so there is nothing to compare.)

**Mutation arms (each control has one that must go RED):** the certificate disabled (accept the first iterate) -> C-CERT or no-harm goes RED on the navtrain smoke; the repair applied to ALL tokens -> C-UNCH RED; margin forced to -0.05 (inward shift) -> repair rate collapses and C-REG-style check RED; a 2e-6 perturbation of one unchanged plan -> C-UNCH RED; a pinned file edited -> C-PIN RED.

## 8. Estimator
Paired log-cluster bootstrap, 136 logs, 10,000 draws, seed 20260927, 95 % percentile (`eval/proxy_eval.boot`, as TF-N2-A1). It answers **"would another draw of LOGS give the same result?"** The rollouts, labels, repair and harness are deterministic (C-DET, C-ID-C), so no inference or training variance hides behind it. It says nothing about a student trained on these labels. Bonferroni intervals beside every one with K = 45.

## 9. K ledger
Current max n in `ledger_K.json` at tip `924901b7` is **44** (TF-N2-A1). This read is **n = 45**: `{"key": "TF-N4:<sha256 of this file>:E3", "n": 45}`. The MM re-reads the ledger max at freeze and appends.

## 10. Four metric families (binding; STRATEGIC PAUSED)
`families_run.py --per-window` (executed_tracked frame, W-TRANSP on the harness metric cache via `--metric-cache-root`, i.e. the harness scenery: D-REFE-WTRANSP-CACHE-1/2, 12,146 / 12,146 identity on T; the family metrics themselves are not harness numbers) for N2-R and N4-D, with `fam_ci.py` paired CIs, labelled "not harness-identical" as in N2-A1. Longitudinal (speed bias / MAE, accel MAE, headway / TTC), lateral (cross-track, heading, curvature, yaw rate), tactical (decision quality) reported per family, never pooled. Navtrain expectation (DEV sec. 4): 0 of 61 paired metrics separated.

## 11. Run (for the MM; CPU only, at most 2 processes, 0 GPU) and cost
1. **Build:** builder tracked scoring of the N2-R plans on all 12,146 tokens (W-TRANSP + harness metric cache, ~1 to 2 s/token, ESTIMATED from the W-TRANSP docstring) about 5 to 7 CPU-h, about **2.5 to 3.5 h wall at 2 processes**; REPAIR-D on about 116 to 200 tokens, at most 6 label builds of ~0.7 s each (MEASURED on navtrain), minutes; N4-X and N4-P2 plans the same.
2. **Score:** the changed-token subset plus the 200 identity tokens through the pinned harness, minutes to under 1 h (ESTIMATED); the full-seam option adds about 1 to 2 h per seam (ESTIMATED from TF-N2-A1's chain length).
3. **Read + families:** about 1.5 h (N2-A1's families stage was about 80 min).
Total about **4 to 5.5 h wall, 0 GPU** (ESTIMATED, `chain/CHAIN_N4.md` sec. 7: derive ~1-1.3 h, three full seams ~1.7 h with D and DM in parallel, families ~1-1.5 h), no pod, no box 2. Launch: `n4_nt_chain.py --run-dir <RUN> --card <RUN>/n4/PREREG_TF_N4.md --card-sha256 <this card> --pins-sha256 <N4_PINS> --ledger <RUN>/inputs/ledger_K.json --k 45 --procs 2`, detached (Win32_Process Create, CreateFlags 0x08000200, stdin NUL), `--check-only` first. Success line `ZZN4_CHAIN DONE`. The chain refuses any argv with a different margin / iteration count / sign (read back from the built repair).

## 12. Evidence and its limits
| claim | class |
|---|---|
| N2-R 95.61 / DAC 99.04; 116 zero-DAC; CERT2 79 none-pass | MEASURED (RESULT_N2_A1.md, navtest, harness) |
| REPAIR-D repairs 5/5 (RJ5) and 30/30 (six arms) form-R DAC failures, 0 harm on NC/DDC/TTC/C | MEASURED (navtrain, W-TRANSP DB rebuild; `DEV_N4_DAC.md`) |
| PDMS +1.07 [0.00, 2.39] on navtrain | MEASURED, **cannot separate** (5 tokens, 2 of 13 logs) |
| DB-route DAC = harness DAC | MEASURED on navtest N1-T (WTRANSP_DIAG.md:85), INHERITED for navtrain and for N2-R plans (C-DACMAP closes it) |
| the 79 navtest none-pass tokens are the "teacher intent is outside" class | INFERRED from navtrain anatomy (all 30 failing instances have the untracked teacher plan outside, by 4 to 24 cm) |
| helps a student | not measured (P5-C's question) |

## 13. Pins (filled by the MM at freeze, 2026-11-12)
- **Code:** base tip `13e30630` plus the chain files landed with this card under `PK/raw/2026-11-11-teacher-fidelity/n4/chain/`; every run file is pinned by sha256 in `N4_PINS.json` (C-PIN checks them at launch and again in the read).
- **Run tree:** `F:/tanitad_scratch/n4nt/run` on the dev box: TF-N2 A1's `w/` and `repo/` (its pinned harness tree) with `w/pk/eval/instruments/transp_score.py` replaced by the tip copy, `n2/` (the two A1 analyzers), `n4/` (chain, analyzer, builder, repair code, `fam_ci.py`, this card, pins), `inputs/` (A1's, minus rollouts) and `inputs/n2r/`.
- **Pins file:** `n4/N4_PINS.json`, sha256 `1085ee43516af8d00beefa743dcc95d3c339c3be45848b280521689c6a0e9c50`, 946 files; metric-cache listing digest `2ce6c713d4bbe2696990f7e60dd36fa43d1b81bfc46d7331e558084db90ddac6` over 12,146 files.
- **Code sha256 (prefixes; full values in the pins):** `n4_repair.py` 1a4be60a, `n4_geom.py` d3f9d5df, `n4_common.py` 639b6ff6, `n4_nt_chain.py` 9e273f6c, `n4_nt_build.py` 091de499, `n4_nt_analyze.py` f5c1b929, `fam_ci.py` 8f7ada4e, `transp_score.py` a6c727a1.
- **Banked inputs:** N2-R plans `b501cc66...`, N2-R seam `ade308c8...`, N2-R CSV `2a2d4cbc...`, N2-C CSV `a3cd0218...`, RJ5 rows `cdd4c1f9...` / `a5bef38d...`. Plans, seam and RJ5 rows were single-copy on dev-box scratch and are now banked: private HF `Sayood/tanitad-refe-scorer1p6`, `evidence/2026-11-12-tf-n2-a1-inputs/n2r_rj5_inputs.tar` (sha256 `e909e04e...`, members verified). The CSVs are in the repo (`n2/navtest_a1/run/harness_csv_n2.tar.xz`).
- **Tests at freeze (MEASURED by the chain agent, navsim venv, pristine tip export):** `test_n4_repair.py` 5 OK; `test_n4_chain.py` 29 OK. Navtrain smoke (DB route, a stand-in for the harness reference, so its C-DACMAP is not a harness check): real arm GREEN (REPAIR-D 5/5, N4-DM 5/5, N4-X 0/5); every mutation arm RED (certificate off: C-CERT-B and C-UNCH; repair-all: C-UNCH; margin -0.05: refused at read-back; 2e-6 perturbation: C-UNCH; pinned file edited: C-PIN). Harness-side controls are exercised on synthetic CSVs only before this read.
- **K:** ledger max n at tip `13e30630` = 44; this read is **n = 45**.
