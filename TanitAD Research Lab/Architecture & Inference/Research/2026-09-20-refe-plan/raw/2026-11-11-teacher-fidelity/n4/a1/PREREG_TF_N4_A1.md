<title>PREREG TF-N4 A1</title>

# PREREG TF-N4 Amendment A1: rerun TF-N4 with the label pipeline on the nuPlan-DB ego, after its C-DERIV void

**Status: FROZEN by the MM 2026-11-13 (sha256 in `PREREG_TF_N4_A1.sha256`), before the rerun.** It amends `PREREG_TF_N4.md` (sha256 `1d051cd1…`, frozen 2026-11-12T22:50:38Z, K n = 45). Everything not named here is unchanged: question, population, arms (N4-D, N2-R, N4-X, N4-DM, N4-P2 derived), REPAIR-D (margin 0.05 m, cap 5, certificate), scorer of record and full-seam scoring, every bar of section 6, every control of section 7 and its bar, the estimator, the four-family guard and the naming rule.

## 1. What happened (MEASURED)
- TF-N4's chain ran on 2026-11-12/13 and stopped at its **build** stage: `ZZN4_CHAIN STOP`, `STAGE build FAIL` (`n4run/chain.log`, 2026-11-13T00:48:55 local).
- **C-DERIV FAILED.** The builder's re-derivation of the N2-R label for the 200 smallest-sha1 tokens differed from the banked N2-R plans: max **6.36 mm**, median 2.7e-5 m, **0 / 200** within the 1e-9 m bar (`n4run/build/out/n4_build_report.json`; per-token values in `build/derive_s*of2.jsonl`).
- The other build controls passed: C-COVB, C-DACMAP (12,146 / 12,146), C-DET, C-CERT-B, C-UNCH.
- **By the card, the read is VOID** (section 7: "a failure voids the read"). No harness score of any N4 arm exists. The chain stops before the score stage.
- K entry n = 45 stays counted.

## 2. Diagnosis (MEASURED, `diag/cderiv_route.py`, `diag/cderiv_route.json`, 40 tokens: the 8 worst plus 32 further C-DERIV tokens)
- TF-N2 built N2-R from the ego of the nuPlan-DB scenario (`tfn2_nt_build.py:153-157`, `sc.initial_ego_state`).
- TF-N4's builder took the ego from the harness metric cache (`--route cache`, `Ctx.from_parts`) for everything, label construction included.
- **Labels rebuilt from the DB ego equal the banked N2-R plans on 40 / 40 tokens, max 0.0.** Labels from the cache ego differ on 40 / 40, max 6.36 mm.
- The two ego state arrays differ by at most 4.5e-7 (median 8.1e-8). The iterative label pipeline (Tc20s `iterate20` + P5-R `R.solve` around `tracked8`) amplifies that to millimetres.
- The chain's own values equal the diagnostic's cache-route values exactly, so the build is deterministic. The defect is the ego SOURCE, not noise.
- Class: **a builder input that differs from the comparator's builder input** (same family as TF-N2's C-R0 void, where the thread count differed).

## 3. The amendment
1. **Label construction uses the DB ego.** That is `Ctx.labels`: `iterate20` and the `tracked8` closure inside P5-R. It covers C-DERIV and every REPAIR-D rebuild of N4-D, N4-X and N4-DM, so it is exactly TF-N2's N2-R pipeline input.
2. **Everything else stays on the harness metric cache:** base scoring, drivable union, return vector, certificate scoring, as section 3 of the card says.
3. **New control C-LEGO (build stage; a failure voids the read).** Every row that built a label records `label_ego: "db"`, and the build report counts them. Count = C-DERIV tokens + every token that entered REPAIR-D, in all three arms. A token that needs a label and has no DB is REFUSED (the build stops).
4. **C-DERIV is unchanged:** 200 tokens, bar 1e-9 m. Its mutation arm (label ego forced back to the cache ego) must go RED on real navtest tokens, as the void itself measured.

## 4. Disclosure: what was seen before this amendment
- The void run's build report printed builder-side repair counts: N4-D 91 / 116, N4-DM 96 / 116, N4-X 3 / 116 zero-DAC tokens.
- These are builder counts on labels built from the wrong ego, not harness numbers. No bar, arm, margin, cap or certificate is changed after seeing them.
- No harness PDMS, DAC or family number of any N4 arm was computed.

## 5. K
This read is **K n = 46**. The ledger key is `TF-N4:<sha256 of this file>:E3`, the form the pinned chain checks (`n4_nt_chain.py:132`), with nominee `TF-N4-A1`. It amends TF-N4 (n = 45, void at C-DERIV, still counted). Bonferroni intervals use K = 46. The chain runs with `--card <this file> --k 46`.

## 6. Pins (filled by the MM at freeze, 2026-11-13)
- **Code:** the A1 copies of `n4_geom.py` (sha256 `1966bb6d…`) and `n4_nt_build.py` (`e74ebf9f…`), landed under `PK/raw/2026-11-11-teacher-fidelity/n4/a1/`. They add `label_ego()` (DB ego, fetched lazily through `build_scenarios_for_log` → `initial_ego_state` for the tokens that build labels), `LabelEgoMissing` (re-raised: never an error row and never a cache fallback), the per-row `label_ego` field and the merge control C-LABELEGO (= C-LEGO of section 3). Smoke-only flags `--mutate` and `--cderiv-only` never enter the chain argv. `n4_repair.py`, the chain and the analyzer are byte-identical to TF-N4's.
- **Run tree:** `F:/tanitad_scratch/n4nt/run_a1` (TF-N4's `run` minus its `n4run/` outputs, with the two A1 files). **Pins:** `n4/N4_PINS.json`, sha256 `1dd35d360198a5eae6e5e5137aac26570fc929f54e1cd42fa3a68d41c64041a9`, 946 files. Versus TF-N4's pins, exactly `n4/n4_geom.py` and `n4/n4_nt_build.py` changed; the metric-cache listing is unchanged (`2ce6c713…`, 12,146 files).
- **Checks at freeze** (MEASURED by the A1 fix agent, `a1/A1_FIX.md`):
  - **Real navtest C-DERIV, all 200 tokens:** with the DB ego, 200 / 200 at exactly 0.0 (bar 1e-9), `label_ego` db on 200. Mutation (cache ego): 0 / 200, max 6.358e-3 m, which reproduces the void.
  - **Tests:** `test_n4_a1.py` 8 + `test_n4_chain.py` 29 = 37 OK; reverting `labels()` to the cache ego fails 5 of the 8.
  - `test_n4_repair.py` was not re-run: it needs the navtrain rows pack, and `n4_repair.py` is byte-identical to TF-N4's, where it ran 5 OK.
- **Extra wall time (ESTIMATED):** 316 tokens need the DB ego (116 builder-DAC-0 + 200 C-DERIV, overlap 0), at about 7.4 s each, so about 20 min wall at 2 processes. Total about 4.5 to 6 h.
