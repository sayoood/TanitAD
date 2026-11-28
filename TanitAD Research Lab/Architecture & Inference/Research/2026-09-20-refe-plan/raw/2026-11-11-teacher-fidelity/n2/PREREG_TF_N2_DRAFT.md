<title>PREREG TF-N2 draft</title>

# PREREG TF-N2 (DRAFT, not frozen, not run): the teacher with a P5-R label of a 5 % progressive time-warp, map-certified, on navtest through the pinned harness

**Status:** DRAFT, written 2026-11-12 by the TF design agent.
- No navtest score of any N2 arm exists, and this agent opened no N1 per-token row.
- Freeze order, mechanically checked by the chain as in N1/A1:
  1. The MM records sha256(this file).
  2. The MM appends the §9 K entry and lands it.
  3. The MM launches the chain.
- Evidence: `TFN2_EVIDENCE.md` (navtrain held-out only).

## 1. Question
On navtest, does the N2 candidate reach the paper level? The candidate is: the teacher's banked rollout → a 5 % progressive time-warp → the P5-R regularised tracker inversion, with a map-certified fallback to the unwarped labels.
- The paper level is DriveRL Table 4: PDMS **95.8**, DAC **99.9**.
- The candidate must also beat N1-C (Tc20s, 94.10) by a pre-committed margin.

**Known before this read:**
- N1-A1 on navtest, pinned harness (`../n1/run_a1/RESULT_N1_A1.md`):
  - N1-C 94.10 [93.60, 94.59], DAC 99.01, EP 88.76;
  - CERT 94.75, DAC 99.73;
  - the gap to the paper is EP first (+1.13), then DAC (+0.84).
- navtrain held-out (469 tokens, 13 logs, exploratory, `TFN2_EVIDENCE.md` §2):
  - N2-S − C = **+1.59 [+0.64, +2.66]**, with ΔEP +2.78 and ΔDAC +0.85;
  - RW105 alone: +1.18;
  - CERT over [R, C, T]: +0.96;
  - R alone: +0.55.
- **Expected (ESTIMATED, written before any read):** N2-S ≈ 95.1 to 95.7, DAC ≈ 99.7 to 99.8 ⇒ most likely **NEAR**.

## 2. Population
The 12,110 primary navtest tokens in 136 logs. This is N1's population, with the same 36 C1-known exclusions (`frame_control_navtest_full.json`). All 12,146 tokens are scored and also reported as a sensitivity read. Every comparison is paired with N1.

## 3. Inputs (banked; nothing is re-rolled)
- **N1's pinned inputs** (`../n1/N1_PINS.json`):
  - the teacher rollouts `PK/raw/2026-10-05-teacher-navtest/full/rows_s{0..3}of4.jsonl(.xz)`;
  - the T seam `refe_p4_teacher_navtest.npz`;
  - the banked teacher CSV (the C-ID-H reference);
  - the human CSV;
  - the known-36 file;
  - the token → log map;
  - the nuPlan test DB and maps.
- **N1-C's banked harness CSV** `../n1/run_a1/csv/refe_a1_tfn1_c.csv`. It is reused as the comparator, **not re-scored**.
- **Code.**
  - P5-R: `../../p5r/p5r_labels.py` (frozen with PREREG_P5R, λ* = 3.0).
  - The warp `tfn2_probe.warp`, copied verbatim into the build.
  - `iterate20`, cubic interpolation.
  - The A1 harness route `../n1/a1/score_p4_a1.py` → `score_e.py` (`HARNESS_PINS.json`).

## 4. Arms (all 8 × 0.5 s W3 seams, the T seam's tokens and fingerprints)
| arm | plan | role |
|---|---|---|
| **N2-S** | per token, the first of [RW105, R, C, T] whose **harness DAC** = 1; if none passes, RW105 | **PRIMARY candidate** |
| RW105 | `p5r.solve(w, Tc20s(w) − w, [3.0])` with `w = warp(traj20, 0.05)` | CERT input, non-selection read |
| R | `p5r.solve(traj20, Tc20s − traj20, [3.0])` | CERT input, non-selection read |
| C | N1-C, banked CSV (identical plan) | comparator and CERT input, no new read |
| T | the registered T seam, re-scored | identity (C-ID-H) |
| Wreg | Tc20s of `warp(traj20, −0.30)` | deliberate regression (C-REG) |
| human | banked human CSV | descriptive |

**Admissibility, stated per arm:**
- RW105, R and Wreg read the teacher's own path, the t0 ego and NAVSIM's public tracker. They read no log future and no score.
- N2-S selects by the harness DAC, which is map-only (N1 ruling: partly guaranteed by construction).
- ⚠ **The warp is a label speed bias, not the teacher's policy.** Every N2-S number is quoted as **"teacher + P5-R + 5 % time-warp + map-certified selection"**: never as "the teacher" and never as checkpoint identity.

## 5. Scorer of record
The **pinned W3 harness** (`score_p4_a1.py` → `score_e.py`), exactly A1's route.
- A per-arm CSV holds NC, DAC, DDC, EP, TTC, C and PDMS.
- N2-S is assembled per token from the arm CSVs by the DAC rule. The none-pass count is printed; on a none-pass token PDMS = 0 for every arm, so the tie-break cannot move PDMS or DAC.

## 6. Bars and outcomes (committed before any navtest number)
**PRIMARY.** N2-S − N1-C, paired PDMS (×100) on the 12,110 tokens. **CONFIRMED** requires all three:
1. Δ ≥ **+0.8**;
2. the 95 % CI lower bound > **0**;
3. N2-S DAC ≥ **99.5**.

Otherwise: **NOT CONFIRMED.**

**PAPER-LEVEL read** (N2-S, on point values, with the interval beside them; unchanged from N1):

| outcome | PDMS | DAC |
|---|---|---|
| **PAPER LEVEL** | ≥ **95.8** | ≥ **99.9** |
| **NEAR** | ≥ **95.3** | ≥ **99.5** |
| **BELOW** | anything else | anything else |

On BELOW, the gap is decomposed by one-at-a-time sub-score swaps to the paper's NC 99.8, DAC 99.9, EP 91.5, TTC 99.1 and C 99.0, with the residual printed.

**Secondary (decides nothing):**
- RW105 − N1-C and R − N1-C, paired;
- N2-S − N1-CERT (94.75), paired;
- the choice counts;
- the human-anchored paper level (95.55, ESTIMATED).

**What each outcome sends the ladder to:**

| outcome | next step |
|---|---|
| PAPER LEVEL | the TF navtest goal closes. **The PI rules on the warp's family cost (§10) before the label goes to the student.** |
| NEAR | the next rung is the teacher re-roll levers (`TFN2_EVIDENCE.md` §4), which move EP inside the policy and would replace the warp |
| BELOW or NOT CONFIRMED | a D1-style anatomy of N2-S's DAC and EP losses, then the re-roll levers. No GPU. |

## 7. Controls (a failure voids the read; the analysis exits before any N2-S, RW105 or R aggregate exists)
**Before scoring:**
- **C-PIN.** Every input and code file matches `N2_PINS.json`.
- **C-DERIV (navtrain).** The build, pointed at the 469 navtrain probe rows, must reproduce `to_navsim(label)` of R and RW105 against the banked `raw/probe_rows.jsonl.xz` / `raw/extra_rows_RW105.jsonl.xz` `fam.reference8`, within **1e-9 m**.
- **C-BUILD.**
  - 12,146 of 12,146 tokens derived, with 0 errors;
  - T equals the banked `traj20` bit for bit;
  - `to_navsim(traj20)` as float32 equals the seam `poses`.

**After the T re-score, before any other arm is scored:**
- **C-ID-H.** The re-scored T seam equals the banked `refe_p4_teacher_navtest.csv`: |Δ| ≤ **1e-6** on 7 metrics × 12,146 tokens, 0 missing.
- **C-ID-C.** Re-score N1-C's seam on a **200-token** stable-hash subset; it must equal `refe_a1_tfn1_c.csv` within 1e-6. This proves the reused comparator is the same plan. ⚠ The re-score is an extra read of an already-read arm, logged under this K entry.

**After scoring:**
- **C-COV.** 12,146 tokens, every arm present, 0 error rows.
- **C-EST.** The draws reproduce `proxy_eval.boot` to ≤ 1e-9.
- **C-REG.** PDMS(Wreg) ≤ PDMS(N1-C) − **5.0** on the 12,110 (navtrain: −8.73).
- **Diagnostic, never voids.** The human CSV is re-read, and the count of tokens over 1e-6 is printed.

## 8. Estimator
- Paired log-cluster bootstrap over the 136 logs, 10,000 draws, seed 20260927, 95 % percentile: `proxy_eval.boot`'s scheme, exactly.
- Bonferroni at **K = 43** beside every navtest interval.
- It answers *another draw of LOGS* only. Plans, labels, tracker and harness are deterministic: no training or inference seed applies.

## 9. K ledger
**N2 is ONE entry, n = 43.** The ledger holds n = 42 (`TF-N1-A1`) at tip `be6567aa`; if another entry lands first, n and K shift before freezing.

```json
{"key": "TF-N2:<sha256 of the frozen card>:E3", "n": 43, "nominee": "TF-N2", "family": "TF", "set": "E3", "smoke": false,
 "arms": ["N2-S (candidate, PRIMARY)"],
 "non_selection_reads": ["RW105 (CERT input)", "R (CERT input)", "T (identity, C-ID-H)", "C 200-token re-score (C-ID-C)",
                         "Wreg (deliberate regression, C-REG)", "human (descriptive)"]}
```

## 10. Four metric families (binding; strategic PAUSED)
**Instrument.** W-TRANSP **with the A1 harness-cache repair**, `../n1/a1/transp_score_a1fix.py --metric-cache-root F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest`, then `families_run.py`. It runs on N2-S, RW105, R and N1-C's plans, plus human, on the 12,110.
- ⚠ That command-line path is **untested** (`WTRANSP_DIAG.md` §4: only the scoring function is tested). Before use: run its `--limit 200` smoke on the non-N2 human sub200 seam, with 0 of 200 tokens over 1e-6 against `refe_e0b_human_sub200.csv`.
- If the smoke fails, the families come from the DB route and are flagged "not harness-identical". The PDMS verdict is unaffected: it comes from the harness CSVs.

**Primary frame:** executed_tracked. Families are never pooled, and each carries its n.

**The warp's longitudinal cost.** It is reported and has no bar, because no paired family CI exists. It is printed beside the verdict as: N2-S − N1-C in speed MAE, speed bias, accel MAE, progress ratio, min TTC (with n closing) and goal-point error.
- navtrain reference: +0.17 m/s, +0.27 m/s, +0.08 m/s², +0.055, −0.9 s, +0.67 m.
- **A navtest speed-MAE increase above +0.25 m/s is flagged to the PI as a label-quality regression** before N2-S may become a student label.

**STRATEGIC:** PAUSED (PI, 2026-09-19).

## 11. Run (for the MM; CPU only, at most 2 processes, 0 GPU)
**Chain** `tf_n2_chain.py`: a copy of `../n1/a1/tf_n1_a1_chain.py` with these stages.

| stage | what it does |
|---|---|
| `prep` | export at the landed tip and write `N2_PINS.json` |
| `cderiv` | the navtrain C-DERIV |
| `build` | `tf_n2_build.py`: rollouts → RW105, R and Wreg plans → `refe_tfn2_{rw105,r,wreg}.npz` W3 seams |
| `cidh` | T re-score plus the C 200-token subset; **STOP on FAIL before any N2 arm is scored** |
| `score` | RW105, R and Wreg through `score_p4_a1.py` |
| `cert` | assemble N2-S from the CSVs |
| `families` | as in §10 |
| `read` | the analysis, a copy of `tf_n1_analyze_a1.py` |

- Launch: WMI detached, `CreateFlags 0x08000200`, `< NUL`, as N1-A1.
- The chain waits for FreeVirtualMemory > 10 GB before each child.

**Not yet written:** `tf_n2_build.py`, `tf_n2_chain.py`, `tf_n2_analyze.py`, `N2_PINS.json`. They are the implementation step before the freeze (see `HANDOFF.md`). Each needs a mutation-red test like A1's (22 tests).

**Cost (ESTIMATED from this probe's and A1's MEASURED per-token times):**

| stage | estimate |
|---|---|
| build | 2 × P5-R + 2 × iterate20 ≈ 2.5 to 3 s/token ⇒ **9 to 10 CPU-h** |
| harness | 3 seams + the T re-score, 0.5 s/token each ⇒ **≈ 6.8 CPU-h** |
| families | 5 arms ⇒ **≈ 8 to 11 CPU-h** |
| **total** | **≈ 24 to 28 CPU-h, ≈ 12 to 14 h wall on 2 processes, 0 GPU** |

## 12. Evidence and its limits
- navtrain held-out only: 469 tokens in 13 logs, on DB-rebuilt scenery, which is not harness-identical.
- **The warp factor and the CERT order were chosen on those rows** (W110 was refuted on them). That is why this navtest read is a confirmation with its own bars, and the factor 0.05 and the order are frozen here.
- TF-2b showed CERT's navtrain gain is log-dependent (0.000 on fresh logs).
- The standstill-heavy logs were not in the probe.
- E3 outcome A (94.85 vs 94.50) places the gap in protocol and labels, not weights. The warp is a label-side stand-in for a protocol difference this agent could not identify. The re-roll levers are the test of that hypothesis.
