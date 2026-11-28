# PREREG_HSGATE_MDFIX -- the HSGATE gate re-labelled to "gating fixes a missed decel", threshold under a FIT-side FD cap (Master Mind, sealed 2026-10-26 08:19:04 Berlin)

Successor arm to `PREREG_HSGATE` (sha256 2331e1f3...), whose `HSGATE8` read **FAIL (G2)** on 2026-10-26 ~08:10
(`result/HSGATE_VERDICT.{json,md}`, 2a14587e / d2fa5556): MD -0.050 / -0.045 (separated below H2CN), FD +0.0135 / +0.0124 (separated
above), G4 attribution PASS (the MD gain is window selection, not a mixing rate), G3 SHIP-R8 PASS. Its committed s5 next arm (same gate,
threshold under the FD cap) is INFEASIBLE FIT-side -- it collapses to `H2CN_DEPLOY8` (`next/FDCAP_FEASIBILITY_FIT.json`).

## 0. Disclosures (travel with every artifact of this arm)
* **Forking paths.** The MDFIX label was chosen AFTER the `HSGATE8` eval139 read, on FIT-row arithmetic only
  (`next/FDCAP_FRONTIER_FIT.json`; `next/DRAFT_NEXT_ARM_HSGATE_MDFIX.md`). This is the SECOND eval139 read of the gate family. The
  multiplicity is paid in the bars (s3): each bar is read at the interval level that makes it STRICTER.
* No MDFIX output on eval139 exists at sealing. The FIT-side descriptive numbers in the draft are 5-fold episode CV on 7,147 FIT rows, one
  training seed, one fit, no interval.

## 1. The arm `MDFIX8`
* Emitted positions, features `x` (the same 15 inference-available features), standardisation, FIT rows (L1C fresh split, 7,147 S6
  compliant rows), the fresh SELFSHARP trace, out-of-fold `P_H` (H2cn OOF, `folds_of(eid, 0)`): **exactly PREREG_HSGATE s1**.
* **Label (FIT rows only; GT allowed in labels):** `y = 1[GT-decel AND lonclass(P_F) == decel AND lonclass(P_H-OOF) != decel]`, with
  `lonclass` = `l2_common.lon_indicators` exactly as the MD / FD cells use it.
* **lambda = 0.1, FIXED** (HSGATE's registered value; not re-selected -- fewer forks).
* **Threshold (FIT side only).** On held-out OOF logits (5-fold episode-grouped, fold seed 1, lambda 0.1): `tau` = the logit threshold
  minimising FIT MD subject to `FD_FIT <= FD_FIT(H2cn-OOF) + c`; ties -> the HIGHER tau (fewer gated windows).
* **c is chosen by a power rule, FIT side, before any SCORE output (s1a).** Final gate = the all-FIT-rows fit at lambda 0.1, applied
  per window as `g(x) = 1[w.x + b >= tau]` (a deployable per-window logit threshold; no batch statistic). Its in-sample FIT F-rate is
  reported beside.

### 1a. The power rule for c (FIT side; computed and recorded BEFORE the gate is frozen)
For c in {0.002, 0.005} (ascending): compute the FIT-side MD and FD differences `MDFIX-OOF - H2cn-OOF` with an episode-cluster bootstrap
(B 2000, seed 0) over FIT episodes; project the eval139 standard error as `SE_eval = SE_FIT * sqrt(E_FIT_decel / 79)` for MD
(`E_FIT_decel` = FIT episodes holding a GT-decel row; 79 = eval139 episodes in the MD cell) and `SE_FIT * sqrt(E_FIT / 135)` for FD;
projected power of G1 = `Phi(|dMD_FIT| / SE_eval_MD - 2.241)` (one-sided at the s3 G1 level). **c = the smallest value with projected
G1 power >= 0.50 AND projected FD upper bound `dFD_FIT + 1.96 * SE_eval_FD` <= +0.010.** If neither qualifies, MDFIX8 is
**UNDERPOWERED** and is NOT scored on eval139; that is a terminal outcome (s5).

## 2. Controls (a failed control voids the bars)
PREREG_HSGATE s2 unchanged (K-ID0 / K-ID1, K-FRESH-SS, K-JOIN, K-OOF, K-EVALFEAT), plus:
* **K-LABEL:** the MDFIX label recomputed from the per-row `lon_indicators` arrays matches the stored label on every FIT row, and its
  base rate reproduces the draft's 0.0371 to 4 decimals (an independently derived expectation, not the code's own output).
* **K-TAU:** the FIT-side (MD, FD) at the frozen tau is re-derived by a second, independent scan and agrees exactly.
* **K-MUT:** the label with the `P_H` condition DROPPED (`y = GT-decel AND lonclass(P_F) == decel`) must yield a DIFFERENT frozen gate
  (sha differs) -- the label is load-bearing, not decorative.

## 3. Bars (S6 compliant, paired vs `H2CN_DEPLOY8` on the SAME draw, B 2000 seed 0; PASS = G1-G4 at BOTH inference seeds)
* **G1** MD separated BELOW `H2CN_DEPLOY8` on the **97.5 % percentile interval** (alpha 0.05 split over the family's two looks).
* **G2** FD NOT separated above `H2CN_DEPLOY8` on the **95 % interval** (the stricter level for a not-separated bar).
* **G3** SHIP-R8 (1)-(7)+(1b) on `MDFIX8` -> OVERALL PASS (Amendment 4's clauses, arm swapped; D8's conjunctions as in HSGATE).
* **G4 (attribution)** PREREG_HSGATE s3 G4 with `r` = MDFIX8's own SCORE-side S6-compliant F-rate, 95 % intervals.
* Beside: the 95 % G1 interval, the compliance row, F-rates per surface, E1 / E3, ADE, the four families vs `H2CN_DEPLOY8`, vs
  `HSGATE8` and vs `FACTORED_DEPLOY8`; A4d's fit-seed floor beside every H2cn-relative cell.

## 4. Tier / estimator / deployment note
PREREG_HSGATE s4 unchanged (T1 self-action OPEN LOOP on logged eval139 frames, not driving performance; paired episode-cluster bootstrap
over EPISODES only -- blind to training and inference variance; one refcv8 training seed; one gate fit; inference seeds never pooled;
the self-sharpened decode's latency is UNMEASURED and a PI item before any ship).

## 5. Committed outcomes
* **UNDERPOWERED (s1a)** -> no eval read; the 15 features cannot buy a detectable MD gain at an FD cost the bars tolerate. The refcv8
  post-hoc progress line ENDS; the lever is refcv9's trained lon decision (`PREREG_LONHEAD_C`, in flight).
* **PASS (G1-G4)** -> `MDFIX8` joins the PI's shipped-stage choice (with the latency item), labelled "second look, multiplicity-adjusted".
* **G2 FAIL** -> as UNDERPOWERED, plus: richer gate inputs are an architecture decision (PI / MM), not another relabel.
* **G1 FAIL (G2 PASS)** -> the FIT-side MD gain does not transfer at this F-rate; `H2CN_DEPLOY8` stands; the line ends as above.
* **G4 FAIL** -> "rate effect, not attributable"; same consequence as G1 FAIL.
No third relabel of this gate family is admissible on eval139.
