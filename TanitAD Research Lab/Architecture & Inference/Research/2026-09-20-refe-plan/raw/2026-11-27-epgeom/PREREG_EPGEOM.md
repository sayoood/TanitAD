# PREREG_EPGEOM: a geometric progress (EP) surrogate inside the confidently-safe set (plan L3, `H-REFE-EPGEOM-1`)

REFe-line agent, 2026-11-27. Modelled on `PK/raw/2026-11-27-ensscore/PREREG_ENSSCORE.md`; its estimator, bar and outcome structure are reused verbatim.
**State at freezing (disclosed):**
- Code written; 5/5 synthetic tests PASS, including CTRL-ID and three mutations that each turn the checks RED (§6).
- The navtrain fit **has been run** (§3). Its result is a fact at freezing: **fit_outcome = NULL-HELDOUT** (values in `FROZEN_PARAMS.json`, `FIT_REPORT.json`; the card commits the PROCEDURE, not values).
- **No navtest PDMS or selection outcome exists for any L3 rule.** Navtest files were opened for schema only (keys, shapes, token join).
- Disclosed prior navtest read of a related quantity: `propscore/EP_GEOM_CHECK.json` (descriptive means of C1's p_ep / path length / end_x at its own pick, snooped). It is not a rule outcome.

## 1. Question
C1 = 92.5548 (MODEL_REGISTRY 14.6, T1, 12,146 tokens). Plan L3 targets the EP (progress) share of C1's selection regret (2.47 pts, oracle-of-64 decomposition). Does choosing, among proposals C1's own safety heads call confidently safe, the one with the most geometric progress raise tracked PDMS? Parameter-light, CPU, table lookup.

## 2. The rule (exact)
- Inputs: C1's own head logits [64,6] (order NC, DAC, EP, TTC, C, DDC), the executed 8-pose plan `props8` [64,8,3], C1's own pick. **Vision-only rule:** the inputs are model outputs plus the nav command (already inside C1's forward); the geometry comes from the proposal itself. No map, agents, future, PDM-Closed or metric cache is read at inference. No goal input is added; no situation-classifier output enters.
- Safe score `q_k = sig(NC)·sig(DAC)·sig(TTC)·sig(DDC)` (float32). `S_tau = {k : q_k >= tau}`.
- If `S_tau` is empty: C1's own pick. Else: argmax over `S_tau` of the surrogate `g_k`. Ties (exact float32 equality of g): C1's pick if tied, else argmax of C1's `navsim_v1` aggregate over the tied.
- Surrogates (pre-declared): **(a) `arc`** arc length of the 8-pose plan including the origin→pose-1 segment; **(b) `x8`** final forward displacement; **(c) `pep`** C1's own sig(EP).
- tau grid (12): 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995.
- Code: `epgeom_lib.py::select`.

## 3. Fit procedure (committed; values are NOT in this card)
- Data: navtrain cache `F:/safehead_cache/out/<log>.npz` (index `F:/TanitAD_safehead_tmp/adapted/index.csv`), keys `props8, logits, pick, tr_*` only. 2,272 tokens, 35 logs, 6 held-out.
- Fit: on the **train** split only, over 3 surrogates × 12 taus, maximise the mean tracked `tr_pdms` of the pick. Exact ties: earlier surrogate, then higher tau.
- Checks, in this order:
  1. **NULL-FIT** if the fitted rule's train mean ≤ C1's train mean.
  2. **NULL-HELDOUT** if the fitted rule's mean on the 6 held-out logs is below C1's pick there (point, `not_below_c1` false).
  3. Otherwise **PASS-HELDOUT**: the navtest decision read may run.
- The fitted values go to `FROZEN_PARAMS.json` (+ `.sha256`) **before any navtest read**; the read refuses a frozen-file sha256 mismatch and any `fit_outcome` other than PASS-HELDOUT.
- Caveat (stated up front): C1 trained on navtrain, so these are train-distribution tokens for C1; the held-out logs are held out from this fit, not from C1's training. Evidence MEASURED, tier T0 (single-step open-loop planning labels, tracked PDM; labels are a REBUILT metric cache, mean |d| 0.0008 vs NAVSIM's, INHERITED).

## 4. Tier, estimator, population (decision read, only if PASS-HELDOUT)
- **T1**, NAVSIM v1.1 tracked PDMS from the banked C1 64-proposal tables `full_C1_s{0,1}of2.jsonl` (CTRL-TAB: C1's pick reproduces 92.55477798611808). All 12,146 navtest tokens, 136 logs; geometry from `F:/safehead_cache/navtest/<log>.npz` joined by token; the dump must be COMPLETE or the read refuses.
- **Paired log-cluster bootstrap**, 10,000 draws, seed 20260927, 95 % percentile, full_set means (`Boot` copied verbatim from `p5c_read.py`). It answers *"would another draw of LOGS say this?"* only; it is blind to training and to a different checkpoint.
- Measures: **M1** Δ PDMS ×100 with CI; **M2** sub-score deltas NC, DAC, TTC, EP, C (judged), DDC (reported); **M4** the four families (LONGITUDINAL, LATERAL, TACTICAL, STRATEGIC in full, paired, each with CI and n) computed by the pinned family instruments on the confirmation-seam rows at the new picks; **M5** switch rate.
- F_REFe: 2·max(F, F_REFe) = **0.2316** (ENS card §4).

## 5. Committed reading (both outcomes written before any navtest number)
- **P1** CI-lo(Δ) > 0. **P2** Δ point ≥ 0.2316. **G** no separated-adverse sub-score (CI-hi < 0) among NC, DAC, TTC, EP, C. **R** (replicate; there is no second checkpoint's table) = Δ point > 0 on both log-parity halves (sorted logs, even vs odd). R is weaker than ENS's R and ADOPT is labelled *single-checkpoint*. **All §6 controls PASS**, else INVALID.
- **ADOPT** = P1 ∧ P2 ∧ G ∧ R ∧ M4 with no separated-adverse family component ∧ CTRL-T(c) on the confirmation seam. A SPEC amendment that the MM lands; the registry row prints BELOW GOAL if the point is below 94.55. Next: L3-SH (below) and L2.
- **SEPARATED-SMALL** = P1 ∧ G, P2 or R fails. Not adopted. Next: EP-HEAD.
- **NULL** = CI of Δ contains 0. Not adopted. Next: EP-HEAD.
- **ADVERSE** = CI-hi(Δ) < 0 or G fails. Not adopted; each separated-adverse sub-score is named. Next: EP-HEAD.
- **NULL-FIT / NULL-HELDOUT** = §3 outcomes. **No navtest decision read is run**; navtest is untouched. Next: EP-HEAD.
- **INVALID** = any control fails; all numbers descriptive.
- **Committed next arm on every non-ADOPT outcome (Rule Zero): EP-HEAD.** Head-only retrain of C1 `score_head` row 2 (EP) on navtrain tracked `tr_ep`, same recipe as L2 SAFE-HEAD (`PK/raw/2026-11-27-safehead/PREREG_SAFEHEAD.md`), then selection by C1's aggregate with the new EP row. It needs a GPU or CPU-fit decision under its own card.
- **L3-SH (conditional secondary arm):** the same rule with `q` built from the L2 SAFE-HEAD probabilities for rows 0, 1, 3, 5 instead of C1's logits. **Run only if L2 is ADOPTed.** `epgeom_lib.safe_q_from_probs` is the hook (tested via `select(q=...)`); `epgeom_read.py` is NOT wired to L2 outputs yet, because they do not exist.

## 6. Controls (any FAIL = INVALID)
- **CTRL-ID:** tau above every q (1.5) reproduces C1's pick bit for bit, on every navtrain token (in the fit) and every navtest token (in the read), for the frozen surrogate; the unit test covers all three surrogates.
- **CTRL-RED:** the surrogate misaligned by one proposal index changes the pick on ≥ 1 % of tokens (the surrogate reaches the pick).
- **CTRL-JOIN:** the navtest dump's stored `pick` equals the table's `pick` on every token (same proposal order and model); coverage 12,146.
- **CTRL-TAB:** the table's C1 pick PDMS = 92.55477798611808 to 1e-6.
- **CTRL-AGG** (descriptive): the numpy aggregate argmax equals the stored pick (navtrain: 1.0000 at freezing).
- **CTRL-CODE:** every file equals `CODE.sha256`. **CTRL-SHA:** frozen-file sha256.
- **CTRL-T(c)** (confirmation seam): harness CSV at the new picks equals the table at the new picks to 1e-6.
- **Unit tests** (`test_epgeom.py`, 5/5 at freezing): CTRL-ID, tau=0 equals argmax g, mask excludes the unsafe, empty-set fallback, origin segment in the arc length (hand value 10), tie rules, RED, planted-signal fit, seeded Boot, aggregate by hand. **Mutations** that each go RED: mask ignored; origin segment dropped; empty-set fallback removed.
