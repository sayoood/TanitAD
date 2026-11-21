# PREREG (DRAFT, not yet adopted) - L1: is test-time search on REFe's own scorer admissible for EP?

Written 2026-10-18 by the LIT-EP-RL literature pass. **No outcome number exists for this study; nothing in it has been computed.**
It is a DRAFT for the Master Mind to adopt or amend, and the sha256 beside it (`PREREG_L1_SEARCH_ADMISSION.md.sha256`) is taken before any
run. Whoever runs it copies this file into the run package unchanged and may only ADD an addendum, never edit this text.

## 0. Why this study, in one paragraph (each number carries its evidence class and source in `raw/published_rows.py`, `raw/verify_published_rows.json` and the LIT-EP-RL hand-back of 2026-10-18)

TOAD (arXiv 2606.07170, Table 5 p14) searches around the planner's pick with a cross-entropy method against a DrivoR-type scorer and reads
EP +1.1 on iPad, +1.7 on Hydra-MDP, +2.0 on GTRS and +7.4 on ZTRS (that last one with NC -0.4, TTC -0.8), but only +0.2 on DrivoR itself
(PUBLISHED, authors' evaluation, NAVSIM v1 navtest PDMS). Its text says search needs a scorer that is accurate OFF the proposal set.
REFe's scorer has two properties the paper did not test: its measured top-8 EP concordance is 0.887 and TTC 0.722 (MEASURED,
D-REFE-SELRANK-1, in-fan only), and it scores a trajectory IN ITS SET (`refe/model.py:699` docstring: the decoder self-attends over the set).
The programme's one earlier search result is on the teacher, not the scorer: DriveRL value-guided TTS (N=16) changed the action in 1 of 120
failure tokens (P4 F1, `eval/RESULT_P4_TEACHER_NAVTEST.md`). So the first question is admissibility, and it costs no search.

## 1. Fit set and what is never touched

* **Fit set:** navtrain held-out sets on which the scorer under test (P2-s0, `model_final.pt` sha256 `ea31db22...`) has not been trained.
  The existing dev table (3,137 sets / 24 logs, `D-REFE-SELRANK-1`, `D-REFE-HN-CENSUS-1`) is the candidate; **whether P2-s0 saw any of those
  sets is UNVERIFIED in this pass** and is gate G0 below. Sample 400 sets, stratified by log, seed 20260927, before reading any label.
* **navtest is not opened by Step A or Step B.** It is opened once, only on outcome O1 of Step B, for the confirm.
* Scorer under test: P2-s0 only (one training seed; H-ESTIM-SEED-1 applies and is named in every verdict line).

## 2. Gates that must read before any outcome (each has a control that must fail)

* **G0 provenance:** the 400 sets share no log with P2-s0's training manifest. Control: a deliberately overlapping list must make G0 fail.
* **G1 label identity:** the CPU labeller (`refe/navsim_pdm_targets.score_plans` on the metric cache) reproduces the banked PDM sub-scores of the
  64 in-fan proposals on 20 sets to <= 1e-6. Control: a shifted trajectory must change the label.
* **G2 set scoring identity:** scoring the unperturbed fan through the perturbation code path reproduces the banked scorer output bit for bit.
  Control: dropping 1 of 64 proposals from the set must change the other 63 scores (it is expected to, because of set self-attention; if it
  does NOT, the set-dependence premise is false and section 5's caveat is withdrawn).
* **G3 reachability from the caller:** the perturbation flags reach `score_trajectories`; a mutation that deletes the perturbation line must make
  the check go RED (CLAUDE.md "built, tested, and unreachable" rule).

## 3. Step A - admission (no search)

For each sampled set and each of the scorer's top-8 proposals plus 8 random proposals (so the anchor choice is not the scorer's own), build
perturbations: speed-profile scale s in {0.85, 0.92, 1.08, 1.15} (longitudinal) and lateral offset in {-0.3, +0.3} m (lateral). Replace the
anchor by each perturbation inside the full 64-set, score the set with the scorer, label the perturbation with the true PDM sub-scores.

Measured per family (never pooled):
* **A1 EP concordance (longitudinal):** among (anchor, perturbation) pairs whose true EP differs by >= 0.05, the share where the scorer's EP
  head orders them the same way. Reference: the in-fan 0.887.
* **A2 TTC concordance (longitudinal):** same for TTC. Reference: the in-fan 0.722.
* **A3 aggregate concordance:** the scorer's aggregate (`refe/planner.py:537 aggregate`, V1_W=(5,5,2)) against the true PDMS.
* **A4 exploit rate (lateral and longitudinal):** among pairs where the scorer aggregate rises by >= 0.02, the share where the true PDMS FALLS by
  >= 0.10. This is the quantity a search climbs.
* **A5 DAC/NC exploit rate:** same, restricted to pairs where the true NC or DAC goes from 1 to 0.
* Tactical (manoeuvre class of the perturbed plan) is reported as the share of perturbations that change the command-class; STRATEGIC is PAUSED (PI ruling 2026-09-19, ITEM 25), reported as PAUSED, not as a pass.

Estimator for every interval: paired log-cluster bootstrap, 10,000 resamples, seed 20260927. It answers "would another draw of LOGS say
this" and nothing else: not another training seed (H-ESTIM-SEED-1), not another search seed.

## 4. Pre-committed outcomes and the arm each one launches

| outcome | rule (fixed now) | next arm, committed now |
|---|---|---|
| **O1 ADMIT** | A1 point >= 0.80 AND A4 point <= 0.10 AND A5 point <= 0.05 | **Step B**: CEM search, K=5, M=64 per iteration scored as a full set, elites M/8, warm start at the scorer's pick, 3 search seeds; endpoint = true PDMS of the search output minus true PDMS of the pick, on the same 400 sets, paired log-cluster bootstrap. Bar: dPDMS >= +0.30 with the interval's lower end > 0, AND point estimates of dNC, dDAC, dTTC, dC each >= -0.25. Only a Step-B pass earns the single navtest read. |
| **O2 NOT ADMITTED** | A1 < 0.80 OR A4 > 0.10 OR A5 > 0.05 | **Do not search.** Step A has just produced a labelled set of off-proposal pairs: use it as the training data of **L2** (the SR-TTC extension: EP-output contender pairs and designed negatives, `raw/lever_ranking.csv` row L2). Search is retried only after L2's scorer re-passes this Step A. |
| **O3 INCONCLUSIVE** | the interval of A1 or A4 contains its gate | widen the fit set with a second carve of navtrain held-out logs (not navtest); re-run Step A once; if still straddling, treat as O2. |
| **O4 INERT SEARCH** (Step B only) | the search output differs from the pick by > 0.5 m ADE on < 5 % of sets | report INERT, not refuted: the precedent is P4 F1 (1 of 120). Next arm: raise M to 256 and widen the sampling covariance once; if still < 5 %, close L1 and go to L2. |

A Step-B result that clears the bar on EP but breaches a safety-family bound is reported as a TRADE, with the numbers, and the next arm is the
same search restricted to the DAC/NC-safe subset of candidates by the scorer's own DAC and NC heads (no new head).

## 5. Design notes the runner must keep

* **Scoring is set-conditional.** Every candidate population must be scored as one complete set; a single trajectory scored alone is not the
  quantity the scorer ranks. Elites are re-scored inside the original 64-fan for the final pick. (Source: `refe/model.py:699` docstring.)
* **Third variance:** the search samples, so the same scorer evaluated twice differs. Step B reports the spread across the 3 search seeds beside the
  interval; an effect below that spread is not an effect.
* **Goal and situation paths stay information-disjoint** (CLAUDE.md goal rule): the search uses the scorer's output only; no situation-classifier
  output enters the objective.
* **Compute:** Step A is CPU labelling plus CPU or dev-box scoring on the cached `visual_ctx` if `refe/ctx_cache.py` covers the sampled sets
  (UNVERIFIED whether it does); Step B <= 4 dev-box GPU-h, ESTIMATED. No A40 time. The dev-box GPU lock protocol applies.
