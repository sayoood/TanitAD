# Addendum A6 to PREREG_LEVELFLOOR_A5: the bounded-lift floor at κ 1, read with the logged speed and with the heads (written 2026-11-06 after the A5 read, before any A6 rollout or tool change)

**Why.** A5 read OPERATOR-IS-THE-PROBLEM (`raw/2026-11-06-levelfloor-a5-lf5-result/lf5_read/RESULTS.md`, landed 6048f4df).
* **LFO100 (exact level).** It closes 0.738 / 0.739 of the M1 excess inside the ADE6 guard, but G_O fails on DK-VIOL and cross-track
  MAE at both seeds.
* **Noise arms.** A priori noise of σ 0.10 already harms broadly (ADE6 +0.6140 / +0.5848 m).
* **The heads (LF100).** They overshoot: ADE6 +1.2559 / +1.0891 m.

The A5 card committed, for CEILING-HARMS, an arm that changes F in its own addendum. The descriptive diagnosis of the A5 rollouts, made
after the read (`a6_diag/A6_DIAG.md`), finds two mechanisms in the operator itself:
1. **max(s, s_f) turns level error of either sign into one-sided overshoot.** The heads' harm sits in windows where v_f exceeded the
   logged speed.
2. **With the exact level, the harm sits where F holds the first second's level constant over the 6 s plan** while the human slows
   later in it. All of DK-VIOL and of cross-track sit in those windows.

Both mechanisms are lifts larger than O's own plan supports. This card bounds the lift relative to O's planned speed.

Zero training, dev-box CPU only, no box-2 job. PI go of 2026-11-05 as in A3. Claim row: H-LEVELFLOOR-1 (A6).

**Adaptivity, disclosed:** the form and the δ doses were chosen after looking at eval-141 rollouts, and the A6 read is on the same
windows. A counted DOMINANT from this card carries that disclosure into the report.

## 1. Hypotheses
* **[H1]** With the exact level, the bounded floor closes at least half the M1 excess and passes G_O.
* **[H2]** With the heads, the bound removes the overshoot harm while keeping enough closure for P, G and D.

## 2. The operator F_B and the arms (κ = 1.0 in every arm; everything else as in A5)
The operator **F_B** is A5's F with one added term in the floor speed. v_O(τ) is O's planned speed after S, taken from consecutive knot
distances along its polyline / DT (0.2 s), with knot 0 at the origin:

v_f(τ) = min( κ · v_hat (with the lead cap exactly as in A3), v0 + 1.0 · τ, v_O(τ) + δ )

The rest is unchanged: s_f(t) = ∫_0^t v_f; s' = max(s, s_f); s' is resampled along O's polyline exactly as in A3; the floor applies
only when v_hat >= 3 m/s; placement as in LF (bare = S → F_B, composite = S → F_B → kbox).

Since v_f ≤ v_O + δ, the lift is bounded: s' − s ≤ δ · t at every knot.

**Doses:** δ ∈ {0.5, 1.0} m/s, fixed a priori. The M1 ticks need a median lift of about 0.7 m/s to leave the crawl (v0 median
4.48 m/s, `beside/M1_GEOMETRY.md` of A4), so δ 1.0 covers that median and δ 0.5 is half of it.

| arm | level input | class |
|---|---|---|
| **LFOB05_s{0,1}**, **LFOB10_s{0,1}** | v_hat := v_log, exactly as in A4/A5 | CEILING |
| **LFB05_s{0,1}**, **LFB10_s{0,1}** | v_hat from LLR's level head, exactly as in LF100 | ADMISSIBLE, vision-only at inference; the only arms that can count |

**Known answers before any A6 rollout** (each on a 64-window smoke, compared smoke against smoke and bit for bit, since the
batch-composition caveat of A5 forbids comparing a smoke with a banked full rollout):
* **δ = 0 is an identity.** F_B never binds, and the rollout reproduces the floor-off rollout. This is the analytic zero control.
* **δ = 10^9 is the A5 operator.** It reproduces the LF5 tool's κ 1 rollout for both level sources.
* **δ = 0.5 is partial.** Its bind rate lies strictly between those of δ = 0 and δ = 10^9, and the bound s' − s ≤ δ · t + 1e-6 m
  holds on every knot of every bound tick.

## 3. Reads (T1-kin; paired episode-cluster bootstrap B = 2000, seed 0, eval-141)
* **Rollouts:** 24 in all.
  * every arm, dose and seed, bare and composite, on `reindexed` (deciding): 16;
  * LFB05 and LFB10 also on `logged`, bare and composite, for the LEAK-BORNE rule: 8.
* **Bases:** the banked arrays, unchanged: RIC0, RIC0C, C0, C0C, C0REP, P10, and from A5 RILF100, RILF100C, RILFO100, RILFO100C.
* **Instruments (beside):** as in A5, plus the share of bound ticks where the δ term is the binding minimum, and the mean lift where
  bound.
* **Attribution (beside):** LFB{δ} − RILF100 and LFOB{δ} − RILFO100, per dose.

## 4. Rules (deciding scene `reindexed`)
* **The ceiling arms LFOB05 and LFOB10:** P_O, G_O and F_O exactly as in A4 (with LFO := the arm). Each gets one label:
  CEILING-CLEARS, CEILING-HARMS, CEILING-SHORT or F-UNREACHED. The composite − P10 is reported as the CEILING of D and is never counted.
* **The admissible arms LFB05 and LFB10:** the A3 rules exactly (P, G, F, D read A over four families, LEAK-BORNE on `logged`, reach
  control), with LF := the arm.
* **Multiplicity, named:** the heads-floor family will have read five doses (κ 0.50, 0.75 and 1.0, then δ 0.5 and 1.0 at κ 1). A
  counted DOMINANT names its δ and says that five were read.

## 5. Outcomes and the next arm, committed now (Rule Zero; read in this order)
1. **LFB05 or LFB10: D holds at both seeds, with P and G, and the arm is not LEAK-BORNE ⇒ counted DOMINANT.**
   * It names κ 1, its δ, the five doses read and the adaptivity.
   * Signal the PI for scale (data and size), with the video evidence.
   * If both doses hold, both are reported, and δ 0.5 (the smaller intervention) is named first.
2. **Otherwise, LFOB05 or LFOB10 is CEILING-CLEARS ⇒ the bounded form works given a good level, and the heads are the bottleneck.**
   * The estimator levers (the A40, a stronger estimator on more data) go to the PI, with the ceiling and the heads' measured error as
     the case.
   * The next CPU arm is a temporal filter of the heads' level at that δ, in its own addendum.
3. **Otherwise, LFOB is CEILING-HARMS at both doses ⇒ bounding the lift does not remove the operator's harm.**
   * The progress-floor family is exhausted on CPU, and T2 re-perception (option (c)) and the A40 are reported to the PI.
   * The next CPU item is pulled from BACKLOG.md.
4. **Otherwise (CEILING-SHORT at both doses, or SHORT at one and HARMS at the other) ⇒ the bound removes the closure.**
   * The floor's closure needs a lift larger than O's plan supports, and no measured estimator supports such a lift.
   * The family is exhausted on CPU, and T2 and the A40 are reported to the PI.
5. **F-UNREACHED at a dose:** the instrument is checked first.

## 6. Variance named
* **Episodes:** the paired episode-cluster bootstrap.
* **Training:** O is C0's own checkpoint in every contrast with C0, and the heads come from one LLR run per seed (not measured).
* **Inference:** every A6 arm is deterministic (no noise arm). The interval answers another draw of EPISODES only.

## 7. Freeze
This file's sha256 is recorded in `a6_prereg/FROZEN_SHA.txt` before any A6 rollout or tool change.
