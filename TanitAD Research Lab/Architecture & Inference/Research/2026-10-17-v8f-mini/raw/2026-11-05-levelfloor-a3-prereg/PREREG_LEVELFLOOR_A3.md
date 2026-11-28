# Addendum A3 to PREREG_LEVELHEAD_CL / A2: arm LF, the vision heads read OUTSIDE O as a speed floor (written 2026-11-05 21:1x after the LLR read, before any LF rollout)

**Why.** Three arms trained the heads and three read-outs measured O ignoring them. CL in C0's recipe: share closed 0.030 / 0.006. CLR in
RAPR-P01's recipe: 0.233 / 0.130, with nothing added over RAPR-P01. LLR, with both heads as a separate token: 0.235 / 0.195, with nothing
added over CLR. O's plan moves 0.016 / 0.005 m/s for a +3 m/s level, although the token's norm is 1.5 times the ego token
(`raw/2026-11-05-levelhead-llr-result/llr_read/SENSITIVITY.md`).

The A2 card's committed branch closes the family "at these forms". The one form not tried is the heads' outputs used WITHOUT O having to
learn to read them. The composite RILLRC fails D2 on the M1 crawl share alone (+0.0474 / +0.0504 against P10). Every other row is
better or not separated. This card puts the trained heads' estimates directly into the composite as a lower bound on the plan's
along-track progress: the decoupled longitudinal target-speed form of the end-to-end literature, with the path still O's.

Zero training, dev-box CPU only, no box-2 job. PI go: "Do c then a, contibue autonmously until scalabiliry readiness is achieved"
(2026-11-05). Claim row: H-LEVELFLOOR-1.

## 1. Hypothesis
The crawl persists because O's speed level follows its own executed history. A vision-only floor on the plan's progress breaks that
loop without O reading anything new. Inside the crawl, where O's speed is below half the logged speed, a floor at a fraction of the
head's estimate lifts the plan out of the crawl. Outside it, where O is faster than the floor, the plan is unchanged bit for bit.

## 2. The arms (no training; every checkpoint exists and is banked)
* **LF050_s{0,1}** and **LF075_s{0,1}**, for seed s.
  * **Planner:** C0_s{s}, `v8f-recover/M2b_s{s}_recft_C0/ckpt_last.pt`. It is the base's own checkpoint, so LF − C0 isolates the floor
    and no O training variance enters it.
  * **Floor heads:** from LLR_s{s} (`v8f-recover/M2b_s{s}_recft_LLR/ckpt_last.pt`, sha256 as `LLR_PRECHECK.json`). LLR's encoder,
    adapter, `lvl_head` and `lead_head` read the SAME tick batch, in a second forward. LLR's O is not used.
  * **Dose:** κ = 0.50 (LF050) and κ = 0.75 (LF075), both fixed now.
* **The floor operator F**, applied at every replan:
  * **Inputs:** the plan p [30, 2] (ego frame, knots k = 1..30 at DT = 0.2 s); v0, the tick's measured ego speed (`batch["p10f"][:, -1, 0]`,
    measured ego history); and, from the floor heads on the tick frame, v_hat (m/s), π = sigmoid(presence logit) and g = 40 · gap_hat (m).
  1. Floor speed: v_f = κ · v_hat if v_hat >= 3 m/s, else no floor (3 m/s is the crawl band's lower bound).
  2. Lead cap, if π >= 0.5: v_f = min(v_f, max(0, (g − 2 m) / 1.5 s)). This is the car-following equilibrium speed with a 2 m standstill
     gap and a 1.5 s time gap, a priori constants in the range of standard car-following models (not tuned, not quoted as PUBLISHED).
  3. Floor progress: s_f(t_k) = ∫₀^{t_k} min(v_f, max(v0, 0) + a_f t) dt with a_f = 1.0 m/s² (a priori comfort acceleration).
     Where v0 >= v_f, s_f = v_f t_k.
  4. New progress: s'_k = max(s_k, s_f(t_k)), with s_k the plan's cumulative arc length from the origin. It is non-decreasing because
     both terms are.
  5. The plan is resampled along its own polyline (origin prepended) at s'_k, extrapolated beyond its last knot along the last
     segment's direction. The path geometry is O's. If s'_k = s_k at every k, the plan object is returned unchanged.
* **Placement:**
  * bare arm X: `--smooth off --feasible off`, F on O's raw plan;
  * composite XC: O's plan → S (sg72) → F → kbox (`--smooth sg72 --feasible kbox`), the composite of every earlier D read with F
    inserted before the curvature clamp.
* **Vision only at inference:**
  * v_hat, π and g come from LLR's heads, whose vision-only guards are in the model build (83c9ee0).
  * v0 is the measured ego speed.
  * No nav, route, limit, label or situation-classifier output enters F, so the goal-input ruling does not apply.
* **Nothing here was chosen on eval-141.** κ ∈ {0.50, 0.75}, a_f, the 2 m / 1.5 s cap, the 0.5 presence threshold and the 3 m/s band
  are fixed in this card, before any LF rollout.
  * κ 0.50 is the crawl definition's own ratio.
  * κ 0.75 is halfway between that ratio and the estimate itself (κ 1).

## 3. Reads (T1-kin on the RI tool + F; paired episode-cluster bootstrap B = 2000, seed 0, eval-141)
* **Rollouts:** per arm and seed, X and XC on `reindexed` (**deciding**; 12 threads) and on `logged` (LEAK-BORNE only; 16 threads).
  16 rollouts in all.
* **Bases:** the banked arrays of the CL / LLR reads, unchanged: RIC0, RIC0C, C0, C0C, C0REP, P10.
* `hold_now` is not read. A frozen frame makes v_hat constant, so it carries no information about a frame-reading floor. This is
  disclosed.
* **Reported beside each arm** (instrument, never deciding):
  * the share of replans where F binds (s' ≠ s at any knot), overall and inside the M1 crawl ticks;
  * the share where the lead cap binds;
  * the mean lift of the 0–3 s plan speed where F binds;
  * v_f against the logged speed at the matched route point on the deciding scene: MAE, and the share where v_f exceeds it.
* **Reach control:** F must bind on at least 1 % of the replans inside the M1 crawl ticks at each seed. If it binds on fewer, the read is
  labelled F-UNREACHED and no rule below is read.

## 4. Rules (the CL card section 4 with CL := LF, each dose separately, deciding scene `reindexed`)
* **P:** share closed = (RIC0 − RILF)[M1] / (RIC0 − P10)[M1] >= 0.5 at both seeds, with RILF − RIC0 on M1 separated-better at both.
* **G:** on RILF − RIC0, ADE6 point <= +0.0604 m and no non-crawl row of the 19 separated-worse, at both seeds.
* **F (floor rule):** |RILF − RIC0|[M1] > |C0REP − C0|[M1] at each seed.
* **D (counted DOMINANT):** RILFC − P10 read A: D1, D2 (no row of 19 separated-worse) and D3 (four families, STRATEGIC included), at
  both seeds, with P and G holding.
* **LEAK-BORNE:** if the `logged` share minus the `reindexed` share exceeds half the `logged` share at either seed, the arm is labelled
  LEAK-BORNE whatever else it reads.
* **Multiplicity, named:** two doses are read. A counted DOMINANT names its dose and says that two were read.

## 5. Outcomes and the next arm, committed now (Rule Zero)
* **D holds for a dose at both seeds:** counted DOMINANT. Signal the PI for scale (data and size), with the video evidence.
* **P holds, G or D fails:** the failing rows name the harm.
  * Failures only in the DK rows: the next arm is the same dose with a 2.0 s time gap.
  * Failures in ADE6, speed MAE or along-track MAE: the next arm is the dose midway to the lower dose (or κ 0.375 below 0.50).
  * Each next arm gets its own addendum, and costs zero training.
* **P fails at both doses, M1 separated-better at either:** the floor binds but the head's estimate is too weak to lift the crawl.
  The level estimator is the bottleneck. Its levers are motion from vision (option (c): T2 re-perception, the A40) or a stronger
  estimator trained on more data. These are reported to the PI with the binding and v_f-error instruments.
* **P fails, M1 not moved, or F-UNREACHED:** the floor does not bind where the crawl is; the instrument is checked first.
* **LEAK-BORNE:** reported as the camera-clock effect, never as a result. Next as for "P fails".

## 6. Variance named
* **Episodes:** the paired episode-cluster bootstrap answers another draw of episodes.
* **Training:** O is C0's own checkpoint in both arms of every LF − C0 contrast, so O's training variance is absent from it. The heads
  come from one LLR training run per seed. That variance is not measured, and rule F reads against the C0REP floor as before.
* **Inference:** the planner and F are deterministic, so no inference seed applies.

## 7. Freeze
This file's sha256 is recorded in `a3_prereg/FROZEN_SHA.txt` before any LF rollout.
