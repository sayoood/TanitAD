# PREREG_EXEC_STANDSTILL: does the executor's standstill heading drive v8F's launch crawl? (Master Mind, 2026-11-06; frozen when written, before any rollout of the variant)

Claim row: `H-EXEC-STANDSTILL-1`. Tier: T1-kin (own executed motion, logged cameras), eval-141. Zero training, CPU.

## 1. Why
* **The mechanism.** MEASURED by AGENT F, `t2v8f/AUDIT_NUREC_SCORE.md`.
  * At standstill, a plan moves about 1 cm. The playback executor sets the car's heading from the tangent of any step longer than
    `EPS_M = 1e-3` m.
  * The executed history then carries yaw rates of hundreds of °/s: P10's median on 18 standstill-launch windows of T1 eval-141 is
    283 °/s.
  * A stopped car cannot rotate, so these values are physically impossible.
* **Why it may matter.** v8F reads its ego features from that history, including curvature κ = ψ̇ / v, which diverges as v → 0. The
  one failing row of the adopted line is the launch crawl M1. Its excess enters through the ego-history input (report v41 §7.35), and
  it sits exactly at launch from near-standstill. Training data never shows a stopped car yawing at 283 °/s.
* **Not yet tested.** The 2026-09-26 yaw-rate mask fixed the METRIC (scoring tangent-less steps). Nobody has tested the INPUT side.

## 2. Hypotheses
* **[HA] ARTIFACT-CONTRIBUTES.** With a physically consistent executor, C0's M1 excess against P10 shrinks.
* **[H0] NO-EFFECT.** The excess does not move.

## 3. Design
* **Variant executor `hold`.** The executor's heading follows the tangent only when the executed step exceeds ε = 0.1 m per 0.2 s
  knot. Below that, the heading is held, so the yaw rate is 0. ε equals the reader's own standstill mask (ds ≤ 0.5 m/s × 0.2 s).
  Everything else is unchanged:
  * positions are executed exactly as planned;
  * same replans (3 × 10 knots), same windows, same causal bucket.
* **Arms.** C0_s0, C0_s1 and P10, each under `base` (the banked T1-kin executor) and under `hold`. DAGOACC_s0/_s1 are run beside.
* **Primary.** E = M1 crawl share of C0 minus P10, per seed, under each executor. The contrast is ΔE = E_hold − E_base.
* **Metric and estimator.** The T1 reader's own M1 function, all 19 rows of Read A beside. Paired episode-cluster bootstrap, B = 2000,
  seed 0, eval-141. Lateral and heading rows are reported per start class (launch / moving), never pooled.
* **Diagnostic.** The share of launch windows whose executed-history yaw rate exceeds 60 °/s, under base and under hold.

## 4. Gates (each with a deliberate-regression arm that must go RED)
* **G-ID.** With ε = 1e-3 (the base threshold), `hold` reproduces the banked base rollouts bit for bit. RED arm: ε = 0.1 must differ on
  launch windows.
* **G-MOV.** On windows whose every executed step exceeds 0.1 m, hold equals base exactly.
* **G-PHYS.** Under hold, no executed step shorter than ε changes the heading. RED arm: hold disabled.
* **G-P10.** P10's prior is unchanged. Only its executed history differs.

## 5. Outcomes, committed now (Rule Zero)
1. **ARTIFACT-CONTRIBUTES.** At both seeds, ΔE is separated-negative and |ΔE| ≥ 0.5 · E_base.
   * The executor is an instrument defect at standstill. The physically consistent executor becomes the T1-kin and T2 executor, by
     dated amendments beside the frozen T2 card and the T1 tool.
   * Read A's D2 is re-read under it for C0, with all 19 rows. If D holds at both seeds, that is a counted DOMINANT at T1-kin. It goes
     to the PI with this read as its instrument case.
2. **PARTIAL.** ΔE is separated-negative at both seeds with |ΔE| < 0.5 · E_base, or is separated at one seed only.
   * Adopt `hold` as instrument hygiene (it is physically correct) by the same amendments, and report the share explained.
   * T2 stage 2 runs under hold, with base beside.
3. **NO-EFFECT.** ΔE is not separated at either seed.
   * The executor artifact does not drive the crawl. T2 stage 2 proceeds under the frozen card as written.
   * `hold` may still be adopted for the lateral rows, by its own amendment.
4. **Any gate fails:** the instrument is fixed first, and no contrast is read.

## 6. Variance and validity named
* **The bootstrap** answers another draw of episodes. The training-seed question is answered by |ΔE_s1 − ΔE_s0|. The planner is
  deterministic.
* **What a positive read would mean.** The T1-kin crawl was partly manufactured by the instrument. That is an INSTRUMENT finding, not
  a model improvement, and it must be reported that way. It also means every T1-kin launch row banked so far carries this artifact.
  A blast-radius list of affected rows is part of outcome 1 or 2.
