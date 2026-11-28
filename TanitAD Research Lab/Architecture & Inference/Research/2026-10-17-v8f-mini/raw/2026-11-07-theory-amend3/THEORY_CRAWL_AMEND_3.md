# THEORY_CRAWL, Amendment 3: T1′ fell at P1′b; T5 (regression dilution of a noisy acceleration input) is pre-registered with P5 (Master Mind, 2026-11-07, dated amendment beside the frozen note `6e83451c` and Amendments 1 `9d39ac53` and 2 `0427c325`; written before any reliability of the acceleration input is measured)

## 1. Where the theory stands ([M] unless named)
* **T1** (contraction of the speed level) fell at P1. **T1′** (the planned acceleration is a mean over unresolved intent) fell at P1′b: Δ median ρ(CONFIDENT) − median ρ(UNSURE) is +0.245 [−0.028, +0.407] at s0, not separated (`wsp_read/P1PB_RESULT.md`). The pooled descriptive read leans the predicted way and decides nothing.
* **T2** (the data-processing frontier) stands: P3 put the admissible downstream levers on one chain, and the edit WSP-GAIN, which restored the kinematic gains by a weight edit, HARMS in all four cells (`D-V8F-WSPGAIN-1`).
* **The weight reading** (`wsp_read/WSP_READ.md` lines 31–44, held-out windows with v0 in 3–8 m/s):
  * the plan is s = s_base + s_net + s_res;
  * the non-learned base `kd_x` has ds/dv0 = 1.000 and ds/da0 = 1.899;
  * the learned parts bring the total to ds/da0 = 1.265 (s0) and 1.112 (s1), a gain ratio of 0.666 and 0.586;
  * **a0 is the 0.2 s backward difference of measured speed** (line 31).
* So the under-acceleration is, in the weights, a shrunken gain on one input that is a short finite difference. That is the textbook signature of attenuation bias.

## 2. T5. Regression dilution of the acceleration input [H]
* O is fitted by least squares. Let a0 = a + e, with a the true acceleration and e the error of a 0.2 s difference of a measured speed, independent of a.
* Then the fitted coefficient on a0 is the true one multiplied by the **partial reliability** λ = Var(a | other inputs) / Var(a0 | other inputs) < 1 (classical errors-in-variables). The base applies 1.899 to the noisy a0, so the learned parts must remove the excess: total ds/da0 ≈ λ · 1.899.
* **Why WSP-GAIN harmed under T5:** restoring the gain to 1.899 amplifies e by 1/λ. The plan then follows the noise of the input, which costs guards. That is consistent with READ 2, and stated post hoc, not as a test.
* **The lever T5 implies:** a better estimate of a from the same admissible source, the measured ego history (more speed samples, or a longer fit). That changes z, so T2 allows it to move the frontier, unlike every lever on the frontier so far.

## 3. P5 (pre-registered now; it can fail)
* **Windows:** the train windows of the base's clips (never eval-141), with v0 in [3, 8) m/s, all of them or a seeded sample of at least 20,000 from at least 500 episodes (name which).
* **a0 as the model sees it:** reproduce the model's own a0 from the logs on these windows. **Gate:** max |a0_reproduced − a0_model| ≤ 1e-4 m/s² on at least 1,000 windows, else NOT EVALUABLE.
* **a (label side; future allowed):** a local quadratic (Savitzky–Golay, order 2) fit of logged speed over ±0.5 s around the window's present tick, evaluated at that tick. This is the verdict estimator. Windows of ±0.3 s and ±0.8 s are reported beside, deciding nothing.
* **λ:** 1 − Var(e⊥) / Var(a0⊥), where e = a0 − a and ⊥ means the residual after a least-squares regression on the model's other ego inputs (v0, curvature or yaw rate, whichever the model reads; name them). The estimator is an episode-cluster bootstrap, B 2,000, seed 0.
* **P5 HOLDS iff** the λ interval's upper bound is below 0.90, and the gain ratio at each seed (0.666, 0.586) lies within λ ± 0.10. The tolerance is set above this rig's seed spread of 0.08.
* **P5 FAILS if** the upper bound is ≥ 0.90 (the input noise is too small to make the shrinkage), or if either seed's ratio lies outside λ ± 0.10.
* **NOT EVALUABLE** if the a0 gate fails, or fewer than 500 episodes qualify.

## 4. What each outcome sends next (committed now)
* **HOLDS:** the arm "M2b with a denoised acceleration input" is written as its own card before training. Its input is a longer backward fit over the measured speed history, the window chosen on train windows by reliability alone, never by a driving metric. Its argv is M2b's with only the ego-history computation changed, under the launch gate. The prediction: total ds/da0 rises toward 1.899 without separated-worse guard rows at T1-kin, and the M1 crawl share falls. Venue: the pod after B′ (about 1 h per seed at the measured 0.18 s/step) or box 2 (about 4.4 h per seed at the MEASURED M2b wall of 15,741 s).
* **FAILS:** dilution does not explain the shrinkage. The remaining explanation inside the weights is a learned response to the data, intent or the frontier. Then the perception lever (LoRA, P4) carries the crawl alone, as T2 already ranks it.
* **Beside, deciding nothing:** the R² of a on the frozen pooled tokens (a ridge fit, train folds only) says whether vision could denoise a, which T3 says a single frame cannot.
