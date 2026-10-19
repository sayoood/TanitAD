# PREREG_BRAKE_LOCUS -- AMENDMENT 3 (2026-10-19): two inference-only ALLOCATION-HEDGE arms (H1, H2) + the re-roll plan

Recorded BEFORE any re-rolled number exists: `code/brake_locus_reroll_r8.py` has produced no banked window when this file is sealed (seal `PREREG_BRAKE_LOCUS_AMENDMENT3.sha256`; the roll and its analysis REFUSE to run unless the original prereg and Amendments 1, 2, 3 all match their seals).  The original, A1 and A2 are unchanged.  Amendments add; they move no threshold and not the section-5 label rule.  Requested by the coordinator 2026-10-19; the 50 % bar below is the coordinator's.

## 0. Honest limits (read first)

* **Pre-seal knowledge (disclosed).**  I have read `refc.py` `_r8_extend` (2486-2570) and `refcv8_conditioning.allocate` (395-432).  Banked, already-known numbers used below: r8 / r7 false-decel on GT-cruise compliant S6 windows, seed 0, 0.265625 / 0.091056 (`L2_HEADROOM.json`, literal C1 of the original), i.e. an r8 - r7 EXCESS of **0.174569**.  No re-rolled number is known.
* **Mechanism being tested is a SOURCE-READ HYPOTHESIS (HYPOTHESIS, not measured).**  TRAINING always forces the GT joint hypothesis into the 32 allocated candidates with floor `force_min` = 4, ungated by the teacher coin (`refc.py:2512-2513`, `refcv8_conditioning.py:410-415`; the trainer's `tch["min"]` = 2 x `alloc_min_each` = 4).  INFERENCE allocates top-4 joint hypotheses (floor 2 each, `alloc_top_k` 4, `alloc_min_each` 2, `n_alloc` 32 from the run's `config.json`) plus the remainder in proportion to the head posterior, and allocated candidates are emittable (`alloc_emit` True).  A false decel call therefore fills the pool with braking candidates, a situation training never saw.
* H1 / H2 are **eval-only probes of pool composition**, not deployable fixes: H1 hard-codes a lon class (CRUISE) and is not a goal input.  Admissibility: it uses no GT, no situation-classifier output, no ego state beyond what the model already consumes; the lateral half of the forced joint is the model's OWN argmax lateral hypothesis.
* Tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**.  Evidence class: MEASURED when run (ours, CPU re-roll, EMA-30k `E:/refcv8_eval_kit/ckpt/ckpt_ema_30000.pt`).  Estimator: paired episode-cluster bootstrap (B 2000, seed 0, one shared resample matrix per comparison); it answers "another draw of EPISODES" only; NOT another training run (single seed, `H-ESTIM-SEED-1`), and the INFERENCE-seed question is answered only by the N2 arm (one extra draw).  The CPU re-roll is **not bit-equal to the GPU banked dumps** (stochastic DDIM, different RNG stream and fp32 numerics, R6 of the design); every comparison below is re-roll vs re-roll on the same windows, same per-window seed.

## H1. HEDGE-FLOOR

`refcv8_conditioning.allocate(joint, m, top_k, min_each, force, force_min)` is called with `force` = the joint id `lat * 6 + 1` and `force_min = 4`, per batch row.  Exact indices from source: `LON6 = (FOLLOW, CRUISE, BRAKE_TO, CREEP, HOLD, ACCELERATE)`, **CRUISE = LON6 index 1 (v7 id 1)**, `refcv8_conditioning.py:63-64`; `LAT3 = (LANE_KEEP, TURN_L, TURN_R)`; joint id = `lat3 * N_LON6 + lon6`, `split_joint` (`:94-95`); `lat` = argmax over the row's joint marginal (= argmax of the head's lateral posterior).  This is the training mechanism (`force` + `force_min` 4) with the GT joint replaced by (own argmax lat, CRUISE).  Native `allocate` semantics then apply unchanged: the forced joint gets its floor of 4 (taking the place of the 4th-ranked hypothesis if it was not already in the top 4), the other three selected hypotheses keep floor 2, and the remaining 22 candidates are shared in proportion to the posterior over the selected set (native: 24 over the top 4).  Total allocated stays 32, fan stays 149.

## H2. FLAT-ALLOC

`allocate` receives `log_softmax(0.5 * logp_joint)` (posterior raised to the power 0.5 = temperature 2, renormalised over the 18 joint hypotheses); everything else native.  Order of the top 4 is unchanged (monotone), only the proportional shares flatten, so H2 can NOT introduce a hypothesis outside the native top 4 (stated in advance).

## Implementation constraint (committed)

Inference-only, a patch applied inside the re-roll PROCESS ONLY: `refcv8_conditioning.allocate` (looked up as `r8c.allocate` at call time, `refc.py:2513`) is replaced by a wrapper that is the identity unless an arm flag is set for that single forward, and is restored afterwards.  No repo file is edited; no default code path changes.  Controls: the wrapper is installed for EVERY arm including N, and the N arm must stay bit-equal to force_lon's banked N on every window the two rolls share; `S0id` (parameters touched and restored, no change) must equal N bit-for-bit.

## Populations and plan (fixed here, before any number)

Window sampling seed **20261020**; episode-stratified proportional picks as in `force_lon_reroll_r8.apportion`.  S6 compliant windows of EMA-30k seed 0, the banked classes of the original (section 2): **FB** = all banked r8 false-brake windows (493) in phase A, of which **FB250** (a seeded stratified 250) carry the phase-B arms; **CT** = 250 seeded GT-cruise non-false-brake windows in phase A, of which **CT100** carry phase B; **DC100** = 100 seeded GT-DECEL compliant windows (plan class of `g` decel; the real-decel stratum the missed-decel read needs).  The class of each window under the RE-ROLLED arms is taken from that arm's own plan (same rule, `cruise_brake_retime_r8.plan_class_thr`, 1.5 m/s).

## Readouts per hedge arm X in {H1, H2}, vs N on the same windows (paired), each with the N2 floor beside it

* **(a) false-decel on GT-cruise windows** = weighted share of plans of class decel over FB250 u CT100, weights `w_FB = 493 / 250`, `w_CT = 1363 / 100` (1,856 GT-cruise compliant windows = 493 FB + 1,363 non-FB; the weights restore the banked population).  ESTIMATED approximation: weights are fixed per window inside the bootstrap.  `dFD = FD_N - FD_X` (positive = X reduces the false brake).
* **(b) missed-decel on real-decel windows** = share of DC100 windows whose plan class is NOT decel.  `dMD = MD_X - MD_N` (positive = X misses more).
* **(c) S6 ADE** (the six S6 slots vs `g`), per stratum (FB250, CT100, DC100) and weighted over the rolled strata, arm minus N, and arm vs the banked refcv7-50,400 `os` (the embedded `b_refcv7_50400_s0` arm) on the same windows.  Descriptive; not in the rule.
* **(d) share of windows whose emitted plan changes vs N** (any S6 slot differs by more than 1e-4 m), per stratum.
* **N2 floor:** the same four statistics for `N2 - N` (second CPU draw, sampler offset 7919, all phase-B windows).  Stamp `ABOVE_N2_FLOOR` iff |dFD_X| exceeds the larger of |lo|, |hi| of the N2 `dFD` interval.

## Outcome rule (committed; per arm)

* **NO-EFFECT** if the CI of `dFD` includes 0 (change not separated).  Checked FIRST (reading choice, flagged: the coordinator's two clauses can both fire when the point drop is large but the CI covers 0; the stricter reading wins).
* **HEDGE-WORKS** if `dFD >= 0.5 * 0.174569 = 0.087285` (point estimate; "drops >= 50 % of the r8 - r7 excess") AND the CI of `dFD` excludes 0 AND missed-decel is not separated worse (`lo(dMD) <= 0`).
* **MIXED** otherwise (separated change smaller than 50 %, or missed-decel separated worse).
Per-arm labels only; no combination across H1 and H2.  A HEDGE-WORKS label would say the false brake is decided by the pool composition at INFERENCE; it would not by itself say the retrained model with a GT-free floor is better.

## Re-roll controls restated for this plan

`E` and ALONG-RAND (A1 A3), `fan_cells` and `along_stats` are unchanged.  Positive controls: `Fbrake` (force BRAKE_TO, LON6 index 2) on CT windows must raise the decel share; `S0x` (the four `r8_sel` scalars x 10) must change the selected candidate on some windows while `S0id` changes none.  Neither is a threshold of the rule; both are reported before any intervention number.  Phase-B order (priority, so a stop at 2026-10-20 15:00 loses the least): hedge arms H1/H2/N2, then F2/Fbrake, then S0a/S0b/S0id/S0x.
