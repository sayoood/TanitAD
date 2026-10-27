# RESULT_V8F_T1_LATCOND: on well-conditioned lateral rows, HACC's T1-kin curvature deficit SHRINKS but stays real at speed. Outcome: CONDITIONED LATERAL DEFICIT

`PREREG_V8F_T1_LATCOND.md` (`bdbd69a0`) with Amendment 1 (`72dd3df0`), both landed before any model arm was read on its rows. Read 2026-10-26 by `raw/2026-10-26-t1-latcond/t1klc_read.py`, applied verbatim. Claim row `H-V8F-T1-LATCOND-1`. Zero training: it re-reads the banked T1-kin m = 10 dumps.

* **The rows:**
  * L1: lateral-acceleration MAE, \|v ω − v_G ω_G\| on the canonical pair mask, with pair speed = ds_mid / dt;
  * L2: curvature MAE on pairs whose GT speed is ≥ 5.2 m/s;
  * L3: yaw-rate MAE on the same pairs.

  The threshold was read from the instrument's noise floor alone.
* **Estimator:** paired episode-cluster bootstrap, B = 2,000, arm − P10 per seed.
* **Decisive configuration:** HACC with the causal speed bucket (the T1 headline configuration).

## Controls (all PASS)

| control | result |
|---|---|
| N-1: L1 is well-conditioned (5 mm noise floor below 2 m/s ≤ 3× above 11.8 m/s) | 0.2418 vs 0.2419 m/s², ratio 1.00. The canonical curvature row's ratio is about 230× |
| N-2: the masks | L2 / L3 with the threshold at 0 equal the canonical rows, max \|d\| = 0.0, with the same NaN pattern. All 293 windows whose GT is below 5.2 m/s at every pair give NaN |
| N-3: the known-value circle (v 10 m/s, r 50 m) | against the chord identity 2.3e-13; against v²/r 1.3e-4 (bar 1e-3) |
| The float32 dumps | ADE6 delta −0.7680 / −0.7336, equal to the banked read |

## The read (HACC with the causal bucket − P10, T1-kin m = 10)

| row | s0 | s1 |
|---|---|---|
| ADE6 (m) | −0.7680 ✅ | −0.7336 ✅ |
| **L1: lateral-acceleration MAE (m/s²)** | −0.107 [−0.606, +0.388], n.s. | −0.115 [−0.703, +0.487], n.s. |
| **L2: curvature MAE at GT ≥ 5.2 m/s (1/m)** | **+0.0210 [+0.0099, +0.0351] WORSE** | **+0.0082 [+0.0026, +0.0148] WORSE** |
| **L3: yaw-rate MAE at GT ≥ 5.2 m/s (deg/s)** | **+1.98 [+0.34, +3.88] WORSE** | +1.01 [−0.30, +2.35], n.s. |
| heading MAE (deg) | −0.63, n.s. | −0.93, better |
| cross-track MAE (m) | −0.557, better | −0.434, better |
| canonical curvature (1/m), beside | +0.0302, worse | +0.0130, worse |

**Outcome (verbatim): CONDITIONED LATERAL DEFICIT.** L2 is separated-worse at both seeds, and L3 at s0.
* Masking the speeds where millimetres dominate shrinks the curvature excess by 30–37 % (+0.030 → +0.021, +0.013 → +0.008 1/m).
* But a part remains at speed, about 3× the 5 mm noise floor of the 5.2–8.6 m/s band.
* The lateral ACCELERATION row is not worse at either seed.

**Beside, every configuration reads the same:**

| configuration | s0: L2 / L3 / L1 | s1: L2 / L3 / L1 |
|---|---|---|
| logged bucket | +0.0195 / +2.74 / +0.06 | +0.0087 / +1.16 / −0.10 |
| strict | +0.0205 / +1.99 / −0.09 | +0.0099 / +1.55 / −0.03 |
| HACC-LAT arm B | +0.0200 / +2.80 / −0.36 | +0.0050 / +0.35 / −0.44 |

## What this settles

* ⇒ **The lateral deficit is real at speed, and it is in curvature, not in lateral acceleration or position.** HACC's executed track at speed bends with a curvature profile further from the human's than the floor's piecewise-arc track does. Its heading and cross-track are better, and its lateral acceleration is no worse.
* ⇒ **"The canonical rows are noise-dominated below 5 m/s" stands** as an instrument fact: N-1 and the noise probe. **But it explains only about a third of the deficit.**
* **Next (committed by the outcome row):** the box-2 lateral training lever, under its own PREREG, aimed at L2 (curvature at speed).
  * ⛔ It is blocked on box-2 compute: FLAT is training now and OACC is queued behind it.
  * It also needs a diagnosis first. On train ticks, HACC's plan curvature from the same state is no worse than P10's, so the at-speed excess arises in the LOOP (state, seams), not in a single plan.
  * A root-cause diagnosis of the at-speed excess (zero-GPU) is the next unblocked step. It decides what the training lever, or a cheaper zero-training lever, must target.
