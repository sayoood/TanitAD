# PREREG_V8F_T1_LATCOND: is HACC's T1-kin lateral deficit real once the lateral rows are well-conditioned? (2026-10-26)

**Written before any model arm is read on the rows defined here.** Master Mind, 2026-10-26. Claim row `H-V8F-T1-LATCOND-1`. Zero training: it re-reads banked T1-kin dumps.

## 0. Why (MEASURED, `RESULT_V8F_HACC_LAT.md`)

* **The canonical curvature and yaw-rate MAE rows have a millimetre noise floor** that is the size of HACC's T1-kin lateral deficit against P10.
  * The measurement: the eval-141 GT, perturbed by 5–10 mm of i.i.d. position noise per knot and scored against itself, gives a pooled curvature MAE of 0.021–0.038 1/m and a yaw-rate MAE of 2.4–4.7 deg/s.
  * The deficit is +0.040 / +0.016 1/m and +4.2 deg/s.
  * Almost all of that floor sits below 5 m/s (curvature 0.20–0.34 1/m below 2 m/s at 5–10 mm), because curvature divides by speed.
* **The deficit is also not in O's plan:**
  * from the same state, HACC's plan curvature is no worse than P10's on train ticks;
  * forcing the cross axis to P10's below 8 m/s leaves the eval excess unchanged.
* ⇒ The open question is whether any lateral deficit remains on rows that a few millimetres cannot move. ⛔ This does not re-score any earlier result. Every earlier outcome stands as read.

## 1. The conditioned lateral rows (ADDED; the canonical rows stay reported beside)

All rows are computed with `taniteval.four_families._seq_geometry` (origin prepended, dt = 0.2 s), exactly as the canonical rows are, on the same windows. P is the arm's track and G is the GT.

| row | definition | mask |
|---|---|---|
| **L1: lateral-acceleration MAE (m/s²)** | \|v_P ω_P − v_G ω_G\|: speed × yaw rate, per step | the canonical yaw-rate mask (`pair_valid` of both) |
| **L2: curvature MAE at speed (1/m)** | the canonical per-step curvature error | ALSO GT step speed ≥ **5.2 m/s** |
| **L3: yaw-rate MAE at speed (deg/s)** | the canonical per-step yaw-rate error | ALSO GT step speed ≥ **5.2 m/s** |

* **The threshold comes from the instrument alone.** 5.2 m/s is the lowest band edge of the noise probe at which the 5 mm curvature floor falls below 0.01 1/m (2–5.2 m/s: 0.033; 5.2–8.6 m/s: 0.0068). No model arm was read to choose it.
* **The mask reads the GT's speed only,** so an arm cannot change which steps it is scored on.
* **Pooling** is the canonical one: the masked mean over every valid step, per window for the bootstrap.
* Heading and cross-track keep their canonical definitions.

## 2. Configurations read

All are banked T1-kin m = 10 dumps on eval-141 (2,399 windows, 141 episodes):

| configuration | dumps | role |
|---|---|---|
| **(D) HACC with the causal speed bucket** (the T1 headline configuration, `RESULT_V8F_T1KIN_CAUSALBUCKET`) | `raw/2026-10-26-t1kin-causalbucket/run/dumps/dumps_HACC_s{0,1}_m10_causal.npz` | **decisive** |
| HACC with the logged bucket (`RESULT_V8F_T1KIN_HACC`) | `raw/2026-10-26-t1kin-hacc/run/dumps/dumps_HACC_s{0,1}_m10.npz` | beside |
| HACC strict (cameras frozen + causal bucket), once `PREREG_V8F_T1KIN_STRICT` is read | its dumps | beside |
| HACC-LAT arm B | `raw/2026-10-26-hacc-lat/run/dumps/` | beside |

* **Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0, arm − P10, per seed.
* F = 0.1207 m, 2F = 0.2415 m.

## 3. Controls (read before the outcome; a failure ⇒ REFUSED)

* **N-1 (conditioning):** on the noise probe (GT + 5 mm, seed 0), L1's floor below 2 m/s is ≤ 3× its floor above 11.8 m/s. The canonical curvature row's ratio is about 230×.
* **N-2 (the masks):**
  * L2 and L3 with the threshold set to 0 equal the reader's canonical curvature and yaw-rate rows to ≤ 1e-9 on the decisive dumps.
  * A window whose GT speed is below 5.2 m/s at every step contributes NaN to L2 and L3.
* **N-3 (known value):** a synthetic constant-speed circle (v = 10 m/s, r = 50 m) gives L1 = |v²/r − 0| = 2.0 m/s² against a straight line at the same speed, to 1e-6, away from the first step.
* **The float32 dumps:** the decisive dumps' canonical ADE6 paired delta equals `RESULT_V8F_T1KIN_CAUSALBUCKET`'s −0.7680 / −0.7336 m to ≤ 1e-3 m (they are the float32 copies).

## 4. Outcome (no bar moves after data)

**C-DOM**, per seed, on (D), arm − P10:
* ADE6 point ≤ −2F, with the interval excluding 0;
* AND none of {ADE2, speed MAE, along-track MAE, accel MAE, heading MAE, cross-track MAE, L1, L2, L3} separated-worse.

| outcome | condition | next (committed) |
|---|---|---|
| **CONDITIONED DOMINANCE** | C-DOM at both seeds | v8F's T1 claim becomes dominance over P10 on the conditioned rows, with the canonical curvature / yaw-rate rows and their noise floor reported beside, and the low-speed caveat stated. The box-2 lateral training lever of `RESULT_V8F_HACC_LAT` is dropped, with this as the reason. The evidence video is re-rendered on T1 executed tracks |
| **CONDITIONED LATERAL DEFICIT** | any of heading, cross-track, L1, L2, L3 separated-worse at either seed | The deficit is real at speed. The box-2 lateral training lever runs, under its own PREREG, aimed at the conditioned row(s) that fail |
| **OTHER** | anything else (e.g. a longitudinal row worse, or the ADE6 bar) | Reported per seed and per row |

**Beside (decides nothing):** every configuration of §2 on every row; the canonical curvature and yaw-rate rows; L1–L3 per GT-speed band.

## 5. What this does NOT license

* ⛔ Re-scoring `H-V8F-T1KIN-HACC-1`, `H-V8F-CAUSALBUCKET-1` or `H-V8F-HACC-LAT-1`. Their outcomes stand.
* ⛔ Choosing the threshold, the rows or the bar after a number is seen.
* ⛔ Dropping the canonical rows from any report. They are reported beside, with their noise floor.
* ⛔ Closed-loop or "drives" wording.

## Amendment 1 (2026-10-26, before any model arm is read on the rows of §1)

**What changes:** N-3's tolerance, and one implementation detail of L1, both written as definitions before any number.

1. **N-3 is wrong by construction as written.** It says the circle gives "L1 = v²/r to 1e-6". The canonical geometry measures CHORDS: speed is chord length / dt, and the heading step is the chord-to-chord angle Δθ. So the discrete lateral acceleration of a sampled circle is exactly 2r·sin(Δθ/2)·Δθ / dt². For v = 10 m/s, r = 50 m and dt = 0.2 s, Δθ = 0.04 rad, and that is 1.99987 m/s², not 2.0. A 1e-6 bar on v²/r could never pass. **N-3 is now two checks:**
   * against the discrete identity 2r·sin(Δθ/2)·Δθ / dt², to ≤ 1e-9;
   * against v²/r, to ≤ 1e-3 (relative 5e-4).

   This is the class of `R74`: a control tolerance written without computing the identity it tests.
2. **L1's "speed" at a heading step (a PAIR of steps) is the pair's mean speed** v_mid = 0.5 (v_k + v_{k+1}) = ds_mid / dt. That is the same ds_mid the canonical curvature uses, so L1 = curvature × v_mid² on every masked pair. L2's and L3's "GT step speed ≥ 5.2 m/s" is the GT's v_mid at that pair.

Nothing else moves: the rows, the threshold, the configurations, the bar and the outcome table are unchanged.
