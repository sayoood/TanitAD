# PREREG_V8F_SMOOTH: the verdict (2026-10-20, both seeds)

**Status:**
* The pre-registration was sealed at `e58534a` (sha256 `a2f19e48…`) before any smoothed number.
* The seed-0 interim read is `RESULT_INTERIM_S0.md`. This read adds seed 1 from box 2's full chain (`raw/full/eval/`).
* Instrument: `code/v8f_smooth_read.py` (4 tests; 6/6 mutations RED). Raw: `smooth_full.json`.
* Tier: T0 open loop, eval-141 (2,399 windows, 141 episodes).
* Estimator: the paired episode-cluster bootstrap, B = 2,000 (another draw of EPISODES; the two seeds answer the training variance).

**Control C: OK.** SG(7, 2) on the human path moves its ADE@6s by 0.0076 m (bar 0.05).

| arm | seed | ADE@6s S − arm (m) | accel MAE S − arm (m/s²) | yaw-rate MAE S − arm (deg/s) | D | P |
|---|---|---|---|---|---|---|
| M1 | 0 | +0.0002 [−0.0001, +0.0005] | −0.0493 [−0.0611, −0.0369] | −1.5306 [−1.9084, −1.2183] | met | met |
| M1 | 1 | +0.0005 [+0.0003, +0.0007] | −0.0215 [−0.0300, −0.0130] | −0.8945 [−1.1591, −0.6817] | met | met |
| M2b | 0 | +0.0002 [−0.0000, +0.0005] | −0.0474 [−0.0609, −0.0341] | −1.1817 [−1.4548, −0.9383] | met | met |
| M2b | 1 | +0.0006 [+0.0004, +0.0009] | −0.0090 [−0.0174, −0.0003] | −0.8172 [−1.0424, −0.6374] | met | met |

**Verdict: SMOOTH-PASS for M1 and for M2b**, as committed.
* Both bars hold at both seeds.
* At seed 1 the ADE cost is separated but tiny (+0.0005 / +0.0006 m), well inside the +0.01 m bar.

**What the pass does and does not say:**
* **It does say:**
  * the yaw-rate roughness of v8F-mini's paths is high-frequency residual noise;
  * a fixed post-filter removes 0.8–1.5 deg/s of yaw-rate error at no position cost.
* **It does NOT say the acceleration defect is fixed.**
  * S moves accel MAE by only −0.009 to −0.049 m/s².
  * The accel gap to P10 is +0.29 to +0.61, separated in all four runs (`RESULT_FULL.md`).
  * M2b seed 1's accel interval only just clears 0 (upper −0.0003).
  * ⇒ The bar was met on accel by a small margin; the substance of the pass is yaw rate.
* **Committed consequence:** S is a CANDIDATE post-processor. It is not adopted into any K-1 / WM-1 claim without its own confirm on a fresh read. The training lever it names is a smooth output basis (fewer upper cosine terms, or a penalty on them), at the PI's go.
* **The acceleration defect is the lower-frequency speed-profile oscillation.** The descriptive SG(11) dose removes −0.30 (seed 0) and −0.18 to −0.19 (seed 1) of it, with ADE within ±0.001 m. It belongs to the longitudinal lever (`H-V8F-LON-1`), not to this filter.
