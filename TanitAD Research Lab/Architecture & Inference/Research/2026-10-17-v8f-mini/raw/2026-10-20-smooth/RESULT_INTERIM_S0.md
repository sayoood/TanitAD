# PREREG_V8F_SMOOTH: seed-0 read (2026-10-20, INTERIM, no verdict)

**Status:**
* The bars decide only at BOTH seeds. This is seed 0 alone, so neither arm has a verdict.
* The pre-registration was sealed at `e58534a` (sha256 `a2f19e48…`) before this read.
* Instrument: `code/v8f_smooth_read.py`.
  * Tests: `stack/tests/test_v8f_smooth_read.py`, 4 passed.
  * Mutations: `mutate_smooth.json`, 6/6 guard deletions RED. Two of them (D's sign, P's bar) first SURVIVED and were closed by a test arm that smoothing must make worse.
* Raw: `smooth_s0.json`.
* Tier: T0 open loop, the model's own outputs, eval-141 (2,399 windows, 141 episodes).
* Estimator: the paired episode-cluster bootstrap, B = 2,000. It answers another draw of EPISODES only; training-seed variance is answered by seed 1.

**Control C: OK.** SG(7, 2) applied to the human path moves its ADE@6s by 0.0076 m (bar 0.05).

| arm, seed 0 | ADE@6s S − arm | accel MAE S − arm (m/s²) | yaw-rate MAE S − arm (deg/s) | D at s0 | P at s0 | mean \|accel\| arm → S |
|---|---|---|---|---|---|---|
| M1 | +0.0002 [−0.0001, +0.0005] | −0.0493 [−0.0611, −0.0369] | −1.5306 [−1.9084, −1.2183] | met | met | 1.547 → 1.416 |
| M2b | +0.0002 [−0.0000, +0.0005] | −0.0474 [−0.0609, −0.0341] | −1.1817 [−1.4548, −0.9383] | met | met | 1.488 → 1.365 |

Human mean |accel| is 0.751.

**The other families (S − arm, seed 0; separated unless marked):**
* curvature MAE: −0.0144 (M1), −0.0100 (M2b) 1/m;
* heading: −0.09 deg;
* speed MAE: +0.006 / +0.004 m/s (worse, separated, small);
* along-track and cross-track: below 0.001 m.

**Dose SG(11, 2) (DESCRIPTIVE):**
* accel MAE −0.30 m/s² for both arms;
* yaw rate −1.78 / −1.42 deg/s;
* speed MAE −0.03 m/s;
* ADE@6s within ±0.001 m.

## Reading (seed 0 only, so it may still be withdrawn at seed 1)

* **Yaw rate and curvature: the roughness is free high-frequency residual noise.**
  * S removes about 1.53 of M1's +1.60 deg/s yaw-rate gap to P10 and about half of its curvature gap.
  * The ADE cost is +0.0002 m.
* **Acceleration: the roughness is mostly NOT in that band.**
  * S(7) removes only 0.05 of the +0.61 m/s² gap to P10.
  * The wider dose removes about half (−0.30), still at no ADE cost.
  * ⇒ The speed profile carries a slower oscillation, at roughly a 2 s period: the 12-term cosine basis's upper terms, not per-step noise.
* **If seed 1 agrees:**
  * the zero-training post-filter is a CANDIDATE for the lateral dynamics;
  * the training lever is a smooth output basis: fewer upper cosine terms or a penalty on them, at the PI's go;
  * the speed-profile oscillation joins `H-V8F-LON-1`'s longitudinal lever.

**Next:** run the same command with `--tags M1_s0 M1_s1 M2b_s0 M2b_s1` when the v8F-mini chain delivers `eval_*_s1.npz`. That read carries the verdict.
