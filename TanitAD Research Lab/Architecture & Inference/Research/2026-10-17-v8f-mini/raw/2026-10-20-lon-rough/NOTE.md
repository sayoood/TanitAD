# v8F-mini: where the remaining 6 s error lives (2026-10-20, seed 0, INTERIM, DESCRIPTIVE)

**What this is:**
* A diagnostic with no bar, read before the full verdict exists. It decides nothing. H-MINI-K-1 and H-MINI-WM-1 are read only by `PREREG_V8F_MINI` Amendment 4's verdict.
* Instrument: `code/lon_rough_diag.py`. Identities are pinned by `stack/tests/test_lon_rough_diag.py`: the truth reads 0, a uniformly slowed path reads oracle 0, and a mixed-sign speed error is fixed by the oracle but not by one global scale.
* Ground truth is rebuilt from the repo's eval inputs. Every run's saved ade6 equals ||pred − gt||, max |diff| 0.0.
* Tier: T0 open loop, the model's own outputs, on eval-141 (2,399 windows, 141 episodes).
* Runs: box 2's seed-0 files (`eval_M1_s0`, `eval_M2b_s0`, `eval_M1-navshuf_s0`). Seed 1 is re-read by the same command when the chain delivers it.
* Raw output: `lon_rough_s0.json`.

## 1. Path roughness: the learned arms are about twice as rough as the human

| run | mean \|accel\| (m/s²) | mean \|step speed change\| (m/s) | ADE@6s (m) |
|---|---|---|---|
| human (GT) | 0.751 | 0.100 | 0 |
| P10-yaw-4512 | 0.826 | 0.104 | 2.4870 |
| M1 s0 | 1.547 | 0.232 | 2.2466 |
| M2b s0 | 1.488 | 0.222 | 2.1753 |
| M1-navshuf s0 | 1.907 | 0.294 | 3.3739 |

* This is the same signal as the paired family rows (`paired_families_s0.json`, `paired_family_block` over the same windows, B = 2,000): M1 − P10 is worse on acceleration (+0.61 m/s²), yaw rate (+1.60 deg/s) and curvature (+0.027 1/m), all separated, while it is better on heading (−1.46 deg) and cross-track (−0.43 m).
* ⇒ **The K-1 ADE win comes with paths whose dynamics are worse than the kinematic prior's.** ADE alone cannot see it.

## 2. Longitudinal: a per-window speed choice, not a bias

| run | ADE@6s | with an ORACLE per-window arc scale | with ONE global scale | mean final arc bias (m) | median final arc ratio |
|---|---|---|---|---|---|
| P10 | 2.4870 | 2.0868 | 2.4828 | −0.305 | — |
| M1 s0 | 2.2466 | 2.0493 | 2.8202 | −2.884 | 0.974 |
| M2b s0 | 2.1753 | **1.8880** | 2.5916 | −2.392 | 0.979 |

* Both counterfactuals use the ground truth. They size the lever; neither is a predictor.
* A perfect per-window speed choice, with the path's shape unchanged, would take M2b from 2.175 to 1.888 m (−0.29 m, 13 %).
* A bias correction would make it WORSE (+0.42 m). The mean bias comes from a tail, while the median window is within 2–3 %.

**M2b s0 by stratum** (dv is the human's speed change over 6 s; "gain share" is the share of the oracle-speed gain):

| stratum | windows | ADE M2b | ADE P10 | arc bias M2b | arc bias P10 | gain share |
|---|---|---|---|---|---|---|
| dv < −3 (braking) | 15.0 % | 2.761 | 3.662 | +1.43 | +9.76 | 24.4 % |
| −3 ≤ dv < −1 | 13.7 % | 2.442 | 2.315 | −1.40 | +1.66 | 13.2 % |
| −1 ≤ dv < 1 (cruise) | 40.9 % | 1.522 | 1.651 | −1.62 | −1.05 | 29.8 % |
| 1 ≤ dv < 3 | 17.5 % | 2.221 | 2.613 | −4.54 | −3.66 | 13.4 % |
| dv ≥ 3 (accelerating) | 12.9 % | 3.221 | 3.782 | −7.41 | −7.18 | 19.2 % |
| launch (v0 < 1, dv > 3) | 2.0 % | **3.058** | **2.460** | −9.22 | −6.87 | 9.4 % |
| stop (v0 > 3, v_end < 1) | 3.9 % | 2.214 | 3.837 | +2.30 | +11.41 | 5.8 % |

**Reading:**
* M2b has learned braking. On stops and hard braking it is far better than the kinematic prior, which overshoots by 10–11 m.
* It is slow everywhere else:
  * it under-progresses in cruise (−1.6 m over 6 s, 30 % of the gain);
  * it does not anticipate acceleration (−4.5 to −7.4 m);
  * it is **worse than P10 at launch from standstill**.
* ⇒ The longitudinal error is a **speed DECISION** spread over the strata. That matches the tactical family: path-derived longitudinal decision accuracy is 0.56–0.58 (kappa 0.33–0.36), and the tactical lon head's macro-F1 is 0.47–0.50.

## 3. The levers this names, ranked by measured size, and what blocks each

1. **The longitudinal speed decision**, worth about 0.29 m of the 2.18 m (oracle).
   * Candidate arms are training-side:
     * a separate longitudinal decision (go / hold / brake) feeding the speed profile, not the mixed 5-way softmax;
     * a multimodal output with winner-takes-all, which would test the cruise under-progress as mode averaging;
     * launch cues: the v7.x traffic-light labels never reach training (`D-TLIGHT-1`).
   * ⛔ Each is an architecture change: the **PI's go**, and a box-2 or A40 slot after RA-1 Amendment 1.
   * ⚠️ A longitudinal decision feeding the goal must stay information-disjoint from the situation classifier (the goal-input rule).
2. **Path smoothness**, which moves the dynamics families and not ADE.
   * Zero-training form: a fixed smoother on the residual over P10, pre-registered with one config, no selection on eval-141, read at both seeds.
   * Training form: a smooth output basis.
   * The zero-training form is CPU-only and runs once seed 1 exists. Its pre-registration comes first.

**Seed 1:** re-run the same command on `eval_*_s1.npz` when ZZV8FMINI-DONE lands. If the stratum shares or the roughness gap change materially at seed 1, this note's reading is a one-seed artefact and is withdrawn.
