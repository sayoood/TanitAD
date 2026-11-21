# TF-1 result: tracker-aware teacher targets, scored under NAVSIM's own LQR tracker (2026-11-11)

Stream: Teacher Fidelity (TF), opened by the PI on 2026-11-11 ("it must reach the paper level"). This is the stream's first measurement.
Target: the paper's DriveRL row on navtest, **PDMS 95.8 / DAC 99.9** (Table 4).
Evidence class: **MEASURED, EXPLORATORY**, on **navtrain held-out only** (1,018 rank-0 teacher targets, 11 logs; the same population as D1 / P5 / P5-S / E2). This is not navtest, and none of these numbers is a navtest claim. The K ledger is unchanged.
Tier: NAVSIM v1.1 PDMS, non-reactive pseudo-simulation (LQR + kinematic bicycle), vendored navsim @ 3e8291b.
Estimator: log-cluster bootstrap over the 11 logs, 10,000 draws, seed 20260927, 95 % percentile interval, ratio mean (E2's estimator verbatim). It answers *another draw of LOGS*. Labels, tracker and scorer are deterministic, so there is no training or inference variance on these arms. The bit-identical rerun of shard 4 confirms that.

## What was scored

There is one PDMScorer call per (log, token). Every label comes from the banked P5-S npz (`raw/2026-10-11-p5s-smoothed/p5s_cubic.npz`, blob 6568837d; `p5s_linear.npz`, blob ecac7aae). Nothing was re-derived.

| arm | what it is |
|---|---|
| T_tracked | the teacher's 20 Hz rollout, resampled to NAVSIM's 8 poses, then LQR-tracked. This is what a student imitating the teacher is scored as. |
| **Tc20s_tracked** | **the P5-S tracker-aware cubic target (a command pre-distorted so that, once LQR-tracked, the executed path follows the teacher's), LQR-tracked. PRIMARY.** |
| Tc20_tracked | the P5 linear variant, descriptive |
| Tw_tracked | deliberate-regression arm; it must score below T |
| H_tracked | the logged human poses, LQR-tracked |
| T_transparent | the teacher's path scored as executed, without the tracker (E2's builder) |
| Tc20s_transparent | the pre-distorted command scored as a path (descriptive: it is not meant to be driven untracked) |

## Controls (all PASS; `tf1_result_full.json` → `controls`)

- **Identity, bit-for-bit, 0.0 max abs diff:** T_tracked = E2 teacher_tracked; H_tracked = E2 human_tracked; T_transparent = E2 teacher_transparent.
- **Joins with P5-S's own failure flags, 1,017/1,017 agree:** T DAC-zero 82 = P5-S sim8 fails 82; Tc20s 10 = 10; Tw 347 = 347.
- **Deliberate regression:** Tw − T = −36.50 PDMS [−42.61, −26.55].
- **Liveness:** Tc20s differs from T on 704 rows (PDMS) and 80 rows (DAC): 76 fixed, 4 broken.
- **Coverage 1,018/1,018.** The first pass lost 1 row (shard 4, `MemoryError`) when TF1's five shards and E3's four Ray workers all loaded torch at once at 11:30. Shard 4 was re-scored at 22:18Z: the 202 rows scored both times are **bit-identical**, and the lost row was recovered. The first pass is banked as `rows/tf1_s4_with_memerror.jsonl`.

## Result

| arm | PDMS | DAC | NC | EP | TTC | C | PDMS-zero rows |
|---|---|---|---|---|---|---|---|
| T_tracked | 86.73 [79.87, 92.83] | 91.95 [85.36, 97.70] | 99.26 | 83.04 | 96.46 | 100.0 | 88 |
| **Tc20s_tracked** | **94.36 [93.07, 95.79]** | **99.02 [98.08, 100.0]** | 99.90 | 89.69 | 98.23 | 99.90 | **11** |
| Tc20_tracked (linear) | 94.37 [93.08, 95.79] | 99.02 | 99.90 | 89.70 | 98.23 | 99.90 | 11 |
| H_tracked (human) | 94.53 [93.07, 96.15] | 100.0 | 100.0 | 87.06 | 99.80 | 100.0 | 0 |
| T_transparent | 94.95 [93.78, 96.15] | 99.41 [98.67, 100.0] | 100.0 | 90.08 | 98.63 | 100.0 | 6 |
| Tw_tracked (regression) | 50.26 | 65.91 | 86.59 | 51.21 | 68.37 | 88.21 | 421 |

Paired contrasts (PDMS, then DAC):

- **Tc20s − T, tracked: +7.64 [+2.29, +13.60]; DAC +7.07 [+1.83, +13.26].** The tracker-aware target closes **92.8 % [84.2, 95.4]** of the tracked-to-transparent gap.
- Tc20s(tracked) − human(tracked): **−0.16 [−1.68, +1.03]**, i.e. level with the human on this population.
- Tc20s(tracked) − T_transparent: **−0.59 [−0.91, −0.24]**. That is the residual the tracker still costs, carried by 10 DAC-zero rows and 1 NC-zero row.
- Linear − cubic: +0.004 [−0.001, +0.011]. The smoothing is free.
- **Where the gain comes from:** on the 927 rows that are alive in both arms, EP is T 90.91 against Tc20s 90.64. The retime costs about 0.3 EP points there. The whole gain is the failures it removes: PDMS-zero rows fall from 88 to 11, DAC-zero from 82 to 10, NC-zero from 7 to 1.

## Reading against the paper target

- On held-out navtrain, the tracker-aware teacher target reaches **human level under NAVSIM's own tracker** (94.36 against 94.53). That is about 1.4 points under the paper's 95.8, and **DAC 99.02 against 99.9**.
- The remaining DAC failures have two sources:
  - **6 are the policy's own** (T_transparent fails DAC there too, with no tracker involved);
  - **about 4 are tracker residue** (fail tracked, pass transparent).
- Reaching DAC 99.9 means at most 1 DAC-zero row in 1,018. The tracker interface alone cannot do it: **the policy's boundary margin must also improve.** This agrees with E1, where the 0.37 m margin is the policy's.
- These logs run hotter than navtest: our tracked teacher reads 86.73 here against 89.81 on navtest. The held-out level does not transfer one-for-one. A navtest confirmation needs the teacher's navtest rollouts with Tc20s applied, under a frozen pre-registration (the next TF deliverable).

## Four metric families

- **LONGITUDINAL:** EP 89.69 vs 83.04; TTC 98.23 vs 96.46 (Tc20s vs T, tracked). Target-speed accuracy and headway are not computed in this read. *Work item:* per-row speed error of the tracked execution against the teacher's 20 Hz path.
- **LATERAL:** DAC 99.02 vs 91.95. Heading, curvature and yaw-rate error of the tracked path against the teacher's path are not computed. *Work item,* same instrument.
- **TACTICAL:** not computable here (n = 1,018). NAVSIM teacher targets carry no manoeuvre label.
- **STRATEGIC:** PAUSED by the PI's ruling of 2026-09-19 (PI_DECISION_QUEUE item 25). This is not a pass.

## Next levers (ranked by measured effect, for the TF design)

1. **Navtest confirmation of Tc20s-tracked** (frozen pre-registration, one read, K+1). This is the largest measured lever (+7.6), and it needs no training.
2. **Policy boundary margin (L2):** 6 of the 10 residual DAC rows. Two candidates:
   - privileged map-margin selection among the teacher's own test-time samples (`tts_num_candidates=8`) using only t0-privileged state, never the log's future;
   - margin shaping of the targets.
3. **Tracker residue:** about 4 rows; Tc20s refinement.
4. **E3 checkpoint identity** (running): is our `checkpoint_2400` the paper's checkpoint? Test14-random NR, against Table A10's 94.50.

## Artifacts

| what | where |
|---|---|
| scorer | `tf1_score.py` |
| analysis | `tf1_analyze.py` |
| launcher | `launch_tf1.cmd` |
| result json | `tf1_result_full.json` |
| per-row rows | `rows/tf1_s{0..3}.jsonl` and `rows/tf1_s4r.jsonl` |
| first-pass shard 4 and logs | `rows/tf1_s4_with_memerror.jsonl`, `rows/*.log` |

Inputs are already banked: the P5-S npz/json (blobs above); `refe/navsim_pdm_targets.py`, `navsim_dac.py`, `navsim_lane.py`, `navtrain_scenarios.py` and `pdm_route_correction.py` (blob-identical to the tip); E2's rows (`raw/2026-11-11-teacher-review/e2e1/json/e2_rows_1018.jsonl`).
