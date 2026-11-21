# TF-2b result: CERT is NOT SHOWN on fresh logs, because the tracker-aware target alone already reaches DAC 100 there

2026-11-12, Master Mind, finishing the TF-2b run after the dev-box reboot at 00:33Z. CPU only, no GPU, no navtest. This file supersedes the interim "NO VERDICT (no local DB)" version.

- **Pre-registration:** `PREREG_TF2B.md`, sha256 `b0ed43c201d42d9e09fbb4505d0c7b64b80f17c8e3b09534ab0edab798bea14b`, frozen 2026-11-11T23:22:42Z, before any fresh row. Not edited.
- **Population:** 687 held-out navtrain rank-0 samples in **13 logs, none of them in TF-2's 11** (`tf2b_samples.txt`, sha256 `d2dcfae9...56`). One log holds 34 % of the rows, as the card discloses.
- **Data:** 11 of the 13 logs had no local nuPlan DB. The MM copied those 11 from the pod's `navtrain_dbs` (1.0 GB, sha256 11/11 equal to the pod). All 687/687 samples then indexed.
- **Evidence class:** MEASURED, EXPLORATORY, navtrain held-out, **not navtest**. The K ledger is unchanged.
- **Tier:** NAVSIM v1.1 PDMS, non-reactive pseudo-simulation, vendored navsim @ 3e8291b.
- **Estimator:** log-cluster bootstrap over the 13 logs, 10,000 draws, seed 20260927, 95 % percentile interval, ratio mean (TF-2's estimator). It answers *another draw of LOGS*. Labels, tracker and scorer are deterministic.

## Run record

- **Rolls:** 687/687, 0 errors, registered teacher protocol (10 Hz, CPU), 2 shards.
- **Scoring:** 585 rows before the reboot killed the shards. The remaining 102 were scored after it with `tf2_score.py --resume` on the same 2-shard partition (`resume_score.cmd`). No row was written twice, no line was truncated, and coverage is 687/687.

## Controls: all PASS (`tf2b_result.json`, `ZZTF2B_CONTROLS PASS`)

| control | reading |
|---|---|
| C0: the roll script reproduces banked 10 Hz targets | 30 rows, max \|dxy\| 0.0 m |
| C1: scoring identical to TF-2 on the control rows | PASS (30 rows, the original banked npz) |
| C2: Tc20s re-derivation identity | deviation 0.0 m (bar 1e-9) |
| C3: unshifted rows identical | PASS |
| C4: deliberate regression Mw | DAC-zero 49 vs Tc20s 0; PDMS 87.81 |
| C5: liveness | PASS |
| C6: coverage | 687/687 |

## Result (687 fresh rows, 13 logs)

| arm | PDMS | DAC | NC | EP | TTC | C | PDMS-zero / DAC-zero / NC-zero |
|---|---|---|---|---|---|---|---|
| T tracked (teacher, raw) | 92.68 [90.13, 94.57] | 98.69 | 99.85 | 88.55 | 95.78 | 100.0 | 10 / 9 / 1 |
| **Tc20s tracked** | **94.80 [94.35, 95.46]** | **100.0** | 99.85 | 89.33 | 98.25 | 100.0 | 1 / 0 / 1 |
| M tracked (margin shaping) | 91.66 [89.05, 93.81] | 97.23 | 99.71 | 87.08 | 96.80 | 99.71 | 21 / 19 / 2 |
| **CERT tracked** | **94.80 [94.35, 95.46]** | **100.0** | 99.85 | 89.33 | 98.25 | 100.0 | 1 / 0 / 1 |
| Mw tracked (regression) | 87.81 [83.26, 92.20] | 92.87 | 99.78 | 82.81 | 97.53 | 99.71 | 50 / 49 / 1 |
| H tracked (human) | 95.42 [94.96, 95.98] | 100.0 | 100.0 | 89.01 | 100.0 | 100.0 | 0 / 0 / 0 |
| T transparent | 95.04 [94.43, 95.68] | 100.0 | 99.85 | 89.31 | 98.84 | 100.0 | 1 / 0 / 1 |

- **CERT − Tc20s = 0.000 [0.000, 0.000], unrounded.** CERT chose Tc20s on all 687 rows: Tc20s already passes the tracked drivable-area check on every fresh row, so CERT's fallback never fires.
- **Verdict as pre-registered: CERT NOT SHOWN.** CERT DAC-zero ≤ 2 holds (0); ΔPDMS CI-lo > 0 does not.
- **What it means:** on these 13 logs there is no residual DAC failure for CERT to repair. The finding is about this population, not a defect of CERT.
- **Tc20s on fresh logs:** Tc20s − teacher raw = +2.12 PDMS, DAC 98.69 → 100.0. That repeats TF-1's direction on new logs, though the logs run cooler.

**Pooled, descriptive only (not a pre-registered verdict):** 1,705 rows, 24 logs (TF-2's 1,018 + TF-2b's 687).
- CERT 95.03 [94.35, 95.67] vs Tc20s 94.54 [93.71, 95.36]: CERT − Tc20s **+0.49 [0.00, +1.01]**.
- DAC-zero: CERT 0, Tc20s 10. PDMS-zero: 3 vs 12.
- CERT never lowers a row.

## Consequence for N1, as the card commits

**N1's candidate stays Tc20s.** CERT did not reach WORKS. MM ruling at write-up: N1 also scores **CERT as a pre-registered SECONDARY arm**.
- Its cost is about 0.1 s per token in the same pass. It decides nothing.
- It is quoted only as "teacher + map-certified selection", never as the teacher alone or as checkpoint identity (ruling of 2026-11-12).

## Four metric families

These are NAVSIM sub-scores on the tracked execution: LONGITUDINAL = EP, TTC; LATERAL = DAC.
- The family instrument (W-FAM, `scorer_gym_20261107/instruments/`) was not run on TF-2b. That is a work item for N1, where it is a read requirement.
- TACTICAL: not computable (no manoeuvre label on these rows; n = 687).
- STRATEGIC: PAUSED by the PI's ruling of 2026-09-19.

## Artifacts (`teacher_fidelity/tf2b/`)

- `PREREG_TF2B.md` + `.sha256`
- `tf2b_samples.txt`
- `tf2b_roll.py`, `tf2b_mknpz.py`, `tf2b_analyze.py`
- `resume_score.cmd`
- `tf2b_result.json`
- `rows/`: `roll_s0.jsonl`, `roll_s1.jsonl`, `c0.json` (re-banked with `derivation_max_dev_m`), `tf2b_s0.jsonl` (sha256 `34aa30b7...`), `tf2b_s1.jsonl` (`44b7c4f5...`), `c1_rows.jsonl` (`f6f8bb44...`)

The scorer is TF-2's `tf2/tf2_score.py`, unchanged.
