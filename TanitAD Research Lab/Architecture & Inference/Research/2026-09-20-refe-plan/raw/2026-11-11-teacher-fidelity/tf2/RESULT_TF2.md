<title>TF-2 result</title>

# TF-2 result: margin shaping is refuted as specified, DAC-certified fallback removes every DAC zero (NOT SHOWN at its CI bar), and a 20-pose submission does not explain the paper's number

Run 2026-11-11 to 12. CPU only: 0.65 CPU-h, ≤ 3 processes, no GPU, no navtest.

**Pre-registration.** `PREREG_TF2.md`, sha256 `65189b98841d3ae5fa0e0c0ba4d436ea55380373f4c63801c852e48ca9eca2bf`, frozen at
2026-11-11T22:34:31Z, before any TF-2 row existed. No rule moved.

**Evidence and population.** MEASURED, EXPLORATORY, on the 1,018 held-out rank-0 **navtrain** targets (11 logs). This
is the same population as D1, P5, P5-S, E2 and TF-1. The K ledger is unchanged.

**Tier.** NAVSIM v1.1 PDMS, non-reactive pseudo-simulation, vendored navsim @ 3e8291b.

**Estimator.**
* Log-cluster bootstrap over the 11 logs, 10,000 draws, seed 20260927, 95 % percentile interval, ratio mean. This is
  TF-1's estimator verbatim.
* It answers *another draw of LOGS*. Labels, tracker and scorer are deterministic.
* ⚠ With 11 logs the cluster count is the limiting fact. 9 of the 11 TF-1 zero rows sit in 2 logs.

## Controls: all PASS (`tf2_result.json` → `controls`)

| control | read |
|---|---|
| C1: T, Tc20s, H tracked and T transparent equal TF-1 per row | max abs diff **0.0** on every sub-score, 1,018 / 1,018 |
| C2: Tc20s re-derived (`iterate20`, cubic) equals the banked label | label max deviation ≤ 1e-9 m; tracked scores identical |
| C3: unshifted rows give M ≡ Tc20s | 559 rows unshifted, 0 differ |
| C4: deliberate regression Mw | DAC-zero **154** vs Tc20s 10 (bar ≥ 20); PDMS −13.76 [−18.73, −7.48] |
| C5: liveness | 459 rows shifted |
| C6: coverage | 1,018 / 1,018, 0 errors |

## Result

| arm | PDMS | DAC | NC | EP | TTC | C | PDMS-zero / DAC-zero / NC-zero |
|---|---|---|---|---|---|---|---|
| T tracked | 86.73 [79.87, 92.83] | 91.95 | 99.26 | 83.04 | 96.46 | 100.0 | 88 / 82 / 7 |
| **Tc20s tracked** (baseline) | **94.36** [93.07, 95.79] | 99.02 | 99.90 | 89.69 | 98.23 | 99.90 | 11 / 10 / 1 |
| **M tracked** (L2b, PRIMARY) | 92.02 [89.12, 94.30] | 97.74 | 98.43 | 88.09 | 95.78 | 99.71 | 34 / 23 / 15 |
| **CERT tracked** (L3, SECONDARY) | **95.18** [94.02, 96.20] | **100.0** | 99.80 | 90.52 | 98.13 | 99.90 | **2 / 0 / 2** |
| T_sub20 tracked (probe P-S20) | 86.55 [79.28, 92.90] | 91.95 | 99.17 | 82.88 | 96.37 | 100.0 | 90 / 82 / 8 |
| Mw tracked (regression) | 80.60 | 84.87 | 98.77 | 77.23 | 95.87 | 99.90 | 158 / 154 / 8 |
| H tracked (human) | 94.53 [93.07, 96.15] | 100.0 | 100.0 | 87.06 | 99.80 | 100.0 | 0 / 0 / 0 |
| T transparent | 94.95 [93.78, 96.15] | 99.41 | 100.0 | 90.08 | 98.63 | 100.0 | 6 / 6 / 0 |
| M transparent | 85.91 | 95.19 | 96.56 | 85.04 | 93.12 | 77.01 | 68 / 49 / 35 |

Paired contrasts, in PDMS points:
* **M − Tc20s: −2.35 [−5.15, +0.02].**
* **CERT − Tc20s: +0.82 [0.00, +1.53]**, with DAC +0.98 [0.00, +1.93].
* CERT − human: +0.66 [−0.53, +1.54].
* **T_sub20 − T tracked: −0.18 [−0.62, +0.26]**; DAC 0.00 [−0.44, +0.44].
* T_sub20 − T transparent: −8.41 [−15.04, −2.61].

## Registered outcomes (PREREG §6)

* **PRIMARY M: C, REFUTED IN THIS FORM.**
  * The ΔPDMS point is negative.
  * NC-zero rose from 1 to 15, and DAC-zero rose from 10 to 23.
  * The EP-alive clause alone held: 91.09 vs 90.98 on 975 rows.
  * **Mechanism (descriptive).** M repaired the 4 sharp-turn policy rows, but it created 32 new zero rows: 17 DAC, 9 NC,
    4 both, 2 NC-partial. Of those, **26 are at v0 < 3 m/s**, with a median shift of 0.84 m. Ungated shaping moves slow
    or stationary cars that sit close to a kerb, where 0.60 m of margin is neither needed nor feasible. It pushes them
    into parked agents or across the other edge.
  * The shaped paths are also uncomfortable as drawn (M transparent C 77.0).
* **SECONDARY CERT: NOT SHOWN.**
  * DAC-zero went from 10 to **0**, which meets the ≤ 4 clause.
  * The ΔPDMS CI lower bound is **0.000**, and the rule requires > 0.
  * The reason, measured rather than used to move the bar: by construction CERT can never lower a row. It changes a
    row only when Tc20s already scores 0. So every row's Δ is ≥ 0, and the 10 changed rows sit in 3 of the 11 logs (veh-40 5, veh-50 4, veh-08 1). A
    resample that misses those 3 logs has a probability of about (8/11)^11 ≈ 3 %, so the 2.5th percentile is exactly 0.
  * **The remedy is more logs, not a looser bar** (§ Next arm).
  * CERT chose Tc20s on 1,008 rows, M on 6 and T on 4.
* **PROBE P-S20: S20-NO.** Submitting the teacher's 20 poses at 0.2 s through stock NAVSIM gives the same result as the
  8-pose plan (identical DAC, PDMS −0.18). The tracker gap is the LQR's, not the sampling's. ⇒ **A stock 20-pose
  submission does not explain the paper's 95.8 for our checkpoint.** Comfort reads 100.0, so it does not produce the
  paper's 99.0 signature either.

## Residual-failure anatomy: TF-1's 11 Tc20s-tracked zero rows (`tf2_result.json` → `anatomy`)

**Policy-owned: 6 rows, all DAC.** They also fail tracker-transparent.

| token | where | conditions | violation | human |
|---|---|---|---|---|
| `634f2012` | veh-40 | intersection turn 83°, 4.4 m/s | at 3.4 s, −0.13 m, front-right corner | 0.66 |
| `6e7663e8` | veh-40 | turn 80°, 4.3 m/s | at 3.0 s, −0.18 m, rear-right | 0.31 |
| `f2c6dba4` | veh-40 | intersection turn 85°, 4.5 m/s | at 3.6 s, −0.07 m, front-right | 0.67 |
| `f8a815a1` | veh-40 | near-standstill (v0 0.06), straight | at 3.1 s, −0.30 m, front-right | 0.56 |
| `a7b14cd0` | veh-50 | intersection turn 95°, 3.9 m/s | at 1.8 s, −0.19 m, rear-right | 0.09 |
| `d7587000` | veh-50 | intersection turn 97°, 3.6 m/s | at 1.5 s, −0.06 m, rear-right | 0.06 |

⇒ **5 of 6 are sharp (80 to 97°) intersection turns at 3.6 to 4.5 m/s.** The footprint's right corner clips the edge
**late in the plan (1.5 to 3.6 s) and shallowly (−0.06 to −0.30 m)**. In two of them the human keeps only 0.06 to
0.09 m.

**Tracker-owned: 5 rows.**
* **4 DAC grazes of −0.6 to −2.8 cm** under Tc20s, at 2.4 to 3.9 s: `a23ebc72`, `85772c23`, `42a94a9c`, `fa99d6a4`.
  The transparent path keeps 0.12 to 0.34 m. Two of these rows (`a23ebc72`, `85772c23`) were **broken by Tc20s
  itself**; the plain T label passes there.
* **1 NC**, `006134f9`. It starts at standstill (v0 0.02); every tracked arm, the 20-pose one included, collides at
  3.3 s; the transparent path does not.

**How CERT handled them.** It repairs **all 10 DAC rows**: M on the 4 sharp-turn policy rows and on 2 tracker rows, T on
the other 4. It leaves 2 NC zeros: `006134f9`, and `85772c23`, where the T fallback collides.

## EP anatomy (descriptive; 895 rows alive in all five arms)

| arm | EP, alive in all five arms |
|---|---|
| T transparent | 90.93 |
| Tc20s tracked | 90.97 |
| M tracked | 91.09 |
| T_sub20 tracked | 91.28 |
| human | 87.54 |

In turns ≥ 45° (64 rows), the human reads 81.33 and the teacher about 90.6.

⇒ **The teacher's progress does not depend on the conversion, the tracker or the sampling: every arm lies within 0.4 EP
points of the others.** Whatever separates our EP from the paper's 91.5 is upstream of NAVSIM: the checkpoint, the
rollout protocol or the route.

## Next arm, not run here: TF-2b

The cheapest experiment that could make CERT's evidence sufficient is **more held-out logs**.
* D1 used 1,018 of the 1,705 rank-0 held-out samples, the ones whose 10 Hz targets are banked locally. The other 687
  samples sit in other logs.
* Roll them with the registered protocol (`PK/eval/teacher_navtest_seam.py --heldout`, about 1.4 s per row; torch
  processes, so the WinError-1455 memory check comes first). Score them with `tf2_score.py`, which needs the P5-S
  derivation on the new rows: `iterate20` runs inside `tf2_score.py`, so only a `p5s`-format input npz is needed.
* Cost: ≈ 1 CPU-h.
* Pre-register CERT − Tc20s on the fresh logs with the same bar.

It was not run here because this agent is near its context budget (rule 6b). It is the first item of the handoff.

## Artifacts

All in `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/tf2/`, **ONE place only**; the MM lands them.

| file | contents |
|---|---|
| `PREREG_TF2.md`, `PREREG_TF2.sha256` | the pre-registration |
| `tf2_score.py` | the scorer. `--resume` was added after shard 2 was stopped at 69 rows to keep commit memory above 10 GB. It filters rows only and changes no computation; C1 and C2 pass on all rows |
| `tf2_analyze.py` | the analysis |
| `launch_tf2.cmd` | the WMI launcher. Its three WMI launches died silently, cause UNVERIFIED; the run used session background tasks instead |
| `finish_s2.sh` | the shard-2 finisher |
| `tf2_result.json` | the full result |
| `rows/tf2_s{0,1,2,2r0,2r1}.jsonl` | per-row output, sha256 in the banking log below. Each row carries every arm's sub-scores and its anatomy |
| `rows/*.log`, `rows/s2_rest*.txt` | shard logs and the shard-2 remainder lists |

Row sha256: s0 `70d40d0d…`, s1 `ed320e63…`, s2 `957c6847…`, s2r0 `e7fcb8be…`, s2r1 `95dedbcb…`. Each was verified as
equal to the scratch copy before the scratch was deleted.
