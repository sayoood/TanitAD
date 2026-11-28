# RESULT_V8F_HACC_LAT: neither zero-training arm removes HACC's T1-kin lateral deficit. Outcome: NOT FIXED. The deficit is not in O's lateral plan, and the canonical curvature / yaw-rate rows are millimetre-noise-dominated below 5 m/s

`PREREG_V8F_HACC_LAT.md` (`96604184`, landed before any number of either arm). Read 2026-10-26 by `raw/2026-10-26-hacc-lat/t1kl_read.py`, applied verbatim. Claim row `H-V8F-HACC-LAT-1`.

* **Tier T1 (T1-kin)**, the settings of `PREREG_V8F_T1KIN_HACC`: m = 10, `--nav deadreckon --scene logged --limit logged --schedule periodic`; eval-141 (2,399 windows, 141 episodes). Dev-box CPU. ⛔ Not closed loop.
* **Estimator:** paired episode-cluster bootstrap, B = 2,000. F = 0.1207 m, F/2 = 0.0604 m, 2F = 0.2415 m.
* **Built by** a Sonnet build agent, read by the MM. The pipeline:
  * `--dump-ticks` and `--windows train` on `v8f_t1_eval.py`;
  * an optional gate scalar `hacc_gate_v` beside `hacc_w` in `V8FMini`;
  * `v8f_hlat.py`, with HACC's fit functions copied verbatim and pinned by an AST test.

  The chain ran from a frozen copy and asserted the sha256 of every imported file before and after each run.
* **Artifacts** (`raw/2026-10-26-hacc-lat/`):
  * the reads, fits and bake records, controls, logs and helper scripts;
  * the code as run (`code/`, not integrated into `stack/`: neither arm ships);
  * float32 dumps of the 12 eval runs.

  Checkpoints, train tick dumps and float64 run dumps: `E:/Projects/TanitAD-artifacts/v8f-mini-hacc/hacc_lat_2026-10-26.tar` (2,046 files, 1,078,712,320 B, sha256 `66767eb1…09e250f7e69`, `tar -d` verified).

## The fits (TRAIN tick samples only; 141 training clips, 28,746 ticks per seed at m = 10)

* **Arm A** (HACC's weights and λ rule re-fitted on the tick samples): λ0 = 408.3. No grid λ qualified at either seed, so the rule's pre-registered fallback chose **m\* = 0 (λ = 0)**.
* **Arm B** (the cross axis falls back to P10 below a speed gate v\*):
  * **v\* = 2.0 m/s at s0**, qualified;
  * **v\* = 8.0 m/s at s1**, the fallback (none qualified).
  * The two readings of "within F/2" choose the same v\* at both seeds (`reading_sensitivity.same_as_used` true), so the clause decided nothing.

## The read (arm − P10 at T1-kin m = 10; lower is better)

| arm, seed | ADE6 (m) | LAT-DOM | arm − HACC ADE6, upper bound (KEEP ≤ +F/2) | rows separated-WORSE than P10 |
|---|---|---|---|---|
| A, s0 | −0.2226 [−0.3321, −0.1187] | ✗ | +0.7616 ✗ | ADE2, speed, accel, yaw rate, curvature |
| A, s1 | +0.0531 [+0.0103, +0.0926] | ✗ | +1.0013 ✗ | ADE6, accel, heading, yaw rate, curvature, cross-track |
| B, s0 | −0.8759 [−1.0165, −0.7361] | ✗ | −0.0240 ✓ | yaw rate, curvature |
| B, s1 | −0.7605 [−0.8923, −0.6437] | ✗ | +0.0990 ✗ | curvature |

**Outcome (the table, verbatim): NOT FIXED.** No arm removes the separated-worse lateral rows at both seeds.
* Arm A is far worse than HACC (+0.61 / +0.85 m ADE6, +2.2 / +0.9 m/s² accel).
* Arm B at s0 is separated-better than HACC on ADE6 (−0.047), speed, accel, heading and cross-track, but curvature stays worse than P10.

**Controls (all PASS):**
* C-T: the tool without `--dump-ticks` is bit-identical, 49 / 49 files.
* C-T2: the tick-0 samples equal the open-loop τ / P / v0 exactly.
* C-A1 and C-B1: the pipeline reproduces the T1KIN HACC dumps, max \|d\| = 0.0.
* C-B2: with the gate always on, the cross plan equals P10's, 0.0.
* C-A2: λ = 0 equals the clipped closed form to 4.8e-14 / 8.6e-14; the GT := P regression gives w = 0.0; KKT holds.
* The frozen suite: 152 passed.
* R-1 and R-2 on every arm run; C-1 and C-2 on the m = 30 runs; the floor equals T1KIN's floor.

## Why it is not fixed (DESCRIPTIVE, decides nothing): the deficit is not in O's lateral plan

1. **From the same state, HACC's plan is no worse than P10's** (`lat_tick_probe.log`: TRAIN ticks, the first 10 knots, i.e. the part a tick executes, against the tick GT).
   * At tick 0 HACC is better in every speed band: −0.24 / −0.21 1/m below 2 m/s.
   * At the own-state ticks 1–2, HACC − P10 is between −0.15 and +0.006 1/m in every band.
2. **Arm B at s1 makes the cross axis literally P10's below 8 m/s**, and the eval curvature excess does not move: B − HACC is −0.0000 [−0.0055, +0.0059] 1/m.
3. **The canonical lateral rows have a millimetre noise floor that is the size of the deficit** (`curv_noise_floor.log`, INSTRUMENT probe, no model arm read). The eval-141 GT is perturbed by i.i.d. Gaussian noise per 0.2 s knot and scored against itself with the reader's own row function.

| position noise | curvature MAE, pooled (1/m) | below 2 m/s | 2–5.2 m/s | above 11.8 m/s | yaw-rate MAE, pooled (deg/s) |
|---|---|---|---|---|---|
| 2 mm | 0.0080 | 0.074 | 0.014 | 0.0004 | 0.96 |
| 5 mm | 0.0213 | 0.205 | 0.033 | 0.0009 | 2.44 |
| 10 mm | 0.0377 | 0.342 | 0.068 | 0.0018 | 4.70 |

* HACC's "deficit" against P10 is +0.040 / +0.016 1/m and +4.2 deg/s. That is **the noise of 5–10 mm of position jitter**, and it sits in the band where the floor is 0.2–0.3 1/m.
* ⇒ Near standstill, a wiggle that no occupant could notice ranks two planners on these rows. A loop that matches the human's slow speeds pays for it: HACC's speed MAE is 0.36–0.43 m/s better than the floor's.
* This is the root-cause class of `RETR-2026-09-26-YAWMASK` (yaw-rate cells scoring standstill jitter), now on curvature as well.

⚠ **This does not re-score this result or any earlier one.** The pre-registered rows were read as written, and NOT FIXED stands. Re-reading DOMINANCE on well-conditioned lateral rows is a new question, under its own pre-registration (`PREREG_V8F_T1_LATCOND`), with the canonical rows still reported beside it.

## Next

* **Committed by the outcome row:** a TRAINING lever on O's lateral output at low speed, on box 2, under its own PREREG.
  * ⚠ The diagnosis above says it would be aimed at a deficit that is not in O's plan. Box 2 is also allocated to FLAT and then OACC.
  * ⇒ The MM holds this lever until `PREREG_V8F_T1_LATCOND` is read. It is named and ordered, not dropped.
  * If the conditioned rows still show a lateral deficit, it runs.
* **`PREREG_V8F_T1_LATCOND`** (zero training, dev-box CPU, re-reads banked dumps): the cheapest discriminating experiment, pre-registered in the same commit as this result, before any arm is read on its rows.
