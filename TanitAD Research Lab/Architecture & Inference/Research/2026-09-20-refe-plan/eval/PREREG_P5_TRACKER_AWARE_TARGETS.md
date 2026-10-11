<title>PREREG P5 — tracker-aware and margin-aware targets</title>

# PREREG P5: tracker-aware and margin-aware targets, held-out label test (written 2026-10-11, BEFORE any P5 number exists)

**Status at writing:**
- No P5 label, metric or control has been computed.
- The scripts `eval/p5_target_labels.py` and `eval/p5_families.py` land in the same commit as this file.
- The only runs before landing are a **controls-only smoke on 40 rows** and six **deliberate-regression arms** on 6
  rows (§4). They print and write the control booleans and `n`, and nothing else.
- This file is written first. It is not edited after the first full-run number exists, except in the clearly dated
  **§9 decisions log**.

**Origin:** `eval/RESULT_P4_TEACHER_NAVTEST.md` §6, **PI decision (b)**, verbatim:
> Reframe P5 from "re-query failing targets" to **margin-/tracker-aware targets**. Dropping the ~8 % of targets
> that fail NAVSIM's simulated DAC would discard paths that are compliant as driven (D1-navtrain); the better
> candidates are (i) a margin term (≥ ~0.6 m, the human's median) when choosing or re-querying the target, or (ii)
> training the WTA on the tracker-compensated reference. Both need a pre-registered held-out test before GPU.

This file is that pre-registered held-out test. It uses 0 GPU.

**What P4 already measured, and why it motivates this test:**
- D1-navtrain (`raw/2026-10-05-teacher-navtest/d1_navtrain.json`):
  - 82 of 1,018 held-out teacher targets (8.06 %) fail NAVSIM's simulated DAC.
  - Only 0.59 % fail on the teacher's own 20-pose path.
  - 95 % of the failures are class C: they are made by NAVSIM's tracker.
- F2: compensating the teacher's 8 NAVSIM poses for the tracker raises its navtest PDMS by +4.30 [+3.33, +5.32].
- D3: the teacher's turn margin is 0.37 m, against 0.60 m for the human.

## §1 Question

Two label-side questions, both on the held-out navtrain targets that D1 already characterised. Each must be answered
before either becomes a GPU arm.

- **(ii) Tracker-compensated labels.** Take the WTA's training label, the 20-pose teacher target at 0.2 s (`t20`),
  and pre-distort it so that NAVSIM's tracker, executing it, lands on the teacher's intended poses. Does that label:
  - fail NAVSIM's simulated DAC less often than the plain label;
  - keep the executed trajectory at least as close to the teacher's intent;
  - stay learnable (no new jerk)?
- **(i) A margin criterion.** Is the drivable-area margin of the plan's own footprint, with no tracker, below 0.6 m a
  usable flag for the targets that NAVSIM's tracker will push out of the drivable area?
  - "Usable" means it catches most of them without discarding a large share of good targets.
  - This is the tracker-free signal that a margin training term would read.

## §2 Arms (all 0-GPU, CPU, dev box, PY_D through `eval/run_p4.py`)

**The token set** (§5) is the 1,018 held-out rank-0 navtrain targets with a banked 10 Hz target, as in D1. For each
token, every reference below is an **8-pose NAVSIM plan**, simulated by NAVSIM's own LQR + kinematic bicycle
(`refe/navsim_dac.py::simulated_states`, vendored verbatim).

| ref | what it is | role |
|---|---|---|
| **T** | `to_navsim(t20)`: the plain teacher label as NAVSIM receives it | baseline |
| **Tc20** | `t20` pre-distorted on its **own 0.2 s grid**, then `to_navsim` | **primary arm (ii)** |
| Tc8 | P4's 8-pose compensation `p4_tracker_comp.compensate(T)` | descriptive: the 20-pose variant against P4's 8-pose one |
| Tw | the wrong-sign iteration (`r20 -= e20`) | negative control: must make things worse |
| ident | the iteration with the error forced to 0 | identity control: must reproduce T bit for bit |
| H | the logged human future, `teacher_lane_check.human_p8` | reference |
| Hc | `compensate(H)` | descriptive |

**The Tc20 iteration**, committed here:
- Start with `r20 = t20`.
- Simulate `to_navsim(r20)` and take the executed 8 poses `s`, in the ego frame, at the rear axle
  (`p4_tracker_comp.tracked8`).
- The tracking error at the 8 NAVSIM times is `e8 = [T_xy − s_xy, wrap(T_h − s_h)]`.
- Spread it onto the 0.2 s grid: `e20` = linear interpolation of `[0, e8]`, knots at t = 0, 0.5, …, 4.0 s, queried at
  t = 0.2, 0.4, …, 4.0 s.
  - The 1, 2, 3 and 4 s samples (indices 4, 9, 14, 19) equal `e8[1, 3, 5, 7]` exactly.
  - The half-second NAVSIM poses are means of two 0.2 s samples, so one step is not exact. That is why it iterates.
- Update `r20 += e20`. The heading is not wrapped, so that `ident` stays bit-exact.
- Run 3 iterations. Keep the iterate with the smallest max-xy deviation over k = 0..3, with k = 0 (plain T) included,
  so Tc20 can never deviate more than T.
- Tw takes `r20 -= e20` and keeps the **last** iterate, so its worsening is not hidden by the same selection.

## §3 Scoring

Every reference is scored on **its executed trajectory**: the NAVSIM simulation of its 8 poses.

| metric | definition |
|---|---|
| **sim8 fail** | NAVSIM DAC verdict on the executed footprint: `state_array_to_coords_array` corners, CENTER excluded, against `drivable_polygons` (exactly REVIEW_7 and D1) |
| interp8 fail | DAC on the plan's own footprint (`at_10hz` of the 8 poses, no tracker), as D1 |
| DDC, lk10 | `navsim_lane.navsim_ddc_and_lane` on the same simulation, with the PDM-corrected route (exactly REVIEW_7) |
| comfort | `navsim_dac.comfortable_from_states` (NAVSIM v1's six comfort metrics) |
| **executed vs intended** | against **T** (the teacher's intent) for T, Tc20, Tc8, Tw and ident, and against **H** for H and Hc: max xy deviation; signed along-track error at 4 s and mean \|along\|, mean \|cross\|, all projected on the intended heading; heading MAE (°); path-length ratio |
| **learnability** | per row, on the 20-pose label with the origin prepended at 0.2 s: max \|acceleration\| (m/s²) and max \|heading rate\| (°/s); reported as median and p95 over rows, for Tc20 against t20 |
| margin | `p4_dac_margin.margin`: the minimum over steps and the 4 corners of the signed distance to the union of the drivable polygons, on T's plan footprint (`at_10hz`, no tracker); positive = inside |

⭐ **Families are read on the EXECUTED trajectory against the INTENDED one.** P4 F2 read the student's families on
the pre-distorted reference. That inflated its yaw-rate (3.23 → ~8.4 °/s), heading and curvature, and left its PDMS
gain NOT PROVEN. Scoring executed against intended removes that ambiguity. It also gives evidence for **PI decision
(a)**'s prerequisite. The decision itself stays the PI's.

**Interval:** paired log-cluster bootstrap over the logs of the token set, 10,000 resamples, seed 20260927.
- For a rate it is the ratio estimator (sum of fails over sum of rows, under the same cluster weights).
- For a continuous metric it is the difference of cluster-weighted means.
- Two-sided 95 % percentile CI.
- ⚠️ **There are only 11 logs, so this interval is coarse.**
- It answers *another draw of logs* only. There is no training here; the simulation is deterministic, and the
  batch-independence control (§4) checks it.

## §4 Controls (any failure ⇒ `ZZP5_GATE_FAIL`, and no verdict is read)

1. **Identity:**
   - `ident`'s 20-pose label equals `t20` bit for bit.
   - Its 8 poses equal T's.
   - Its sim8 verdicts equal T's on every row.
2. **Knot exactness:** `e20[[4, 9, 14, 19]] == e8[[1, 3, 5, 7]]` exactly, on every row and every iteration checked.
3. **Wrong sign:** Tw's median max-xy deviation is **larger** than T's, and Tw is **not** CONFIRMED by the §6 rule.
   - If the wrong-sign iteration also helped, the iteration would not be measuring tracking error.
4. **The D1 join:**
   - T's sim8 equals D1's `sim8` on **100 %** of the rows.
   - The row set is D1's: same (log, token) pairs and the same n.
5. **REVIEW_7 agreement** (`raw/2026-10-04-lane-discipline/teacher_check_heldout.jsonl`):
   - T's and H's DAC agree on **100 %** of rows.
   - T's and H's DDC and lk10 agree on **≥ 99.5 %** each. REVIEW_7 ran on the pod, so its route-correction library
     may differ in the last digit.
6. **Margin direction:** every row with margin < 0 is an interp8 violation.
   - This is one way only. The union contains every polygon, so the converse is not implied.
7. **Batch independence:** T simulated alone and T simulated in the batch agree to within 1e-6 m on every executed
   pose.

The controls-only smoke (`--limit 40 --controls-only`) runs controls 1, 2, 5, 6 and 7, and control 4's join, on 40
rows.
- Control 3 is evaluated on the full run only, because it reads a verdict.
- The smoke writes no metric.

**Each control is proven able to fail.** `--mutate {knot,ident,d1,r7,margin,batch}` (controls-only, refused
otherwise) breaks exactly one control, and `ZZP5_CONTROLS` must then read FAIL with that control, and only that
control, false. Run before landing, 6 rows each (`raw/2026-10-11-p5-target-labels/mutate_*.{log,json}`):

| `--mutate` | what it breaks | control that went FALSE | all others |
|---|---|---|---|
| `knot` | interpolation knots shifted by 0.05 s | C2 knot exactness | TRUE |
| `ident` | `ident`'s correction sign 0 → 1e-9 | C1 identity | TRUE |
| `d1` | T's sim8 flipped on the first row before the D1 compare | C4 D1 join | TRUE |
| `r7` | T's DAC flipped on the first row before the REVIEW_7 compare | C5 REVIEW_7 | TRUE |
| `margin` | T's margin forced negative | C6 margin direction | TRUE |
| `batch` | "alone" compared against the wrong batch row | C7 batch independence | TRUE |

The clean arm (no `--mutate`) read PASS on the same 6 rows and on the 40-row smoke (`smoke40_controls.{log,json}`).

## §5 Token set

- D1's 1,018 rows (`raw/2026-10-05-teacher-navtest/d1_navtrain.json`): the REVIEW_7 held-out rank-0, step-0 targets
  that have a banked 10 Hz target in `E:/Projects/TanitAD/data/refe_navtrain10`, over **11 logs**.
- n = 1,018, 82 of them sim8 failures (8.055 %).
- Nothing is re-selected.
- Rows that raise are counted and reported, and leave both arms (paired).

## §6 Decision rules (committed)

**Arm (ii): Tc20 against T**, primary metric Δ sim8 fail rate (Tc20 − T).

**CONFIRMED** iff:
- the upper CI bound is **< 0**, AND
- no guard is adverse. Each guard is a paired Tc20 − T CI:
  - DDC rate or comfort rate: adverse if the upper bound is < 0;
  - lk10 rate: adverse if the lower bound is > 0;
  - executed vs intended (mean \|along\|, mean \|cross\|, max-xy deviation, heading MAE): adverse if the lower
    bound is > 0.

**REFUTED** iff the lower bound of Δ sim8 is **> 0**: the compensated label fails more often.

**NOT CONFIRMED** otherwise, including a CI that crosses 0, or a separated Δ with an adverse guard. The adverse
guard is named.

**Learnability flag** (does not change the verdict): **LEARNABILITY RISK** iff Tc20's median per-row max \|heading
rate\| or median per-row max \|acceleration\| is more than **2×** t20's. A CONFIRMED arm with this flag goes to GPU
only after a pre-registered smoothed-label amendment.

**Arm (i): margin < 0.6 m on T's plan footprint, as a flag for T's sim8 failures.**
- **VIABLE** iff the flagged share is **≤ 25 %** of rows AND recall is **≥ 80 %** of the sim8 failures.
- **NOT VIABLE** otherwise.
- The comparison point is drop-if-sim8, which flags 8.06 % with recall 100 % by construction but needs the tracker at
  label time.
- The 0.3 / 0.45 / 0.6 / 0.75 m sweep is reported as descriptive and decides nothing.
- ⛔ The 0.6 m threshold is the PI's and is not tuned on this data.

**Decision map:**

| outcome | next arm (each needs its own prereg, A40-days and the PI's go) |
|---|---|
| (ii) CONFIRMED | **P5-C**: fine-tune the WTA on Tc20 labels, with **P5-D** (drop-if-sim8) as its control |
| (i) VIABLE | **P5-M**: a margin term on the WTA's predicted footprint, ≥ 0.6 m |
| neither | P5-D only, or deprioritise P5 |

⚠️ **Priority is unchanged:**
- P1 and P2 stay ahead of P5 (`RESULT_P4_TEACHER_NAVTEST.md` §6).
- The student sits ~9.6 PDMS below the trackable teacher.
- Only 29 % of its DAC failures are tracker-repairable (83 % for the teacher).

## §7 Reported

**Per reference (T, Tc20, Tc8, Tw, ident, H, Hc), each with n and n_logs:**
- sim8, interp8, DDC, lk10 and comfort rates;
- the executed-vs-intended metrics;
- the learnability medians and p95.

**Arm (ii), Tc20 − T:**
- paired CIs on every §6 metric;
- the D1 class breakdown of what Tc20 repairs and breaks (class A teacher-path / class C tracker);
- per-row iteration traces (max-xy deviation for k = 0..3);
- `max |to_navsim(Tc20) − Tc8|`.

**Arm (i):**
- flagged share and recall at 0.6 m, with the sweep;
- the human's margin on the same rows, descriptive.

**Four families** (`eval/p5_families.py`, `taniteval/adapters/navsim.py` `four_families_block`, tier T1, 2,000
resamples, cluster unit `log_name`):
- executed against intended for T, Tc20, Tc8 and Tw;
- executed against H for T, Tc20 and Tc8.
- STRATEGIC is **UNAVAILABLE** in NAVSIM and **PAUSED** by the PI's 2026-09-19 ruling.
- A family that raises is reported as RAISED with its error.
- The families are descriptive and decide nothing here.

**Raw output:** `raw/2026-10-11-p5-target-labels/`:
- `p5_labels.json` (metrics, controls, verdicts);
- `p5_exec.npz` (executed and intended poses, logs, tokens, v0);
- `p5_families.json`;
- the run log.

## §8 What this does NOT license

- ⛔ **Not a PDMS claim.** No model is trained or scored. The test is about the **labels**: whether a label NAVSIM's
  tracker executes cleanly exists and stays faithful.
- ⛔ **Not a lever claim for the student.** CONFIRMED is a necessary condition for P5-C, never a sufficient one. A
  student can fail to learn a pre-distorted label, and the student's own DAC failures are mostly not tracker-made.
- ⛔ **Not a ruling on PI decision (a).** Tc20 embeds the tracker into the labels, which is a softer form of
  test-time compensation. Whether to read families on the reference or the executed trajectory is the PI's call. This
  test supplies evidence for it.
- ⛔ **Not a navtest result.** These are navtrain held-out targets, 11 logs. A CONFIRMED label effect still has to
  survive P5-C's own navtest read.

## §9 Decisions log

### 2026-10-11 · Amendment P5-S: smoothed correction (registered after P5, before any P5-S code or run)

**Why.** The P5 result (`eval/RESULT_P5.md`) read arm (ii) **CONFIRMED** with **LEARNABILITY RISK**:
- Tc20's median per-row max |acceleration| is 2.824 against t20's 1.371 m/s², which is 2.06×.
- Heading rate is 1.68×.

§6 says such an arm goes to GPU only after a pre-registered smoothed-label amendment. This is that amendment.

**Hypothesis** (HYPOTHESIS, from §2's construction, not measured). `e20` is a piecewise-linear interpolant of
`[0, e8]`, so its slope jumps at every half-second knot. Each jump is a second-difference spike in the label, which
`learnability()` reads as acceleration. A C²-continuous interpolant through the same knots should remove the
spikes without moving the knots.

**Arms.** Same 1,018 tokens, same `ITERS = 3`, same best-iterate selection, same bootstrap (10,000 resamples, seed
20260927, log clusters). The only change is the interpolant that spreads `e8` onto the 0.2 s grid in `iterate20`,
selected by a new flag `--interp`:

| `--interp` | interpolant of `[0, e8]` at knots t = 0, 0.5, …, 4.0 s | role |
|---|---|---|
| `linear` | `np.interp`, unchanged | **control**: must reproduce the landed P5 run |
| `cubic` | `scipy.interpolate.CubicSpline(..., bc_type="natural")` | **primary**, giving Tc20s |
| `pchip` | `scipy.interpolate.PchipInterpolator` (C¹, no overshoot) | **secondary** |

- The interpolant applies to all three rows of the batch: Tc20, Tw and ident.
- Tc8 and Hc use P4's 8-pose `compensate` and are unaffected; they are rerun only as a byte-level check.

**Controls**, in addition to §4's C1 and C3–C7, which stay as written:
- **C2′ (knot exactness under a spline):** `max |e20[KNOT_Q] − e8[KNOT_E8]| ≤ 1e-9`, with the max recorded.
  - `linear` keeps C2's `array_equal`.
  - `--mutate knot --interp cubic --controls-only` must read FAIL.
- **C8 (reproduction):** `--interp linear` must reproduce the landed `raw/2026-10-11-p5-target-labels/p5_labels.json`.
  - That means every per-row record, every `per_ref` entry and every CI, compared with `"seconds"` stripped.
  - A mismatch is a gate failure for the whole amendment, and no P5-S verdict is read.
- **C3** stays binding: Tw under `cubic` must not be CONFIRMED.

**PASS bar for a smoothed arm, committed. All three are required:**
- (a) **Δ sim8 (Tc20s − T): upper CI bound < 0.**
- (b) **No adverse guard**, with §6's guards and §6's directions, on Tc20s − T.
- (c) **Learnability:** Tc20s median per-row max |acceleration| **≤ 2×** t20's, **AND** median per-row max |heading
  rate| **≤ 2×** t20's.

**How the outcome is read:**
- `cubic` PASS ⇒ **P5-S PASSED**. Tc20s (cubic) is the label P5-C pre-registers.
- `cubic` FAIL and `pchip` PASS ⇒ **P5-S PASSED on the secondary**. Tc20s (pchip) is the label, and the doc says
  the primary failed.
- Both FAIL ⇒ **P5-S FAILED**. The failed letter (a, b or c) is named for each arm. The committed next lever is
  **P5-R, a regularised inversion**:
  - solve for `r20` minimising the 8-pose tracking error plus λ·‖Δ²r20‖², rather than interpolating the error;
  - it gets its own §9 entry before any code.

**Descriptive reads (no verdict):**
- Tc20s − Tc20 paired CIs on sim8 and interp8. This asks: does smoothing cost tracking?
- Dual compliance (sim8 AND interp8) for T, Tc20 and Tc20s.
- Learnability p95.
- **Tsel-s**, DERIVED post hoc as in the P5 result: Tc20s where T fails sim8, else T.
  - Report its dual compliance and its learnability on the selected rows.
  - It stays an input to the P5-C design, never a verdict.
- The four families (`eval/p5_families.py`) on the new `exec_Tc20` against T and H. STRATEGIC is UNAVAILABLE in
  NAVSIM and PAUSED (PI ruling 2026-09-19).

**Not licensed:** the same four limits as §8 apply. A P5-S PASS clears the learnability gate for a P5-C
pre-registration. It is not a claim that a student learns Tc20s, and it is not a PDMS claim.

**Outputs:**
- `raw/2026-10-11-p5s-smoothed/p5s_{linear,cubic,pchip}.json` (+ `.npz`, `.log`), `mutate_knot_cubic.{json,log}`
- `eval/p5s_compare.py`, `eval/RESULT_P5S.md`
