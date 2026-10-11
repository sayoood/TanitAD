<title>P4 — our teacher on navtest</title>

# RESULT P4: our DriveRL teacher protocol on NAVSIM navtest (2026-10-05)

**Pre-registration:** `eval/PREREG_P4_TEACHER_NAVTEST.md` (written before any teacher number; §9 logs every later
decision with its time). **Tier:** NAVSIM v1.1 PDMS, a non-reactive pseudo-simulation of a 4 s open-loop plan against
logged agents. **Estimator:** paired log-cluster bootstrap over 136 logs (`proxy_eval.boot`, 10,000 resamples, seed
20260927). It answers *another draw of LOGS*. The teacher is deterministic given device and rate, so there is no
inference-seed variance on this arm. **Evidence classes:** MEASURED (artifact path) · PUBLISHED (DriveZero arXiv
2609.06055, banked PDF sha256 `82a00048…`, Table 4 p. 11) · ESTIMATED · HYPOTHESIS. Every navtest number here is
**exploratory** for the student programme: navtest has been read many times.

## 0 · Verdict

**TEACHER CAPS** (registered rule: CAPS iff PDMS < 93.0 OR DAC < 98.0). Our teacher protocol scores
**PDMS 89.81 [88.56, 91.00]** and **DAC 95.23 [94.24, 96.16]** on the 12,110 primary tokens. Neither threshold lies
inside its interval, so the verdict is not sensitive to log sampling. It is **−4.74 [−6.03, −3.50]** below the paired human
and **−5.99** below the paper's DriveRL row (95.8, PUBLISHED). The student's DAC **equals the teacher's**
(student − teacher DAC **−0.50 [−1.89, +0.80]**, not separated).
(MEASURED, `raw/2026-10-05-teacher-navtest/p4_result.json`)

**But the cap is mostly not the teacher's driving.** Of the teacher's 578 DAC failures, **504 (87.2 %) are
compliant in the teacher's own rollout** and fail only after NAVSIM's LQR tracker re-simulates the 8-pose plan.
Only **72 (0.59 % of tokens)** leave the drivable area in the rollout itself (D1). The tracker is equally inexact
on human plans (D2), but the teacher's turn paths keep **less margin** to the road edge (0.37 m vs the human's
0.60 m, D3). The same holds on held-out navtrain targets: 95 % of their simulated-DAC failures are compliant
paths. **Made trackable (F2), the teacher scores 94.10 [93.59, 94.60], indistinguishable from the human.** TTS re-query
(F1) repairs 1 of 120 failures, so re-query is not the lever. P5 should be reframed as margin- or tracker-aware targets,
and REVIEW_7's P1/P2 stay first (§6).

## 1 · Controls (all passed)

| control | known value | read | artifact |
|---|---|---|---|
| C3 teacher pipeline: re-roll 47 held-out navtrain samples vs the BANKED rank-0 targets (10 Hz, CPU) | median ≤ 0.05 m, ≥ 90 % ≤ 0.25 m | median **8.1e-7 m**, max 0.026 m, 47/47 ≤ 0.25 m | `c3_heldout_10hz.json` |
| C3 mutation arm: the same at 20 Hz must FAIL | FAIL | median 0.63 m, 27.7 % ≤ 0.25 m: **FAIL** | `c3_heldout_20hz_mutation.json` |
| C1 frame/time: logged future vs W3 `human_future_poses`, 1 mm | only the 36 known DB≠OpenScene tokens over | 36 over, **all 36 the known ones, 0 unexpected**; median 9e-16 m | `refe_p4_teacher_navtest.report.json` |
| C2 human through the harness on the same 12,146 tokens | 94.55 [93.80, 95.24] | **94.55 [93.78, 95.23]**, status PASS | `score_logs/refe_p4_human_navtest_status.json` |
| C4 coverage | misses ≤ 1 % | **0 misses** (12,146/12,146 rolled and scored, harness PASS) | same |

The local `checkpoint_2400.pt` (md5 `0b4236016856704b564bdbe33043185a`) reproduces the banked targets (median deviation
8e-7 m, max 0.026 m), so **no pod file was needed and nothing ran on the pod.** 24 tokens had less than 2.0 s of DB history
and ran with the longest available buffer (`history_clamped`). The to_navsim plan differs from the rollout's own
0.5 s states by ≤ 0.036 m.

## 2 · Registered read (primary set = 12,146 − the 36 C1-known = 12,110 tokens, 136 logs)

| | PDMS | NC | DAC | EP | TTC | C | DDC* | zero rate % |
|---|---|---|---|---|---|---|---|---|
| **teacher (ours)** | **89.81** [88.56, 91.00] | 99.37 [99.10, 99.61] | **95.23** [94.24, 96.16] | 85.18 [83.78, 86.58] | 97.15 [96.49, 97.79] | 99.92 [99.84, 99.98] | 98.97 [98.64, 99.25] | 5.34 [4.28, 6.47] |
| human (paired) | 94.55 [93.78, 95.23] | 100.0 | 100.0 | 86.94 [85.09, 88.58] | 100.0 | 99.96 [99.91, 99.99] | 98.82 [98.38, 99.20] | 0.0 |
| **teacher − human** | **−4.74** [−6.03, −3.50] | −0.63 [−0.90, −0.39] | **−4.77** [−5.76, −3.84] | −1.76 [−3.35, −0.09] | −2.85 [−3.51, −2.21] | −0.04 [−0.13, +0.04] | +0.14 [−0.05, +0.35] | +5.34 [4.28, 6.47] |
| student (deployed, route-fix) | 84.47 [83.16, 85.69] | 98.12 [97.71, 98.50] | 94.73 [93.69, 95.72] | 76.70 [74.62, 78.62] | 93.68 [92.75, 94.57] | 99.93 | 97.61 | 6.87 [5.95, 7.82] |
| **teacher − student** | **+5.34** [+4.00, +6.78] | +1.25 [+0.86, +1.65] | **+0.50 [−0.80, +1.89]** | +8.48 [+6.96, +10.17] | +3.47 [+2.57, +4.39] | −0.02 | +1.36 [+0.88, +1.87] | −1.53 [−2.91, −0.23] |
| paper DriveRL (PUBLISHED, Table 4) | 95.8 | 99.8 | 99.9 | 91.5 | 99.1 | 99.0 | — | — |
| ours − paper (point) | −5.99 | −0.43 | −4.67 | −6.32 | −1.95 | +0.92 | | |

*DDC has weight 0 in v1.1 PDMS (reported separately). All-12,146 read: teacher PDMS 89.79 [88.55, 90.97], DAC 95.20 —
same verdict. Teacher better / worse / tied than the human on 5,689 / 3,507 / 2,914 tokens.

**Reading.** The teacher beats the student by +5.34, almost all in EP (+8.5) and TTC (+3.5). **On DAC it does not:
the student sits at the teacher's level.** The teacher's EP 85.2 is below the paper's 91.5 *and* below the human's
86.9.

## 3 · Where the teacher fails (splits as REVIEW_7 defines them; MEASURED, `p4_result.json` → `splits`)

| scene type | n | teacher PDMS | teacher DAC | teacher zero % | human | student (review run) |
|---|---|---|---|---|---|---|
| far-goal bearing < 10° | 5,772 | **94.80** [94.33, 95.26] | 99.39 | 0.92 | 95.13 | 85.95 |
| 10–30° | 2,307 | 89.81 [87.91, 91.51] | 94.93 | 5.38 | 94.14 | 81.12 |
| 30–60° | 2,388 | **83.29** [80.20, 86.36] | **89.03** | 11.77 | 94.29 | 83.34 |
| ≥ 60° | 1,643 | **81.72** [79.15, 84.19] | **90.02** | 11.50 | 93.47 | 76.58 |
| on-route (≤ 20 m) | 11,635 | 89.94 | 95.26 | 5.29 | 94.55 | 84.38 |
| off-route (> 20 m) | 475 | 86.56 [81.83, 91.15] | 94.32 | 6.74 | 94.68 | **55.45** |
| v0 1–5 m/s | 4,546 | 86.70 | **92.43** | 8.29 | 94.16 | 85.27 |
| v0 ≥ 10 m/s | 963 | 96.52 | 99.06 | 1.04 | 97.03 | 81.81 |
| **student dirty fan (< 50 % safe)** | 1,846 | **75.58** [71.30, 79.85] | **81.58** | 20.48 | 94.82 | 54.38 |
| student clean fan (≥ 95 %) | 5,302 | 94.68 | 99.68 | 0.34 | 94.63 | 90.90 |

* **On straight goals the teacher is at human level (94.80 vs 95.13); its whole deficit is in turns and at low speed.**
  The paper's teacher does not show this (DAC 99.9, PUBLISHED).
* **Off-route goals are a student-only problem**: the teacher corrects its own route (DriveRL's
  `correct_route_roadblock_ids`), so it scores 86.6 where the student scored 55.5 before A9.
* **Inheritance is real but partial** (`p4_result.json` → `inheritance`). Where the teacher fails NC·DAC (647 tokens),
  the student's fan is dirty 58.4 % of the time vs 12.8 % elsewhere (×4.6). The student's pick zeroes 24.7 % vs 7.1 %.
  The teacher fails on 20.5 % of dirty-fan tokens. **But only 16.4 % of the student's zero picks fall on teacher-fail
  tokens**, so most student zeros happen where the teacher drives cleanly.

**Families** (`p4_teacher_families.json`, `p4_student_deployed_families.json`; families6 episode-cluster bootstrap):

| family | metric (vs the logged human) | teacher | student (deployed) |
|---|---|---|---|
| LONGITUDINAL | speed MAE m/s | 0.770 [0.730, 0.811] | 0.886 [0.856, 0.919] |
| | speed bias m/s | +0.329 [+0.233, +0.418] (faster) | −0.324 [−0.385, −0.267] (slower) |
| | progress ratio / under-progress rate | 1.096 / 0.308 | 0.981 / 0.595 |
| | distance keeping (headway/TTC to lead) | **UNAVAILABLE** in families6: no lead track supplied (n = 0, work item); the harness TTC sub-score above is the only distance-keeping read | same |
| LATERAL | heading MAE ° | 2.48 [2.28, 2.66] | 2.71 [2.50, 2.90] |
| | yaw-rate MAE °/s | 2.50 [2.32, 2.68] | 3.23 [3.03, 3.43] |
| | curvature MAE 1/m | 0.0115 [0.0107, 0.0123] | 0.0237 [0.0217, 0.0258] |
| | cross-track MAE m | 0.388 [0.358, 0.416] | 0.424 [0.396, 0.451] |
| TACTICAL | lateral decision acc / κ | 0.934 / 0.880 | 0.918 / 0.849 |
| | longitudinal decision acc / κ | 0.683 / 0.525 | 0.642 / 0.459 |
| | 5-way manoeuvre acc / κ | 0.803 / 0.749 | 0.770 / 0.701 |
| STRATEGIC | — | **UNAVAILABLE**: NAVSIM scores no strategic decision and has no route-setting label (n = 0 by design) | same |

## 4 · Anatomy of the teacher's DAC failures (EXPLORATORY, after the verdict; PREREG §9)

**D1** (`eval/p4_dac_anatomy.py` → `d1_dac_anatomy.json`). Three nested checks were run with NAVSIM's own
polygons and footprint (`refe/navsim_dac.py`): the teacher's own 20 rollout poses, the 8-pose plan interpolated, and
NAVSIM's LQR + bicycle simulation of the plan.

| class (of 578 harness-DAC-fail tokens) | n | % |
|---|---|---|
| A: the teacher's own rollout leaves the drivable area | 72 | 12.46 |
| B: 0.5 s plan subsampling | 2 | 0.35 |
| **C: rollout and plan compliant; NAVSIM's tracker pushes the footprint off** | **504** | **87.20** |

Controls: the human plan reads **0/778** violations; the 200 pass-control tokens agree with the harness 200/200; the
578 fail tokens agree 578/578. Class C sits at 1–10 m/s (488 of 504).

**D2** (`d2_*`). NAVSIM's tracker deviates from the plan by a median **0.81 m (teacher) vs 0.79 m (human)** on the
same fail tokens, so the tracker is equally inexact for both. **D3** (`d3_margin.json`). On class-C tokens the
teacher's own footprint keeps a median **0.37 m** margin to the drivable-area edge, against **0.60 m** for the human
(teacher closer on 73.6 %). On pass controls the two are equal (0.96 vs 0.97 m).
⇒ **The teacher cuts closer to the road edge in turns than the human, by less than NAVSIM's tracking error.**
That is where 87 % of its DAC loss comes from.

**D1 on held-out NAVTRAIN, the training distribution** (`eval/p4_d1_navtrain.py` → `d1_navtrain.json`).
Inputs: the banked 10 Hz rank-0 targets that exist locally for **1,018 of the 1,705** rank-0 held-out samples
(11 logs, so not a random subset). The rate at which targets fail NAVSIM's simulated DAC is
**8.06 % [2.30, 14.64]**; the rate at which **the target path itself** leaves the drivable area is
**0.59 % [0.00, 1.34]**. **78 of 82 failures (95.1 %) are tracker-interface (class C).** The simulated verdict agrees
with REVIEW_7's recorded `teacher.dac_viol` on 1,018/1,018 (a determinism check of the join and pipeline, not an
independent validation).
⇒ **REVIEW_7's "teacher targets fail DAC on 4.97 %" is, by mechanism, ~95 % a property of NAVSIM's tracker plus thin
margin, not of the teacher's path.** Cleaning targets by NAVSIM's simulated DAC would discard paths that are
compliant as driven.
⚠️ The first run of this check read 528 "errors"; they were a DB-lookup bug (navtrain DBs also sit flat in `splits/`),
fixed, and the run repeated. The voided output is kept as `d1_navtrain.first_run_528_missing_db.json`.

## 5 · Next levers, run in the same session (RULE ZERO)

### F2 · tracker-aware plan compensation (zero training; registered in §9 before any compensated number)

The 8 reference poses are corrected so that NAVSIM's own vendored tracker reproduces the intended plan
(`eval/p4_tracker_comp.py`, 3 iterations).

**Instrument level** (`f2/*.report.json`, all 12,146): the tracker's deviation from the intended plan falls from a median 0.45 → 0.05 m for the teacher (p95 1.00 → 0.17) and 0.53 → 0.14 m for the student (p95 1.11 → 0.55).
* Teacher: DAC violations **583 → 123** (486 repaired, 26 broken).
* Student: **643 → 521** (188 repaired, 66 broken).

**Harness** (MEASURED, status PASS on all 12,146 for both; `raw/2026-10-05-teacher-navtest/f2_result.json`; paired
on the 12,110 primary tokens):

| | PDMS | NC | DAC | EP | TTC | C | zero % |
|---|---|---|---|---|---|---|---|
| teacher, compensated | **94.10** [93.59, 94.60] | 99.68 | **99.01** [98.60, 99.36] | 88.76 [88.07, 89.47] | 98.77 | 99.82 | 1.27 [0.88, 1.71] |
| compensated − uncompensated | **+4.30** [+3.33, +5.32] | +0.31 [+0.14, +0.51] | +3.78 [+2.85, +4.76] | +3.58 [+2.59, +4.60] | +1.62 [+1.24, +2.03] | −0.10 [−0.22, +0.01] | −4.07 [−5.11, −3.08] |
| compensated − **human** | **−0.45 [−1.21, +0.36]** | −0.32 | −0.99 [−1.40, −0.64] | **+1.82** [+0.48, +3.34] | −1.23 | −0.14 | +1.27 |
| student, compensated | 85.54 [84.37, 86.65] | 98.37 | 95.74 | 77.11 | 94.41 | 99.83 | 5.69 |
| compensated − uncompensated | **+1.07** [+0.68, +1.45] | +0.24 [+0.07, +0.42] | +1.01 [+0.63, +1.37] | +0.41 [−0.04, +0.84] | +0.73 [+0.45, +1.02] | −0.10 [−0.15, −0.05] | −1.18 [−1.57, −0.77] |

* **(i) Teacher, registered F2 read: "the cap is mostly the plan↔tracker interface"** (compensated PDMS 94.10 ≥ 94.0
  AND DAC 99.01 ≥ 99.0). ⚠️ Both thresholds lie inside their intervals, so the read is **sensitive to log sampling**.
  It is not separated from the human (−0.45 [−1.21, +0.36]), and it beats the human on EP. Still **−1.7 below the
  paper's 95.8** (EP 88.8 vs 91.5, PUBLISHED).
* **(ii) Student, registered F2 read: NOT PROVEN.** ΔPDMS is **+1.07 [+0.68, +1.45]**, lower bound > 0. But **7 family
  components separate adversely** against the uncompensated seam: yaw-rate MAE 3.23 → ~8.4 °/s, heading MAE
  2.71 → ~4.6°, curvature MAE 0.024 → ~0.066 1/m, and speed MAE, speed RMSE, accel MAE and target-speed accuracy.
  ⚠️ Mechanism, stated rather than used to move the goalpost: families6 scores the **seam's poses**, which after
  compensation are the deliberately pre-distorted **reference** handed to the tracker, not the executed motion (the
  tracked path is within 0.14 m median of the original plan). Whether the families should be read on the reference
  or on the executed trajectory is a **definition the programme has not made**. That is a PI/instrument decision,
  not a pass. The tactical κ also drops (lateral 0.849 → 0.677) for the same reason. Comfort costs −0.10.
* The student gains a quarter of what the teacher gains (+1.07 vs +4.30). At the instrument level only **188 of
  643** student DAC violations are tracker-repairable, against **486 of 583** of the teacher's.
  ⇒ **Most student DAC failures are real path failures, not tracker artefacts.** The student's DAC (94.73) equals the
  teacher's for a different reason than the teacher's.

### F1 · DriveRL test-time search (TTS N = 16) as P5's re-query mechanism

This was registered in §9 before any teacher score. The released `tts.py` was run at N = 16 with their seed, on a
`random.Random(20261005)` sample of **120 of the 647** teacher NC·DAC-fail tokens plus **80 pass tokens** (harm control).
CPU, **≈ 120 s/token** (70× the plain teacher). Harness status PASS. MEASURED: `f1/f1_result.json`,
`csv/refe_p4_teacher_tts16_f1.csv`.

| | read |
|---|---|
| **repair rate** (fail set → NC·DAC = 1) | **0.83 % [0.00, 2.80]**: 1 of 120 |
| **harm rate** (pass set → NC·DAC = 0) | 0.0 % (0 of 80) |
| fail set, TTS − teacher PDMS | +0.83 [0.00, +2.78]; better / worse / tied 1 / 0 / 119 |
| pass set, TTS − teacher PDMS | +0.51 [−0.07, +1.72]; 12 / 12 / 56 |
| repairs by D1 class (fail sample) | A (teacher path off-road) **0 / 18** · B 0 / 1 · C (tracker) 1 / 87 · NC-only 0 / 14 |

**Registered outcome: "RE-QUERY IS NOT THE LEVER: P5 uses the DROP arm (or a different repair)."** TTS almost never
changes the teacher's action: 119 of 120 failures are byte-tied. That matches its 2–5 % switch rate on val14
(`raw/tts_sweep_N1_8_16_32.txt`). It does not repair even the 18 class-A failures where the teacher's own path leaves
the road.

## 6 · What this means for the plan (REVIEW_7 §4)

1. **The registered verdict stands as written: TEACHER CAPS** (89.81 / DAC 95.23 on the registered protocol). Per
   PREREG §6 this names P5. **But D1–D3 and F2 change what P5 must be.**
   * **Re-query (P5 as REVIEW_7 framed it) can reach at most class A**: 72 tokens, 12.5 % of the teacher's DAC failures,
     0.59 % of navtest. **And it does not reach even that**: F1 repairs 1 of 120 failures and 0 of 18 class-A ones.
   * **87 % of the teacher's DAC loss is the plan↔tracker interface**: its paths in turns keep 0.37 m to the edge, the
     human keeps 0.60 m, and NAVSIM's tracker misses both by ~0.8 m. **Made trackable, the teacher scores 94.10,
     indistinguishable from the human** (−0.45 [−1.21, +0.36]).
   * ⇒ **P5 should be reframed from "re-query failing targets" to "tracker-aware / margin-aware targets"**, i.e. judge
     and clean targets by NAVSIM's **simulated** footprint, or by a margin requirement, not by the rollout.
     REVIEW_7's "teacher fails DAC on 4.97 %" was measured on NAVSIM's simulation of the target, and
     **D1-navtrain now MEASURES the same mechanism there**: 95.1 % of the held-out target failures are compliant as
     paths (0.59 % raw-path fail rate; 1,018 samples, 11 logs).
2. **The student is not capped by the teacher's paths; it sits ~9.6 points below the trackable teacher** (84.47 vs
   94.10). Its DAC failures are mostly **real path failures** (only 29 % tracker-repairable vs 83 % for the teacher),
   and 84 % of its zero picks (review run, pre-A9) fall where the teacher drives cleanly. ⇒ **REVIEW_7's P1/P2 (NAVSIM-label scorer
   supervision; un-detached scorer context) stay first.** P5 in its re-query form is deprioritised against them.
3. **A zero-training test-time lever exists for the student, +1.07 [+0.68, +1.45]** (F2 ii). It is **NOT PROVEN** at
   the registered rule because the families read the pre-distorted reference.

**Decisions for the PI** (escalated, not taken):
* **(a)** Adopt tracker-aware compensation of the student's executed plan at test time? It puts NAVSIM's public LQR +
  bicycle model inside the agent, as PDM-Closed does, which is a departure from the paper's student. Prerequisite:
  decide whether the four families are read on the **reference** or on the **executed** trajectory.
* **(b)** Reframe P5 from "re-query failing targets" to **margin-/tracker-aware targets**. Dropping the ~8 % of
  targets that fail NAVSIM's simulated DAC would discard paths that are compliant as driven (D1-navtrain); the better
  candidates are (i) a margin term (≥ ~0.6 m, the human's median) when choosing or re-querying the target, or
  (ii) training the WTA on the tracker-compensated reference. Both need a pre-registered held-out test before GPU.
* **(c)** The remaining −1.7 to the paper's teacher row (EP 88.8 vs 91.5) is unexplained. HYPOTHESIS: their navtest
  teacher row uses their own rollout or plan format; not testable without their evaluation code.

**Where this turn stops (RULE ZERO).** P4 is a measurement, and it is complete with every control passing. Of the levers it
exposed, re-query (F1) is refuted, and the cheap ones were run in the same session: D1/D2/D3, D1-navtrain, F2.
**Every remaining lever is blocked on a named item:**
* test-time compensation for the student: blocked on the PI decision (a) plus the families-on-executed-trajectory
  definition. Once decided it costs CPU-hours;
* margin-/tracker-aware targets: blocked on PI decision (b) and on a WTA fine-tune (A40-days);
* P1/P2 scorer supervision: REVIEW_7's own plan, GPU, PI-gated.

## 7 · Artifacts (all under `raw/2026-10-05-teacher-navtest/` unless noted; staged, not committed)

| artifact | what |
|---|---|
| `eval/PREREG_P4_TEACHER_NAVTEST.md` | the pre-registration + dated §9 decisions log (token set, F1, F2) |
| `eval/teacher_navtest_seam.py`, `eval/run_p4.py` | teacher rollouts → seam (C1 same breath), C3 mode, merge; launcher with the bank's environment |
| `teacher_navtest_seam.launched_a9532c19.py` | byte copy of the exact script the 4 registered shards ran |
| `eval/score_p4.py` | harness wrapper (retry on RAM guard; `--tokens all` for the full set, see WinError 206 in `score_logs/failed_winerror206/`) |
| `eval/teacher_navtest_analyze.py` | registered readout (verdict, paired sub-scores, splits, inheritance, families) |
| `eval/p4_dac_anatomy.py`, `eval/p4_dac_margin.py`, `eval/p4_d1_navtrain.py` | D1, D3, D1-navtrain |
| `eval/p4_tracker_comp.py`, `eval/p4_merge_seams.py`, `eval/p4_f2_analyze.py` | F2 |
| `eval/p4_f1_tts.py` | F1 selection + readout |
| `c3_*` | C3 and its 20 Hz mutation arm, with the 47 banked targets used |
| `smoke24*`, `full_s*of4.log`, `full/rows_s*of4.jsonl.xz` | smoke, shard logs, all 12,146 teacher rows (xz; raw jsonl kept unstaged, 32 MB) |
| `refe_p4_teacher_navtest.npz` / `.report.json` | the registered teacher seam + its C1/coverage report |
| `csv/refe_p4_{teacher_navtest,human_navtest,teacher_trackcomp,student_trackcomp,teacher_tts16_f1}.csv` | per-token harness output (all status PASS) |
| `score_logs/*_status.json` | the harness status lines |
| `p4_result.json`, `p4_*_families.json` | registered readout + families |
| `d1_dac_anatomy.json`, `d2_*`, `d3_margin.json`, `d1_navtrain.json` | diagnostics |
| `f2/`, `f2_result.json`, `f2_student_*_families.json` | F2 seams (shards + merged), instrument reports, harness readout |
| `f1/` | F1 tokens, selection, rows, seam, readout |

