<title>TF-N2 evidence</title>

# TF-N2 evidence: where the converted teacher loses ego progress, and the levers ranked (navtrain held-out only)

2026-11-12, TF design agent. CPU only, 0 GPU. **No navtest token was scored or opened** (no `C:/Users/Admin/n1_run/fam/*.json`, no N1 per-token row).

- **Evidence class:** MEASURED, EXPLORATORY (the warp factor and the CERT order were chosen on these rows), navtrain held-out.
- **Tier:** NAVSIM v1.1 PDMS, non-reactive pseudo-simulation, LQR-tracked (T1), vendored navsim @ 3e8291b.
- **Population:** 469 tokens in 13 logs: 361 of TF-2's 1,018 + 108 of TF-2b's 687. Drawn as `random.Random(20261112).sample(1,705, 600)`; the 131 drawn tokens in the 11 TF-2b logs whose nuPlan DB is no longer local are absent (362 of the 1,705 are in those logs).
- **Estimator:** paired log-cluster bootstrap over the 13 logs, 10,000 draws, seed 20260927, 95 % percentile, ratio mean (TF-2's). Answers *another draw of LOGS* only; labels, tracker and scorer are deterministic.
- **Scorer and its fidelity.** W-TRANSP's own functions (`eval/instruments/transp_score.py` at tip `be6567aa`, imported unchanged) on the **nuPlan-DB rebuild** of the metric-cache parts. navtrain has no harness cache, so the A1 repair (`../n1/a1/transp_score_a1fix.py --metric-cache-root`) cannot apply here. On navtest this route differs from the pinned harness in EP on 27.5 % of tokens (median |dEP| 9.9e-5, p99 2e-2) and flips NC/TTC on 7 of 12,146 (`../n1/a1/WTRANSP_DIAG.md`). Every arm of a token is scored on the **same** scenery, so the paired contrasts are internally consistent; **levels are not harness-identical**.
- **Controls.** C-DERIV: Tc20s re-derived from `t20` equals the banked `p5s_cubic` Tc20s within **1.3e-10 m** (bar 1e-9) on the 361 TF-2 rows. Deliberate regression Wreg (a 30 % slowed plan): ΔEP **−23.2 [−25.6, −21.2]**, ΔPDMS −8.73. Arm T reproduces TF-1/TF-2's direction (−4.98 vs C).
- **Code:** `tfn2_probe.py`, `tfn2_analyze.py`, `tfn2_ep_rows.py` (this folder). Raw: `raw/probe_rows.jsonl.xz` (sha256 `6fb235cf…`), `raw/extra_rows_RW105.jsonl.xz` (`3650abb0…`), `raw/tfn2_probe_read.json` (`3f80daf2…`), `raw/fam/fam_*.json`.

## 1. The EP mechanism (baseline C = teacher → Tc20s, LQR-tracked; 469 tokens)

**EP = 90.59; the loss to 100 is 9.41 points.**
- **1.28 points come from rows the plan fails** (NC·DAC = 0 ⇒ EP = 0; here all are DAC). Fixing DAC returns EP as well: this is why the N1 mean-field swap's EP and DAC terms overlap.
- **8.13 points come from rows that pass**, where the tracked plan covers less centerline than PDM-Closed in 4 s: 59.9 % of tokens have 0 < EP < 1.
- 4.9 % of tokens get EP = 1 by the 5 m normaliser rule; on 3.4 % PDM-Closed itself fails, so the plan's EP is 1.

**The conversion does not slow the plan.** Raw centerline progress, paired:
- C tracked − T transparent (the teacher's rollout as executed) = **+0.015 m [+0.008, +0.022]**, median ratio 1.001;
- T tracked − T transparent = −0.068 m [−0.148, +0.006].
- The 1,705 banked TF-2/TF-2b rows agree: EP on passing rows is 90.18 (Tc20s tracked) vs 90.14 (T transparent).

**The horizon and the rate map correctly.** Neither is the cause:
- `rollout_from_frame` appends the state after each 0.1 s step (`refe/build_teacher_rollouts.py:281-286`);
- `poses[stride-1::stride][:20]` is therefore t = 0.2 … 4.0 s (`eval/teacher_navtest_seam.py:121`);
- `to_navsim` keeps the exact 0.4/0.8/… poses and interpolates 0.5 s between 0.2 s samples (`refe/navsim_dac.py:44-56`). C-SEAM already pinned it against the rollout's own 0.5 s states.
- NAVSIM's EP is centerline progress of the final 4.0 s pose (`pdm_scorer._calculate_progress`). Plan and normaliser both cover 4 s.

**It is the teacher's own speed profile against PDM-Closed's.**
- PDM-Closed is an IDM toward the speed limit, accelerating at up to 1.5 m/s² with no jerk limit. The teacher's action space limits positive jerk to 1.0 m/s³ while a ≥ 0, and caps a ≤ 1.5 m/s² (`release/configs/driverl_teacher.yaml:23,27`, loaded into the planner at `driverl/nuplan/planner.py:140-147`).
- Median per token: the teacher's 4 s path is **0.942** of PDM-Closed's, and its speed at 4 s is 0.145 m/s lower.
- Against the human, the teacher already drives further: progress ratio 1.10 vs the human's 1.02.

| where the 9.41 EP points are lost (C) | n | EP (C) | EP human | loss, points of total | teacher / PDM path at 4 s (m) |
|---|---|---|---|---|---|
| v0 [0, 0.5) standstill | 36 | 86.5 | 88.3 | 1.04 | 7.1 / 7.2 |
| v0 [0.5, 3) | 117 | 90.6 | 86.7 | 2.34 | 9.0 / 9.0 |
| **v0 [3, 6)** | 146 | 89.5 | 89.2 | **3.26** | 16.5 / 17.2 |
| v0 [6, 9) | 101 | 90.9 | 84.6 | 1.96 | 30.6 / 32.0 |
| v0 [9, 12) | 59 | 94.0 | 88.3 | 0.76 | 40.4 / 42.4 |
| v0 ≥ 12 | 10 | 97.6 | 98.1 | 0.05 | 45.9 / 47.1 |
| lead at t0 (lead gap 10–30 m) | 252 (152) | 90.2 (89.3) | 88.0 (85.5) | 5.26 (3.46) | 16.7 / 17.6 |
| no lead | 217 | 91.0 | 87.1 | 4.15 | 25.1 / 25.8 |
| **red light on the route at t0** | 81 | **86.9** | 86.9 | 2.26 | 12.7 / 14.3 |

**Standstill, from the 1,705 banked rows** (`tfn2_ep_rows.py`; TF-2/TF-2b sub-scores, no rescoring):
- n = 155: EP on passing rows is **79.5** for the teacher against **86.8** for the human. That is 1.84 of 10.45 loss points.
- From rest the teacher covers a median 6.3 m in 4 s. Its jerk-limited ceiling is about 7.5 m (ESTIMATED); PDM-Closed covers about 10 m.
- Only 36 standstill tokens reached the probe: the standstill-heavy TF-2b logs are among those without a local DB.

⇒ **The EP loss is diffuse.** It is a systematic ~6 % shortfall of the teacher's 4 s progress against PDM-Closed, concentrated where the car is accelerating: below 6 m/s, behind a lead 10–30 m ahead, at red lights and from standstill. It is upstream of every conversion. The tracker, Tc20s, the 8-pose sampling and the 4 s horizon are each eliminated (MEASURED above).

## 2. Levers ranked by measured effect (paired against C, PDMS ×100)

| rank | lever | ΔPDMS [95 % CI] | ΔEP | ΔDAC | ΔNC | ΔTTC | ΔC | PDMS | admissible? |
|---|---|---|---|---|---|---|---|---|---|
| **1** | **N2-S: CERT over [RW105 → R → C → T], first with tracked DAC = 1** | **+1.59 [+0.64, +2.66]** | +2.78 [1.71, 4.00] | +0.85 [0, 1.77] | −0.21 | 0.00 [−0.57, 0.60] | −0.21 | **95.46** | yes, with the W and CERT caveats below |
| 2 | RW105 = P5-R label of the 5 % warped teacher path | +1.18 [+0.59, +1.94] | +2.40 [1.68, 3.16] | +0.43 | −0.21 | 0.00 | −0.21 | 95.05 | yes (W caveat) |
| 3 | CERT over [R → C → T] (no warp) | +0.96 [+0.30, +1.72] | +0.67 | +0.85 | 0.00 | +0.43 | 0.00 | 94.83 | yes (CERT caveat) |
| 4 | CERT over [W105 → C] | +0.83 [+0.16, +1.58] | +2.29 | +0.21 | −0.21 | −0.43 | −0.21 | 94.71 | yes |
| 5 | W105: 5 % progressive time-warp, then Tc20s | +0.62 [+0.15, +1.05] | +2.08 [1.42, 2.64] | 0.00 | −0.21 | −0.43 | −0.21 | 94.49 | yes (W caveat) |
| 6 | CERT over [C → T] (N1's form) | +0.61 [0.00, +1.48] | +0.56 | +0.85 | −0.21 | 0.00 | 0.00 | 94.48 | yes (CERT caveat) |
| 7 | R: P5-R (λ* = 3.0) instead of Tc20s | +0.55 [+0.09, +1.22] | +0.29 | +0.43 | 0.00 | +0.43 | 0.00 | 94.42 | **yes, clean** |
| ✗ | W110: 10 % warp | −0.40 [−1.19, +0.26] | +2.87 | −0.85 | −0.21 | −1.28 | **−3.20** | 93.47 | yes, refuted: comfort and DAC break |
| ref | C (Tc20s, N1-C form) | — | — | — | — | — | — | 93.88 [92.34, 95.39] | — |
| ref | human (DB future, tracked) | +0.95 [−0.53, +2.59] | −3.01 | +1.28 | 0.00 | +3.41 | +0.21 | 94.82 | — |
| ⛔ | argmax of harness PDMS over {T, C, R, W105, W110} | ceiling 96.67 | | | | | | | **NO: an oracle on the benchmark** |

**How the levers are built:**
- **W105:** pose(t) := teacher(τ(t)) with τ(t) = t·(1 + 0.05·t/4). This keeps the t0 speed (τ′(0) = 1) and ends at the teacher's own 4.2 s position. Beyond 4 s it uses the constant velocity and yaw rate of the last 0.2 s (≤ 0.2 s of extrapolation).
- **RW105:** P5-R (`p5r_labels.solve`, λ = 3.0, warm start = the W105 Tc20s).
- **CERT:** reads the tracked DAC of each candidate (the scorer's DAC, map-only), as N1's CERT does.

**Admissibility (the brief's rule):**
- **R** reads the t0 ego and NAVSIM's public LQR/bicycle model: clean.
- **W105** reads only the teacher's own path, never the log future or any score. ⚠ **It is a deliberate label speed bias, not the teacher's policy.** It must be quoted as "teacher + 5 % time-warp", and its family cost is below.
- **CERT** selects by part of the test metric (map-only DAC), so its DAC is partly guaranteed by construction (N1 ruling).
- ⛔ Any selection or shaping by harness EP/PDMS, or a target built from PDM-Closed's own trajectory (the EP normaliser itself), is an oracle and is not proposed.

## 3. Four metric families (W-FAM, `families_run.py` unchanged; executed_tracked vs the logged human; 469 windows, 13 logs, B = 2,000)

| family | metric | C | R | W105 | RW105 | **N2-S** | human (tracked) |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | speed MAE (m/s) / bias | 0.699 / +0.36 | 0.696 / +0.35 | 0.875 / +0.63 | 0.871 / +0.63 | **0.869 / +0.63** | 0.123 / +0.04 |
| | within 1 m/s / accel MAE (m/s²) | 0.770 / 0.456 | 0.769 / 0.444 | 0.703 / 0.551 | 0.703 / 0.535 | 0.703 / 0.534 | 1.000 / 0.133 |
| | progress ratio vs human | 1.101 | 1.098 | 1.159 | 1.157 | 1.156 | 1.018 |
| | min headway (m) / min time gap (s) / min TTC (s, n closing), 244 lead windows | 16.92 / 5.86 / 14.26 (181) | 16.89 / 5.86 / 14.29 (181) | 16.55 / 5.78 / 13.20 (183) | 16.54 / 5.79 / 13.35 (182) | 16.54 / 5.79 / 13.35 (182) | 17.15 / 5.91 / 15.89 (166) |
| LATERAL | heading (deg) / yaw rate (deg/s) | 2.61 / 2.95 | **2.52 / 2.75** | 2.73 / 3.10 | 2.62 / 2.88 | 2.62 / 2.88 | 1.48 / 1.84 |
| | curvature (1/m) / cross-track (m) | 0.0158 / 0.301 | 0.0149 / 0.299 | 0.0156 / 0.341 | 0.0148 / 0.337 | 0.0148 / 0.334 | 0.0107 / 0.136 |
| TACTICAL | lateral decision acc (κ) | 0.917 (0.852) | 0.934 (0.883) | 0.910 (0.843) | 0.923 (0.865) | 0.923 (0.865) | 0.981 (0.965) |
| | longitudinal decision acc (κ) | 0.670 (0.509) | 0.670 (0.510) | 0.657 (0.495) | 0.653 (0.489) | 0.653 (0.489) | 0.938 (0.906) |
| | 5-way acc (κ) / goal-point error (m) | 0.774 (0.712) / 2.69 | 0.793 (0.737) / 2.68 | 0.785 (0.727) / 3.38 | 0.797 (0.743) / 3.37 | 0.797 (0.743) / 3.36 | 0.949 (0.934) / 0.48 |
| STRATEGIC | — | PAUSED (PI 2026-09-19; also unavailable in NAVSIM) | PAUSED | PAUSED | PAUSED | PAUSED | PAUSED |

These are point values; per-family CIs are in `raw/fam/`. Across arms they are descriptive: no paired family CI exists (the work item N1 named).

**Reading.**
- **P5-R improves everything it touches.** Lateral and tactical-lateral improve, and nothing gets worse.
- **The warp buys EP by driving further from the human.**
  - Speed MAE +0.17 m/s; speed bias doubles (+0.36 → +0.63 m/s).
  - Min TTC to the lead −0.9 s.
  - Goal-point error +0.68 m.
- That is the price of chasing PDM-Closed's speed-limit progress.
- ⚠ **The student learns what the labels do.** A warped label teaches a faster, less human student. **This is a PI-visible trade-off, not a free gain.**

## 4. Levers this agent could not measure, and what unblocks each

| lever | why it is the honest alternative to W105 | blocker | unblock |
|---|---|---|---|
| **re-roll the teacher with its protocol knobs:** positive-jerk limit 1.0 → 5.0 (= `max_jerk_long`); `route_goal_min_speed_mps` 5 → 8 with horizon 12 s; policy at 5 Hz (the engine's `target_sample_rate_hz: 5.0`) | moves EP inside the teacher's own policy, so no label bias | re-roll ≈ 9.4 s/token (TF-2b `sec`): about 1.2 CPU-h per arm on 470 tokens, longer than an agent's 4-minute window | a detached CPU run (≈ 2 h wall, 3 arms, 2 processes), then score with `tfn2_probe.py`; no PI decision needed |
| release TTS, N = 8 | the paper's teacher may report TTS | 60–120 s/token: 8 to 16 CPU-h on 470 tokens | detached CPU run; F1's pass-set +0.51 is descriptive only |
| **paired family CI** | needed to put a bar on the warp's longitudinal cost | instrument not built (N1 §10) | extend `families_run.py --per-window` into a paired bootstrap |

## 5. Limits
- **Coverage.** 13 logs. The standstill-heavy logs are missing (no local DB).
- **Selection.** The warp factor (0.05 over 0.10) and the CERT order were chosen on these rows, so every navtrain number here is exploratory, and N2's navtest read is the confirmation.
- **Scenery.** DB-rebuilt, not the harness's.
- **CERT on fresh logs.** TF-2b measured CERT − Tc20s = 0.000 on fresh logs: its navtrain gain is log-dependent.

**ESTIMATED navtest projection:**
- N1-C 94.10 plus about +1.0 to +1.6 gives **≈ 95.1 to 95.7**, with DAC ≈ 99.7 to 99.8 (N1's CERT reached 99.73).
- ⇒ the most likely outcome is **NEAR**. PAPER LEVEL needs DAC 99.9, which no measured lever has reached on navtest.
