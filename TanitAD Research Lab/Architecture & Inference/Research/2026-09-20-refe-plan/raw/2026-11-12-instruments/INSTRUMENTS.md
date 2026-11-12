# W-TRANSP and W-FAM instruments

Built 2026-11-12, CPU only, navtrain held-out only (the TF-1 population, 1,018 targets, 11 logs). **No navtest token was read.**
Evidence class for every reading below: MEASURED (ours). Final readings: `raw/controls.json` (written by the chain in `HANDOFF.md`).
Readings marked PARTIAL come from the first 252 tokens (`raw/controls_partial.json`) and are replaced by the full-1,018 values.

## Files
| file | role |
|---|---|
| `transp_score.py` | W-TRANSP. CLI, sharded, resumable. Per token and per arm: LQR-tracked and tracker-transparent NAVSIM v1.1 sub-scores, plus the family payload. |
| `families_run.py` | W-FAM. Four families on the executed trajectory (primary) and the reference (diagnostic). |
| `validate_controls.py` | the controls below. Exit 0 only if all pass. |
| `test_instruments.py` | unit tests (builder, sharding, resume, plan views) and a banked-controls test. |
| `hold/` | the repo drop (`CHANGED.txt`, `files/`). The MM lands it. |

## What was reused (file:line in the tip `origin/agent/arch-inf-20260803`)
* `transparent_states` and `human_rel`: E2's `e2_score.py:49-84`, `:87-97`, arithmetic unchanged; the pose spacing `dt` is now a parameter (0.2 reproduces E2). Proven bit-for-bit below.
* The scorer call: TF-1's `tf1_score.py` `score()` (= E2 `score_states` plus DDC). Vendored NAVSIM 3e8291b via `refe/navsim_pdm_targets.py::_setup`, one call per arm on `[PDM-Closed, plan LQR-simulated, plan as executed]`.
* Tracked conversion: `refe/navsim_dac.py::to_navsim` (no A7 by default; `--a7 1` applies it).
* Families: `taniteval/adapters/navsim.py` `scenes_to_win` + `four_families_block` (the `families6.py:84-86` pattern, tier T1, cluster unit = log). Tactical labeller `tanitad.refs.refc_tactical` through that adapter.
* NEW, one piece: `transp_score.lead_track` builds the **lead block** the adapter documents as UNAVAILABLE (`navsim.py` "lead_block"). It uses the lead rule of `lead_metrics.py` (corridor |lat| < 2.0 m, gap in [0, 80] m) on nuPlan tracked VEHICLE objects.
* Answer to the brief's open question (PREREG_P5C §8.2 [U]): `families6.py` reads whatever poses it is handed. Our rows hand it the **LQR-simulated** states (primary) and the transparent states separately, so the ambiguity does not arise here.

## W-TRANSP definition
Plan file: npz with `tokens`, `logs` (or `--token-log navtest_inputs.json.gz`), `plans20` [N,20,3] at 0.2 s (P5-C / TF-N1 form) and/or `plans8` [N,8,3] at 0.5 s. Several `--plans name=file` are scored **in one pass**, so the PDM-Closed and metric-cache work (the dominant cost) is paid once for all arms.
* **tracked**: `to_navsim(plan20)` (or `plans8`) goes to NAVSIM's LQR + bicycle, then the scorer.
* **transparent**: E2's cubic spline through the origin and the 20 poses, giving 41 states at 0.1 s with analytic velocity, acceleration and yaw, no tracker. With `plans8` only, the same builder runs at dt 0.5 (about 1.5 cm coarser on an 8 m/s arc, `test_transparent_dt_parameter_is_consistent`).
* Sub-scores per scorer: nc, dac, ddc, ttc, ep, c, mult, pdms. PDMS = NC·DAC·(5 EP + 5 TTC + 2 C)/12. EP is normalised against PDM-Closed in the same call.
* `--split navtest` builds scenarios with `history_rows=1, future_rows=80` as `p4_tracker_comp.py:99` does; `--db-root` points at the nuPlan splits dir.
* Resumable: finished tokens are skipped, errored ones retried, a torn last line is terminated. Shards are by a stable SHA-1 of the token (`--shard i/N`), so resuming never moves a token.
* ⚠ E2's caveat stands: transparent comfort `c` depends on how the states are derived. Never quote it as a driving result.

## W-FAM definitions (per family, never pooled; each carries n and the log-cluster bootstrap CI)
Frame of every pose: ego rear-axle frame at t0, x forward, y left, 8 poses at 0.5 s, against the **logged human future**. dt = 0.5 s is declared, never assumed. Three frames per arm: `executed_tracked` (**primary**, the LQR states at 0.5 s), `executed_transparent` (diagnostic), `reference` (diagnostic, the plan as handed to the tracker).
* **LONGITUDINAL.** Speed MAE / bias / RMSE (m/s), target-speed accuracy within 0.5 / 1 / 2 m/s, along-track MAE and bias (m), accel MAE (m/s²). Distance-keeping, **where a lead exists**: min headway (rear-axle to lead rear face, m), min time-gap (headway / v0, s, NaN below 0.5 m/s), min TTC (s, capped at 30, `n_closing` printed beside it). Where no lead is in the corridor the window is NO_LEAD and is not in that n; the lead state counts are printed.
* **LATERAL.** Heading MAE (deg, path tangent, not vehicle yaw), curvature MAE and bias (1/m), yaw-rate MAE (deg/s), cross-track MAE and bias (m). Steps below 0.5 m/s are excluded from curvature and heading and counted.
* **TACTICAL.** Computable as the adapter defines it: lateral (lane-keep, left, right) and longitudinal (brake-stop, steady, accelerate) decision accuracy and kappa, with the confusion matrix. Both are **trajectory-derived** from the programme's own labeller applied to the plan's path and to the human path. **Not computable** from these tokens: tactical goal-setting and command adherence. NAVSIM carries no manoeuvre label beyond the driving command, and REFe has no tactical goal head. Command adherence needs the NAVSIM `driving_command` and `command_adherence.py` (P6's read); it is not built here. n is printed per block.
* **STRATEGIC.** **PAUSED** (PI, 2026-09-19, item 25). Printed as PAUSED in every block. Not a pass.
* Planned-vs-executed deviation is read by comparing `executed_*` against `reference` in the same arm: on the human arm the reference side is exactly 0 and the executed side is the tracker's deviation.
* Not built: a **paired** C1 − C0 family CI. Run `families_run.py --per-window` per arm and pair on the per-window arrays (`taniteval/per_window.py`, `lead_metrics.paired_distance_keeping`). The MM decides if that is a pre-read blocker.

## Controls (navtrain held-out; all must read the stated value)
| control | statement | reading |
|---|---|---|
| C-ID-TRACKED | arm T tracked equals TF-1 `T_tracked`, 8 sub-scores | max abs diff **0.0**, PARTIAL n=252 tokens |
| C-ID-TRANSPARENT | arm T transparent equals TF-1 `T_transparent` (E2's builder) | **0.0**, PARTIAL n=252 |
| C-ID-TC20S | Tc20s tracked and transparent equal TF-1 | pass, PARTIAL |
| C-ID-HUMAN | human tracked equals TF-1 `H_tracked` | **0.0**, PARTIAL n=252 |
| C-RED-SIM | LQR states fed to the executed slot reproduce the tracked score | **0.0**, PARTIAL |
| C-RED-SHIFT | plan shifted 0.5 m left | DAC changes on 3 (tracked) / 5 (transparent) of 252 tokens; mean PDMS 0.927 → 0.918 tracked, 0.958 → 0.843 transparent |
| C-RED-TW | deliberate-regression label Tw is worse than T | tracked DAC 0.972 → 0.853, PDMS 0.927 → 0.681; transparent DAC 1.000 → 0.837, PDMS 0.958 → 0.670; lateral family worse on all four (heading, cross-track, curvature, yaw rate), both frames |
| C-FAM-REF0 | arm human, reference side | all 7 errors read **exactly 0.0** |
| C-FAM-EXEC | arm human, executed side | reads the tracker's deviation: heading 1.36 deg, cross-track 0.107 m, curvature 0.0120 1/m, yaw rate 1.68 deg/s, speed 0.109 m/s, along 0.219 m, accel 0.107 m/s². Equals P5's banked `exec_H` / `exec_T` to 2.7e-6 (float32 storage). |
| C-RERUN | a rerun is bit-identical (40 tokens, rows without `t_s`) | see `raw/controls.json` |

Two controls of PREREG_P5C §10.5 W-TRANSP (i) **cannot run without navtest** and are **not done**: tracked equals the pinned harness per token on the 24-token u400 control and on all 12,146 base tokens. The tracked arm here is the same scorer code TF-1 and E2 used and matches them exactly, but its equality with the **pinned W3 harness** on navtest scenarios is unproven. `--split navtest` builds scenarios as `p4_tracker_comp.py` does, which the harness-equal F2 work used. MM: run this control as the first step of the P5-C read (re-scoring tokens whose plans are already banked; it adds no new arm read).

## Throughput (MEASURED on the first 279 rows, CPU contended by E3 and others, 2 processes)
* Per token, 3 plan arms + human arm + controls = 7 scorer calls: `t_s` mean **2.79 s** (median 2.58, p90 4.25), excluding scenario build (about 1 s/token amortised, 6 s for the first token of a log).
* The cost is the **arm-independent** `metric_cache_parts` (PDM-Closed + GT observation, 1.1 to 2.3 s); each extra arm costs about 0.05 to 0.15 s (profile: score 0.04 to 0.06 s, families payload 0.05 to 0.08 s).
* Wall: about 40 rows/min for two processes at best, about 12 rows/min averaged over the run (contention, per-log set-up).
* **Projection** (ESTIMATED from the above): N1 or P5-C, any number of arms in one pass, no controls: about 2.4 s + 1 s per token = **3.4 s/token, about 11.4 CPU-h for 12,110 tokens** (12,146: 11.5), **about 5.7 h wall on 2 processes**. A second pass for more arms costs about 0.1 s/token/arm only if it is a separate invocation **not** sharing the cache, so put all arms in one `--plans` list. This is **far above** PREREG_P5C's 0.9 process-h per arm (0.27 s/token), which counted only the derivation, not PDM-Closed.
* Optional speed-up, **not built and not validated (navtest cache would have to be read)**: load NAVSIM's existing navtest metric cache (`F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest`) instead of recomputing PDM-Closed. It would cut roughly 2 s of 3.4 s per token. The DB is still needed for the human future and the lead track.

## Usage
```
set PYTHONPATH=F:\Archive\devbox-C\navsim\navsim-3e8291bfa89ff247231e0227778840cd0a036896;<refe dir>
set CUDA_VISIBLE_DEVICES=-1
C:/Users/Admin/navsim-crun/venv/Scripts/python.exe transp_score.py --refe <refe dir> --split navtest ^
   --token-log F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz ^
   --plans N1C=<c.npz> N1W=<w.npz> N1T=<t.npz> --out rows_s0.jsonl --shard 0/2 --human-arm
C:/Users/Admin/navsim-crun/venv/Scripts/python.exe families_run.py --rows "rows_s*.jsonl" --arm N1C --label N1C --out fam_N1C.json
```
`families_run.py` needs `<repo>/taniteval` and `<repo>/stack` (default `DEFAULT_REPO`, a scratch export of the tip: change it on landing). Use `;` in PYTHONPATH on Windows.
* **N1** (PREREG_TF_N1 §6, §9): arms N1-C, N1-W, N1-T as `plans20` from the TF-N1 build script (plans must be built there; this repo does not build them). C-ID for N1-T is the pinned-harness control above. Families read `executed_tracked` as primary (the draft's open PI decision (a) is still the PI's).
* **P5-C** (PREREG_P5C §7, §8.2): arms C1, C0, base, plans **after A7** (the pick the tracked path converts), so leave `--a7 0`. Print both scorers for every number; the verdict uses `tracked` and the qualifier uses `transparent`. G2 reads `executed_tracked` only.
* Add `--human-arm` for the standard human row. `--controls` is for validation only (3 extra scorer calls per arm).
* The paired bootstrap of PDMS and sub-scores over logs is not in these scripts; use the P4/F2 harness estimator (`proxy_eval.boot`) on the rows.
