# Absolute tolerance of the RL-S1 fan-equivalence gate: MEASURED, and NOT CHOSEN (2026-10-12, dev-box RTX 4060)

**Verdict (`verdict.json`): `CHOSEN_ATOL = null`.** The pre-registered rule (docstring of `run_calibration.py`, written before any arm was dumped) accepts a tolerance only if every
deliberate-regression arm has max delta >= 10x the tolerance AND >= max(5, 10 %) of the 48 sets beyond it. Five of the six decision arms miss the 10x bar at the rule's
tolerance (0.5 = 2 x the 0.25 floor), and one of them cannot be seen at any tolerance. So no number is written into the chain: `rls1_pod.sh` stage `fancmp` takes it from
`RLS1_FANCMP_ATOL` and refuses when it is unset (runbook D-RLS1-5).

Instrument: `run_calibration.py --phase prep|dump|analyse` (every arm is dumped by the CLI dumper itself, `--compare-sets <A40 banked step-10,075 sets> --compare-atol 0.5
--compare-min-matched 40`, behind `wait_idle.py`). Sample: 48 keys, every 28th of the 1,365 keys that have both a dev-box image set and an A40 banked set (`keys48.tsv`,
sha256 in `prep_manifest.json`). Delta of a set = max |new - ref| over every x, y (m) and yaw (rad) value. Evidence class: MEASURED (ours); the A40 reference is the banked
`onpolicy_sets_logs_full` extract (`sets_10075_proxy.jsonl`, dev box only).

## The floor (null arms: a correct dumper)

| arm | sets | max delta | p50 | p90 |
|---|---|---|---|---|
| N_full: identity run, batch 4, all 1,365 keys, cross-GPU (CPU re-analysis) | 1365 | 0.25 | 0.125 | 0.25 |
| N_b1 / N_b2 / N_b4 / N_b4shift / N_b4w2: identity run's 24-key invariance dumps | 23-24 | 0.25 | 0.125 | 0.25 |
| N0_b4 / N1_b1 / N2_b2 (this run, CLI) | 48 | 0.25 | 0.06-0.08 | 0.25 |

0.25 is one bf16 ulp for |value| in [32, 64) and is the maximum in all nine null dumps. The largest |x| in the 1,365 banked sets is 59.25, and the 512-scene slice of the
chain (`splits/rl_fancmp_keys.tsv`) is drawn from them; a slice with |x| >= 64 would see 0.5 ulp flips. Not measured: the pod's own floor (A40 vs the A40's banked sets, same GPU
model, torch / cuDNN possibly different), which may be far below 0.25. UNVERIFIED.

## Regression arms at tol = 0.5 (the rule's value)

| arm | class | CLI exit | sets beyond 0.5 of 48 | max delta | x tol | would separate (10x) only at tol <= |
|---|---|---|---|---|---|---|
| A1 another vehicle's rig (real rig, wrong vehicle) | DECISION (a) | 0 | **0** | 0.25 | 0.5 | 0.025 (below the floor: impossible) |
| A2 intrinsics fx fy cx cy x 0.5 | DECISION (a) | 5 | 40 | 4.125 | 8.25 | 0.41 |
| B1 all four cameras mirrored | DECISION (b) | 5 | 45 | 7.766 | **15.5** | 0.78 |
| B2 BGR / RGB swapped | DECISION (b) | 5 | 10 | 1.875 | 3.75 | 0.19 |
| C1 `traj_head.2` + 10 % noise | DECISION (c) | 5 | 29 | 1.625 | 3.25 | 0.16 |
| C2 `snap_epoch021.pt` (step 7,348, another banked checkpoint) | DECISION (c) | 4 (no reference at that step) | 47 (offline, step relabelled) | 4.438 | 8.9 | 0.44 |
| S_focal105 fx fy x 1.05 | sensitivity | 0 | 0 | 0.25 | 0.5 | |
| S_yaw3 rig rotated 3 deg | sensitivity | 0 | 0 | 0.375 | 0.75 | |
| S_head_noise1 +1 % | sensitivity | 0 | 0 | 0.25 | 0.5 | |
| S_snap027 (step 9,763, 312 steps early) | sensitivity | 4 | 0 | 0.328 | 0.66 | |
| S_cam_reversed camera order reversed | sensitivity | 5 | 48 | 7.688 | 15.4 | 0.77 |
| N3_jpeg every image re-encoded at q95, nothing else (informational null) | not in the floor | 0 | 0 | 0.375 | 0.75 | |
| S_wrong_scene | NOT RUN | | | | | the bank resolves frames by (log, member), so another scene's frames never resolve (exit 4, 3 s) |

`RESULTS_TABLE.md` / `arms_summary.json` hold every number (quantiles, beyond-counts at 0.25 ... 5, x/y RMS). The CLI's own max delta equals this analysis' on all 13 arms where
the checkpoint step matches (`cli_vs_analysis_max_delta_equal`).

## What the result means

* **The fan is almost blind to the camera calibration.** Another real vehicle's rig moves the fan by exactly the floor (0.25, median set 0.125): no absolute tolerance can see it, and
  for RL-S1's labels it is arguably harmless (the labels move by no more than the numerical noise does). A 5 % focal error and a 3 deg rotation are also at or near the floor.
  A units bug (half-scale intrinsics) moves it by up to 4.1.
* **The per-set max is the wrong statistic for the weak arms.** It is dominated by one ulp-quantised outlier per set, the same 0.25 in every null.
* A tolerance of 0.5 therefore catches gross defects (mirrored images, reversed cameras, halved intrinsics, an earlier checkpoint, a corrupted head, swapped channels) with 10-48
  of 48 sets beyond it, but only mirrored / reversed images clear the 10x bar. It is a catastrophe screen, not an equivalence proof.

## Post-hoc lead (NOT pre-registered; `posthoc_mean_abs.py`, `posthoc_mean_abs.json`)

The per-set MEAN |delta| over the x/y values has a much tighter floor: 0.0122 at worst over the 1,365 cross-GPU sets (p50 0.0042). Sets with mean |delta| above 0.025:
nulls 0 of 1,365 and 0 of 48 each (JPEG-null 0), A1 0, A2 47, B1 47, B2 31, C1 48, C2 48, S_cam_reversed 48, S_yaw3 22, S_head_noise1 9, S_snap027 1, S_focal105 0 (of 48).
The worst set's mean |delta| is 14x to 135x the null floor for A2, B1, B2, C1, C2 and 1.7x for A1. This was computed AFTER the verdict, so it is a lead and not a result: it
needs its own pre-registered rule and a replication on disjoint keys (`HANDOFF.md`). Another vehicle's rig stays invisible under it.

## Files

`run_calibration.py` (prep / dump / analyse; the pre-registration is its docstring) - `verdict.json`, `arms_summary.json`, `RESULTS_TABLE.md` - `arms/<arm>/{cli.json,cli.log,
rl_fan_dump_cal_s0of1.manifest.json}` (the CLI runs) - `prep_manifest.json`, `keys48.tsv`, `dump_phase.log` - `posthoc_mean_abs.{py,json}` - `HANDOFF.md`.
The dumped sets themselves (6 MB per arm) and the perturbed copies (calibration tables, re-encoded pixel zips, two 1.3 GB checkpoints) stay on the dev box in
`D:/Projects/TanitAD-artifacts/_rlfan_work/atol_cal/` and are reproducible from `--phase prep`.
