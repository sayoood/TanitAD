# HANDOFF — run the rung-β gate when two trained M2 checkpoints exist

State: the gate is **built and unit-tested on CPU; it has not been run on a real checkpoint** (no trained M2 exists; `H-MINI-K-1` is untrained). Nothing here trains anything or needs a GPU. Spec and both outcomes: `Project Steering/PREREG_V8F_BETA.md`.

## What it needs (per M2 arm: M2a and M2b are read separately)
* **Two M2 checkpoints, training seeds 0 and 1**, from `v8f_mini_train.py --arm M2a|M2b --kin-mode yaw` (recipe: `…/2026-10-17-v8f-mini/HANDOFF.md` §2). `wm='off'` (M1) and `kin_mode=steer` checkpoints are **refused** by the run.
* The same eval inputs as the K-1 read: the S4 eval cache, `raw/eval141_kin.npz`, `raw/eval_grid_windows.json`, `raw/join_eval141.npz`, the v9 eval labels, and `native_tracks.pt` (`stack/scripts/build_native_tracks.py`; required for a yaw checkpoint). Use the same `--ages` as the K-1 read (default `cycle`).
* `PYTHONPATH` = `stack;taniteval;stack/scripts`. Any CPU is enough; the scorer rolls 118 candidates through the WM per window (2,399 windows; full-scale runtime NOT measured). Run it on the dev box or box 2 after that box's GPU job, never on a training pod.

## Commands (per arm; `<ARM>` = M2a or M2b; paths use `/`)
```
python stack/scripts/v8f_mini_eval.py run --ckpt OUT/<ARM>_s0/ckpt_last.pt --tag <ARM>_s0 --plan beta --families \
  --eval-cache CACHE --tracks TRACKS/native_tracks.pt --out-dir OUT/eval_beta --n-boot 2000 [--fan-artifact FAN/anchors_117_alat_declared.pt]
python stack/scripts/v8f_mini_eval.py run --ckpt OUT/<ARM>_s1/ckpt_last.pt --tag <ARM>_s1 --plan beta --families \
  --eval-cache CACHE --tracks TRACKS/native_tracks.pt --out-dir OUT/eval_beta --n-boot 2000 [--fan-artifact FAN/anchors_117_alat_declared.pt]
python stack/scripts/v8f_mini_eval.py compare-beta \
  --runs <ARM>_s0=OUT/eval_beta/eval_<ARM>_s0.json <ARM>_s1=OUT/eval_beta/eval_<ARM>_s1.json --out OUT/eval_beta/verdict_<ARM>.json
```
`run --plan beta` also writes the ordinary M2 evaluation, so one command per checkpoint gives both M2 (the comparator) and β on the same windows. `--plan none` (the default) runs the scorer zero times.

## What to read, in this order
1. `eval_<tag>.json` → `plan_beta.score_calls.candidates` **> 0** (else the scorer never ran: the read is void) and `plan_beta.fan.max_abs_diff_vs_pinned_grid` (**0.0**; a non-zero distance means the shipped artifact is not the pinned grid; report it, it is not a PASS).
2. `plan_beta.diagnostics_T0.signal.recovery` — the sharpest single number (≥ 0.5 means z² ranks the fan with a perfect goal) — and `plan_beta.wm_diag` (`replaced_frac`, `fan_endpoint_spread_ratio`).
3. `verdict_<ARM>.json` → `verdict` and `next` (the named next lever), with `P1`, `P2`, `P3`, `seed_floor_F_m`, `families_separated_worse`.
4. `plan_beta.paired_vs_m2.families` — the 13 components, β − M2, paired episode-cluster bootstrap. Quote every number with its estimator (CLAUDE.md); never ADE alone.
5. `eval_<tag>_beta.npz` — per-window plans (`beta__pred`, `m2__pred`, `beta__replaced`, `beta__fan_idx`, `<arm>__fw_<component>`).

## Caveats that travel with any result
* **The fan.** Without `--fan-artifact` the pinned grid is used and the file `anchors_117_alat_declared.pt` (on a pod) is **not** checked against it. Pull it and pass it for the real read.
* **The 60-clip holdout is IN-SAMPLE for M2** (the trainer has no holdout option). β selects nothing, so this gate is unaffected; a β′ selected on the holdout would be optimistic. Decision for the PI: retrain M2 with the 60 clips withheld, or accept and label.
* **The goal is derived, not trained** (S's route × P10 pace through the WM). A FAIL-GOAL is the trigger for the plan's trained ĝ² head, which needs its own PREREG and launch gate; it is **not** built here.
* The chord-vs-measured speed convention in `wm_action_sequence` lags the cache's by about one and a half frames; it is the same for every candidate, the goal and M2's plan, so it cannot favour one of them.

## CPU tests (the instrument)
```
$env:CUDA_VISIBLE_DEVICES="-1"; $env:PYTHONPATH="<repo>\stack;<repo>\taniteval;<repo>\stack\scripts"
python -m pytest -q stack/tests/test_v8f_beta.py stack/tests/test_v8f_mini.py
```
