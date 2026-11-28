# HANDOFF — a0k0_headroom (EvalFlyWheel analyst, 2026-10-19 ~00:10 Berlin)

State: instrument + prereg + pytest WRITTEN; compute GATED on dev-box free commit (rule 7: needs >= 6 GB; read 5.4-5.7 GB at 00:08,
a foreign ~5 GB-commit python job holds it). Nothing computed on real data yet.

## Files (all NEW, nothing existing edited)
* `raw/a0k0_headroom/PREREG.md` — original prereg + **Amendment 1** (PI ruling 2026-10-18 ~23:58: a0 + yaw rate admissible, curvature not;
  ha0_imu) + **Amendment 2** (premise correction: refcv8 argv carries `--ego-history` and `--residual-prior ha0_ext_pose`; PRIMARY arm =
  residual scaling RS on ha0_ext_pose; L-BFGS-B fitter). Both amendments written BEFORE any real-data number.
* `code/a0k0_headroom_r8.py` — `compute` / `render`. Subset args write `A0K0_HEADROOM.smoke.json` only; the full default run writes
  `A0K0_HEADROOM.json` + `.md` (refuses → `CONTROL_FAILED.json` if any control fails). Partial per-tree JSON: `A0K0_HEADROOM.partial.json`.
* `code/test_a0k0_headroom_r8.py` — analytic fit targets, identities, leak detection + deliberate-regression (leaky cross-fit must be
  caught), imu integrator analytic + library a0=0 limit, deliverable controls (incl. exact 2026-09-26 reproduction +0.0055 [+0.0036, +0.0076]).
* `raw/a0k0_headroom/logs/` — run logs.

## To resume
```
SP=C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/f0c4bd12-5464-4952-995c-062072b3b8d0/scratchpad
bash $SP/gate_run.sh <PKG>/raw/a0k0_headroom/logs/smoke.log compute --trees step30000ema_unc --seeds 0     # smoke (~3 min)
bash $SP/gate_run.sh <PKG>/raw/a0k0_headroom/logs/full.log compute                                         # full (~20 min est.)
cd <PKG>/code && CUDA_VISIBLE_DEVICES=-1 PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;<PKG>/code' \
  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_a0k0_headroom_r8.py
```
(gate_run.sh polls Win32 FreeVirtualMemory every 30 s up to 2 h, then runs the tool with CUDA off, OMP 4, the battery PYTHONPATH.)
If the scratchpad script is gone: wait for free commit >= 6 GB and run the python line from the tool docstring.

## Update 00:13
* SMOKE PASSED (EMA-30k, seed 0, S2+S6, compliant+all; `A0K0_HEADROOM.smoke.json`, `logs/smoke.log`): controls a 15/15 exact, b, V1 = V2 = 0.0 m,
  2026-09-26 reproduction EXACT (+0.0055 [+0.0036, +0.0076], 4,754 / 139). Smoke numbers are seed 0 only -- not the deliverable.
* FULL run queued behind the memory gate (`logs/full.log`, background). If it never starts, rerun the full line above.

## Update 00:25 (Master Mind 00:15 request)
* PREREG Amendment 3 written (frozen gains + forward test on EMA 40,000 / 48,852; bars FZ-E1, FZ-E1P, FZ-NR, FZ-E3) BEFORE any 40k/48,852 data.
* Tool gained `freeze` (-> `FROZEN_GAINS_ema30k.json`) and `forward --tree <tree>` (-> `FORWARD_<tree>.json`).
* Pipeline running in background (memory-gated): `bash $SP/pipeline.sh` = compute -> freeze -> forward step30000_unc -> forward
  step30000ema_unc -> render -> pytest (`logs/full.log`, `logs/pytest_a0k0.log`). If interrupted, rerun the pipeline.
* When 40k / 48,852 EMA trees exist: `python a0k0_headroom_r8.py forward --tree step40000ema_unc` (name as the battery banks it), then `render`.

## FINAL 00:40 -- DONE
* Pipeline completed: A0K0_HEADROOM.json/.md (cells by script sha a70fb24c = the surviving run; a stale duplicate run, pids 43772/62276,
  was killed at ~00:17:30 -- logs/full.log lines before the NOTE are interleaved), FROZEN_GAINS_ema30k.json, FORWARD_step30000_unc.json,
  FORWARD_step30000ema_unc.json. Summary blocks recomputed by `resummarise` (sha 2f63b47d; direction-aware rows -- the first summary
  listed two ACCURACY rows' improvements as FAIL); no cell changed. pytest 21/21 (logs/pytest_a0k0.log).
* Pending only: `forward --tree <EMA 40k / 48,852 tree>` when those trees are banked (Amendment 3 bars), then `render`.
