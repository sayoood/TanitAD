# P6 on the L40: the first launch-gate refusal (2026-11-06)

PREREG_P6 Amendment (e) cites this folder. All four files were copied off the pod at 13:13Z, before the relaunch's G-LIVE-MEM stage could overwrite them. Each copy's sha256 equals the pod's.

| file | what it is |
|---|---|
| `FAIL_f2f604f5d806.json` | the launch gate's FAIL verdict. Commit `f2f604f`, profile `refe_p6`. Every check PASSES except G-LIVE-LAB. |
| `evidence_G-LIVE-LAB.json` | G-LIVE-LAB's evidence: throughput 8.405 sets/s against the 6.735 bar (OK); identity OK on 64 sets; stale proposal refused. The **stall judged on the G-LIVE-MEM run (accum 64): 0.71 / 5.36 / 3.28 / 1.48 %**. The first window is excluded by rule, and the maximum, 5.36 %, is above 5 %. |
| `mem_metrics.jsonl` | the G-LIVE-MEM instrument run (the real launch argv, `--label-workers 6`, scratch output, 4 steps). sha256 `97637a5e57a8c2d0…`, equal to the gate's `inputs_read`. |
| `lab.json` | the G-LIVE-LAB instrument at 6 workers. sha256 `853f4595d69dee1d…`, equal to the gate's input. |

## Readings (MEASURED, these files)

- **The pod's step time.** `mem_metrics.jsonl` shows optimizer steps (batch 4 x accum 64) of 51.6 / 51.1 / 50.5 s after the first, which includes compile. The A40's median was 58.6 s (Amendment (d)).
  - This is an instrument run, not the training run.
- **Labeller CPU.** `cpu_s_per_set` is 0.89-0.96 s. A step labels 256 sets, so the labeller needs about 236 CPU-s per 51 s step, or about 4.6 CPUs on average. Six workers leave little headroom for bursts.
- **CPU throttling is NOT the cause.** The pod's cgroup `cpu.stat` shows 72 throttled periods out of 120,267.

## The lever (Amendment (e))

`--label-workers 13`, the chain's own `CORES - 2` under `cpu.max` 15.3.

- `label_workers` is not in `train.py`'s `IDENTITY_ARGS` (`train.py:996`), so the resume accepts it as the same run.
- The relaunch's ARGS sha is `f74aee47cf0b708f`, against `570619873b1ebaa0` for steps 1-6,266.
- The chain re-keys G-LIVE-LAB (`w13-c2`), G-LIVE-MEM and G-EGR (the ARGS sha) and re-runs the gate by its own rules.
