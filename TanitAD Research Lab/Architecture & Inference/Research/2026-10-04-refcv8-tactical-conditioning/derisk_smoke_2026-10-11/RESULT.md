# refcv8 full-size de-risk smoke — Thor, 2026-10-11 (NON-BINDING)

**Purpose.** Before the R2 ladder ends, run the refcv8 launch argv at full size through the trainer once. It is
serialized with the ladder on the shared GPU lock (`/home/nvidia/refcv7_post/thor_gpu.lock`), never run concurrently.
**Non-binding:** the recipe values are placeholders (the assembler's self-test recipe, `k_tac` 1, `w_map_hires` 1.0),
so the gate's binding G-LIVE re-runs on the final argv after the ladder.

**Code.** Commit `1df3503`, shipped to `/home/nvidia/refcv6_run/1df35031db` (2,931 files md5-verified + 5 ladder files).

**Argv.** `raw/smoke.argv.json` (210 tokens, md5 `c1a420ed…`). It was built with `refcv8_assemble_argv.py --stage
measure`, plus `launch_gate.smoke_argv(argv, 100, 50)`. It carries:
- the step settings: `--steps 100`, `--grad-share-every 50 --grad-share-at 50 --grad-share-terms tac_v6,map_hires`,
  `--r8-alloc-emit-start 50`;
- warm-up and the B1 guard: `--ema-decay 0.9998 --b1-guard-ref-ap2m 0.21211 --b1-guard-step 100`;
- the PI's input rulings: `--r8-no-rc` and `--r8-speed-input label` with refcv7's two road-type sidecars;
- the warm start: `--init-from` refcv7-r101-s0 `ckpt.pt`.

## Result (MEASURED, `raw/metrics.jsonl`, `raw/summary.json`, `raw/gs_measure.json`)

| check | reading |
|---|---|
| trainer exit / done marker | rc 0; `summary.json` `done: true`, step 100; wall-clock 2,286.6 s |
| s/step EXCLUDING the gradient-share reading | **9.212** (SPEC ceiling ≤ 10.5) |
| one fp32 gradient-share reading at full size (2 kept terms) | **838.9 s** |
| peak device memory (`cuda_max_mem_gb`, in-process) | **31.2 GB** |
| EMA | 100 updates; `ema_eval_*` rows present at the step-100 in-run eval |
| B1 guard | armed (ref 0.21211); **decided `keep`** at step 100 on `ema_eval_box3d_ap2m` |
| in-run eval set = refcv7's B1 reference set | `eval_box3d_n_pos` **339** = refcv7-r101-s0's step-50,400 `box3d_n_pos` 339 (`inrun_curve.json`) |
| live `eval_box3d_ap2m` at step 100 (warm start) | 0.2256 (refcv7 @ 50,400 in-run: 0.22867) |
| emission switch at step 50 | steps 51–100 completed with emission on, no crash |
| gs at step 50 (instrument output, not a scored outcome) | `gs_trunk_proj_tac_v6` −0.0003, `gs_bev025_proj_map_hires` 0.0226 |
| errors | 0 tracebacks / non-finite in `smoke.log` and `train.log` |

## Consequences
- **Gradient-share schedule for the launch, by the registered rule** (`pick_gs_schedule`; SPEC 10.3):
  - The cost is 4 readings × 838.9 s = 3,356 s.
  - The budget is 3 % of 30,000 × 9.212 s = 8,291 s.
  - **`--grad-share-at 1000,10000,20000,30000`** (the long schedule).
- **Projected run:** 30,000 × 9.212 s ≈ **76.8 h (3.2 days)**, plus 3,356 s of readings plus evals.
- The binding G-LIVE, the gate's s/step check and the overfit records are re-run on the final argv after the ladder.
