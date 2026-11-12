# Test results (MEASURED 2026-11-12 ~01:10, venv `C:/Users/Admin/venvs/tanitad`, torch 2.11.0, `CUDA_VISIBLE_DEVICES=-1`, scratch `git archive f7e790ab stack taniteval` exports; patched side = the same export + `codefix/v8f_mini.py` + the new test)

| suite | f7e790ab (+ new test) | patched |
|---|---|---|
| `test_v8f_slowwm_device.py` (new, 4 items) | **3 failed, 1 passed**: `RuntimeError: Tensor on device meta is not on the expected device cpu!` in `built_basis`; the two level/lead source-pin items fail; the bit-identity item passes | **4 passed** |
| `test_v8f_slowwm.py` (SLOWWM suite) | not run (unchanged file, same code path) | **21 passed** |
| `test_v8f_lead_head.py` + `test_v8f_level_head.py` (they call the two touched vision-only probes) | 2 failed, 70 passed | 2 failed, 70 passed (the same two) |

The two failures are `test_the_snapshot_is_the_tip_module` in each of the lead/level files: identical on both sides, so pre-existing at f7e790ab in this export (not caused by the fix; not investigated further: likely a snapshot-vs-tip comparison that needs a file outside the `stack taniteval` export).
Meta-device note: `torch.equal` has no meta kernel, so the test substitutes a shape/dtype comparison for meta tensors inside `built_basis`; the planted VALUES are pinned on the CPU instead (bit-equal to `torch.randn(1,4,256, generator seeded 0)`).
Not run: the full 771-file suite, any GPU, box 2.

## GPU evidence on box 2 (MEASURED 2026-11-12, child 7296df50, read on the dev box 07:01)

The pilot stage job `PILOT_SLOW_s0` (gid `20261112_011153`, launched 01:11:56) ended **DONE** at 01:58:43 (`D:\tanitad\slowwm\runs\PILOT_SLOW_s0\stage.log`: `SLOWWM-OK PILOT_SLOW_s0 mean 0.4704 s/step, 12.5 h planned`). This was the first time `built_basis` and the level/lead probes ran on a CUDA module since the fix; the crash of gid `20261112_005134` (`gate-fail`, layer_norm device mismatch) did not recur.
* Gates at pilot: all five full argvs (`SLOW_s0`, `PCA-B_s0`, `SLOW_s1`, `PCA-B_s1`, `SLOW_rep`) **true** (`PILOT.verified.json`, `full_run_gates_at_pilot`).
* Timing: 200 steps, mean 0.4704 s/step, median 0.4809; interval means 0.4856 / 0.4809 / 0.4447 s/step; first 50 steps incl. startup 27.4 s; peak GPU 4,272 MiB. Re-derived on the dev box from the pulled `train_log.jsonl` (sha256 `1e8893b3…`) to the same mean.
* Projection: 5 runs x 19,164 steps = 95,820 steps, **12.52 h planned** (2.50 h per run, ESTIMATED from the 200-step mean), inside the card's 72 h limit; `permit` passed at 07:02:31 and `SLOW_s0` launched on box 2 at 07:02:33.
* `gpu_util_median_pct` reads **0.0** in the pilot record. That is the sampler's reading, not interpreted here; the step time is the measured quantity.
* The dev-box driver did not see this DONE live: the dev box restarted at 01:32:46 (Windows Update, event 1074) while it waited. The finished job was adopted at 07:01 by the chain's own post-DONE steps (`../box2/restart_adopt_2026-11-12/ADOPT.md`), not run a second time.
