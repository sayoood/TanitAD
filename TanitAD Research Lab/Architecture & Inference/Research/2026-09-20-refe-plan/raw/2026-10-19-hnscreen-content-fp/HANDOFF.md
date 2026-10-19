# HANDOFF -- HNSCREEN-CACHE-CONTENT-FP: the sibling-suite confirmation is still running (written 2026-10-19 16:45)

**State.** The fix, the tests, the mutation matrix, the cost measurement and RESULT.md / COST.md are DONE (`C:/Users/Admin/par18/hnfp2/out/` is a complete package, `paths.json` lists 33 entries). One item is open: the measured "0 new failures"
over every other test file that names an edited file (28 files + `test_launch_gate.py` full form), base vs mine. Priority files already finished, 0 regressions (section 6 of RESULT.md): launch_gate static 1147 -> 1167, ctx_cache 15/15, hn_screen_wiring 120/120,
hn_screen_read 31/31, dump_scores 16/16, sft4r_gate 153/153, carve_feed_gate 10/10, p2r_chain 139/139, rl_s1b 18/18, rl_r0c 26/26, rl_pickcon 20/20; new `test_hn_content_fp.py` 81 (62 arms + 19 mutants).

**Why a handoff.** The remaining files need about 90 min of CPU (`test_gate_suite_prerun.py` alone ~16 min, `test_hnscreen_chain.py` ~12 min); no agent waits on them.

**The jobs (detached, WMI, one child python at a time each, RAM-gated, `CUDA_VISIBLE_DEVICES=-1`, `PATH` has Git `usr/bin`).**
* mine: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe C:/Users/Admin/par18/hnfp2/suite_runner.py C:/Users/Admin/par18/hnfp2/cfg_mine.json` (log `logs/runner_mine.out`, results `suite_mine.json`, per-file logs `logs/suite_mine/`)
* base: the same with `cfg_base.json` (the tip's bytes in `base/`).
* Both are RESUMABLE (a file already in the JSON is skipped) and write `suite_{mine,base}.json.DONE` at the end. Started ~16:28; ETA ~18:00.

**How to check.** `ls C:/Users/Admin/par18/hnfp2/suite_*.DONE`; `tail logs/runner_*.out`. If no `.DONE` and no process whose command line contains `par18\hnfp2\suite_runner.py` (match the exact script name; `Get-CimInstance Win32_Process`), re-launch the same command: it resumes.
If a file hangs, run it by hand: `cd "base|work/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe" && CUDA_VISIBLE_DEVICES=-1 python <file>`.

**Exact next step (a fresh Sonnet agent, <= 15 min).**
1. `python C:/Users/Admin/par18/hnfp2/finish_pkg.py` -> writes `suite_compare.md`, replaces the PENDING block in section 6 of `RESULT.md` with the generated table, rebuilds `out/` and `paths.json` (it also copies the final `test_hn_content_fp` log and the sha256 of the six code files is printed).
2. Read the last line of the table: `NEW FAILURES` and `LOWER COUNTS` must both be `none`. For every row `FAILS ON BOTH`, read `tail` in `suite_mine.json` and decide environment/layout vs real (a layout gap shows as FileNotFoundError on a `raw/` or `taniteval/` path: export it from the mirror
   with `export_extra.py <tip> raw/<dir>` into BOTH trees and re-run that file in both). A row with NEW FAILURE is a real finding: root-cause it before anything else.
3. Replace the PENDING sentence under the table with one line ("0 new failures" or each failure with its cause), update the "Regressions so far" line, and put the final counts in the hand-back's first line.
4. `rm -rf C:/Users/Admin/par18/hnfp2/tmp/*` (the runners' temp dirs), keep `base/` and `work/` until the Master Mind has landed the files. No `git add` / commit / push.

**Constraints that still hold.** CPU only; never the GPU, a lock, a pod, or `par18/{l2*,sh5*,podstage,r0eb9}`, `D:/b9`; never `pkill -f` (kill by explicit PID, and only processes whose command line contains `par18\hnfp2`); at most 2 python processes of mine;
write nothing to D: (exFAT, ~19 GB free); keep `par18/hnfp2` under 300 MB (131 MB at hand-off); never run the full stack suite.

**Known non-passes (identical on base and mine, not caused by this change).** `test_aux_g12_bf16.py` SKIPS (no CUDA; the bf16 G12 probe did NOT run); `test_gate_gpu_probes.py` prints `SKIP_CUDA`; `test_s1aux_chain.py` prints one `ZZTEST_SKIP` (sandbox_runs: the navtest list under `FlyWheels/` is absent in this layout).
The code files must not change after this hand-off; if one does, re-run `test_hn_content_fp.py` (120 s) and `test_launch_gate.py --static --no-banked` (70 s) and the siblings that touch it.
