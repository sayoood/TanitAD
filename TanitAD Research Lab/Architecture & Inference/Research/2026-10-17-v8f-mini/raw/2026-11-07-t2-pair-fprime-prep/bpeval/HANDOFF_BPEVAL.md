# HANDOFF_BPEVAL (dev-box clock 02:50, 2026-11-07; tip a9da6e8f)

Package `E:/Projects/TanitAD-artifacts/dagland/motion_c/lora_pod/bpeval/` (nothing staged, nothing committed, box 2 untouched: read-only ssh only).

## State
* `compute_fprime.py` DONE and tested: `--banked-selftest` gives F = F' = 0.12074 m (reads 0.1207); a 2 mm perturbation of one window in the identity eval REFUSES (rc 3); a missing file REFUSES (rc 2).
* `box2/{env_bp,bp_job,bp_evals,bp_hash}.cmd` DONE (LF in the repo copy; the script ships CRLF copies).
* `run_bpeval.sh` WRITTEN, `bash -n` clean, DRY run (`BPEVAL_DRY=1`, fake box-2 root `dry/run1/fakebox`, stub evals = copies of the banked files) was RUNNING when this was written: output `dry/run1.out`. Read its tail; the expected last line is `ZZ BPEVAL DONE ZZ compute_fprime rc=0`, `dry/run1/out/FPRIME.json` present, the busy marker in `dry/run1/busy_aside/`.
* If the dry run shows a bug, fix `run_bpeval.sh` (edit, `bash -n`, rerun with the same env as in `dry/run1.cmd.txt`).

## Facts found (MEASURED, quote file:line in the report)
* Invocation of record: `code/box2_ops/chain_v8f_mini.sh:109-113` queues `eval ARM SEED`; `mk_v8f_job.py:9-14` wraps `v8f_mini_box2_queue.cmd` in `%RUNSAFE% cmd /c` (cwd `%REPO%\...\v8f-mini\code`); `v8f_mini_box2_queue.cmd:219-232` (`:eval`): `python v8f_mini_eval.py run --ckpt D:\tanitad\runs\v8f_mini_full\M2b_s0\ckpt_last.pt --tag M2b_s0 %EC% %ER% --project --families`, EC (line 112) = `--v9-eval/--lead-eval/--speed-eval D:\tanitad\data\v9\... --kin-mode yaw --p10 ...\p10_prior.pt --out-dir ...\eval`, ER (113) = `--eval-cache D:\tanitad\cache_full --tracks D:\tanitad\cache_full\native_tracks.pt`; defaults cycle ages, plan none, tac pred. Box 2 tree at that run: `D:\tanitad\repo`, tip `6e5d3768` (box 2 `TIP_V8F.txt` still reads it). Evals took 2m10s (M2b_s0) and 2m24s (M2b_s1) (`raw/full/box2_logs/chain_v8f_mini.out`).
* F: `v8f_mini_eval.py:559` = |mean ade6(s0) - mean ade6(s1)| over the 2,399 windows of the npz; banked 0.120740 (npz) / 0.1207 (JSON 4 dp). Banked refs: `raw/full/eval/eval_M2b_s{0,1}.{json,npz}`, copied to `bpeval/banked/` with sha256 in `banked/SHA256.txt`.

## Remaining
1. Read the dry-run result, fix if needed.
2. Final report (<= 250 words) to the Master Mind: invocation + source, dry-run result, F from banked, launch command, disk needs, manifest.
## Open flags for the Master Mind
* Pod `native_tracks.pt` is not in `lorapod_eval/eval`: put it at `lorapod_eval/native_tracks.pt` to have it shipped and compared with box 2's; otherwise box 2's copy is used for all three evals and logged UNVERIFIED-equal.
* The script refuses if `C:\lorapod_eval` already exists on box 2 (a leftover run): rename it aside by hand first.
* Distance keeping is UNAVAILABLE inside this eval (as for all banked M2b evals); `code/v8f_dk_read.py` reads it from the saved npz, not run here.
