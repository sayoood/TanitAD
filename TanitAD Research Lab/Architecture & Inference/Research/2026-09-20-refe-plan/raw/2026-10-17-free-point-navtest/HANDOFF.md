# HANDOFF: free-point navtest read (2026-10-17). State, artifacts, the exact next commands. <= 2k tokens.

`<pod>` = the A40 pod; its ssh prefix is in `C:/Users/Admin/freept/pod_ssh.txt` (OUTSIDE the package; never copy it in). Work dir on the dev box: `C:/Users/Admin/freept/work` (= `W`). Code: `code/` in this package (copy at `C:/fp/PK`-independent paths, run with the driverl-eval python).
PK checkout used by the code: `C:/fp/PK` = `git archive 19785c8` of the refe-plan package (re-create it if missing: `git -C C:/Users/Admin/tanitad-push archive 19785c8 "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan" | tar -x -C C:/fp` then move the dir to `C:/fp/PK`).

## State at hand-off
(see section STATE at the bottom; filled in last)

## Next commands (in order; each is resumable)
1. **Gate**: the pod worker may start ONLY when `/workspace/ops/bank3c/full/started_epoch` exists (the P6 re-roll launched). Check: `ssh <pod> 'ls -la /workspace/ops/bank3c/full/started_epoch'`. If absent: END (do nothing on the pod).
2. **Pod prep (once, 1 min, no GPU)**: `python code/fp_ship.py prep-pod --work W` (ships fp_forward.py, inputs, rig, the two scorer states; verifies md5/sha on the pod; prints `ZZSHIP_PREP_OK`).
3. **Start the pod worker** (one command; it refuses itself unless `started_epoch` exists, exit 3):
   `ssh <pod> 'cd /workspace/ops/freept && OMP_NUM_THREADS=2 REFE_BACKBONE_ROOT=/workspace/data/backbones PYTHONPATH=/workspace/refe-plan-rls1/refe nice -n 10 setsid nohup /workspace/venv-teacher/bin/python fp_forward.py serve --work-dir /workspace/ops/freept --inputs inputs_full.json --calib calib.npz --ckpt /workspace/data/rls1/base/model_sft4_B_u400.pt --state C0A=states/c0_u1000.pt:A --state R0B=states/r0_u1000.pt:B --device cuda --repeat-first 128 > serve.log 2>&1 < /dev/null & echo $! > worker.pid'`
   Stop it (if ever needed) by pid only: `kill $(cat /workspace/ops/freept/worker.pid)`. Never `pgrep -f` / `pkill -f`.
4. **Start the uploader + the dev-box tail detached (WMI, CreateFlags 0x08000200)**:
   `python code/fp_ship.py run --work W > W/ship.log 2>&1` then `python code/fp_chain.py --work W --wait-for-pull > W/chain_stdout.log 2>&1`
   (the chain waits for `ZZSHIP_ALL_PULLED`, then runs picks -> harness x4 -> C0A subset -> controls -> rows; `W/chain.log` ends with `ZZCHAIN_DONE <VALID|IDENTITY-FAILED|INVALID>`).
5. **Do not hold an agent for it.** ~2.3 h pod stage (upload-bound), ~1.6 h harness. Poll with: `ls W/out | grep -c npz` (chunks pulled, of the plan), `tail -3 W/ship.log`, `tail -2 W/chain.log`, pod: `ssh <pod> 'tail -2 /workspace/ops/freept/serve.log; df -h /workspace | tail -1'`.
6. **When `ZZCHAIN_DONE` appears**: read `W/result/freepoint_result.json` + `W/result/RESULT_TABLES.md` and `W/gates_summary.json`; write `RESULT.md` in this package applying the prereg rows (FP-R0B / FP-M25 / FP-NONE) mechanically; copy `W/result`, `W/*.json`, `W/chain.log` and the small outputs into the package `raw/`; bank the 9 chunk npz (about 0.45 GB) by sha256 in a `MANIFEST.sha256`; delete the pod's `/workspace/ops/freept` only after the dev-box copies are verified.

## Failure handling
* Pod disk < 6 GB free: the uploader holds (log line `holding`); do not free space by deleting anything that is not under `/workspace/ops/freept`.
* `ZZFWD_IC2_FAILED`: INVALID, stop. A killed worker: re-run step 3 (finished chunks are skipped by their `.done`).
* `ZZSHIP`/ssh drops: `fp_ship.py run` retries the chunk; re-launch the same command.
