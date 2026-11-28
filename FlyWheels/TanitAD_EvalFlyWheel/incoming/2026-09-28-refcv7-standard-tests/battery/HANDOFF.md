# HANDOFF - X8 fan control (D-REFCV7-FANCEIL), detached job

Written 2026-10-11 by the EvalFlyWheel battery analyst. NOT staged, NOT committed. The Master Mind lands.

## The detached job
* What: a refcv7 step-50,400 roll that records the FULL 117-candidate anchor fan (not just the selected pick).
* Started: 2026-10-11T16:18:25+0200, shell pid 5135, via `sh code/chain_x8_fanctl.sh 0`
  (wait_commit --min-gb 6 -> with_gpu_lock.py --job x8-fanctl-s0 -> code/fan_roll_r7.py).
* Lock: `REFCV7_GPU_LOCK=C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock` (dev-box RTX 4060). Never break a held lock.
* Output dir: `D:/refcv7_eval_kit/fanctl/s0/` (chain.log, child.log, lock_rec.json, fan_roll.json, dump/fan_extras.npz, dump/epNNN.npz).
* Done marker: `ZZFANDONE` in `D:/refcv7_eval_kit/fanctl/s0/chain.log`, ALSO assert content: `dump/fan_extras.npz` and `fan_roll.json` non-empty,
  `fan_roll.json` has traj_equals_sidecar_nav_true true, 4754 windows, fan shape (4754,117,4,2).
* Failure markers: `ZZFANNOCOMMIT` (no memory), `ZZFANMISSING` (child ended without outputs). Re-run is safe: `sh code/chain_x8_fanctl.sh 0`.
* Second inference draw (optional, for the inference-seed variance question): `sh code/chain_x8_fanctl.sh 1` after seed 0 is done.

## Analysis (CPU only, minutes)
`python -I code/fan_control_x8.py refcv7 --fan-npz D:/refcv7_eval_kit/fanctl/s0/dump/fan_extras.npz --gt-dump <banked s0 dump dir>
 --xcheck-dump <banked s0 dump> <banked s1 dump> --ckpt-md5 b418d0fc4a92a6848c246a6a7c50207b --roll-json .../fan_roll.json --out-json <out>`
Banked dumps used for GT: `C:/Users/Admin/AppData/Local/Temp/claude/x8dump/dump_s0/` and `dump_s1/` (extracted from raw/step50400_a6text/dumps_step50400_a6text.tar.gz).
Reproduction control (no GPU): `python -I code/fan_control_x8.py oldpt --pt taniteval/results/fan_refc-base-30k.pt --out-json ...`

## Where things live (NOT in git)
* The whole package `incoming/2026-09-28-refcv7-standard-tests/` is UNTRACKED; it exists on the E: and D: clones only.
* The checkpoint `D:/refcv7_eval_kit/ckpt/ckpt_50400.pt` (md5 b418d0fc4a92a6848c246a6a7c50207b) and all chain logs are on D: only.
* Small result JSONs of the fan control are copied under `raw/fanctl_x8/` and listed in LANDING_READY_X8.txt.

## ADDENDUM 2026-10-11 (EvalFlyWheel) -- seed-1 inference replicate + the two follow-on instruments
* Seed-1 roll (inference-seed replicate of the SAME checkpoint, answers "would another INFERENCE run say this?"):
  started 2026-10-11T16:52:58+0200, shell pid 7270, `nohup sh code/chain_x8_fanctl.sh 1 > D:/refcv7_eval_kit/fanctl/chain_s1.out 2>&1 &`.
  Outputs: `D:/refcv7_eval_kit/fanctl/s1/` (chain.log, child.log, lock_rec.json, fan_roll.json, dump/fan_extras.npz). Same markers as seed 0
  (`ZZFANDONE` / `ZZFANNOCOMMIT` / `ZZFANMISSING` in s1/chain.log); assert content, not the marker: fan_extras.npz and fan_roll.json non-empty, 4754 windows.
  Analysis: the same `fan_control_x8.py refcv7` argv as seed 0 with `--fan-npz .../s1/dump/fan_extras.npz --roll-json .../s1/fan_roll.json`
  and `--out-json D:/refcv7_eval_kit/fanctl/ctl/ctl_refcv7_50400_fan_s1.json`. GT dump for the windows stays `x8dump/dump_s0` (same windows).
* New CPU-only instruments (12 s and 9 s on the seed-0 fan), both write a JSON under `D:/refcv7_eval_kit/fanctl/ctl/` and a copy under `raw/fanctl_x8/`:
  `code/fan_headroom_x8.py` (oracle-in-top-K vs random-K) and `code/fan_mbr_x8.py` (inference-only MBR pick, 36-cell grid chosen on episode-half A, reported on half B).
  Both take `--fan-npz --gt-dump --out-json`; run them on the seed-1 fan to replicate. Set `OMP_NUM_THREADS=2` while a roll is live.
* Doc fix owed by the Master Mind (this agent does not edit Project Steering): GOALS_AND_CLAIMS.md row D-REFCV7-FANCEIL and MODEL_REGISTRY.md section 15.3 quote
  refc-base-30k numbers (0.1914 / 0.4728 / +0.2813) as refcv7; the true refcv7 values are in RESULT.md, section D-REFCV7-FANCEIL.
* Windows trap hit and recovered: `D:/Projects/TanitAD` and `E:\Projects\TanitAD` are ONE tree; a `cp` between the two spellings truncates the file to 0 bytes.
  Copy only from `D:/refcv7_eval_kit/...` (a different tree) into the package, never across the two project spellings.
- 2026-10-11 (EvalFlyWheel, final): the seed-1 inference roll FINISHED and its analysis is banked -- RESULT.md section
  "D-REFCV7-FANCEIL -- inference-seed replicate (seed 1)"; artifacts under raw/fanctl_x8/ (ctl_refcv7_50400_*_s1.json,
  ctl_refcv7_50400_seedcmp.json). No process of mine is running; the dev-box GPU lock is released.
