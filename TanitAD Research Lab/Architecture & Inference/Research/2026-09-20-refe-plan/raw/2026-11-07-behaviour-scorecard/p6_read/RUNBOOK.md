# RUNBOOK -- the pre-registered P6 scorer-speed read (H-P6-SPEED-1), tier T0

Card: `../PREREG_P6_SCORER_SPEED.md` (sha256 `8e8aa3cac3762ae4351dc09ef8842945b2f8d6ff82aeb849ee22811b594c0e6a`, frozen, unchanged by this package).
`$R` = `E:/Projects/TanitAD-artifacts/refe_behaviour_20261107/p6_read` (cwd for every command). `PYT` = `C:/Users/Admin/venvs/tanitad/Scripts/python.exe`, `PYD` = `C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe`.
Git-bash: never `python -I` and never a `:`-joined PYTHONPATH (memory: windows-pythonpath-separator). The scripts set their own `sys.path`.

**Everything is frozen before P6 exists.** `CODE_PIN.json` pins the code; `p6_read.py` runs `p6_control.py` first (u400 control must PASS, G6-PIN included).
Any edit to a pinned file after today needs an `AMENDMENT_1.md` written BEFORE a P6 value is read, and a re-pin (`python make_pin.py`).

## 0. Precision requirement (Master Mind): the dump runs in FP32, never bf16
bf16 autocast quantises raw plan x to 0.25 m at 32-64 m, which moves the 0.2 s chord v4 in 1.25 m/s steps and corrupts v4/v0 < 0.85. Enforced three ways:
`selc6_dump.py` refuses to run under autocast or with any non-float32 parameter and checks output dtypes; **gate G6-FP32** fails if more than 1 % of `traj[..., 19, 0]` values in [32, 64) m are exact multiples of 0.25
(needs >= 500 values in the band; u400's fp32 reference: 0 of 16,712, MEASURED); and the control's RED arm (trajectories snapped to the 0.25 m grid) must fail it. Do not pass any autocast/bf16 flag to anything.

## 1. Inputs, where they come from, how long (ESTIMATED unless marked)
| # | input | source | cost |
|---|---|---|---|
| 1 | P6 final checkpoint `ckpt_last.pt` (written when `run/summary.json` has `"done": true`; chain marker `ZZP6_DONE`) | **pod or HF** private repo `Sayood/tanitad-refe-p6-paper-faithful`, path `run/ckpt_last.pt` (P6_POD_RUNBOOK sec. 4). Pull to `F:/TanitAD_p6read_work/ckpt/ckpt_last.pt` (two folders deep). Size UNVERIFIED. Record `sha256sum` and compare to the HF LFS oid; pass it as `--expect-ckpt-sha`. | pull: minutes (HF ~118 MB/s INHERITED) |
| 2 | `config.json` of the run (G6-ARGV: `args.labeller`, `args.label_backend`, `args.nav_mode`) | same repo `run/config.json` (train.py:1510 writes it) | seconds |
| 3 | `train_logs.txt`: the logs P6 trained on (G6-DISJOINT) | **pod**: `ls $DATA/refe_navtrain/metric_cache \| sort > train_logs.txt` (`MC` of `p6_pod.sh:32`; one directory per log, `<log>/<token>.pkl.xz`; `$DATA` = the pod data root). This is the only pod access the read needs; one `ls`. (`splits/navtrain.yaml`, local, is checked in addition; 0 of 123 C3 logs are in it, MEASURED today.) | seconds |
| 4 | the **P6 navtest read**: full-navtest seam npz + PDMS csv of the SAME checkpoint (G6-1, G6-3) | dev box: `eval/eval_checkpoint.py --ckpt <ckpt> --name p6final --tokens <full navtest tokens>` -> `D:/Projects/TanitAD/data/refe_navtest/seams/<name>_navtest_full.npz` and `.../score/<label>/<label>.csv` (names UNVERIFIED for a P6 run: take the paths its report prints). It is the P6 headline read anyway and is probably queued; this package does not run it. If it was run with `--rule` or a different checkpoint file, G6-1/G6-3 will go red. | 12,146 tokens, GPU hours (UNVERIFIED) |
| 5 | C3 token lists, ego probe, C3 table, selection-probe code | banked locally, pinned by sha256 (`CODE_PIN.json`, `selc6_analyze.PINS`) | -- |
| 6 | u400 reference (pinned content) | `p6_read/u400_reference.pinned.json` (sha `1ad5776a...` = the card's pin) | -- |

Do NOT open the checkpoint, any snapshot or any P6 eval output before step 2 below. Open nothing for the read other than through `p6_read.py`.

## 2. Commands after `ZZP6_DONE` (dev box)
```bash
cd /e/Projects/TanitAD-artifacts/refe_behaviour_20261107/p6_read
sha256sum -c ../PREREG_P6_SCORER_SPEED.sha256                       # MM has landed it; must be OK
PYT=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; PYD=C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe
CK=F:/TanitAD_p6read_work/ckpt/ckpt_last.pt
# (a) control + pins, CPU, ~1 min.  Must print PASS:
CUDA_VISIBLE_DEVICES=-1 $PYT p6_control.py --out F:/TanitAD_p6read_work/control_result.json
# (b) GPU smoke, 3 tokens, FIRST use of the P6 variant on a real forward (shapes (64,6), command mode, fp32). ~3 min incl. model load. Takes the YAX lock (waits, never forces).
$PYD selc6_dump.py --ckpt $CK --out F:/TanitAD_p6read_work/smoke --smoke --expect-ckpt-sha <sha>
#     success line: "consolidated: ..."; failure = REFUSED/RuntimeError text. Then delete F:/TanitAD_p6read_work/smoke.
# (c) the whole read, DETACHED (one command; resumable; ends in p6_speed_result.json). Launch via WMI so it survives the shell and has stdin NUL:
powershell -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd /c C:\Users\Admin\venvs\tanitad\Scripts\python.exe E:\Projects\TanitAD-artifacts\refe_behaviour_20261107\p6_read\p6_read.py --ckpt F:\TanitAD_p6read_work\ckpt\ckpt_last.pt --out F:\TanitAD_p6read_work\run --run-config <config.json> --train-logs <train_logs.txt> --p6-seam <seam.npz> --p6-csv <p6.csv> --expect-ckpt-sha <sha> > F:\TanitAD_p6read_work\run_stdout.log 2>&1 < NUL'; CurrentDirectory='E:\Projects\TanitAD-artifacts\refe_behaviour_20261107\p6_read'}"
```
Poll with a script, not model turns: wait for `F:/TanitAD_p6read_work/run/markers/p6read.DONE` or `.FAILED` (a Monitor until-loop on those two files). Do not wait alive; write a HANDOFF.md and end.
Result: `run/p6_speed_result.json` (+ `P6_SPEED_RESULT.md`, `control_result.json`, `CHAIN.log`, `logs/`). Exit of the analyze stage: 0 = verdict issued, 2 = INSTRUMENT FAILURE (a reportable outcome; no number is quotable).

## 3. What the chain does (stage -> time -> resource)
| stage | what | time | resource |
|---|---|---|---|
| preflight | G6-PIN + u400 control PASS | ~1 min | CPU |
| dump | `selc6_dump.py`: 1,058 tokens, 64 proposals, 6 logits, FP32, one segment per invocation (<= 1,100 tokens; lock released between segments) | ~0.7 h | **GPU**, YAX lock; waits on `devbox_gpu.chain_active` marker and a foreign lock, never forces, never touches `devbox_gpu.lock` itself |
| consolidate | `dump/dump_p6.npz` | ~1 min | CPU |
| score | `selc6_score.py`: W3 NAVSIM v1.1 harness, 2 chunks x (64 columns + 1 RED column) = **130 calls**, 3 workers, work dir `F:/TanitAD_p6read_work/score`, label prefix `refe_p6read` (distinct from the confirm run's `refe_selcf`) | 1.3-1.8 h (scaled from POWER.md, the card's figure) | CPU, >= 3.5 GiB RAM, >= 3 GiB free on C:/E:/F: |
| analyze | gates G6-PIN, G6-ARGV, G6-DISJOINT, G6-0..4, G6-FP32, G6-REPRO, then the section 5 verdict | ~2 min | CPU |
**GPU estimate: ~0.7 h (+ model load, + any wait for the lock). Total wall-clock about 2.0-2.5 h after the P6 navtest read exists.**

## 4. After the verdict
1. Quote only from `p6_speed_result.json`. If `red_gates` is non-empty: INSTRUMENT FAILURE, no number quoted; if G6-ARGV shows `labeller != online`, H-P6-SPEED-1 is VOID (written into the result).
2. Report the four families per the card: LONGITUDINAL and LATERAL as defined in section 6 (T0), SAFETY, TACTICAL **n/a, n = 0** (no manoeuvre-decision label on these tokens), STRATEGIC **PAUSED** (PI 2026-09-19, item 25).
3. Register: update `GOALS_AND_CLAIMS.md` in the same turn (H-P6-SPEED-1). Rule Zero: REFUTED is not a finished turn; the card names the next arm for each outcome.
4. Deletion (CLAUDE.md, 2026-11-07): after the result folder is copied into the package and banked, delete `F:/TanitAD_p6read_work/score` (harness run dirs), `.../smoke`, and the pulled `ckpt/` (source: HF), by literal absolute path, logging path and size in the handoff. Never delete `ckpt_last.pt` on the pod/HF, the P6 navtest read's outputs, or `markers/`.

## 5. Failure modes and what to do
- `REFUSED: checkpoint sha256`: the file is not the one named; re-pull.
- G6-1/G6-3 red: the P6 navtest read used a different checkpoint, rule or token set, or its paths are wrong. Do not edit the gate.
- `GPU BUSY`: another job holds the lock; the chain waits up to 10 h. Never remove a foreign lock; never kill a process you did not start.
- Dump crash (rc 1): retried 3 times; the stage is resumable per log (`dump/chunks/<log>.npz`).
- UNVERIFIED: `selc6_dump.py` has not run a real forward (no P6 checkpoint, GPU held by another job). Step (b) is the first run; its failure is cheap and happens before the 0.7 h dump.
