# P2R pod runbook: a fresh A40 to a banked P2 checkpoint (PI, 2026-10-11)
Chain `refe/p2r_pod.sh` (tests `refe/test_p2r_chain.py`). What the result decides: `eval/PREREG_P2_AMENDMENT_1.md`.

## 1. Create the pod
- **GPU:** 1 x A40 (48 GB), Linux, a `/workspace` volume; bash, flock, curl, tar, python3 and apt or pip (which template the
  old pod used is UNVERIFIED).
- **`HF_TOKEN`:** a pod environment variable, **write-scoped** for `Sayood` (it reads the two private source repos and
  creates `Sayood/tanitad-refe-p2-joint-pdm`). Never in a file or in chat.
- **Disk: at least 400 GB** (ESTIMATED: DBs 163.82 GiB MEASURED 2026-09-23, pixels ~84 GB MEASURED on one shard, ~60 GB for banks,
  venv and checkpoints ESTIMATED; the old pod's 449 GB of navtrain is INHERITED and unexplained). **Prove it with `dd`, not `df`:**
  `dd if=/dev/zero of=/workspace/.t bs=1M count=$((400*1024)) conv=fsync; rm /workspace/.t`. The chain proves only 8 GB (`P2R_DD_GB`).
- **CPU, RAM:** read `cat /sys/fs/cgroup/cpu.max`, not `nproc` (96 cores shown on a 7.65-CPU quota, MEASURED, `code/pod_dataprep.sh`).
  The relabel uses `min(quota, 16)` workers (`P2R_RELABEL_WORKERS`). Ask for **16+ CPUs and 64+ GB RAM**: `train.py` loads the
  whole 174,912-set bank and the old pod's 50 GB cgroup was an OOM hazard.

## 2. Ship the code (the package is staged, not landed; the pod is assumed to have no git)
scp/rsync `code/`, `refe/`, `splits/` of `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/` to
`/workspace/refe-plan/` (skip `__pycache__`). Check `ls /workspace/refe-plan/refe/p2r_pod.sh`.
The repo is public: never write the pod's host or port into a repo file.

**BINDING since 2026-09-26: the launch gate** (`eval/REFE_LAUNCH_GATE.md`). The package must be LANDED first, then on the dev box, before shipping:
`python refe/launch_gate.py ship-manifest --pkg <package root> --commit <the landed 40-hex> --git-dir D:/Projects/TanitAD/.git --repo-prefix "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/" --out <package root>/SHIP_MANIFEST.json`
(it REFUSES any file that is not its blob at that commit). Ship `SHIP_MANIFEST.json` at the package root next to `code/ refe/ splits/`. The chain refuses to start without it,
mints the PASS token on the pod (stage `launch_gate`, ESTIMATED 15-25 min, UNVERIFIED) and verifies it against the exact `ARGS` before the preflight and before every `train.py` launch.
A `REFUSED:` line naming `launch_gate.log` / `lg/` / `token_verify.log` means the gate said no: read `lg/evidence_*.json`, fix the cause, delete `stage/launch_gate.ok`, re-run.

## 3. Start (one command, detached)
```
mkdir -p /workspace/data/refe_p2 && nohup setsid bash /workspace/refe-plan/refe/p2r_pod.sh \
  > /workspace/data/refe_p2/p2r_stdout.log 2>&1 < /dev/null &
```
- **Dry run first:** prefix `P2R_STOP_AFTER=recover` (env, weights, restore, recover). Any stage name ends the chain after it.
- **Re-run the same command after ANY interruption, or on a NEW pod:** finished stages are skipped, `recover` pulls the PDM
  labels and `ckpt_last.pt` back from HF, and `train.py --resume` continues.

## 4. Watch without SSH: HF private repo `Sayood/tanitad-refe-p2-joint-pdm`
| file | updated | tells you |
|---|---|---|
| `logs/chain.log`, `logs/pdm_train.tail.log` | every 30 min | stage markers (below); relabel progress |
| `run/metrics.jsonl`, `run/summary.json` | every 30 min | training; `done: true` at the end |
| `run/ckpt_last.pt` | every 30 min if changed | a pod loss costs the 20-min checkpoint interval + up to 30 min + push time |
| `pdm_labels/` | once, before training | the ~11 h of relabel CPU is safe |
| `run/model_final.pt` | at the end | the result, sha256-checked against `summary.json` |

Markers in `chain.log`, in order: `env ok`, `weights ok`, `restore ok`, `recover: ZZRECOVER_OK labels=.. ckpt=..`, `dbs ok`,
`PDM labels complete`, `pixels ok`, `PDM labels banked on HF`, `gate PASS`, `launch gate PASS (token verified)`, `train.py preflight OK; launching`,
`attempt N exited rc`, `ZZP2_DONE`, `ZZP2R_FINAL_OK`, **`ZZP2R_DONE`**.
- `REFUSED: <reason>`: stopped on purpose. Fix the reason, re-run the start command. `ZZP2_FAILED`: six attempts, no `done: true`.
- `ZZP2R_FINAL_FAILED`: the checkpoint is on the pod only. **Do not delete the pod.**
- `ZZSYNC_DEGRADED`: pushes are failing (token or quota); training continues, fix it before the end.
- A `chain.log` on HF older than about an hour means the pod or the chain is down.

## 5. Expected wall-clock (a first run on a fresh pod)
| stage | expected | class and source |
|---|---|---|
| env + weights | 20-40 min | ESTIMATED, no timing banked |
| restore (4.4 GB) + recover | under 15 min | ESTIMATED; the ~118 MB/s pod-to-HF rate is INHERITED |
| DBs (97.57 GiB transit), pixels (304.5 GB, beside the relabel) | not banked | volumes MEASURED, times UNVERIFIED; DBs skipped if labels are restored |
| PDM relabel | **about 11.3 h** | INHERITED, `STOP_CONSEQUENCES_2026-10-11.md`: 174,912 sets at ~258/min, CPU count not recorded |
| gate + preflight | 0.5-1 h | ESTIMATED (first compile) |
| training, 1,209 steps at ~60 s, then the final push (under 15 min) | **about 20 h** | ESTIMATED from the final run's step time, `PREREG_P2.md` |

Sum about 33-36 h on a first run, about 21 h when the labels come back from HF (both ESTIMATED).

After `ZZP2R_DONE`: pull `run/model_final.pt` and `run/summary.json` to the dev box, check the sha256, run R1 then R2 of the
amendment. Stop the pod only after `ZZP2R_DONE` and the PI's say-so.
