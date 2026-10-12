# P6 pod runbook: the same A40 as P2, to a banked 25-epoch P6 checkpoint (2026-10-11)
Chain `refe/p6_pod.sh` (tests `refe/test_p6_chain.py`, helpers `refe/p6_prep.py`). Design: `SPEC_P6_PAPER_FAITHFUL.md`; gate: `eval/REFE_LAUNCH_GATE.md`.
PI ruling P6-Y (2026-10-11): unwrapped yaw, `--yaw-wrap off`. It is the default of one variable (`P6_YAW_WRAP`); a blank or other value REFUSES, and the `refe_p6` profile refuses `on`.

## 1. Create
- No new pod: P6 runs on P2's A40 **after `ZZP2R_DONE`**. The chain refuses while P2's `chain.lock` is held (never add load to a training pod).
- P6-C (`eval/RESULT_M_LAB.md`): >= 24 vCPU, >= 64 GB RAM, >= 400 GB disk. Workers come from the cgroup quota (`cpu.max`), never `nproc`: label workers = min(cores-4, 16), cache workers = min(cores, 24). Under 16 vCPU the chain warns and G-LIVE-LAB probably fails its 6.735 sets/s bar.
- `HF_TOKEN` (write-scoped, `Sayood`) in the pod environment. Never in a file or in chat.
- P2's files are reused in place: bank `refe_navtrain/train_grow`, 1,192 DBs, 32 pixel shards, DINOv3 weights, teacher env. A missing one is fetched by its own `fetch_*` function only; a present one is never re-downloaded.

## 2. Ship (while P2 still trains; it adds no load)
- After landing, on the dev box: `python refe/launch_gate.py ship-manifest --pkg <package root> --commit <40-hex> --git-dir D:/Projects/TanitAD/.git --repo-prefix "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/" --out <package root>/SHIP_MANIFEST.json` (the shipped set now holds `p6_pod.sh`, `p6_prep.py`, `test_p6_chain.py`).
- scp `code/ refe/ splits/ SHIP_MANIFEST.json` to **`/workspace/refe-plan-p6/`**, never over `/workspace/refe-plan/`: P2's token binds that tree and its trainer reads it. The repo is public: no pod host or port in any repo file.

## 3. Start (one command, detached)
```
nohup setsid bash /workspace/refe-plan-p6/refe/p6_pod.sh > /workspace/data/refe_p6/p6_stdout.log 2>&1 < /dev/null &
```
- `mkdir -p /workspace/data/refe_p6` first. Dry run: prefix `P6_STOP_AFTER=<env|weights|recover|inputs|commands|cache|lanes|lab|mem|egr|launch_gate>`. `cache` and `lanes` can run while the PI waits.
- Re-run the same command after ANY interruption, or on a NEW pod: done stages are skipped; `recover` pulls `ckpt_last.pt` back from the P6 repo; `train.py --resume` continues. A rebuilt input or a changed argv voids exactly what depends on it (cache -> lab, mem, egr, gate; mem -> egr, gate).
- Knobs: `P6_LABEL_WORKERS`, `P6_CACHE_WORKERS`, `P6_SYNC_MIN` / `P6_HEAVY_MIN` (30 / 30), `P6_MAX_ATTEMPTS` (12), `P6_REPO`.

## 4. Watch without SSH: private HF repo `Sayood/tanitad-refe-p6-paper-faithful` (new, created by the first push)
`logs/chain.log` and `logs/train.tail.log` every 30 min; `run/metrics.jsonl`, `run/summary.json`; `run/ckpt_last.pt` and `run/snap_epoch*.pt` every 30 min; `gate/` (launch-gate evidence + PASS token, no key); `gate/ckpt_last_probe.pt` (G-EGR probe).
`chain.log` markers, in order: `env ok`, `weights ok`, `recover:`, `inputs ok`, `commands ok`, `metric cache ok`, `D7 rank k:` (or `lanes ok`), `ARGS sha`, `lab ok`, `mem ok`, `egr ok`, `launch gate PASS (token verified)`, `train.py preflight OK`, `attempt N exited rc`, `ZZP6_TRAIN_DONE`, `ZZP2R_FINAL_OK`, **`ZZP6_DONE`**.
- `REFUSED: <reason>`: stopped on purpose. Fix it, delete the named `stage/*.ok` if told to, re-run. `REFUSED: D6:` = no NAVSIM metadata. `REFUSED: D7:` = twins not on their own route (decision: `P6_D7_FALLBACK=1` = trajectory loss only, quarantined, logged).
- `ZZP6_FAILED`: 12 attempts, no `done: true`. `ZZP6_FINAL_FAILED`: the checkpoint is on the pod only; do not delete the pod. `ZZSYNC_DEGRADED`: pushes failing.

## 5. Expected wall-clock (a first run on P2's pod; ESTIMATED unless marked)
| stage | expected | source |
|---|---|---|
| env, weights, recover, inputs (all reused) | minutes | weights are hashed, not fetched |
| commands (NAVSIM metadata + table) | 10-40 min | download size UNVERIFIED; pickles ~19 MB each (MEASURED on one navtest log) |
| cache (D9) + exactness check | 1-1.5 h on 20 workers | ~24 CPU-h, `RESULT_M_LAB.md` |
| lanes (D7) | UNVERIFIED | twin count not measured |
| lab + mem + egr | 20-40 min | 256 labelled sets; 200 micro-batches; one 228 MB push at the INHERITED ~118 MB/s |
| launch gate | 15-25 min | P2's figure |
| training, 10,075 steps | 6.8-8.2 days | PI-accepted (P6-T) |
Pre-training total about 3-5 h. A fresh pod adds the DBs, pixels and bank (the P2 runbook's figures).

## 6. Open, UNVERIFIED, and for the Master Mind
- NAVSIM metadata (`openscene_metadata_trainval.tgz`): MEASURED 2026-10-12, public, **7,052,210,642 B**, sha256 `193138ac...`, pinned to dataset revision `a76f840b` (not `main`); a download with another size or hash is refused. The stage then still verifies content (>= 99.5 % of rank-0 tokens matched). Set `P6_NAVSIM_LOGS_URL=""` to forbid the download.
- D7 acceptance is judged from artifacts (twin route differs from the logged route on >= 90 % of 40 twins); first run on real parts objects is the pod.
- 30-min checkpoint egress, not SPEC section 4's 6 h: ~48 pushes x 0.228 GB a day is ~77 GB of LFS history over 7 days (ESTIMATED), inside the 1 TB quota (199 GB used). `P6_HEAVY_MIN=120` cuts it.
- The labeller workers inherit `OMP_NUM_THREADS=4`; G-LIVE-LAB measures under the same value.
