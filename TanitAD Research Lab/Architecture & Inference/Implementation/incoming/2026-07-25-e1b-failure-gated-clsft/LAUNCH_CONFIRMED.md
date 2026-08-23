# E1b — LAUNCH CONFIRMED (detached)

`2026-07-25 ~12:56 UTC` · `tanitad-pod3` (A40, idle before launch: 0 MiB) · renderer-free.
Sayed reads Europe/Berlin (UTC+2); `e1b_run.log` timestamps are UTC.

## What is running

A single detached job runs **mine → CL-SFT** end to end (`run_e1b.sh`):

| field | value |
|---|---|
| launcher (survives ssh close) | `bash run_e1b.sh`, **PID 2161704** — own session (`setsid`), fds redirected to `e1b_run.log`, `< /dev/null` |
| stage 1 (now) | `e1b_mine.py` **PID 2161706** — mining **400** parity-train episodes @ **K=185** → `/workspace/e1b/mined_buffer.pt` (~1.7 h; ≈16.5 s/window MEASURED) |
| stage 2 (auto after mining) | `e1b_clsft.py` — 4000 steps, lr 2e-5, cl-batch 16 / replay-batch 16, workers 4, encoder frozen → `/workspace/e1b/refc-base-e1b-clsft/` |
| master log | `tanitad-pod3:/workspace/e1b/e1b_run.log` |
| start marker (seen) | `[run_e1b] START 2026-07-25T12:55:46Z` → `[mine] 400/2376 … K=185` |
| disk | `dd` write 200 MB @ 159 MB/s — headroom OK (quota not full) |

**Detachment is sound:** `run_e1b.sh` is a `setsid` session leader with stdout/stderr redirected to the
log and stdin from `/dev/null`; the launching ssh had no tty, so no SIGHUP reaches it on disconnect.
The mined buffer persists, and `run_e1b.sh` **skips mining if `mined_buffer.pt` exists** — so a stage-2
crash costs only the CL-SFT attempt, re-launchable on the saved buffer.

**Stage 1 is SILENT** — `e1b_mine.py` logs only at completion (`[mine] N states … -> mined_buffer.pt`),
not per-episode. During the ~1.7 h mine, **liveness = GPU util > 0 AND the `e1b_mine` PID's `etime`
advancing** (checked at launch: 899 MiB / 15 %). Do not read the quiet log as a hang. If after ~3 h
there is still no `mined_buffer.pt` and no live PID, re-launch `run_e1b.sh` (it resumes at stage 2 if
the buffer landed, else re-mines).

## Check progress (read-only)

```bash
# overall run log (mining %/departures, then CL-SFT step rows)
ssh tanitad-pod3 'tail -30 /workspace/e1b/e1b_run.log'
# CL-SFT step log once stage 2 starts (watch cl_cls fall, cl_anchor_acc rise, rp_loss stable)
ssh tanitad-pod3 'tail -5 /workspace/e1b/refc-base-e1b-clsft/train_log.jsonl'
# mined buffer summary (after stage 1)
ssh tanitad-pod3 'cat /workspace/e1b/mined_buffer.meta.json'
# is it alive? (kill ONLY by explicit PID — never pkill -f)
ssh tanitad-pod3 'ps -C python -o pid,etime,cmd | grep -E "e1b_mine|e1b_clsft"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader'
```

**Done signature:** `E1B_RUN_DONE` in `e1b_run.log` **AND**
`/workspace/e1b/refc-base-e1b-clsft/metrics.json` has `"done": true` **AND** `ckpt.pt` present.
(Verify the marker + artifact — a bare "PID gone + GPU 0 MiB" is the SAME signature for success and
for a crash; check `e1b_run.log` for `FAILED`/`Traceback` before declaring either.)

## Run the paired verdict (the NEXT drumbeat, after stage 2 finishes)

```bash
ssh tanitad-pod3 'cd /workspace/e1b && export PYTHONPATH=/workspace/TanitAD/stack && \
  /workspace/venv/bin/python e1b_eval.py \
    --base-ckpt /workspace/experiments/refc-diffusion-base-v21-30k/ckpt.pt \
    --ft-ckpt   /workspace/e1b/refc-base-e1b-clsft/ckpt.pt \
    --val-dir   /workspace/v4run/valcache/physicalai-val-heldout-79d4e3d2d4c6 \
    --horizons 20,185 --out /workspace/e1b/e1b_eval_result.json 2>&1 | tail -20'
```
Re-rolls base + FT on the **same 44 held-out episodes** (K=20 and K=185) and reports the
PRE-REGISTERED primary — **junction corridor-departure@K185, paired episode-cluster bootstrap** —
plus the open-loop-ADE guardrail and the K=20 (2 s) rows. Wall ≈ 2× the E1a K185 run (~24 min for
base+FT). Then bank `e1b_eval_result.json` + `mined_buffer.meta.json` + `train_log.jsonl` into the
repo and write the verdict against `PRE_REGISTRATION.md §3`.

## Pre-registered success/bound (from PRE_REGISTRATION.md, do not move the goalposts)

- **SUCCESS**: FT junction departure@K185 CI-separated LOWER than base (paired `hi < 0`) **AND**
  open-loop ADE@2s not CI-separated-worse.
- **BOUND/FAILURE (equally publishable)**: not CI-separated, OR open-loop ADE@2s CI-separated worse.
