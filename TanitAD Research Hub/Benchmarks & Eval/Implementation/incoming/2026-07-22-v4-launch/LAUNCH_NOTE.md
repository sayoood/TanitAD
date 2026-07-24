# flagship v4 — 30k run (⛔ STOPPED 2026-07-22, SUPERSEDED by v4.1)

> **⛔ SUPERSEDED — do NOT quote v4 as live.** This run was STOPPED at ~step 4000 (PID 75844
> killed 2026-07-22 ~16:33 UTC). Its WM-integrity canary degraded **0.42 → 0.599 (step 500) →
> 0.814 (step 1000) → 1.305 (step 1500)**, driven by the **trunk LP-FT at `lr_trunk=3e-4`** — it
> kept climbing after the controller zeroed the planner gradient (`lam_mult=0`) at step 2000, so the
> cause is the trunk fine-tuning, not the planner. Replaced by **v4.1** (`--lr-trunk 3e-5`, 10×
> lower); see `../2026-07-22-v4.1-launch/`. Everything below is accurate for what it measured (v4
> *did* train cleanly), but v4 is not the deployed run.

`2026-07-22 16:00:42 local (14:00:42 UTC)` · pod `tanitad-pod2` (NVIDIA A40) · Sayed's go relayed via the launch brief.

Evidence class legend: **MEASURED** (ours + artifact) · INHERITED · ESTIMATED. Every load-bearing
number below is MEASURED on pod2 at launch; nothing here is quoted from a trainer curve as an eval result
(C1). "Launched" is not the claim — **steps observed advancing** is.

---

## STATUS (historical — run since STOPPED): v4 trained cleanly 0→~4000, then stopped for WM-canary degradation

| fact | value | source |
|---|---|---|
| PID | **75844** (`/usr/bin/python3`) | `nvidia-smi --query-compute-apps` + `kill -0` |
| out dir | `pod2:/workspace/experiments/flagship-v4-30k` | — |
| log | `pod2:/workspace/experiments/flagship-v4-30k/train.log` | — |
| PID file | `pod2:/workspace/experiments/flagship-v4-30k/train.pid` | — |
| GPU lock | `v4-train` tied to `--pid 75844` (`gpu_lock.sh`) | lock is authoritative on a live PID |
| datasets | **train 406,099 / val 102,532** windows (parity `e438721ae894`) | `[data]` log line |
| step-0 canary baseline | **`canary_ade@2s` = 0.42148** (n=881), ref 0.452 | step-0 `canary_baseline` row |
| step 0 loss (finite) | total **1202.75**, wm 1.70, planner 24.72, plan_ade 70.24, oracle_ade 8.77 | step-0 row |
| step 50 loss (decreasing) | total **906.73**, wm 1.94, plan_ade 62.98 | step-50 row |
| step advance | **0 → 50 observed** (log rows), no traceback | `train.log` |
| GPU at step 50 | **100 % util, 33.3 GB, 243 W** | `nvidia-smi` |

The step-0 gradient norm is large (gnorm_head ~24.6k) — **expected** for a freshly-initialised head on a
warm trunk under warmup lr (lr_head 5e-8 at step 0); it is clipped to 1.0 and the loss is finite (a
non-finite loss hard-exits at `train_flagship_v4.py:622-623`, which did not fire). Total loss fell
1202.75 → 906.73 over the first 50 steps.

## Timing — MEASURED phase-A rate only (do NOT extrapolate to the gate)

- Steps 1→50 took `93.8 − 14.9 = 78.9 s` ⇒ **~1.58 s/step in phase A** (step-0 = 14.9 s incl. dataloader/
  CUDA warmup). MEASURED.
- ⚠️ **This is the phase-A WM-warmup rate** (`lambda_plan = 0.0`, curriculum A[0,2000)). The planner loss
  turns on in phase B[2000,8000) and the **strategic long-horizon-k=50** rollout in phase C[8000,30000) —
  both raise s/step. The **G1 KILL gate is at step 10000 (phase C)**, so its ETA depends on the
  as-yet-**unmeasured phase-C rate**. Use the GATE_PROTOCOL planning number **~10.9 s/step → ~30 h** as the
  conservative bound until a phase-C rate is observed; a phase-C measurement should replace it.

## Interpreter correction (MEASURED, overrides the brief)

The brief said use `/workspace/venv/bin/python` and that `/usr/bin/python3` is "CPU only". On **pod2 this is
wrong**: `/workspace/venv` **does not exist** (2 probes), and **`/usr/bin/python3` is CUDA-capable** —
torch 2.4.1+cu124, `cuda_avail True`, a real 2048² matmul on the A40 in 0.093 s. The trainer's own
`_staged_command` also emits `python3`. Pods drift; the measured interpreter wins.

## Sync provenance (the v4 code was STAGED-but-uncommitted on the dev box → git would not carry it)

- Synced the whole `tanitad/` package + 5 named scripts dev-box→pod2 via a tarball (atomic; MooseFS
  `Cannot change ownership` warnings are benign, content-faithful).
- **Byte-identity MEASURED**: md5 of all 7 critical modules matches local exactly —
  `train_flagship_v4.py 845ff87d…`, `flagship_v4_data.py 954650fa…`, `flagship_v4.py 5ca5dde3…`,
  `strategic_goal.py 8fbb1985…`, `flagship_v15.py ef6ab811…`, `flagship_losses.py 0f9c373c…`,
  `v4_curriculum.py c8ae5db1…`. (pod2 `train_flagship_v4.py` was **stale 23419 B → now 60310 B**.)
- **PREFLIGHT: OK** (`--print-launch` imported the full v4 graph + passed config invariants: gate 10000 ≥
  phase_b 8000; levers 2/2 door CLOSED).
- **real-smoke on 4 real windows PASSED**: `factorised_ce_trains:true`, `strategic_scalar_trains:true`,
  `all_heads_receive_grad:true`; grads into lat/lon/dist/goal heads (159.6/177.5/128.9/765.9).
- Disk headroom MEASURED by real `dd` (not `df`): 2 GB write at 506 MB/s on `/workspace`.

## Exact launch (as run)

`bash /workspace/run_v4_launch.sh launch` on pod2 (script in this dir: `run_v4_launch.sh`), which runs, detached:

```
PYTHONPATH=/workspace/TanitAD/stack /usr/bin/python3 scripts/train_flagship_v4.py \
  --train-cache /workspace/data/physicalai_phase0/_epcache/physicalai-train-e438721ae894 \
  --val-cache /workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11 \
  --trunk /workspace/experiments/flagship4b-speedjerk-30k/ckpt.pt \
  --anchors-dense /workspace/experiments/flagship_v4_anchors_dense.pt \
  --out /workspace/experiments/flagship-v4-30k \
  --labels v3 --lambda-plan sched --phase-a-steps 2000 --phase-b-steps 8000 \
  --strategic full --long-horizon-k 50 --steps 30000 --gate-step 10000 \
  --batch 16 --lr-head 1e-4 --lr-trunk 3e-4 --eval-every 500 --save-every 1000 \
  --eval-episodes 40 --rollout-k 4 --seed 0 --device cuda
```
(Trainer fills defaults `--warmup 2000 --workers 4 --log-every 50`.)

## Deliverable manifest

| artifact | where it lives |
|---|---|
| this note | `repo: TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-22-v4-launch/LAUNCH_NOTE.md` (STAGED) |
| exact launch/preflight/smoke script | `repo: …/2026-07-22-v4-launch/run_v4_launch.sh` (STAGED) **and** `pod2:/workspace/run_v4_launch.sh` |
| training process | `pod2` PID 75844 (detached, nohup) — ONLY on pod2 (a run, not a file) |
| train log + pid | `pod2:/workspace/experiments/flagship-v4-30k/{train.log,train.pid}` — ONLY on pod2 (grows) |
| checkpoints (future) | `pod2:/workspace/experiments/flagship-v4-30k/ckpt.pt` (save-every 1000) + `ckpt_step10000.pt` at the gate |

## Next / escalation

1. **G1 gate at step 10000** (card `Project Steering/Gates/flagship-v4.card.json`, 8 KILL / 5 REPORT). Run
   `run_gate.py` on the gate checkpoint when it lands. First in-loop canary eval + val at step 500.
2. **Measure the phase-C s/step** once past step 8000 to give a real gate ETA (replaces the ~10.9 s/step bound).
3. Run is detached with `nohup` + lock tied to PID 75844; a dev-box/session restart does not touch it. If it
   dies, recover via `train.log` (swallow-safe: it is under `--out`, not `/workspace` root) and relaunch
   (`ckpt.pt` auto-resumes; controller/phases restored).
4. Do NOT eval on pod2 while it trains; do NOT recycle `flagship-v4-30k/`.
