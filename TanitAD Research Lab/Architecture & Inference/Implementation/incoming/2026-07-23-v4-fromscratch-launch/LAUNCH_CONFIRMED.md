# flagship-v4-fromscratch-30k — LAUNCH CONFIRMED (running)

**Launched:** 2026-07-23 23:54:44 Europe/Berlin (21:54:44 UTC) · **Host:** pod2 (`tanitad-pod2`, A40 46 GB).
**Status:** RUNNING, verified advancing (step 0 → 50, MEASURED). **This is the ~53 h flagship.**
**Authority:** design `…/incoming/2026-07-23-v4-fromscratch/V4_FROMSCRATCH_LAUNCH.md` (§1 command); Sayed go 2026-07-23.

Number discipline: evidence class on every load-bearing claim. **MEASURED** = ours + artifact path.

---

## 0. Why (one paragraph)

v4.2b warm-started v1's already-converged WM and the anchored-diffusion planner's gradient yanked it
off-manifold: WM-integrity canary **baseline 0.421 → 0.520@1500 → 0.860@2000 → 0.697@4000**, stuck
`controller_held_at_floor` (MEASURED, `pod2:/tmp/flagship-v4.2b-train.log`). The pre-probe showed
`g_wm ⊥ g_plan` (cos +0.004) → gradient surgery is a no-op → the coupled path is **from-scratch = v1's
proven co-evolve-from-random recipe** (v1 held its intent-free canary at 0.42). This run trains the WM +
planner **jointly from random init, no warm-start** — there is no pre-converged manifold to fall off of.

---

## 1. Exact launch command (reconstructed by the trainer's own `--print-launch`; PREFLIGHT: OK)

```
PYTHONPATH=/workspace/TanitAD/stack python3 -u scripts/train_flagship_v4.py \
  --from-scratch \
  --train-cache /workspace/data/physicalai_phase0/_epcache/physicalai-train-e438721ae894 \
  --val-cache   /workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11 \
  --anchors-dense /workspace/experiments/flagship_v4_anchors_dense.pt \
  --out /workspace/experiments/flagship-v4-fromscratch \
  --labels v3 --lambda-plan sched --phase-a-steps 2000 --phase-b-steps 8000 \
  --strategic full --long-horizon-k 50 \
  --steps 30000 --gate-step 10000 --batch 16 --accum 4 \
  --lr-head 1e-4 --lr-trunk 1e-4 --lam-mult-floor 0.25 \
  --warmup 2000 --workers 4 --eval-every 500 --save-every 1000 \
  --eval-episodes 40 --rollout-k 4 --seed 0 --device cuda
```

Byte-identical to `V4_FROMSCRATCH_LAUNCH.md` §1. Parity: corpus `physicalai-train-e438721ae894`
(2376 eps, skip-hash `f09e44db`); `--from-scratch` changes only the weights' init, never the episode set.

### ⚠️ One drift surfaced (immaterial, but logged)

`V4_FROMSCRATCH_LAUNCH.md` §1 prose calls `--lam-mult-floor 0.25` "byte-identical to the v4.2b arm," but
v4.2b's actual `config.json` ran **`lam_mult_floor: 0.15`** (MEASURED, `pod2:.../flagship-v4.2b-30k/config.json`).
So the from-scratch run differs from v4.2b in **two** flags (trunk init AND floor 0.15→0.25), not one.
**Launched with the doc's `0.25`** because: (a) it is the §1 command value and the value the §3.1 smoke
validated ("held at floor 0.25"); (b) §4 proves the floor is **inert from-scratch** — the canary controller's
baseline is the untrained (high) canary, the co-evolving WM sits below it, so the controller never cuts
`lam_mult` and the floor never binds. **VERIFIED inert live:** `lam_mult = 1.0` at steps 0 and 50. So the
0.25-vs-0.15 difference has no behavioral effect and attributability is preserved. The "one-lever" prose is
imprecise about this single number; the run is correct.

*Note: the task brief mentioned "+ grad-checkpoint" — `train_flagship_v4.py` has no such flag and the
v4.2b encoder ran `grad_checkpoint:false`; §1 omits it and fit on the A40. Followed §1 exactly.*

---

## 2. Preflight (P0) — all MEASURED on pod2

| check | result |
|---|---|
| pod2 reachable / GPU free | A40, 0 MiB used, no compute procs (v4.2b pid 99197 dead) — **MEASURED** |
| pod2 trainer had the fixes? | **NO** — `--from-scratch` count 0, `g_op_fwd_ade_m` count 0 (stale). Pods drift; verified, not assumed |
| synced current trainer | local working-tree `stack/scripts/train_flagship_v4.py` (sha256 `34b9a85f…`, 76492 B) scp'd; **remote sha matched exactly**. Old file backed up → `train_flagship_v4.py.pre-fromscratch-bak.20260723-214306` (NOT deleted) |
| fixes now present | `--from-scratch` count **9**, `g_op_fwd_ade_m` count **2** (the gate log-fix, so the 10k gate can score speed_benefit) |
| import chain | `--help` RC 0, `--from-scratch` in help, **no ImportError** → full v4 chain (incl. `tanitad.lake` + `tanitad.lake.vocab`, the §3.2 gap that blocked pod1) resolves on pod2 |
| interpreter | `/usr/bin/python3` = torch 2.4.1+cu124, CUDA True, A40 (no `/workspace/venv` on pod2) |
| caches | train `…/physicalai-train-e438721ae894` ✓, val `…/physicalai-val-0c5f7dac3b11` ✓, dense anchors `flagship_v4_anchors_dense.pt` (42550 B) ✓ |
| disk | daemon `dd_500mb_ok:true, tight:false, oom_kill_total:0`; fresh 500 MB `dd` OK @412 MB/s |
| `--print-launch --from-scratch` | **PREFLIGHT: OK**; "2 of 2 encoder levers — door CLOSED" |

---

## 3. Launch verification (P2) — all MEASURED (`pod2:.../flagship-v4-fromscratch/`)

**Processes:** supervisor pid **107985**, trainer pid **108011** (+4 DataLoader workers). Heartbeat
`running`, `restarts:0`.

**`config.json` (startup provenance):**
- `from_scratch: true`; `trunk.init: "from-scratch (random)"`, ckpt `null`, step `-1` — no warm-start.
- **not_frozen: true** — encoder **149/149** require grad, predictor **159/159** require grad,
  `trunk_tensors_frozen: 0`, `trunk_group_lr: 1e-4`.
- **effective_batch: 64** (micro 16 × accum 4); `lr_trunk 1e-4`, `lr_head 1e-4`.
- `lambda_plan_mode: sched`, `strategic: full`, `mult_floor: 0.25`.

**stdout banner:** `[v4][from-scratch] trunk (encoder+predictor) + grounding RANDOM-INITIALIZED — no v1
warm-start; WM + planner co-evolve to a joint optimum (the v1 training regime).` + dense anchors loaded.

**`train_log.jsonl`:**
- step-0 canary **baseline = 15.674** (n=881). HIGH/untrained — the from-scratch signature. (The design's
  "~1.5" was the smoke's toy-model/toy-data scale; the full model on 881 real val windows reads 15.67.)
  The **descent** of this number is the signal, opposite to v4.2b's warm-start baseline (0.421) that ROSE.
  Per design §5 this is expected; judge the gate against the card, not against v1's 0.452.
- **step 0 → 50 advancing** (MEASURED). Both steps: `eff_batch 64`; **gnorm_encoder & gnorm_predictor
  both > 0** (89.3/35.3 → 294.1/129.1, growing with warmup) — Sayed's hard requirement (encoder AND
  predictor training separately, proven by live gradients) is met.
- `lam_mult 1.0` (controller inert, as designed), `lambda_plan 0.0` (Phase A [0,2000), planner not yet
  coupled). **total loss 1230.7 → 923.9 (dropping)** — WM+planner co-evolving, not collapsing.

**GPU:** 33.2 GiB stable, util bursty 11–96 % (normal for rollout-heavy training). Early pace **~7.9 s/step**
(steps 0→50) vs v4.2b's warm 6.4 — likely cold-start (kernel autotune, dataloader + on-the-fly-label FS
warm); expected to settle. If it holds, 30k ≈ ~60–66 h; the **step-10000 gate ≈ ~22 h** is the first read.

---

## 4. Supervision (auto-resume, no self-kill)

Launched **detached** via `setsid` under the ops-standard supervisor
`pod2:/workspace/ops/bin/supervise_run.sh` with manifest `pod2:/workspace/ops/runs.d/flagship-v4-fromscratch.env`.
- **Auto-resume:** on trainer death the supervisor relaunches (exp backoff 10→120 s); the trainer resumes
  bit-exact from `ckpt.pt` (`--out` dir; restores model/grounding/head/opt/controller/step). Stops on the
  trainer's completion print via `DONE_TOKEN='"done": true'` (the v4 trainer writes `metrics.json`, no
  summary.json — token set to match its stdout).
- **No self-kill:** flock single-instance guard + foreign-trainer guard that skips its own `$$`/child PID;
  **no `pkill -f`** anywhere. (This same supervisor is what runs the pod2 ops-daemon.)
- **Durable log:** `TRAIN_OUT` (`$OUT/train.out`) is a symlink to `/tmp/flagship-v4-fromscratch-train.out`
  (survives a `/workspace` swallow-on-death, per the pod2 constraint); structured `train_log.jsonl` +
  checkpoints live in `$OUT` on `/workspace` (durable across pod restart). Heartbeat →
  `/workspace/ops/heartbeats/flagship-v4-fromscratch-30k.json`.
- **The ops-daemon does NOT auto-launch from `runs.d`** (it is a memory-relief/disk-monitor only), so the
  manifest triggers nothing on its own; the supervisor was started by hand. `ENABLED=1` set for future
  boot-hook pickup.

---

## 5. Monitoring & next decision

- Live step/canary: `pod2:.../flagship-v4-fromscratch/train_log.jsonl` (per-row flushed) and the heartbeat.
- Watch the canary **descend** from 15.67 (controller stays `lam_mult 1.0` while WM sits below baseline).
- Phase B ramp (λ_plan 0→1) over steps 2000–8000 is where warm-start v4.2b degraded; from-scratch should
  keep co-evolving (no pre-converged WM to break). **First gate: step 10000** on
  `Project Steering/Gates/flagship-v4.card.json` (needs the v4 held-out eval driver — standing blocker).
- ⚠️ Do NOT add eval/other GPU load to pod2 while it trains.

---

## Deliverable manifest

| artifact | where | status |
|---|---|---|
| this launch note | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-23-v4-fromscratch-launch/LAUNCH_CONFIRMED.md` | **STAGED** (git add; not committed, not pushed) |
| run manifest | `pod2:/workspace/ops/runs.d/flagship-v4-fromscratch.env` | live on pod |
| trainer (both fixes) | `pod2:/workspace/TanitAD/stack/scripts/train_flagship_v4.py` (sha `34b9a85f…`) | installed; prior version backed up alongside as `…pre-fromscratch-bak.20260723-214306` |
| run outputs | `pod2:/workspace/experiments/flagship-v4-fromscratch/` (`config.json`, `train_log.jsonl`, `train.out`→/tmp, `supervisor.log`, ckpts) | writing live |
| stdout (durable) | `pod2:/tmp/flagship-v4-fromscratch-train.out` | writing live |
| heartbeat | `pod2:/workspace/ops/heartbeats/flagship-v4-fromscratch-30k.json` | updating ~30 s |

**Escalation:** none blocking. The gate at step 10000 needs the v4 held-out eval driver (the standing v4
blocker noted in the design §7) — flagged for the gate-reader, not required to launch/train.
