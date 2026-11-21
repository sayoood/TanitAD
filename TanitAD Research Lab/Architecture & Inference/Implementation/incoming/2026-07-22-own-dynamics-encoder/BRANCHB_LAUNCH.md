# Branch B — LAUNCHED (from-scratch camera-conditioned video-SSL encoder)

> 🟥 **OUTCOME 2026-07-24 — Branch B FAILED its decisive held-out-rig transfer eval (`MEASURED`).**
> Best cross-rig speed R² **−0.667** (gate +0.9), yaw R² negative everywhere; **worse than the plain frozen
> flagship-v1 encoder** (+0.657) paired on 3/4 arms; own 40k head reads rig-B speed R² only **0.24**.
> From-scratch, all-block GAIA-2 camera-conditioning **did not** engineer rig-robustness at this scale. The
> "Branch B is the go / YouTube-IDM thesis" framing below is **superseded** — see
> `../../2026-07-24-branchb-transfer-eval/RESULTS_branchB.md` and `Project Steering/MODEL_REGISTRY.md §10`.
> Pivot (HYPOTHESIS, **Sayed-gated new arm, not auto-launch**): a flagship-warm-started, longer-trained,
> augmentation-matched encoder variant.

**Launched 2026-07-23 on pod3 (A40), Sayed-approved.** The camera-conditioning ablation FAILED (Branch A
refuted — `RESULTS_camcond.md`); Branch B is the go: the conditioning is learned **from scratch, at every
block, on a multi-rig corpus**, exactly the GAIA-2 regime the cheap probe lacked.

## Config (MEASURED at launch)

| | |
|---|---|
| model | `DynamicsEncoderModel(DynEncConfig(grad_checkpoint=True))` — **from scratch** (no warm-start) |
| params | deployable **97.4 M** (ViT 87.0 + GAIA-2 all-block cond 7.4 + readout 0.1 + IDM head 2.9), total 105.9 M — sub-300 M |
| conditioning | GAIA-2 **all-block** intrinsics/extrinsics/distortion embeds + known/unknown mask, per-clip `cy` |
| objective | masked-latent SSL + action-cond forward pred + SIGReg + supervised IDM head + odometry metric grounding |
| corpus | **2466 clips** — PhysicalAI rig-A **637** + rig-B **1739** (per-clip cy, fisheye) + comma2k19 **90** (rectilinear). Multi-rig, 3 geometries. **SIDE model — never the parity key.** |
| standardizer | corpus-sampled (fixed, resume-safe) — speed/yaw/steer/accel mean·std from 20 batches |
| optimizer | AdamW lr 3e-4, wd 0.05, warmup 1000, grad-clip 1.0, batch 16, grad-checkpoint |
| loader | memory-safe SHARD loader: 48 clips resident (~5.6 GB, well under the 46 GB cgroup), rotate every 200 steps |
| steps | 40 000 (≈ **11–20 h** at ~1 s/step; extend later if the curve warrants) |

## Robustness

- **Supervisor** `pod3:/workspace/tmp/dynenc_supervise.sh` (staged copy: `dynenc_supervise.sh`) auto-resumes
  from the durable ckpt on any death (up to 300 attempts, 25 s backoff).
- **Atomic checkpoints** every 500 steps to `pod3:/workspace/experiments/dynenc-branchB/ckpt.pt` (write-tmp
  + rename; durable MooseFS), md5-logged. **Milestone** copy at step 2000.
- **Logs to /tmp** (`/tmp/dynenc/train.log`, `supervisor.log`) — /workspace logs are swallowed on hard kill.
- **GPU lock** `own-encoder-branchB` tied to the **supervisor PID** (survives trainer restarts).

## GPU probe (MEASURED, pre-launch)

Real config, batch 8, grad-checkpoint: peak **3.4 GB** / 46 GB (huge headroom), ~1 s/step, camera-cond
live on real data (|Δz| 0.018 on a cy change). 25-step trainer run: loss **8.2 → 3.0**, IDM **5.65 → 1.40**,
stable (the per-batch-standardizer explosion was diagnosed and fixed → fixed corpus-sampled standardizer).

## First milestone (course-correction point) — step 2000 (~40–60 min)

Bank the milestone ckpt + confirm: (1) loss descending, (2) camera-conditioning live (|Δz|>0), (3)
multi-rig batches. If healthy, continue to 40k; else course-correct (lr / objective weights / batch).

## Monitoring

`ssh pod3 'tail /tmp/dynenc/train.log'` · lock `gpu_lock.sh status` · ckpts `ls -la
/workspace/experiments/dynenc-branchB/`. The run is on **pod3 only**; do not add load to pod2 (v4.2),
eval (scenario), pod1 (Orin INT8). NOTE: background notification-monitors have been killed mid-run twice
— **check the log/ckpt DIRECTLY**; the run itself is self-sufficient (supervisor + auto-resume + durable
ckpts) and does not depend on any monitor.

## Step-10k course-correction (MEASURED 2026-07-23, healthy — no adjustment)

- **Loss descending cleanly:** total **10.2 (step0) → 4.5 (25) → 1.4 (4250) → ~1.0–1.4** (7k–10.3k);
  supervised **IDM 5.8 → 1.5 → ~0.3–0.8** — converging. SIGReg ~5–6 (×0.1 = stable anti-collapse).
- **⭐ Camera-conditioning LIVE & rig-discriminating** (CPU check on the step-10k ckpt, GPU masked off):
  per-block inject weight-norms **[17.1, 10.6, 11.5, 12.0, 12.8, 13.9, 15.5, 17.8, 16.6, 16.4, 16.5, 17.1]**
  (all grown from **zero**-init → the all-block conditioning is actively used); injected token-delta for
  rig-A vs rig-B cy (542 vs 753) **2.7–7.5 per block** (conditioning strongly separates the rigs — exactly
  what the suffix-only warm-start ablation could NOT do).
- **Progress:** step **10,300 / 40,000** (~26 %), ~2 s/step, **zero crashes** (supervisor attempt 1),
  GPU 6.7 GB/46 GB. **On track to 40k (~16 h remaining).**

## When complete (integration escalation)

Final encoder → `pod3:/workspace/experiments/dynenc-branchB/ckpt.pt`. On completion: md5, push to a
gated `Sayood/tanitad-dynenc-branchB` HF repo (pods can't be the only copy), stage a `RESULTS_branchB.md`
(final loss + a held-out-rig transfer eval — the real test of whether from-scratch all-block conditioning
recovered cross-rig, vs the ablation's −2.1), and register it. Held-out-rig eval reuses
`run_camcond_ablation.py`'s harness with this encoder in place of the warm-started one.
