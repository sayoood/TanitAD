# REF-C eval path in TanitEval — build + first CORE eval

**Date:** 2026-07-19 · **Author:** tools-devenv agent · **Greenlit:** Sayed
**Arm:** REF-C-XL (Anchored-Diffusion-C, ~252M) · **Ckpt:** step 16000 (mid-training snapshot)
**Verdict:** REF-C eval path landed; REF-C-XL @16k = **0.565 m ADE@2s**, beats the CV floor, lands 2nd of the trained-encoder arms (behind flagship-30k, ahead of REF-B v2).

---

## 1. What was wrong / the gap

REF-C was **train-only**: `scripts/refc_train.py` + `tanitad/refs/refc.py` existed, but
TanitEval (on `tanitad-eval`) had no way to load or decode a REF-C checkpoint — the
eval pod's stack didn't even carry `refc.py`, and `taniteval` had no `refc` arch.
REF-C decodes trajectories with its **own** DiffusionDrive-style anchored decoder
(a fixed anchor vocabulary cross-attends the conv map → per-anchor confidence +
offset; truncated-diffusion steps refine the winning modes; the trajectory is the
argmax-confidence anchor path), **not** the operative grounded step-readout the
flagship/REF-A arms roll out. So it needed a **direct-head** eval path like REF-B's.

## 2. The eval path (files / lines)

All changes on `tanitad-eval` under `/root/taniteval/taniteval/` (originals backed up
`*.bak-refceval-20260719-214204`). The metric/GT/CV/strata code is reused verbatim —
only the per-arch **decode mechanism** is new.

| file | change | lines |
|---|---|---|
| `taniteval/refc_eval.py` | **NEW** — REF-C collector: window → `model(frames, nav_cmd=None, v0, steps)` → read the selected anchor trajectory at the shared `WP_STEPS` (5/10/15/20). Returns the same `pred/gt/cv/eid/speed/head_deg/wp_steps` dict `bench.run` consumes. Mirrors `refb_eval.py`. | 94 (new) |
| `taniteval/loaders.py` | **+refc arch branch** — rebuild `RefCModel` from the scale preset (`config_preset`) + the run's own `config.json` overrides (so every gated graft + the anchor buffer are at the trained shape), STRICT-load `ck["model"]`, `feed="frames"`, `step_readout=None`. | +23 |
| `taniteval/registry.py` | **+`refc-xl` entry** — `arch="refc"`, `config_preset="xl"`, `mode="diffusion"`, `ckpt=/root/models/refc-xl-snap/ckpt.pt`, `speed_input=True`. | +19 |
| `taniteval/runner.py` | **direct-head dispatch** — `direct_head = arch in ("refb","refc")` gates the profiling-only guard; `arch=="refc"` routes to `refc_eval.collect(... mode=...)`. | +11 |

Provisioning (also on `tanitad-eval`, additive / non-destructive):
- Copied `tanitad/refs/refc.py` (md5 `e289c0b2…`, **byte-identical** to pod3's trained
  version, and fully torch-only — no cross-module imports) into the eval stack
  `/root/TanitAD/stack/tanitad/refs/refc.py` (it was absent).
- Pulled **one** milestone ckpt read-only from pod3 → `/root/models/refc-xl-snap/`
  (`ckpt.pt` 3,024,021,445 B + `config.json`). pod3 training left untouched (see §6).

Validated end-to-end before the real run with a CPU smoke (tiny `refc_smoke_config`
model → `loaders.load(arch=refc)` → `refc_eval.collect` → `bench.run`), then the real
252M ckpt STRICT-loaded (673 tensors, anchors buffer `(256,4,2)`, **251.93M params**,
`refc1=False`, `diffusion_steps=2`, `window=8`).

## 3. CORE eval — ADE@2s (open-loop) on physicalai val

`python -m taniteval.runner run --model refc-xl --episodes 40`
Same harness as every arm: `physicalai-val-0c5f7dac3b11`, window 8 / stride 8,
**n=881 windows**, 8-split episode-disjoint jackknife (mean ± CI95), nav=follow.

**REF-C-XL @ step 16000: ADE@2s = 0.565 ± 0.045 m** (held-out) · de@1s 0.329 · FDE@2s 1.108 · miss@2m 0.149 · beats CV **True**.

### By-speed stratum (model vs CV, ade@2s)
| stratum | n | REF-C | CV | beats CV |
|---|---|---|---|---|
| low  | 294 | 0.708 | 0.932 | ✓ |
| med  | 293 | 0.586 | 0.934 | ✓ |
| high | 294 | 0.521 | 0.647 | ✓ |

REF-C beats CV in **every** speed tercile — strongest at high speed (0.521), weakest at low speed (0.708).

### By-curvature stratum (model vs CV, ade@2s)
| stratum | n | REF-C | CV |
|---|---|---|---|
| straight | 634 | 0.523 | **0.439** |
| gentle   | 125 | 0.785 | 1.357 |
| sharp    | 122 | 0.844 | 2.376 |

The edge is on **curves** (sharp 0.844 vs CV 2.376; gentle 0.785 vs CV 1.357). On
straights, constant-velocity is near-optimal and REF-C is slightly behind it
(0.523 vs 0.439) — the expected "CV wins the straight majority" pattern, same shape
the other trained arms show.

## 4. Where REF-C lands vs the other arms (identical harness, n=881, CV=0.825 m)

| arm | step | ADE@2s | beats CV |
|---|---|---|---|
| flagship-30k (v1 FINAL) | 29999 | **0.452** | ✓ |
| **REF-C-XL (this)** | **16000** | **0.565** | **✓** |
| REF-B v2 (30k FINAL) | 29999 | 0.592 | ✓ |
| flagship-speed (v1, 19k relay) | 19000 | 0.628 | ✓ |
| — CV floor — | — | 0.825 | — |
| REF-B v1 | 6000 | 0.868 | ✗ |
| REF-A DINOv2 4B | 29999 | 2.132 | ✗ |
| flagship no-speed (abl.) | 22000 | 2.918 | ✗ |
| REF-A dyn-in 4B | 29999 | 2.920 | ✗ |

**Reading:** REF-C-XL clusters with the **trained-encoder** arms (all beat CV) and
already lands **2nd** — behind only the flagship v1 FINAL (0.452 m @ 30k) and *ahead
of the REF-B v2 FINAL* (0.592 m @ 30k) and the flagship 19k relay — while itself only
**16k of 30k** (mid-training, likely to improve). It massively clears the
**frozen-encoder** REF-A arms (2.13–2.92 m), consistent with the frozen-DINO ceiling.
So the anchored-diffusion trajectory decoder on a trained ResNet-L trunk is a genuine,
CV-beating trajectory learner — currently between flagship and REF-B.

## 5. Assumptions / faithful-reading notes

- **Naming:** the task said "diffusion-XL … ~54.7M, DiffusionDrive-scale", but the
  **live pod3 run is `--config xl`** (`base_width 124`, `d=512`, 6 layers, 256 anchors,
  H15 imagination ON) = **~252M**, *not* the 54.7M `small`/DiffusionDrive-scale preset.
  I evaluated the checkpoint that actually exists on pod3 (the XL run) and labelled the
  registry entry accordingly. If a 54.7M `small` arm is later trained, the same
  `refc` path evaluates it by setting `config_preset="small"`.
- **Diffusion decode (the sampler reading):** faithful to `refc_train.compute_losses` —
  `steps = cfg.decoder.diffusion_steps (=2)` under `mode="diffusion"`, and the model's
  own `forward` selects the **argmax-confidence** anchor trajectory *after* the
  truncated-denoise refinement (`grounded_selector=False`, so score = confidence). At
  `model.eval()` the denoise noise is **zeroed** → deterministic, reproducible decode.
  The one eval-time choice is `nav_cmd=None` (the `follow` command), matching how REF-B
  is evaluated and how the model deploys without an external nav brain.
- **Frame comparability:** REF-C trains on `refb_labels.waypoint_targets`, whose ego
  frame (the `_ego` convention, relative to the last window pose) is **identical** to
  `gt_ego_waypoints`. With `refc1=False` the horizons ARE 0.5/1/1.5/2 s **time**
  waypoints (not fixed-distance path checkpoints), so the row is directly comparable.
  `refc_eval.collect` asserts `refc1==False` and `horizons==(5,10,15,20)` — a refc1
  ckpt is refused here (it needs a path/speed metric, not time-ADE).

## 6. Ops / hygiene

- **pod3 untouched:** the milestone was pulled read-only. To beat the slow dev-box
  relay (0.18 MB/s), the transfer went **direct pod3→eval** via SSH agent forwarding
  (datacenter-to-datacenter, 18 MB/s). No writes to pod3, no GPU load. Training stayed
  alive throughout (advanced 16150→16550, same PID). Atomic-write + open-fd semantics
  guarantee the pulled `ckpt.pt` is a consistent step-16000 snapshot.
- **Non-destructive:** nothing committed. Eval-pod originals backed up
  `*.bak-refceval-20260719-214204`; the added `refc.py` is a new file in the eval stack.
- **Re-run:** `python -m taniteval.runner run --model refc-xl --episodes 40`
  (also picked up by `run-all`). Result at `/root/taniteval/results/refc-xl.json`
  (+ `windows_refc-xl.pt` for A/B and viz).

## 7. Follow-ups (not done here)

- Re-snapshot at the 20k/30k milestone gates for the size-vs-data read; REF-C is still
  mid-training so 0.565 m is a floor, not the final number.
- The `imagination` / `hierarchy` / `pathspeed` panels are world-model-arm shaped
  (they need a grounded rollout / policy); REF-C's direct anchored-diffusion head is
  correctly skipped by their `traj_capable` guards — a REF-C-native anchor/mode-quality
  panel (anchor-selection accuracy, mode coverage) would be the natural next addition.
