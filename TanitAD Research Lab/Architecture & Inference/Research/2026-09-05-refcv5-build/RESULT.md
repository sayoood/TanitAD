# refcv5 BUILD — environment extraction from the front camera, emitted-fan scoring, control-space diffusion

**STATUS: IN PROGRESS (started 2026-09-05).** This is a **BUILD**, not a study. The deliverable is
working code that makes refcv5 trainable. Every component ships with a test that fails without it and
a control that must read a known value.

**author:** Architecture & Inference FlyWheel · **branch:** `agent/arch-inf-20260803`
**PI instruction (verbatim, 2026-09-05):** *"I propose to prepare refcv5 for training, implement all
missing pieces. Let's follow environment extraction from the front camera and train it based on GT
data. If the whole refcv5 is driving and achieves good quality, we can extend it to multi cameras
(first step 4) then to Lidar."*
**Standing correction that governs this stream:** *"our goals in TanitAD programme is not to refute
hypotheses, it's about achieving excellent results and really driving autonomously with a reference
implementation."* ⇒ a clean refutation is not a deliverable; a fixed component is.

**GPU spent by this stream: 0 so far.** `refcv4b-b1-v72-40k` is LIVE on `tanitad-refcv3` — untouched.
The dev-box 4060 is running an RL arm; Thor is running refav1 Stage B. All work below is CPU.

## Component ledger

| # | component | file(s) | status |
|---|---|---|---|
| 0 | STATUS banked | this file | ☑ DONE |
| 1 | Monocular 3D agent detection head (GT = `obstacle.offline`), cylindrical projection | `stack/tanitad/models/agent_det.py`, `stack/tanitad/data/agent_targets.py` | ⬜ |
| 1b | Agent-token cross-attention in the REF-C decoder, zero-init gated | `stack/tanitad/refs/refc.py` | ⬜ |
| 1c | Agent-raster BEV aux target (v5a, no LiDAR) reusing `bev_raster.py` | `stack/tanitad/data/agent_targets.py` | ⬜ |
| 2 | Emitted-fan scorer with four-family heads | `stack/tanitad/refs/refc.py` | ⬜ |
| 3 | Control-space truncated diffusion (anchored Gaussian, DDIM, x0, per-layer AdaLN) | `stack/tanitad/refs/refc.py` | ⬜ |
| 4 | Trainer wiring, seams stamp, smoke run, launch argv | `stack/scripts/refc_v3_train.py` | ⬜ |
| 5 | Blockers: `ha0_ext` on the REF-C surface (M11 integrator), 12 ablation CLI flags | tbd | ⬜ |

## Log

- 2026-09-05 — stream opened; STATUS banked before any code (eight agent generations died mid-work
  in 48 h, so the bank comes first).
