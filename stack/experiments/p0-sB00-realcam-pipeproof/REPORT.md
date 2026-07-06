# p0-sB00 — first real-camera run of TanitAD-4B-M (RTX 4060) — REPORT

**Purpose.** End-to-end proof of the REAL-data path (D-009): comma2k19 HEVC → 2-frame RGB stacks
@256 px → route-level splits → full 261 M model, all five losses, bf16 autocast, on the dev GPU.
6 segments (5 train routes / 1 val route), batch 2, 300 steps. NOT a learning run.

## What passed (mechanics — the actual purpose)

- Full real-camera pipeline runs: decode → contract → windows → 261 M forward/backward → records.
- bf16 + horizon-subset future encoding fit in 8 GB at batch 2 (was 22 GB naive fp32, see git log).
- **I2 = 9.5e-7 (pass, pinned numerics)** on the real-camera encoder; I3 route-level enforced.
- All losses engaged and finite; H15 imagination NLL trained (−0.10 → −2.23; negative is legitimate
  for heteroscedastic NLL).

## What failed honestly (learning signal) — Finding F-2: SigReg starvation at tiny batch

**I4 = 98.9** (imagined step far worse than persistence). Post-hoc checkpoint diagnosis:

| Probe | Value | Healthy target |
|---|---|---|
| step-to-step latent change / norm | 0.0066 | O(0.1+) in a driving scene |
| effective rank | 22.9 / 2048 | hundreds |
| per-dim std | 0.00097 | ~1.0 (LeJEPA isotropic Gaussian) |

Textbook scale/temporal collapse: the encoder maps consecutive frames to nearly identical latents, so
prediction loss falls (0.336 → 0.003) while nothing is learned — **the falling loss was an illusion
and I4 caught it**. Root cause: at batch 2 SigReg receives n = 32 latent samples per step; an
Epps–Pulley test on 32 points has no statistical power over a 2048-dim distribution. ALPS-4B's
validated operating point was 256-effective batch; LeJEPA assumes large n.

**Consequences (implemented):**
1. Training logs now emit live collapse-health rows: `erank`, `dim_std`, `step_ratio` every log step.
2. Loud warning when SigReg receives < 256 samples/step.
3. The A40 run (p0-sB01, batch 64 → n = 1024) is the first run whose learning signal counts.
   Gradient accumulation for small-VRAM machines: backlogged.

## Verdict

Real-data pipeline: **validated**. Learning at batch 2: **invalid by design, correctly flagged by I4**.
Proceed to p0-sB01 on the A40 unchanged except batch. Doctrine note: this is the second time in two
days an instrument row caught what the loss curve hid (F-1 numerics, F-2 collapse) — the rows earn
their keep.
