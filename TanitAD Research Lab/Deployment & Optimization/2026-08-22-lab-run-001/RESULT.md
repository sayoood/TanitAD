# LAB RUN 001 — Deployment & Optimization — literature pass + ideation

`TanitAD Research Lab, daily run 001. Run label 2026-08-22 (Master Mind trigger);
executed 2026-08-23 wall-clock. Literature only, no GPU. Agenda targets:
RESEARCH_AGENDA.md Field 3, items 1 (quantization sweep, paired-eval rule) and 2
(compilation/kernels on Thor).`

## Findings

### F3.1 — Component-wise quantization on Jetson: structure, not size, sets sensitivity ⭐ banked
- **Rethinking Small VLM Quantization: From Component-Wise Analysis to
  Hardware-Aware Edge Deployment** — arXiv **2607.08029** (9 Jul 2026).
  **PUBLISHED, BANKED (library key `2607.08029`, tag `edge-quantization`)**.
- **Finding:** for sub-3B VLMs split into vision encoder / projector / LLM and
  swept over six INT4/INT8 configurations on Jetson Orin NX and AGX:
  (1) quantization sensitivity is governed by architectural paradigm (MoE vs
  dense), not parameter count; (2) SigLIP-class encoders take a
  disproportionate INT8 latency hit on Ampere Jetson; (3) **INT4 cuts memory but
  SLOWS generation** (dequantisation overhead); (4) per-component errors compose
  **additively, except along the modality-alignment path**; (5) energy
  efficiency diverges across platforms with memory bandwidth.
- **Impact:** this is the method template TanitDeploy needs for agenda item 1 —
  a per-component, per-hardware sweep with a paired accuracy read — and its
  non-additivity result has a direct analogue in our stack: the hierarchy
  **seams** (ctx→tactical etc.) are our "modality-alignment paths".
- **What it would change:** TanitDeploy profiles per component AND per seam,
  never whole-model; INT4 is evaluated on measured latency, not assumed faster.

### F3.2 — FP8 is not a free win for inference (the counterweight to 2026 FP8 hype)
- **FP8 versus INT8 for efficient deep learning inference** — arXiv
  **2303.17951** (Qualcomm, 2023; kept as the strongest measured counterweight).
  **PUBLISHED (abstract verified); NOT banked — secondary.**
- **Finding:** FP8 may suit training, but for inference the results "do not
  warrant a dedicated implementation of FP8 in favor of INT8"; FP formats are
  **50–180 % less compute-efficient in dedicated hardware** than INT.
  ⚠️ A widely-quoted "E4M3 covers 92.64 % of workloads vs 65.87 % for INT8"
  figure surfaced in today's search is NOT from this paper and was not verified
  against a primary — it is not quoted here.
- **Impact:** Thor's FP8/FP4 paths are an experiment, not a default; the
  paired-eval + measured-latency rule decides.
- **What it would change:** TanitDeploy's quantization matrix leads with an INT8
  PTQ baseline and treats FP8/FP4 as arms against it.

### F3.3 — Practice note: INT4/FP4 tooling on Thor is immature (non-archival)
- NVIDIA developer-forum thread on INT4/FP4 quantization on Jetson Thor (2026).
  **Non-archival; not citable** — recorded only as a planning signal: expect
  TensorRT/ModelOpt friction on Thor for sub-INT8 formats; budget a tooling
  spike before any INT4 claim.

## Ideation — our own hypothesis

**H-DEPLOY-1 (OPEN, proposed).** *Quantization error in the 4B hierarchy is
seam-dominated.* Under per-component INT8 PTQ, the paired-eval delta of our
hierarchical world model will be dominated by layers adjacent to the hierarchy
seams (ctx→tactical, tactical→operative), not by the encoder or predictor
blocks — because seam activations carry the lowest-participation (most nearly
collapsed) directions, where uniform quantization noise is proportionally
largest relative to signal variance. Corollary: the higher-participation two-term
trunk (`champ30k`, val participation 6.499, register H-RANK-12) quantizes with
a SMALLER delta than the collapsed v6 trunk (5.55 @20k train-pooled, H-RANK-2)
— collapse is also a deployment cost.

- **Cheapest discriminating experiment:** per-component INT8 PTQ of the v7-tiny
  champion checkpoint on the dev-box RTX 4060 (no Thor load): quantize one
  component at a time, read EM-vs-HOLD, val participation and decodability
  deltas (T0 WM diagnostic, stamped as such — not driving performance), paired
  on the same windows. ≈ 1 GPU-hour; run 002+.
- **Outcome A (seam-adjacent layers dominate):** TanitDeploy gets a seam-aware
  mixed-precision policy (keep seams at higher precision) and collapse
  statistics become a deployment-readiness signal.
- **Outcome B (uniform or encoder-dominated):** standard per-block sensitivity
  suffices; the corollary is dropped and the sweep proceeds whole-stack.

## Transfer note (charter §7)

→ **TanitAD_DeployFlyWheel:** adopt F3.1's component-wise protocol as the
TanitDeploy sweep template (per component, per seam, paired eval, measured
latency with admissible probes only — on Thor `torch.cuda.max_memory_allocated()`
alone). INT8-first ordering per F3.2. H-DEPLOY-1 offered as the first sweep's
pre-registered hypothesis. Accept/reject with reason requested.

## Evidence discipline

Paper numbers PUBLISHED from verified abstracts; the banked key is the only
registry-admissible citation; programme participation numbers are INHERITED
from the register/brief.
