# KNOWLEDGE_BASE — Data Engineering

> Curated, deduplicated, newest first. Format:
> `[YYYY-MM-DD] [source] finding (1-3 lines) — impact: H_x / WP_y — link`

- [2026-07-07] [measured] comma2k19 loader (D-009) decode path validated on REAL bytes: `av` decodes
  real HEVC → [200,3,256,256] uint8 @ ~105 fps (py3.13/Win), stack→[199,6,256,256] — impact: D-009/H7 —
  see `2026-07-07-comma2k19-data-card.md` §5
- [2026-07-07] [license/D-002] PhysicalAI-AV **real** sets (-Vehicles/-NCore/-NuRec) = NVIDIA AV Dataset
  License: **internal-dev-only, confidential, 12-month expiry, no public claims** → EXCLUDED from public
  benchmarks (comma2k19/MIT stays the public corpus). Cosmos-Drive-Dreams = **CC-BY-4.0**, ungated → the
  one publicly-claimable AV asset. Internal use needs Sayed+NVIDIA-legal sign-off ("using NVIDIA tech")
  — impact: D-002, all public claims — `2026-07-07-physicalai-av-license-review.md`
- [2026-07-07] [tool+finding] A8 statistics harness shipped (`stack/tanitad/data/stats.py`, 6 tests):
  per-corpus/per-domain `frame_change_fraction` distribution for change-weighting + D-010 mix. Measured
  toy=0.046 (threshold-INsensitive, hard-edged) vs comma-real=0.053→0.012 @0.05→0.10 (threshold-sensitive:
  real change is mostly small gradient, ~1.2 % large) — change-weight the small-but-real residuals —
  impact: H3/A8, D-009, D-010 — `...-validation-and-h7.md` §4a
- [2026-07-07] [measured] A8 on REAL highway camera `frame_change_fraction`≈0.053@0.05 / 0.012@0.10 — only
  ~1.7× the toy floor; raw-RGB under-reads consequence on low-texture highway → change-weighted loss
  justified — impact: H3/A8, W2 bake-off — `2026-07-07-comma2k19-validation-and-h7.md` §4
- [2026-07-07] [measured] comma2k19 = **MIT**, ungated HF `commaai/comma2k19`, ~100 GB (10×8.73 GB chunks);
  real CAN steering (steering-WHEEL deg, ratio ~15.3) + GNSS poses, zero labels; first real batch ≈1–2
  engineer-h, 0 new code — impact: D-009/H7/G-D1 — data card
- [2026-07-07] [finding] Windows `|` path bug: comma2k19 route dirs (`dongle|date`) illegal on Win32 →
  extract/train on Linux A40 pod only; do NOT extract Chunk zips on the Windows dev box — impact: ops —
  data card §6
- [2026-07-07] [arXiv] H7 deltas: LAOF (optical-flow-consistent latent actions, label-sparse gains) +
  Sensorimotor World Models (IDM-as-perception) → add flow-consistency term to seed IDM; log
  steering-ratio calibration residual — impact: H7 — `...-validation-and-h7.md` §3
- [2026-07-05] [kickoff] Initial research baseline for all hypotheses established; discipline agenda
  seeds defined — impact: all — see `../../INITIAL_RESEARCH_SYNTHESIS.md`
