# KNOWLEDGE_BASE — Data Engineering

> Curated, deduplicated, newest first. Format:
> `[YYYY-MM-DD] [source] finding (1-3 lines) — impact: H_x / WP_y — link`

- [2026-07-14] [loader/license] **Cosmos-Drive-Dreams** (`nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams`)
  = **CC-BY-4.0** → the one *publicly-claimable* rich AV corpus (closes the gap left by the real
  PhysicalAI-AV exclusion). RDS-HQ: 5 843 clips + 81 802 synth videos, 7 weathers, 30 fps, per-frame
  4×4 `vehicle_pose`; front_wide_120fov = same 120° HFOV as PhysicalAI (D-016 focal reuse). Loader
  ships (intake pkg, 9 tests): derives steer/accel from geometry (`κ=yaw_rate/v`, low-speed clip),
  D-015 9-ch, `CORPUS_META` byte-identical to comma2k19 (D-017 I7 → admissible in the D-010 mix) —
  impact: D-014/D-002/H7/H4 — `2026-07-14-cosmos-drive-dreams-loader-and-landscape.md`
- [2026-07-14] [doc] `DATASET_LANDSCAPE.md` created (D-012 standing duty, was missing): 3 tiers, per-corpus
  license class / size / actions / urban-richness / cost-to-first-batch. Firewall: public numbers =
  comma2k19 + Cosmos-DD only. Next: verify WorldModel-Synthetic-Scenarios card; add Zenseact ZOD (real-CAN
  #2, H4 arm-B) — impact: D-012/G-D1 — `DATASET_LANDSCAPE.md`
- [2026-07-14] [arXiv] H7 latent-action/IDM surge: **LAWM** (2509.18428, latent actions from unlabeled
  video via world modeling → the labeled-bridge our comma2k19 IDM serves), **Drive-JEPA** (2601.22032,
  V-JEPA latent WM for E2E driving → "world model" no longer differentiates; moat = hierarchy+efficiency+
  imagination+self-monitoring), **HiLAM** (2603.05815, hierarchy×latent-action), **CLAW**/**DeFI**
  (label-free forward/inverse dynamics → flow/forward-consistency term for the IDM). External support
  only, no status upgrade (P8) — impact: H7/H3/H1 — same note §5
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
