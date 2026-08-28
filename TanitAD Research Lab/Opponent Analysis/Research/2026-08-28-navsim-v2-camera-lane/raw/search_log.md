# Search log — 2026-08-28 Opponent Analysis / Benchmarks & Evals pass

Method: primary-first — the reference model's own banked PDF read before any
web search; every quoted number extracted from a banked PDF via PyMuPDF.

## Local primaries read
- lib `2601.22032` (Drive-JEPA): Tab. 1 (perception-free: LAW 83.8/21M,
  World4Drive 85.1/21M, Epona 86.1/1.1B, ours 89.0/307M/330h), Tab. 2 (v1
  navtest PDMS rows incl. iPad 91.7, DriveSuprim 93.5, DrivR 93.1, ours 93.7;
  C&L rows Transfuser 84.0, HydraMDP 86.5, HydraMDP++ 86.6, DiffusionDrive
  88.1), Tab. 3 (v2 EPDMS papers-protocol: Transfuser 76.7, HydraMDP++ 81.4,
  DriveSuprim 87.1(web-corroborated), ours 87.8), Tab. 6 (encoder ablation:
  DINOv2 76.1, SigLIP 83.4, V-JEPA2 86.1, ours 89.0), ego-status input passage.
  "navhard" regex: 0 hits (absence, probe 1); table scan: no navhard block
  (probe 2).
- lib `2601.05083` (DrivoR): "roughly 40M parameters"; register overhead 0.6M,
  250× fewer tokens, "nearly reaches the performances of the no-compression
  model"; ego-status-into-trajectory-queries passage; navhard-two-stage Tab. 3
  incl. post-bug-fix note + GTRS-Dense exclusion; warmup-two-stage validation
  provenance; per-stage sub-metric columns (NC DAC DDC TLC EP TTC LK HC EC).
- lib `2606.07170` (TOAD): v1 table (PDM-Closed 89.1, human 94.8, ZTRS 86.9,
  GTRS 90.4, Hydra-MDP 90.9, iPad 91.7, RAP-DINO 93.8, DrivoR 94.6, +TOAD
  94.7); navhard-two-stage (DrivoR+TOAD 56.3, DriveFuture 55.5, PDM-Closed
  56.6 privileged, iPad 34.7→49.8, RAP-DINO 39.6, GTRS-D 45.0); +3.1 % DrivoR
  gain / +43.6 % iPad gain.

## Queries run
1. `NAVSIM v2 navhard leaderboard 2026 EPDMS camera-only entries GTRS DriveSuprim Centaur latest state of the art`
   → HF spaces AGC2025 (navhard + 2025 challenge), navsim GitHub, DriveSuprim
   AAAI, TOAD (2606.07170, banked), Driving-on-Registers CVPR26 lead.
2. `"Driving on Registers" CVPR 2026 camera-only NAVSIM EPDMS PDMS parameters ViT registers Kirby`
   → arXiv 2601.05083 (banked); valeoai/DrivoR repo; TOAD follow-up confirmed.

## Deliberately NOT quoted
- DrivoR "40M" scope ambiguity (whether the frozen pretrained ViT is counted)
  — quoted verbatim with the ambiguity visible, not resolved by guessing.
- DriveFuture params/inputs, GTRS LiDAR usage — UNVERIFIED, marked as such.
- HF-space leaderboard live rows — not fetched this pass (auth/dynamic page);
  the navhard numbers used are from the two banked 2026 papers instead.

## Papers banked this pass (Opponent Analysis)
2606.07170 (TOAD) · 2601.05083 (DrivoR). Already banked and cited:
2601.22032 (Drive-JEPA), 2406.15349 (NAVSIM), 2506.04218 (pseudo-simulation),
2312.03031 (Is Ego Status All You Need), 2506.06659 (DriveSuprim).
