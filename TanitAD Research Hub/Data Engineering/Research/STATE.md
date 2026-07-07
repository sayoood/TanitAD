# STATE — Data Engineering

LAST_RUN: 2026-07-14 (W2, Tuesday agent)
QUALITY: full (G-A…G-F, G-D1, G-D2 met; pkg 9✓, stack 73✓/1 skip)

## This run (2026-07-14)
- **Cosmos-Drive-Dreams loader shipped** (intake pkg `2026-07-14-cosmos-drive-dreams-loader/`, proposed
  `stack/tanitad/data/cosmos_drive.py`). Rationale: the license review (D-002) excluded real PhysicalAI-AV
  from public claims; **Cosmos-DD is CC-BY-4.0** → the one publicly-claimable *rich* AV corpus, and D-014
  named it the training-mix synthetic arm — but it had no loader. **9 tests pass; stack 73✓/1s.**
  - Novel vs prior loaders: no CAN → `poses_to_signals(veh_to_world[N,4,4], dt)` derives steer (bicycle
    `κ=yaw_rate/v`, low-speed-clipped), accel, yaw, v from geometry. D-015 9-ch, D-016 focal (120°=same
    as PhysicalAI), `CORPUS_META` byte-identical to comma2k19 (D-017 I7) → admissible in the D-010 mix
    (`test_admissible_in_mix`). CLIP-level split (I3); per-weather distinct episode_id.
  - P8: pose glob + FLU/OpenCV axis order asserted-by-doc, pod-verified via `verify_real_clip()` before
    any trained claim; synthetic ≠ off-expert rollouts (that stays with CARLA-on-pod, D-014).
- **`DATASET_LANDSCAPE.md` created** (D-012 standing duty, was missing): 3 tiers, per-corpus license
  class / size / actions / urban-richness / cost-to-first-batch (G-D1). Firewall: public numbers =
  comma2k19 + Cosmos-DD.
- **Lit sweep (H7):** LAWM / Drive-JEPA / HiLAM / CLAW / DeFI — external support, no status upgrade (P8).
  Note: `2026-07-14-cosmos-drive-dreams-loader-and-landscape.md`; ledger H7/H4 changelog row added.

## Next (backlog, priority order)
1. **Steering-ratio calibration log** (H7 artifact) — per-segment residual when seed IDM trains on comma2k19.
   Still the binding H7 evidence; produce on real Chunk_1.
2. **Mirror the loader for PhysicalAI-WorldModel-Synthetic-Scenarios** once its HF card license is verified
   (near-zero cost — shares pose/contract/decode code). Adds emergency/pedestrian/weather_degradation
   long-tail (H6/H15/D9).
3. **On the Linux A40 pod:** (a) run `cosmos_drive.verify_real_clip()` on 3 downloaded clips → settle
   axis order + A8; (b) pull+unzip comma2k19 Chunk_1 (8.7 GB), smoke `build_episode` on 3 segments.
4. **Run the A8 harness on real Chunk_1** → set change-weight schedule + D-010 real-vs-sim from real distros.
5. **Add Zenseact ZOD** as real-CAN #2 (H4 arm-B, EU/night distribution comma2k19 lacks).
6. Focal-canonicalization prototype (backlog #3) — deprioritized until heterogeneous video (GoPro/BDD/OpenDV).

## HANDOFF
None — all work committed and pushed; pkg + stack suites green. No real bytes touched (unattended run);
`verify_real_clip()` is the pod entry point for real-data validation.
