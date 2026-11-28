# MM reading: DSCALE10K block 2, the deciding read (2026-11-17 ~23:20Z)

Evidence class MEASURED (ours). Tiers: LONGITUDINAL / LATERAL / TACTICAL are T1-kin on 141 episodes / 2,399 windows; STRATEGIC is T0 at the window's first tick. Estimator: paired episode-cluster bootstrap B 2,000 seed 0 ("another draw of EPISODES"). The floor F per cell is the seed-0 replicate pair |B_rep - B_s0|. Card `card/CARD_DATASCALE_TRAIN.md` (H-V8F-DSCALE10K-1). Result `out/RESULT_DSCALE10K_B2.{json,md}`; problems none; K1 0.0; K-RE against block 1 passes on every pair.

## Outcome as committed: **DATA-HURTS**

At matched steps (19,164 + 3,000), training on the union (4,512 corpus + 10,000 new clips) instead of the 4,512-clip corpus gives the following at the deciding stage B:

| family | seed 0 (U_s0 - B_s0) | seed 1 (U_s1 - B_s1) |
|---|---|---|
| LONGITUDINAL | none | none |
| LATERAL | cost (operative yaw rate) | none |
| TACTICAL | **cost** (TAC-LAT -0.0504 [-0.0738, -0.0267], F 0.030) | **cost** (TAC-LON -0.0959 [-0.1379, -0.0567], F 0.0204) |
| STRATEGIC | none | gain (S2 skill@80 +0.1025 [+0.0458, +0.1615], F 0.0041) |

* **DATA-HURTS rule:** a family costs at both seeds. TACTICAL costs at both, but on a different cell at each seed. The reader takes "at both seeds" as the same family at both seeds (MM deviation 4, accepted before data).
* **The tactical cost is consistent across cells.** All four tactical points are negative at both seeds:
  * TAC-LAT -0.0504 / -0.0292 (the second is separated but inside F);
  * TAC-LON -0.0375 / -0.0959.
  * So the label does not hang on one noisy cell. The rig's same-seed noise (5 of 48 single cells, C0REP read) would not produce four negative points in one family.
* **Reach precondition:** REACHED (R_s0, T0), not UNIT-SENSITIVE.
* **T mechanism, beside (T0):** NOT T-DATA-LIMITED at both seeds (seed 0 r -3.14 %, not separated).
* **H-V8F-DSCALE10K-1 is FAILED as written.** The claim was a gain in at least 2 families at both seeds with no family cost. The register row is updated in the same land.

## What the read points at (the card's DATA-HURTS row: diagnose dilution)

* The 10,000 new clips carry geometry labels, but their **VLM goal bits are masked** (PI Q2, 2026-11-16). About 69 % of the union's windows therefore have no VLM tactical goal supervision.
* The family that costs at both seeds is TACTICAL, the one that supervision feeds. The LONGITUDINAL family, fed by geometry labels on every clip, does not cost. This is a HYPOTHESIS, and the card's diagnostics test it.
* **Block 3** (UE_s0 / BL_s0 at 4x S_U steps, training on the pod now) separates "more data" from "more epochs per corpus clip". Its read lands after about 06:00Z (ESTIMATED).

## Next (Rule Zero; the card's DATA-HURTS row)

1. **Started now, CPU only:** the per-goal-class F1 (VLM against geometry bits) and the eval-141 read per country and platform, both on the banked dumps.
2. **X_s0** (the 10k clips only, 19,164 + 3,000 steps). It is in the card as the conditional arm for DATA-HURTS, but it was not priced in Q1 (blocks 0-3). Its chain is built and staged now, and it is **asked of the PI** (about 1.5 pod-h, ESTIMATED) to run after NAVRB2.
3. **The mixture arm** (corpus clips drawn at B's rate) needs a sampler-weight flag with its G-DVB entry, so it goes in a card amendment. The amendment is written after the block-3 read, which may change it.
4. **The VLM-CoT pass on the 10k clips** is the direct lever if the F1 read confirms the dilution. It is costed in the data-scale package, and it needs the PI.
