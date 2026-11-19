# MM reading: DSCALE10K two-seed read, UM against UE at seed 0 and seed 1 (stage A), 2026-11-19

**Inputs**
- Card: `card_mix_um` with its amendments. A2 is `reps1/AMENDMENT_UM_A2.md`, which carries the two-seed rule, frozen before any UE_s1 or UM_s1 number.
- Reader: `ds_read_reps1.py`, 8 tests.
- Result: `read/RESULT_REPS1.json` (sha256 708f943f) and `read/RESULT_REPS1.md`. Floors are in `read/FLOORS_REPS1.json` (sha256 1e949bf2).
- Problems 0, blocked 0, unread 0.

**Run**
- All eight stage-A arms were re-run in one environment: the LoRA pod, 2 threads, read 13:26Z to 13:40Z.
- K-ENV passes: 238 seed-0 cells match the banked dev-box rows within tolerance.

**Evidence class, tier and estimator**
- MEASURED (ours).
- LONGITUDINAL / LATERAL / TACTICAL are **T1-kin**: self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene.
- STRATEGIC is T0 and is reported in full. Fog is unscored.
- Estimator: paired episode-cluster bootstrap, B 2,000, seed 0, over 141 episodes / 2,399 windows. It answers "another draw of EPISODES" only. Training-run variance is carried by the floors, and the planner is deterministic.

## Outcome as committed: DILUTION-ONE-SEED, worded "lever plus commit"; UM_s0 and UM_s1 BLOCKED (reach)

| headline TAC-LAT accuracy | delta [95 % CI] | separated |
|---|---|---|
| UM_s0 - UE_s0 | +0.0692 [+0.0283, +0.1092] | yes (below F2) |
| UM_s1 - UE_s1 | -0.0267 [-0.0609, +0.0063] | no |
| UE_s1 - UE_s0 (the union replicate) | +0.0342 [-0.0021, +0.0683] | no |

**Floors**
- F_cell = 0.0767: the card's B pair, B_rep - B_s0.
- F_U = |UE_s1 - UE_s0| = 0.0342.
- **F2 = max(F_cell, F_U) = 0.0767.** Beside it, D2 = 0.0959; the mean of the two UM - UE deltas is +0.0213.

**Reading**
- UM - UE is separated better at exactly one seed. A2 commits that label as: "the seed-0 effect is then not replicated, and no lever effect is quotable."
- Prediction H (UM - UE in [+0.04, +0.10]) holds at seed 0 and does not hold at seed 1.

**Qualifiers that travel with every UM - UE row**
- **K-TREE FAIL (UM-A3 R1).** Three 20-step runs (A = tree cdbff971, B = tree_um 648d4ca3, C = A again) all differ, and A vs B (max rel 7.2e-7) lies inside the same-code repeat envelope (A vs C, 2.0e-6). This device is not bit-reproducible, so a commit difference cannot be separated from the lever. The rows read "lever plus commit". Record: `ktree/KTREE_RESULT.json` (sha256 e0156204).
- **BLOCKED (reach) at both seeds, as committed.** `tac_src_empty_batches` was 10 (s0) and 7 (s1) against the bar of 5. Exact sampler replays explain every empty batch as a ChunkedSampler leftover of 1 to 4 rows; no full batch was empty. The bar was kept as written (A2 section 3).

## Beside, no new label: the union itself costs TAC-LAT at both seeds

| beside | delta [95 % CI] | reading |
|---|---|---|
| UE_s0 - BL_s0 | -0.0855 [-0.1216, -0.0492] | separated worse, above F: COST |
| UE_s1 - BL_s0 | -0.0513 [-0.0808, -0.0246] | separated worse, below F |
| UM_s0 - BL_s0 | -0.0163 [-0.0433, +0.0121] | not separated |
| UM_s1 - BL_s0 | -0.0779 [-0.1131, -0.0446] | separated worse: COST |
| UM_s1 - UM_s0 | -0.0617 [-0.0947, -0.0292] | separated (seed spread of the mixture arm) |

- BL exists at seed 0 only, so the seed-1 rows against BL mix seed variance into the contrast.
- What the beside rows say: adding the 10k clips at equal weight (UE) lowers eval-141 TAC-LAT at both seeds. The mixture weighting (UM) brings it back to BL's level at seed 0 only.

## The four families (UM - UE, per seed, then the both-seeds rule)

| family | result |
|---|---|
| LONGITUDINAL | no two-seed cost or gain |
| LATERAL | no two-seed cost or gain |
| TACTICAL | no two-seed cost or gain |
| STRATEGIC (in full) | no two-seed cost or gain |

- **Seed-0-only cost cells:**
  - tactical 2-6 s accel MAE +0.2344 [+0.0234, +0.4352];
  - tactical 2-6 s headway error +0.2321 [+0.0272, +0.4571];
  - tactical 2-6 s curvature, heading and yaw-rate MAE;
  - S_CKPT under donor nav at 80 m.
- **Seed-1 cost cells:** none.
- STRAT-KEPT: seed 0 false, seed 1 true.
- Against BL: UE_s1 and UM_s1 read STRATEGIC gains (S2 skill at 80 m; UM_s1 also S1 position error at 80 m). UE_s1 - UE_s0 reads one STRATEGIC cost cell (S1 position error at 80 m), a seed spread.
- Full tables: `read/RESULT_REPS1.md`.

## Next (RULE ZERO)

1. **More seeds of UM / UE are not the lever.** At D2 = 0.0959 and F2 = 0.0767, a third seed buys confirmation, not a gain. The card's section 5.6 UM_rep (seed 2) keeps its role, and the MM does not recommend it first.
2. **The lever with the largest measured effect is the union's TAC-LAT cost** (UE - BL, separated at both seeds). The card's own stage B, a DAgger fine-tune of 3,000 steps from each arm's stage-A checkpoint, is the next experiment on UE and UM at s0 and s1.
   - It asks whether the labelled-set finish recovers TAC-LAT while keeping the union's longitudinal gains over B_s0. UM_s0 - B_s0 longitudinal is separated better on 6 of 10 cells (`RESULT_REPS1.md`).
   - Cost ESTIMATED from MEASURED block-2 walls (`b2_next/blockx/LAUNCH_X_s0.md`): on the union's windows, bank plus selfcheck plus mint took 6,738 s for U_s1 (985,048 windows), and stage B about 0.16 h. That is about 2.0 h per arm and about 8 pod-h for 4 arms, and it **needs a PI go**.
   - The coordinator's integration plan (D7, base = 648d4ca3 lineage) decides first whether the union base is carried into integration at all. Both questions were sent to the coordinator.
3. **Banked:** the full read output (T1 dumps of all eight arms, T0 mechanism runs, evidence) is on private HF `Sayood/tanitad-v8f-dscale10k-runs`, `reps1read/reps1read_outputs.tar.gz`, sha256 04c3e20a. The record is `read/BANK_REPS1READ.json`. The two seed-1 T1 files are landed beside this reading (`read/t1/`).
