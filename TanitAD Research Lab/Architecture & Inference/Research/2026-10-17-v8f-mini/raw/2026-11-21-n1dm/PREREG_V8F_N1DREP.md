# PREREG_V8F_N1DREP: a same-code, same-seed replicate of N1D gives the operative guard cells a measured floor at the executor of record

Status: FROZEN 2026-11-22 (MM), before the replicate trains. PI go: "Run both" (2026-11-22, about 07:12Z; RL and N1DREP at the stated costs). This is N1DM-COST's committed next lever (`PREREG_V8F_N1DM.md` 958d143d, section 3; `MM_VERDICT_N1DM.md`). Draft: `PREREG_V8F_N1DREP_DRAFT.md` (landed with land11). Changes from the draft, all made before any N1DREP number exists: (a) code of record is N1D's own 0c3b374f, not 573ad839, so the replicate differs from N1D_s0 in nothing but `--out`; (b) the outcome table is made exhaustive (the draft had no row for exactly one of three cells inside the floor); (c) F2 is named as the guard's own per-cell floor with its file; (d) two controls (K-REP-CODE, K-REDUMP).

## 0. Why (MEASURED, quoted)
* N1DM's eval-141 guard (T1-kin, executor hb0.2, `RESULT_N1DM_EVAL141.json` sha256 ade101e6ac17164a, `guard_eval141_hb02_N1DM-N1D`) read GUARD-COST-BOTH-SEEDS on the operative 0-2 s cells, 2,399 windows, 141 episodes:

| cell | N1DM_s0 - N1D_s0 [CI] | N1DM_s1 - N1D_s1 [CI] | F2 (the guard's floor) |
|---|---|---|---|
| ade_m | +0.01223 [0.0061, 0.0212] | +0.01238 [0.0067, 0.0209] | 0.00047 |
| along_mae_m | +0.01232 [0.0061, 0.0217] | +0.01237 [0.0068, 0.0211] | 0.00053 |
| speed_mae_mps | +0.00836 [0.0033, 0.0155] | +0.00813 [0.0029, 0.0147] | 0.00306 |

* F2 comes from a same-seed replicate of C0, a different arm (no DAgger swap). No replicate of N1D exists at any executor.
* N1D's trainer runs the DAgger swap (`--dagger-p 0.1`). A same-seed run of that trainer is not bit-identical (memory, MEASURED on an earlier pair), so the replicate is expected to measure run-to-run training noise rather than read an identity. Inference is deterministic.

## 1. Arm
* **N1DREP_s0**: kind N1D, seed s0, rendered by the pod's `navrbd/render_argv.py` exactly as N1D_s0 was, minted and trained by `strat1_check.py` from the pod tree `/root/strat1/navrbd/tree` at **0c3b374f** (0c3b374fb1036b0ca4ca7620f98137dd6c59cca4), against N1D's own G-SUITE result (`gsuite_result_navrbd.json`, sha256 20f0279e... in N1D_s0's gate token). Init M2b_s0, 3,000 steps. Runs dir `/root/strat1/n1drep/runs`, so its argv differs from N1D_s0's in `--out` only.
* One seed. The floor is per cell F_REP = |mean N1DREP_s0 - mean N1D_s0| on the same 2,399 windows, applied at both seeds.

## 2. Controls (each refuses before training, or marks the read)
* **K-REP-CODE (pod, before training):** the minted argv equals N1D_s0's `argv.json` token for token except the value of `--out`; the new gate token's `commit`, `trainer_sha256`, `checker_sha256`, `tree_manifest_sha256`, `gsuite_result_sha256`, `nav_report_sha256`, `nav_rb_report_sha256` and `data` equal N1D_s0's. Any difference: the chain stops, no training.
* **Tree and inputs (pod):** the navrbd tree checks as N1D's chain did (`navrbd_tree.sha256`, blob check against `navrbd/all_blobs.txt`, file count), inputs sha256 as N1D's chain.
* **Post-training (pod):** `strat1_check.py post` and the zero-grad gate with the `e_nav_rb.weight` control, as N1D_s0.
* **K-REDUMP (dev box):** N1D_s0 is dumped again in the same session, tree and environment as N1DREP_s0, and must equal the banked `dumps141_hb02/dump_N1DM-N1D_s0_m10_causal_heads` array for array. If it does not, F_REP is read against the re-dump, and the gap is reported (memory: thread drift between environments).

## 3. Read (T1-kin, executor hb0.2, eval-141, 141 episodes)
* Dump: `n1dm_dumps.py N1DM-N1D_s0` (the N1DM read tree, N1DM_KAPPA=0.2), with `N1DM_CKPTS_N1D` pointing at a folder whose `N1D_s0/ckpt_last.pt` is N1DREP_s0's checkpoint (sha256 = the pod's), output to its own folder. Follow dump only.
* Cells and estimator as the N1DM guard: paired episode-cluster bootstrap, B 2,000, seed 0. The bootstrap answers "another draw of EPISODES"; the replicate answers "another TRAINING run" at fixed seed, init and code.
* F_REP is computed and reported on all 24 guard cells per pair (operative 0-2 s, tactical 2-6 s, TAC-LAT, TAC-LON); the outcome below reads the three operative cells. STRATEGIC: the guard carries no strategic cell on this line (N1DM's strategic read is the separate S1-slot view), so this card measures no strategic floor; named here with that reason. For each of the three cells c: floor_c = max(F2_c, F_REP_c); c **exceeds** when N1DM - N1D > floor_c at BOTH seeds (the deltas in section 0, unchanged).
* The verdict of record for N1DM stays N1DM-COST, as written.

## 4. Outcomes, committed in advance (checked in this order)
| name | condition | reading | next |
|---|---|---|---|
| **N1DREP-IDENTICAL** | every array of the N1DREP_s0 dump equals N1D_s0's | the trainer is deterministic at this seed; F_REP = 0 is an identity, not a floor | a second-seed replicate (N1DREP_s1 with a fresh dagger seed) is the floor: back to the PI |
| **N1DREP-COST-HOLDS** | all three cells exceed | the operative cost of N1DM is above run-to-run noise | F_REP becomes the floor of record for operative cells on the N1 line at hb0.2 |
| **N1DREP-COST-MIXED** | exactly two of the three cells exceed | the cost is above noise on two cells; the third is named | F_REP as above; N1DM stays not adopted |
| **N1DREP-COST-IN-NOISE** | at most one of the three cells exceeds | the cost is run-to-run noise; N1DM is still not adopted, because D1-HO shows no gain | F_REP as above; the decoder-path family reads no gain and no cost |
| **N1DREP-REFUSED** | a section 2 pod control fails | no read | fix the cause and re-run the same card, or back to the PI if the fix changes the arm |

## 5. Cost (ESTIMATED)
* Pod: N1D_s0 trained in 672 s (its log); with mint and checks about 0.3 pod-h, under `/root/gpu_queue.lock`, never beside a peer's training.
* Dev box: two eval-141 hb0.2 follow dumps (N1DREP_s0 and the K-REDUMP of N1D_s0, about 6 min each, CPU, one python at a time) and the read.
