# PREREG_V8F_WMPOOL (DRAFT, not frozen): does a spatial pool of the adapter tokens into z1 beat the mean pool? (2026-11-10)

## Why
EGOWM (`egowm/PREREG_V8F_EGOWM.md`, frozen ddd6fc9b) gave the predictor the measured speed. MEASURED IA-0 skill at L1 (`egowm/read/results/arm_*.json`, `ladder_read/results/arm_R35H_s*.json`, field `passes[1].skill`): R35H 0.2594 / 0.2673; E-KIN 0.2664 / 0.2788 (+0.0070 / +0.0115); E-NOW 0.2586 / 0.2799 (-0.0008 / +0.0126). E-KIN s0 vs s1 alone differs by 0.0124 at L1, so the committed floor already covers the effect: NO-EFFECT (formal verdict awaits the seed-2 replicate and the planner read). Its committed next lever is DECINFO's: a spatial or attention pool in place of the mean pool.

**State plainly: the motivation for this lever on speed is weak.** DECINFO (`decinfo/VERDICT_DECINFO.md`): at 53 fit episodes every rung is below the 0.70 speed bar, and R3 (adapter grid, 0.428 / 0.397) is not clearly above the mean-pooled rungs; the POOL-LOSS outcome was not reached. DECN (`decinfo/decn/VERDICT_DECN.md`): at 1,053 fit episodes the mean-pooled adapter token reads speed 0.721 / 0.736 against z1 0.695 / 0.686, a gap under the 0.05 those cards treat as weightless; yaw rate 0.502 (s0) for the pooled token. The adapter grid rungs (R3, R4) were read only at 53 fit episodes, never at 1,053. The trigger is kept as committed; this card spends it, and it may end NO-EFFECT without surprise. The ladder WM reads only `adapt.mean(2)` (`v8f_mini.py` `z1` / `z1_from_mean` / `forward_mean`, lines ~890-929), so anything lost is lost at that mean.

## Question
On the ladder rig (frozen M2b adapter, readers and planner frozen, T0), does replacing the token mean into z1 by a learned spatial pool of the 4 x 10 adapter token grid raise IA-0 k-step skill over R35H (world model of record, PI 2026-11-09) by more than a replicate floor, and does the ladder planner harness then read PLANNABLE?

## Arms (world-model only; everything except the pool as R35H: strides 3 5, k 10 10 6, K = 3, ctx 3 3 3, 19,164 steps, the same 40,358 windows, list sha256 1e4fc4c4...)
| arm | z1 input | seeds |
|---|---|---|
| R35H (control, already trained) | token mean, projected | 0, 1 |
| **P-ATT** (proposed first) | attention pool: one learned query over the 40 grid tokens, then `proj` | 0, 1 |
| P-GRID (second variant) | learned per-position weights over a 2 x 5 pre-pooled grid, flattened, then `proj` | 0, 1 |
| P-ATT_rep | P-ATT flags, seed 2 on the s0 adapter (a same-seed replicate is bit-identical, `ladder_read/R35B_CKPT_IDENTITY.json`) | 2 |
If box-2 time allows only one variant, keep **P-ATT**. Reason: with a zero-initialised query it starts as the mean pool (equal weights) and departs only by gradient, it adds few parameters, and it keeps all 40 tokens, whereas P-GRID's 2 x 5 grid discards the detail DECINFO's grid rungs were pointing at. Runs: three with P-ATT only (s0, s1, rep s2), five with both.
Floor: per level, the max |skill difference| among P-ATT s0, s1, rep s2 (EGOWM's rule). An effect counts only if it exceeds the floor of its level at both seeds.

## Read (EGOWM's read verbatim, T0)
IA-0 skill against Copy-Last at the band ends (L1 2 s, L2 6 s, L3 12 s) by IMAG1B's two-part rule, plus IA-0 health. Planner: `ladder_plantest` harness, forms c1 and c3, FLAT1 latent cost, G-TRUE minus HOLD and minus G-RAND, LONGITUDINAL and LATERAL, paired episode-cluster bootstrap B 2,000 (estimator named; it answers "another draw of episodes", training variance is read against the replicate floor). TACTICAL n = 0 (no decision output). STRATEGIC PAUSED (PI item 25), never a pass. Beside, no bar: IA-3 lateral gain at L1 2 s, WMACT action sensitivity, speed-probe R2 of the rolled-out latent.

## Committed outcomes (before any run)
* **POOL-HELPS:** skill over R35H exceeds the floor at the L1 band end at both seeds AND the planner read is PLANNABLE (separated-better than HOLD and G-RAND) in at least one family. Next: the pool arm becomes the world model of record and HWM-PLAN is read on it.
* **SKILL-ONLY:** skill exceeds the floor, planner stays NO-GAIN. Next: the predicted goal path (HWM-PLAN G4, G5), with the pool kept in the record.
* **NO-EFFECT:** skill does not exceed the floor. The mean pool is not the limit. Next (Rule Zero): DECINFO's ADAPTER-LOSS lever (unfrozen adapter, or a speed and yaw-rate auxiliary loss on it), and the full-n probe of DECINFO's NOT-READ-FROM-VISION branch on box 2.

**What NO-EFFECT says about the world-model stream:** the planner reads PLANTEST (`plantest/RESULT_PLANTEST.json`), PLANTEST2 (`plantest2/RESULT_PLANTEST2.md`), PLANTEST3 (`plantest3/VERDICT_PLANTEST3.md`), PT3N (`plantest3/n1053/VERDICT_PT3N.md`), PT3M (`plantest3/mlp4/VERDICT_PT3M.md`) and PT3F (`plantest3/ceilf/VERDICT_PT3F.md`) were all NO-GAIN (as stated in the task brief; PLANTEST, PLANTEST2 and PT3M/PT3F verdict text was not re-read for this draft, UNVERIFIED). With EGOWM (speed input) and this card (pool) both empty, two predictor-input levers sit below the floor, and the limit moves to the frozen vision front end and to the goal, not the predictor's input.

## Controls (the read is INVALID if one fails; each has a deliberate-regression arm that must go red)
1. **Off identity:** pool lever off reproduces R35H bit for bit (built model and one trainer step on a fixed batch). P-ATT's zero-init query reproduces the mean pool: z1 equals `wm.proj(mean)` on a fixed batch (implementer states bit-exact or 1e-6). Mutation: a non-zero query init goes red.
2. **Reachability:** non-zero gradient on every new pool parameter from the actual trainer path (argparse -> namespace -> `WMAux` -> `z1` -> loss, `file:line` at each hop), and a parameter outside the graph reads exactly 0.0. Mutation: deleting the pool line goes red. The flag's argparse DEFAULT must not disable it, and the flag is named in argv.
3. **Cache identity:** the mean of the cached grid equals today's `zmean` on 8 clips (form of `ladder/ZCACHE_CHECK12.json`); fp16 storage error stated.
4. **Echo guard:** z1 targets and goal latents stay a function of vision only: identical with the lever on and off on a fixed batch; no speed or action channel enters the pool.
5. **Launch:** every lever named in argv and read back from the BUILT model (G-DVB form); per-run token route.
6. IMAG1B's controls 2, 3 and 4 on the read.

## Feasibility (commands and outputs: `FEASIBILITY_WMPOOL.md`)
Grid per frame [40, 256] (MEASURED, `v8f_mini.py:322-326`). Today's mean cache: 0.478 GB per seed (MEASURED, box 2). Full-grid fp16: 9.45 GB per seed, 18.9 GB for both seeds (ESTIMATED). Box-2 D: has 11.23 GB free (MEASURED today): both seeds do NOT fit; a 2 x 5 pre-pool at fp16 is 2.36 GB per seed (4.7 GB both) and fits. The dev box cannot build it (the raw feature cache lives only on box 2). Each run: 8,636-9,161 s of steps (MEASURED), about 3.3 h with start-up. Open risk: loader bandwidth at the full grid (119 MB per step against about 30 MB at 2 x 5). Recommendation: build the 2 x 5 cache first.

## Code changes needed (file, function; no code written)
* `theory/ladder/zcache.py` (`zmean_of`, build and manifest): write a grid (pooled or full) per clip, with the adapter sha.
* `stack/tanitad/data/v8f_wm_windows.py`: `_zmean`, `item`, `collate_wm`, `ok_windows` (return `zgrid[idx]`).
* `stack/tanitad/models/v8f_mini.py`: `V8FMiniConfig` (new lever `wm_pool`, default off), `WMAux.__init__` (pool module), a `z1_from_grid` beside `z1` / `z1_from_mean`, `forward_mean` and the batch dispatch near line 1360, a `built_pool` G-DVB reader.
* `stack/tanitad/models/v8f_beta.py` and `theory/ladder_plantest/lp_lib.py`: the planner and harness must feed the grid at inference.
* `stack/tanitad/models/v8f_mini_train.py`: argparse flag, gate refusal of a pool flag without a grid cache.
* `theory/ladder_read` and `theory/egowm/read` (`eval_speed.py` form): the IA-0 reader builds the pool arm's z1.
* `stack/scripts/launch_gate.py` G-DVB: an entry for the new flag.

## Open before freeze
P-ATT vs P-GRID; 40 tokens or 2 x 5 for the cache; whether the EGOWM replicate and planner read have landed (they gate the NO-EFFECT trigger); which box-2 drive is the NVMe (UNVERIFIED). Needs the Master Mind's integration and a box-2 go.

## Master Mind review (2026-11-10 23:25, before any WMPOOL code or run; decisions to apply at freeze)
1. **One variant: P-ATT on the 2 x 5 pre-pooled grid (10 tokens)**, not on all 40. The 40-token cache is 9.45 GB per seed (ESTIMATED) against 11.23 GB free on box-2 D: (MEASURED by the drafter) plus ~5 GB per run the chain's guard needs, so it does not fit. On the 2 x 5 grid each cell is the mean of a 2 x 2 block of the 4 x 10 grid, so the mean over the 10 cells equals the global token mean exactly, and the zero-initialised query still starts at the mean pool (control 1 holds as written). Cache 2.36 GB per seed (ESTIMATED). P-GRID is dropped. Three runs: P-ATT s0, s1, rep s2 (~10 h of box 2, ESTIMATED from EGOWM's run times).
2. Prior is low and the card says so (Why). It stays the committed trigger of EGOWM's NO-EFFECT. Box-2 slot: after STOPBAL, the PI's ordering call.
3. Freeze after VERDICT_EGOWM (replicate read + verdict_egowm_ia0.py), with the planner read reported beside. The planner half cannot change EGOWM's outcome (SPEED-HELPS and SKILL-ONLY both need skill above the floor).
4. Before freeze, re-read the six planner verdict files the Why cites (the drafter marked them UNVERIFIED) and quote each outcome line.
5. **Review item 4 done (2026-11-11 00:3x, MM, primaries re-read):** PLANTEST (`plantest/RESULT_PLANTEST_body.md`), PLANTEST2 (`plantest2/RESULT_PLANTEST2.md`), PLANTEST3 (`plantest3/VERDICT_PLANTEST3.md`), PT3N (`plantest3/n1053/VERDICT_PT3N.md`) and PT3M (`plantest3/mlp4/VERDICT_PT3M.md`) read **NO-GAIN**; **PT3F reads NOT-READ** in both families (`plantest3/ceilf/VERDICT_PT3F.md`, "Outcome: NOT-READ in both families"), not NO-GAIN. The Why's sentence "were all NO-GAIN" is replaced at freeze by: five NO-GAIN, PT3F NOT-READ.
