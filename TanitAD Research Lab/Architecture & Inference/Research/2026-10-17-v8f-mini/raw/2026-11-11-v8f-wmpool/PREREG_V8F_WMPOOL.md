# PREREG_V8F_WMPOOL: does an attention pool of the adapter tokens into z1 beat the token mean? (Master Mind, 2026-11-11, written before any WMPOOL code or run)

Drafted by a Sonnet agent (`PREREG_V8F_WMPOOL_DRAFT.md`, landed `32c7910f` with its feasibility file and the Master Mind's review); frozen here with the review's decisions applied. Every number carries its evidence class; every eval number its tier.

## Why
* **EGOWM read NO-EFFECT for both arms** (`egowm/VERDICT_EGOWM.md`, T0, MEASURED): measured speed given to the predictor moves IA-0 skill at L1 by -0.0008 to +0.0126 against a replicate floor of 0.0124, at no level above the floor at both seeds. Its committed next lever is DECINFO's: a spatial or attention pool in place of the mean pool.
* **The motivation for this lever on speed is weak, and this card says so before any run.** DECINFO (`decinfo/VERDICT_DECINFO.md`, 53 fit episodes): every rung below the 0.70 speed bar; the adapter grid R3 (0.428 / 0.397) is not clearly above the pooled rungs; POOL-LOSS was not reached. DECN (`decinfo/decn/VERDICT_DECN.md`, 1,053 fit episodes): the mean-pooled adapter token reads speed 0.721 / 0.736 against z1 0.695 / 0.686, a gap under the 0.05 those cards treat as weightless. The grid rungs R3 / R4 were never read at 1,053 episodes. The trigger is kept as committed, and NO-EFFECT is a likely outcome.
* The ladder world model reads only the token mean of the adapter output (`v8f_mini.py` `z1` / `z1_from_mean` / `forward_mean`, about lines 890-929; adapter output [40 tokens, 256] per frame, `v8f_mini.py:322-326`). Whatever spatial layout the predictor would need is lost at that mean. That, not speed, is what this arm tests: whether a layout-aware z1 predicts the scene better.
* The planner reads on this world-model stream so far: PLANTEST, PLANTEST2, PLANTEST3, PT3N and PT3M read NO-GAIN, and PT3F read NOT-READ (each verdict file re-read 2026-11-11; paths in the draft's review item 5).

## Question
On the ladder rig (frozen M2b adapter, readers and planner frozen, T0), does an attention pool over a 2 x 5 pre-pooled grid of the adapter tokens, in place of the token mean into z1, raise IA-0 k-step skill over R35H (world model of record, PI 2026-11-09) by more than a replicate floor, and does the ladder planner harness then read PLANNABLE?

## Arm (world-model only; everything except the pool as R35H: strides 3 5, k 10 10 6, K = 3, ctx 3 3 3, 19,164 steps, the same 40,358 windows)
| arm | z1 input | seeds |
|---|---|---|
| R35H (control, already trained) | token mean, then `proj` | 0, 1 |
| **P-ATT** | the 4 x 10 token grid mean-pooled to 2 x 5 cells (each cell the mean of a 2 x 2 block), one learned query attending over the 10 cells, then `proj` | 0, 1 |
| **P-ATT_rep** | P-ATT's flags at seed 2 on the s0 adapter, for the floor (a same-seed replicate is bit-identical on this rig, `ladder_read/R35B_CKPT_IDENTITY.json`) | 2 |
* The query is zero-initialised, so the attention weights start uniform. Because the 2 x 2 blocks are equal-sized, the uniform mean over the 10 cells equals the global token mean exactly, so P-ATT starts as R35H's z1 and departs only by gradient.
* Why not all 40 tokens: the 40-token fp16 cache is 9.45 GB per seed (ESTIMATED, `FEASIBILITY_WMPOOL.md`) against 11.23 GB free on box-2 D: (MEASURED by the drafter, 2026-11-10) plus about 5 GB per run the chain's guard needs. The 2 x 5 cache is 2.36 GB per seed (ESTIMATED).
* Three runs, ESTIMATED about 3.3 h each on box 2 (EGOWM's runs: 8,636-9,161 s of steps, MEASURED), about 10 h.
* The lever is named in each argv, read back from the BUILT model, and each run passes the per-run token route; default off.

## Read (EGOWM's read verbatim, T0)
* Deciding per level: IA-0 skill against Copy-Last at the band ends (L1 2.0 s, L2 6.0 s, L3 12 s) by IMAG1B's two-part rule, and IA-0 health.
* Floor per level: the largest |skill difference| among P-ATT s0, s1 and rep s2. An effect counts only if (P-ATT minus R35H, same seed) exceeds the floor of its level at both seeds 0 and 1.
* Deciding for planning: the `ladder_plantest` harness (PLANTEST and PLANTEST2 forms, latent cost) on FLAT1 at level 1, contexts 1 and 3 as labelled rows; G-TRUE (vision-only goal) minus HOLD and minus G-RAND; LONGITUDINAL and LATERAL; paired episode-cluster bootstrap over the harness's episodes, B 2,000 ("another draw of episodes"; training variance is the floor above). TACTICAL: n = 0, no decision output. STRATEGIC: PAUSED (PI item 25).
* Beside, no bar: the IA-3 lateral gain at L1 2.0 s, WMACT's action-sensitivity audit, the speed-probe R2 of a rolled-out latent, and the attention weights' entropy per level (how far the pool moved from uniform).

## Committed outcomes (before any run)
* **POOL-HELPS:** skill over R35H exceeds the floor at the L1 band end at both seeds, and the planner read is PLANNABLE (separated-better than HOLD and than G-RAND) in at least one family. Next: P-ATT becomes the world model of record, and HWM-PLAN is read on it.
* **SKILL-ONLY:** skill exceeds the floor at L1 at both seeds, and the planner stays NO-GAIN. Next: the predicted goal path (HWM-PLAN G4, G5), with P-ATT kept in the record.
* **NO-EFFECT:** skill does not exceed the floor. The mean pool is not the limit. Next: DECINFO's ADAPTER-LOSS lever (an unfrozen adapter, or a speed and yaw-rate auxiliary loss on it). Together with EGOWM this would leave two predictor-input levers below the floor; the remaining world-model levers then sit in the frozen vision front end and the goal, and any further world-model arm is reprioritised against the T-side speed levers before it gets box-2 time.

## Controls (the read is INVALID if one fails; each has a deliberate-regression arm that must go red)
1. **Off identity:** with the pool lever off, the built model and one trainer step on a fixed batch equal R35H bit for bit. With the lever on and the zero-initialised query, z1 equals `wm.proj(token mean)` on a fixed batch (the implementer states bit-exact or the measured float tolerance). Mutation: a non-zero query init goes red.
2. **Reachability:** a non-zero gradient on every new pool parameter from the actual trainer path (argparse -> namespace -> model -> z1 -> loss, `file:line` at each hop), and a parameter outside the graph reads exactly 0.0. Mutation: deleting the pool line goes red. The flag's default must not enable it silently, and the flag is named in each argv.
3. **Cache identity:** the mean over the cached 2 x 5 cells equals today's cached token mean on 8 clips (the form of `ladder/ZCACHE_CHECK12.json`); the fp16 storage error is stated.
4. **Echo guard:** z1 targets and goal latents stay a function of vision only: identical with the lever on and off on a fixed batch, and no speed or action channel enters the pool.
5. **Launch:** every lever named in argv and read back from the BUILT model (G-DVB form); per-run token route.
6. IMAG1B's controls 2, 3 and 4 on the read, as in the ladder.

## Code changes the lever needs (from the draft; no code exists at freeze)
`theory/ladder/zcache.py` (a 2 x 5 cell cache with the adapter sha), `stack/tanitad/data/v8f_wm_windows.py` (serve the cells), `stack/tanitad/models/v8f_mini.py` (config lever `wm_pool`, default off; the pool module; `z1_from_cells` beside `z1_from_mean`; a G-DVB reader), the trainer's flag and refusal without a cell cache, the planner harness and the IA-0 readers (build z1 from the cells), `launch_gate.py` G-DVB entry (by its owner; this card does not edit it).

## Log
* 2026-11-11 01:18:11 frozen: prefix 7520 bytes, sha256 01e4dd802e41105ae7ed162d5b2658537ebba86d1cf54674638be6eda5a7a014 (everything above this line). No WMPOOL code or run existed at this time; EGOWM's verdict (`egowm/VERDICT_EGOWM.md`) was written minutes before.
