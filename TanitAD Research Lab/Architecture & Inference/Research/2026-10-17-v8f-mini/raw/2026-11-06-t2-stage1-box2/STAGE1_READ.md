# STAGE1_READ: the v8F T2 harness on box 2, stage 1 (Master Mind, 2026-11-06)

**Tier:** T2 re-perception rollouts on NuRec reconstructions. **PLUMBING ONLY:** these rollouts are never read for H1 or H2 of
`PREREG_T2_CRAWL.md` (frozen `2c4c6557`). Evidence class: MEASURED (ours), `pull_box2/pull/STAGE1_RESULT.json`.

## Run
* Package tar sha256 `1785337f...d3421c`, verified on box 2 by `certutil`; 472 files extracted. Launched 13:06:29, `STAGE1.done`
  13:16:51 (preflight 146 s, T5 59 s, gap 15 s, actor 8 s, plumbing 345 s). One process tree, safe mode.

## Gates
| gate | verdict | reading |
|---|---|---|
| preflight | PASS | gsplat 1.5.3, msgpack 1.2.3, torch 2.11.0+cu128, RTX 4060 (6.94 of 8 GiB free), 471 of 471 shipped files by sha256, DINOv3-L snapshot found |
| T5 determinism | PASS | C0_s0 at 7c72937c f0 22, rendered twice: 40 of 40 frame md5 equal, plans bit equal (max diff 0.0 m), executed track equal |
| actor falsifier | PASS | the lead draws at ticks 31 / 44 / 58 (82,279 / 43,960 / 26,490 changed px against `empty`); the planted `empty` read is absent; harness falsifier ACCEPTED on 3 of 3 frames |
| T3 zero control (real renders) | PASS | ZERO equals P10 exactly, all 19 rows, under T2 and RR |
| plumbing | PASS | 28 of 28 cells (2 scenes x 7 arms x {T2, RR}); dev-box scorer: 14 groups, X1 / X2 / BL status OK at both seeds |

## Render gap (beside, never deciding)
Pooled DINOv3 tokens of the frame rendered at the logged pose against the real camera frame, 9 ticks per scene:

| scene | cosine mean (min to max) | rel-L2 mean |
|---|---|---|
| 00040136 | 0.833 (0.793 to 0.858) | 0.567 |
| 7c72937c | 0.755 (0.708 to 0.824) | 0.683 |

The encoder's own floor on a logged frame is cosine 0.99965, rel-L2 0.0264. The gap is about 20 times the floor in rel-L2, which is why
the card makes X2 = C0(T2) - C0(RR), measured inside one renderer, the attribution contrast.

## Instrument item open before stage 2
Every arm, P10 included (P10 never reads the camera), scores heading MAE about 30 to 45 deg, yaw-rate MAE about 110 to 190 deg/s and
6 s ADE 4.6 to 7.8 m on these 2 starts, identically for P10 under T2 and RR. The level is set by the scenes, the starts or the scorer's
alignment of the logged future on NuRec scenes, not by vision. Both starts sit near standstill (7c72937c is a launch scene), where
heading is ill-defined; the stub rehearsal used straight constant-speed scenes and never exercised this. **Stage 2 waits on an audit:**
per-start breakdown, the alignment of rendered frame, executed tick and logged future, and the same rows computed by the T1 reader's
path on the same logged track.
