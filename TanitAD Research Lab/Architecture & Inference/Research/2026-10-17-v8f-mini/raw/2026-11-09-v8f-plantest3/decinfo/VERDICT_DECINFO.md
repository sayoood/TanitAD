# VERDICT_DECINFO (Master Mind, 2026-11-09)
Card `PREREG_V8F_DECINFO.md` (`7e2e8aa6`) with its amendment 1 (in the card's Log). Tier **T0**, a readout diagnostic, never a driving result. MEASURED: `RESULT_DECINFO.md`, `runs/*.json`, `amend1/amend1.json`. No interval, as the card says; differences under 0.05 carry no weight. 53 fit episodes, 17 held-out episodes. LONGITUDINAL, LATERAL and TACTICAL have no metric here (a readout R2 only); STRATEGIC is PAUSED (PI item 25).

## Validity
The first run is **INVALID as run**: control 2 failed on the four-frame grid rungs (planted-signal R2 R4 s0 0.332, R4 s1 0.568, R6 0.743; bar 0.80). Its ridge reader had 5,120 features for 9,672 overlapping samples. Amendment 1, chosen after seeing that failure and the first-run values, gives the R4 and R6 ridge the top 8 PCA channels per frame (1,280 features). Control 2 and its bars did not move. Re-read: planted 0.912 / 0.907 / 0.838, zeroed -0.118 / -0.014 / -0.001, so control 2 passes. Controls 1, 3, 4, 5 passed (96 of 96 shuffled rows; 4 of 4 reproductions of the PLANTEST3 gate; 2 of 2 chain identities). The read below is valid under amendment 1, and the amendment changed no deciding value: conv was already the best reader on R4 and R6.

## The read: best held-out R2 over the rung's readers (optimistic, a maximum over a small fixed set)
| rung | speed (bar 0.70) | yaw rate (bar 0.50) |
|---|---|---|
| R5 encoder grid, 1 frame | 0.282 | 0.148 |
| R6 encoder grid, 4 frames | 0.355 | 0.291 |
| R3 adapter grid, 1 frame (s0 / s1) | 0.428 / 0.397 | 0.377 / 0.442 |
| R4 adapter grid, 4 frames | 0.450 / 0.464 | 0.503 / 0.418 |
| R2 pooled adapter token | 0.498 / 0.391 | 0.502 / 0.329 |
| R0 M2b z1 | 0.420 / 0.387 | 0.423 / 0.268 |
| R1 ladder z1, s0: C0, S35, R35, R35H | 0.480, 0.416, 0.412, 0.476 | 0.418, 0.527, 0.523, 0.488 |
| R1 ladder z1, R35_s1 (added after the freeze) | 0.445 | 0.253 |

## Outcome per target
* **Speed: NOT-READ-FROM-VISION.** Every rung is below 0.70. The curve is flat from the adapter grid to the final latent, so neither the mean pool (POOL-LOSS) nor the adapter (ADAPTER-LOSS) is where it goes at this n. This bounds these readers at 53 fit episodes and proves no absence.
* **Yaw rate: LADDER-READS by the report's per-checkpoint rule, and not confirmed.** R35_s0 (0.523) and S35_s0 (0.527) clear 0.50, each by less than the 0.05 that carries weight. R35's second seed reads 0.253, so R35 does not clear at every training seed it has. S35 has one seed so far; S35_s1 is the next ladder run. The same reader swings widely by seed on M2b as well (MLP, four frames: 0.423 / 0.001). The PLANTEST3 rerun this outcome triggers waits for S35_s1, because PLANTEST3 requires both seeds and speed is below its bar in any case.

## Next (the card's triggers, Rule Zero)
1. **Speed:** the probe at larger n. The PLANTEST3 decoder was fitted on these same 53 episodes, so if the read is sample-limited, that decoder was too, and PLANTEST3's NO-GAIN carries that limit. `PREREG_V8F_DECN.md` reads this first with 1,000 train clips streamed read-only (no GPU or RAM on box 2), before the full 4,572-clip job the card names.
2. **Every outcome:** the measured-ego card (`theory/egowm/PREREG_V8F_EGOWM_DRAFT.md`), to be frozen after the ladder's T0 read.
3. **Yaw rate:** re-read R1 when S35_s1 is banked (`launch_wmi.ps1` skips finished units).

## Added 2026-11-09 09:39: S35_s1 banked
R1 re-read with the same readers and 53 fit episodes (`runs/R1_S35_s1.json`; T0, no interval): speed 0.444, yaw rate 0.224 (best over readers; every shuffled row below 0). S35's second seed misses the yaw-rate bar of 0.50, as R35's did (0.253), while their first seeds read 0.527 and 0.523. So yaw rate clears on no ladder arm at both training seeds, and the PLANTEST3 rerun on S35 is not run. The speed read and every row above are unchanged.
