# PREREG_V8F_PROOF addendum 6b: HIERWHERE-2, does the window-class pattern hold on a second banked pair (frozen 2026-11-13 01:26:47 dev-box clock, before its data)

HIERWHERE (addendum 6, read; hier/VERDICT_HIERWHERE.md) on DAG against FLAT-DAG: R3 MIXED (at seed 0 three quarters of the difference sits on windows where neither arm has the logged mode), R1 CRIT-CONCENTRATED (selection-critical windows -0.4795 [-0.6222, -0.3445] at seed 0 and -0.1589 [-0.3043, -0.0117] at seed 1; bulk -0.1847 and +0.0702). Its swap control h4 missed its written bar at seed 0. That partition was chosen after the comparison existed, so its pattern is a hypothesis until it is seen on other runs. This card tests it on the second matched pair the programme banked: DAG0 against FLAT-DAG-0 (the same two fine-tunes with the own-state swap off, --dagger-p 0). Same reader (hier/hierwhere.py, unchanged, imported), same thresholds, same estimator. Zero training, zero rollouts. Tier T1-kin on logged frames, a diagnostic, never driving performance. STRATEGIC: PAUSED (PI item 25).

## Data (banked, read-only)
E:/Projects/TanitAD-artifacts/v8f-dagger-t1/dump_DAG0_s{0,1}_m10_causal_heads and E:/Projects/TanitAD-artifacts/v8f-flatdag-t1/dump_FLATDAG0_s{0,1}_m10_causal_heads; 2,399 windows, 141 episodes, paired by window.

## Predictions (from the first pair, fixed before this pair is read)
* P1: on SELECTION-CRITICAL windows ADE6(DAG0) - ADE6(FLAT-DAG-0) is negative with the interval excluding 0, at both seeds.
* P2: the critical-minus-bulk difference is negative at both seeds, with the interval excluding 0 at one seed or more.
* P3: BOTH-MISS is the largest of the three terms in size at every seed where D is separated.
* P4: the differences on logged ACCEL windows and on logged LEFT windows are negative at both seeds.
## Outcomes
CRIT-REPLICATES if P1 and P2 hold; CRIT-PARTIAL if one of them holds, or both hold at one seed only; CRIT-NOT-REPLICATED otherwise. P3 and P4 are reported as held or not held beside it.

## Controls
* h1a: no banked DAG0 minus FLAT-DAG-0 difference exists, so the reproduction control runs on the banked pair that shares the flat dumps: the same reader on (DAG, FLAT-DAG-0) must give -0.4973 [-0.6075, -0.3874] at seed 0 and -0.2450 [-0.3553, -0.1316] at seed 1 (FLATDAG_READ.json) to 1e-4 m. The DAG0 dumps go through the identical loader as the DAG dumps; 2,399 paired windows are asserted.
* h2: the three terms add to D to 1e-9 m.
* h4c (the swap control in calibrated form; the single-seed form of addendum 6 fails about one time in five by construction, four intervals at 95 %): over 20 swap seeds, per training seed, SELECTION and WITHIN both include 0 for at least 15 of 20.
## Limits stated in advance
DAG0 and FLAT-DAG-0 start from the same pretrained parents as DAG and FLAT-DAG, so this is a second fine-tune run per seed and a recipe without the swap; it is not an independent pretrain and not a replicate arm with equal flags. It tests whether the pattern survives the fine-tune run and the swap. One run per arm per seed: a separated interval is necessary and not sufficient. Floors beside every number: F = 0.12075 m, tactical replicate floor 0.287 m.
