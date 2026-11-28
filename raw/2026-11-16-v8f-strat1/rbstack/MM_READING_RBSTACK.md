# MM reading: RBSTACK (stacking-combiner check), 2026-11-19

Card `PREREG_RBSTACK.md` sha256 bf4d5395 (frozen before any RBSTACK code or number). MM rulings `MM_RULINGS_RBSTACK.md` (R1, R2), made ~00:40Z before any stage JSON existed.
Tier **T0 decodability, NOT driving performance**. Evidence class MEASURED (ours, `result/RESULT_RBSTACK.json`, `result/stage_s0.json`, `result/stage_s1.json`).
Estimator: paired episode-cluster bootstrap, B 2,000, seed 0, on per-window correctness. It answers "another draw of EPISODES" only; probe variance is carried by the two probe seeds.

## Reading: **STACK-WEAK at both probe seeds**

Population: RBJK, 3-class kinematic label, 9,236 windows / 395 episodes, majority rate 0.4075.

| contrast (R1: decision on EDV - ED_STR) | seed 0 | seed 1 | bar 0.0383 |
|---|---|---|---|
| **EDV - ED_STR** (decision) | **+0.0234 [+0.0029, +0.0445]**, separated | **+0.0276 [+0.0075, +0.0496]**, separated | below at both |
| EDV - ED (beside; the code's reading) | +0.0217 [+0.0012, +0.0434] | +0.0259 [+0.0049, +0.0484] | below at both |
| EDV_SHUF - ED_STR (K-SHUF) | -0.0107 [-0.0198, -0.0021] | -0.0116 [-0.0210, -0.0026] | below at both: PASS |
| EDV - V (text over vision, beside) | +0.0368 [+0.0080, +0.0633] | +0.0370 [+0.0088, +0.0638] | no bar |
| V - S (K-V) | +0.2180 [+0.1808, +0.2549] | +0.2220 [+0.1858, +0.2577] | PASS |

Arm accuracies: EDV 0.6623 / 0.6665, ED_STR 0.6389, ED 0.6406, V 0.6255 / 0.6295, EDV_SHUF 0.6282 / 0.6273, S 0.4075.

R1 and the code's own reading agree: STACK-WEAK at both seeds under either contrast, so ruling R1 did not change the outcome.

## Controls
* **K-REPRO PASS**: dense ED 0.6406453009961022 and S 0.4075357297531399 equal RESULT_RBJK exactly, on 9,236 windows.
* **K-NEST PASS**: the fit-set fold assertion holds. Three deliberate-regression mutants raise NestingError as required: plain out-of-fold probabilities, a probe that saw the test fold, and the pipeline fed the plain plan. See `result/MUTATIONS_RBSTACK.json`.
* **K-SHUF PASS**: shuffled probabilities add nothing and cost about 0.011. That fits a tree spending splits on noise.
* **K-V PASS**: the probe reads the frames, at +0.22 over the majority rate.

## Beside, no bar: 5-way v9 `lat_cls`, probe seed 0
Population 12,201 / 426. EDV - ED_STR -0.0041 [-0.0226, +0.0145] and EDV - ED -0.0065 [-0.0260, +0.0130], both not separated. On the 5-way label the vision probabilities add nothing over the route-text tree.

## Stage B (held-out D1-HO)
NOT-YET, as the card allows. The D1-HO windows' pooled 8x20 features and ED rows exist only in the pod-side cache, and the card forbids a pod pull for this check. Stage B never changes the section-3 reading.

## What this says
1. On the label the NAVRBD decision reads (3-class kinematic TAC-LAT), vision and route text are **complementary**. Each adds a separated amount over the other: vision over text +0.023 / +0.028, text over vision +0.037. Both additions are small.
2. The size of the vision addition is **below the 0.0383 bar at both seeds**. It is also close to RB-JOINT's RAW8 - S (+0.0266, separated, below the bar), so two independent T0 reads agree on the scale.
3. It is a T0 decodability ceiling for a probe on frozen features, not a driving number.

## Next (RULE ZERO): the vision lever is carded for the PI, with its expected size below the bar
The card's WEAK branch says the vision lever is carded and the PI is told its expected size is below the bar. The card follows.

**Arm N1D8** = the N1D recipe (exit + distance channel) plus `--grid pool8x20`.
* This is an existing `v8f_mini_train.py` flag (`choices=("pool4x10", "pool8x20")`, default `pool4x10`), so the arm needs no new model code.
* The pod's training cache already holds `pool8x20` per clip. MEASURED: shape (101, 160, 1024) fp8, beside `pool4x10` (101, 40, 1024), in `/root/data/cache_pod/train/clips/*.pt`.
* Seeds s0 and s1: 2 arms.
* The decision reads N1D8 - N1R on the D1-HO kinematic TAC-LAT, at the same bar 0.0424 (max(0.0383, F_HO)) and with the same guard. N1D8 - N1D is read beside it, to attribute the grid.

Expected size:
* The grid alone is ESTIMATED at about +0.02 to +0.03 at T0, below the bar (this reading).
* HYPOTHESIS: channel and grid add. N1D - N1R was +0.0291 (s0) and +0.0171 (s1). With at most about +0.025 from the grid, N1D8 - N1R would land around 0.04 to 0.05 at s0 and around 0.04 at s1, which is near the bar, not clearly over it. Additivity is not given: the stack shows complementarity at T0 on frozen features, not inside the trained model.

Cost:
* MEASURED: N1D_s0 trained 3,000 steps in 672 s on pool4x10 (pod `runs/N1D_s0/logs/train.log`, DONE line).
* ESTIMATED: pool8x20 carries 4x the visual tokens (160 against 40), so 2x to 4x the step cost, about 25 to 45 min per arm. With the gate, G-SUITE and the D1-HO dumps, that is about 1.5 to 2.5 LoRA pod-h for both arms.
* A 200-step pilot reads s/step and container memory first. A mean above 4x the pool4x10 rate goes back to the PI.

After N1D8, the next lever is the RB window weight in the training loss. It needs a new trainer flag with its own G-DVB entry, so it comes second.

The FLOOR-HIGH lever (more seeds of N1D / N1R, PI go) would confirm a +0.02 effect but cannot lift it over 0.0383. It is not recommended ahead of N1D8.

## Temp deleted (by the job, logged in `result/deletions.txt`)
* `F:/TanitAD-scratch/strat1/rbstack/out/feat/X8.npy`, 3,998,023,808 bytes. Re-creatable with `rbstack.py extract_x8`.
