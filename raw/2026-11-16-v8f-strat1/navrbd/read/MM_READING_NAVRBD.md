# MM reading: NAVRBD (the roundabout exit + distance channel), 2026-11-19

Card `PREREG_V8F_NAVRBD.md` sha256 881d7f3d (frozen before any N1D artifact). Arms N1D_s0 / N1D_s1 (commit 0c3b374f) and the deliberate-regression arm N1Dx_s0 (channel rows deranged across clips, commit 5d474c91), each against N1R at the same seed. Result `out/RESULT_NAVRBD.json` (FINAL, problems 0, one NOT-YET).

## Outcome as committed: RBD-NO-GAIN and RBD-FLOOR-HIGH
Both labels hold; the bar is reported as written (max(0.0383, F_HO) = 0.042364).

| D1-HO (TACTICAL, kinematic TAC-LAT accuracy; T1-kin; 5,736 windows / 497 held-out episodes) | delta [95 % CI] | separated | gain |
|---|---|---|---|
| N1D_s0 - N1R_s0 | +0.0291 [+0.0174, +0.0409] | yes | no (below 0.0424) |
| N1D_s1 - N1R_s1 | +0.0171 [+0.0044, +0.0300] | yes | no |
| N1Dx_s0 - N1R_s0 (deranged channel) | +0.0014 [-0.0089, +0.0113] | no | no |

Evidence class MEASURED (ours, artifact above). Estimator: paired episode-cluster bootstrap, B 2,000, seed 0, which answers "another draw of EPISODES" only. The training-run question is carried by F_HO = |N1R_s1 - N1R_s0| = 0.0424 (itself separated, [+0.026, +0.058]).

## What the numbers say
1. **The channel carries a real, small effect.** It is separated at both seeds; the deranged control is flat; Z-USE (real minus zero channel on the same windows) is separated at both seeds (+0.0464 [+0.0334, +0.0592], +0.0202 [+0.0064, +0.0331]). The arm's training rule alone does not produce it.
2. **It moves the SIDE of a turn, not its detection.** D2-HO beside: turn recall and false-turn share do not move (not separated at either seed). Side given turn rises 0.693 -> 0.749 (s0) and 0.783 -> 0.808 (s1). That matches RB-JOINT: the text reads which side better than vision.
3. **The size is below the programme margin at both seeds.** More seeds of the pair (the FLOOR-HIGH lever) can firm up the existence of the effect, but cannot lift a +0.017 to +0.029 effect over 0.0383. The larger lever is the effect size itself.

## The four families (eval-141 guard, T1-kin, 2,399 windows / 141 episodes; T0 for STRATEGIC)
* **TACTICAL:** D1-HO above. On the 235 eval-141 roundabout windows (16 episodes) D1 reads +0.0426 and +0.0298, not separated (thin, beside). TAC-LAT / TAC-LON on all 141 episodes not separated at either seed.
* **LONGITUDINAL:** one cost cell at s0 only, time-gap error 2-6 s +0.0316 s [+0.0061, +0.0617] (above F2); s1 +0.0015, not separated. So the guard does not hold under the card's either-seed rule. It is not RBD-COSTS, which needs both seeds. Accel MAE 0-2 s splits by seed (-0.0040 s0, +0.0075 s1, both separated, both inside F2). Speed, headway: not separated.
* **LATERAL:** no cost. s0 heading 2-6 s -0.297 deg separated and yaw rate 0-2 s -0.062 deg/s separated, both inside F2, so not gains. s1 FDE 2-6 s -0.181 m separated, inside F2.
* **STRATEGIC** (T0, real / donor / zero nav, arcs 30 / 50 / 80 m; STRATEGIC reported in full since 2026-11-16): ROUTE-COST none at either seed. Route-following level N1D - N1R is not separated at any arc at either seed (deltas 0.0 to -0.013). Route setting RC-A: s0 position error at 30 m -0.116 m and heading error at 80 m -1.18 deg, both separated gains above the RC-A floor; s1 none separated (all deltas negative). With the channel zeroed, N1D_s0 follows the route less than N1R (-0.025 to -0.053, separated), so it leans on the new channel. The context cells (country, weather, surface, illumination, period of day; fog unscored) are NOT-YET by construction: no S1D arm was carded in this cell.

## Controls
K-FLOOR (live = banked), K-CONST (majority 0.338), K1 self-pairs exactly 0.0, K-RE (48 rows equal to RESULT_NAVRB2), Z-EVAL on eval-141 (2,160 zero-history windows read exactly 0.0) all pass. NOT-YET: held-out Z-EVAL. The B2 tables hold only the train and eval-141 clips, so no zero-history mask exists for the held-out windows.

## Next (RULE ZERO), in the card's order
1. **Running now:** the card's first NO-GAIN lever, the stacking-combiner check (RBSTACK, CPU, zero training). Pre-registration `rbstack/PREREG_RBSTACK.md` sha256 bf4d5395 was frozen before any RBSTACK code or number existed. It uses nested folds and two probe seeds, and asks whether the vision probe's out-of-fold probabilities add to the route-text tree. STACK-CARRIES or WEAK cards the vision lever (RB window weight in the loss and/or the 8x20 grid input) with that delta as its T0 ceiling. STACK-SILENT leaves the decision on seeds.
2. **FLOOR-HIGH:** more seed pairs need a PI go, and on these numbers they buy confirmation, not a gain. The MM does not recommend them ahead of the effect-size lever.
3. All three arms are banked PRIVATE on HF (`Sayood/tanitad-v8f-strat1-runs`, `N1D_s0/`, `N1D_s1/`, `N1Dx_s0/`, 10 files each, LFS sha256 verified: `out/BANKED_NAVRBD_ARMS.json`). So are the read's dumps (`navrbd_read/navrbd_read_outputs.tar`, sha256 143f3b2a, hub equal: `out/BANKED_NAVRBD_READ.json`).
