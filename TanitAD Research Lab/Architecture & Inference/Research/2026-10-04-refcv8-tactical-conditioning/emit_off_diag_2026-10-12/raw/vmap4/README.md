# V-MAP4 (L3 treatment) descriptive read, 2026-10-13 ~18:05Z. POST-HOC; L3 is scored at I-0 (V-MAP4r not yet in).

V-MAP4 = V-R8 with `--w-map-hires 4.0` (V-R8: 1.0). L3's decider is the in-run step-3000 eval
(`eval_map_hires_iou_<class>_<band>`, headline drivable + lane in 0_20 / 20_40), read as E = V-MAP4 − V-R8 against
F = max(|V0r − V0|, |V-R8r − V-R8|). For an OPTIONAL lever the bar is R = E / F ≥ 2 (registration amendment 1), with
the gain separated. F does NOT yet include V-MAP4r, which the SPEC adds for L3; it can only grow F.

| metric (in-run eval, step 3000) | V-R8 | V-MAP4 | E | F | R |
|---|---|---|---|---|---|
| map_hires IoU drivable 0_20 | 0.6791 | 0.6984 | +0.0193 | 0.0124 | **1.55** |
| map_hires IoU drivable 20_40 | 0.5329 | 0.5637 | +0.0308 | 0.0225 | **1.37** |
| map_hires IoU lane 0_20 | 0.1617 | 0.1907 | +0.0290 | 0.0102 | **2.86** |
| map_hires IoU lane 20_40 | 0.0000 | 0.0024 | +0.0024 | 0.0000 | degenerate (every base arm reads 0) |
| box3d AP@2m | 0.0471 | 0.0428 | −0.0043 | 0.0038 | worse by > F |
| agent AP@2m | 0.0211 | 0.0203 | −0.0008 | 0.0006 | worse by > F |

* **Premise (B-PREM): MET, strongly.** `gs_bev025_proj_map_hires` last-50 % median **0.833** (V-R8 0.175; V-R8r
  −0.093, V0r −0.044: at weight 1 the map's share flips sign across replicates).
* **Reads as NOT clearing B-MAPW:** drivable R 1.55 / 1.37 < 2, and box / agent AP worse than V-R8 by slightly more
  than F (both APs are tiny at this rung). Lane 0–20 m clears R ≥ 2.
* **Driving (plan dumps, vs the R8 replicate spread):** route and turns unchanged (turn direction, heading-15 and
  turn cross all ns). Beyond the replicate: `speed_mae_0_2s` +0.014 / +0.013 and `cons_mae_spd` +0.008 worse (both
  marginal), the dump's 0.25 m `map_iou_drivable` +0.039 better, `lat3_acc_gtplan` +0.014 better.
