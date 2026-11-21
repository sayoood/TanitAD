# V-MAP4-roll (L3 deliberate-regression arm) read, 2026-10-13 ~23:10Z. POST-HOC; L3 is scored at I-0.

V-MAP4-roll = V-MAP4 (`--w-map-hires 4.0`) + `--r8-roll-targets map` (argv diff verified on Thor, `--out` aside): the
10 cm map targets are rolled across the batch, so the map head is trained at weight 4 on WRONG targets. B-REG: its
gain over V-R8 must be <= F. Thor batch-8 base dumps (md5 match Thor-side, `V-MAP4-roll_dumps_thor.md5`).

## In-run eval, step 3000 (E = arm - V-R8; F = max(|V0r - V0|, |V-R8r - V-R8|); R = E / F)

| metric | V-R8 | V-MAP4 (R) | V-MAP4-roll | E roll | R roll |
|---|---|---|---|---|---|
| map_hires IoU drivable 0_20 | 0.6791 | 0.6984 (+1.55) | 0.5620 | -0.1171 | -9.43 |
| map_hires IoU drivable 20_40 | 0.5329 | 0.5637 (+1.37) | 0.4672 | -0.0656 | -2.91 |
| map_hires IoU lane 0_20 | 0.1617 | 0.1907 (+2.86) | 0.0000 | -0.1617 | -15.92 |
| map_hires IoU lane 20_40 | 0.0000 | 0.0024 | 0.0000 | 0 | degenerate |
| box3d AP@2m | 0.0471 | 0.0428 (-1.13) | 0.0398 | -0.0073 | -1.92 |
| agent AP@2m | 0.0211 | 0.0203 (-1.51) | 0.0199 | -0.0012 | -2.15 |

`gs_bev025_proj_map_hires` median over steps >= 1500 (**n = 2 readings** per arm): V-MAP4-roll 0.463, V-MAP4 0.833,
V-R8 0.175, V-R8r -0.093.

* **B-REG HOLDS, by a wide margin:** no map gain over V-R8; every map IoU collapses (lane 0-20 m to exactly 0).
* **What it buys:** the in-run map metric responds to CORRECT supervision, not to the weight, so V-MAP4's (small)
  map gains are map learning, not a weight artefact. It does not change V-MAP4's reading (drivable R < 2).
* **Plan dumps vs the V-R8r - V-R8 replicate (`*_describe.json`):** vs V-R8, beyond the replicate on both seeds: only
  the dump's 0.25 m `map_iou_drivable` -0.064 and `map_iou_lane` -0.112 (worse). Vs V-MAP4: map IoUs worse again, and
  `speed_mae_0_2s` **-0.016 / -0.013 BETTER** (beyond replicate both seeds) -- V-MAP4's short-horizon speed cost is
  tied to learning the CORRECT map at weight 4 (capacity shared with the planner), not to the weight number itself.
  Turns, route, tactical: no separated change beyond the replicate.
