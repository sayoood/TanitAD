# PREREG E2-WISE — ADDENDUM 4: WIRE-OFF, K2DC's two planner wires set to zero (MM 4a3d05, 2026-11-18 ~00:50Z)

**Written after PLAN-RESET's family rows and before its verdict file.** No number of the read defined here exists yet.
* PLAN-RESET's `families_s0/s1.json` were written on Thor at 00:03Z. Its dev-box S6c (clause (c)) started at 00:14Z; `E2_VERDICT.md` does not exist.
* (d) is computed from those family rows, so PLAN-RESET's (d) FAIL is already fixed. The verdict file can only add (c).
* This addendum adds one read, plus one conditional follow-up read. It changes nothing in sections 1 to 6, in Addenda 1 to 3, or in their outcomes.

## Why: PLAN-RESET did not remove the deficit (MEASURED, T1, DEV, 70 episodes, paired K2DC − K0, episode-cluster bootstrap)

| row, seed 0 | E2-TAIL (α 1) | WISE-PLAN (planner α 0.5) | PLAN-RESET (planner α 0) |
|---|---|---|---|
| `LAT_cross_mae_m` | +0.0126 W | +0.0061 | +0.0113 W [+0.0048, +0.0179] |
| `LAT_cross_abs_final_m` | +0.0288 W | +0.0137 W | +0.0242 W [+0.0101, +0.0389] |
| `LAT_heading_mae_deg` | +0.1096 W | +0.0505 | +0.1136 W [+0.0410, +0.1928] |
| `LAT_yaw_rate_mae_radps_valid` | +0.0022 W | +0.0012 W | +0.0025 W [+0.0011, +0.0042] |
| `TAC_traj_lat_correct` | −0.0071 W | −0.0088 W | −0.0079 W [−0.0154, −0.0008] |

W = separated worse. Source: Thor `runs/E2WISE_PLAN0/stageB/read/families_s0.json`. Seed 1 is separated worse on the same five rows.

* **Every lateral gap at α 0 is above its α 0.5 value.** That meets Addendum 3's outcome-3 condition on all four rows: the response is not
  monotone, so planner drift is not the simple carrier.
* **Absolute seed-0 cross-track rose in BOTH arms at α 0:** K2DC 0.1191 m and K0 0.1078 m, against 0.1106 / 0.1045 at α 0.5 and
  0.1160 / 0.1034 at the tail. Putting an init planner on a drifted trunk costs both arms, and K2DC more.
* **Addendum 3's committed next step** is the coupling between the planner and the modules outside S_plan. This addendum names that coupling from source and tests it.

## The coupling, from source (tip c18f90b6)

* **K2DC's argv carries `--r9-plan-map-channels` and `--r9-plan-box-tokens`.** K0's argv carries no `--r9-*` lever at all
  (Thor `runs/r9e2-{K2DC,K0}/config.json`). So K2DC's planner reads two inputs that K0's planner never sees.
* **W-MAP-2** (`stack/tanitad/train/plan_wires_flag.py`, docstring and `attach`): the banded head's drv / drv_sdf / sdw / lane / unobs channels
  go through a **zero-init 1×1** `_perception.bev_pool.map_chan_proj` that is **added to the planner BEV pool**.
* **W-BOX** (same file): the top-32 box3d slots become tokens (`core.box_tok_r9`), appended to the planner's and the tactical decoder's
  agent tokens behind a **zero-init gate per consuming layer** (`box_gate`). `stack/tanitad/refs/refcv6_tactical.py:411`:
  "out = base + g * (full - base); g = 0 is the base bit-for-bit".
* **Neither wire was reset by PLAN-RESET.**
  * `map_chan_proj` is outside S_plan.
  * The 6 `box_gate`s are absent from the init, so they kept K2DC's trained values (Addendum 3 named them).
  * So the α 0 planner still received both wires' trained signals. That is one reading of the non-monotone response.
* **MEASURED tensors in K2DC's tail** (`model_ema_tail5k6k.pt`):
  * the six `box_gate` values are −0.0252, −0.0077, +0.0063, −0.0039 (core.decoder layers 0 to 3) and −0.0056, −0.0071 (tac_decoder_v6 layers 0 and 1);
  * `map_chan_proj.proj.weight` (96, 5, 1, 1) has max |w| 0.0177; `map_chan_proj.proj.bias` (96) has max |b| 0.0062.

## The read: WIRE-OFF

* **Weights.**
  * K2DC: its tail with exactly 8 tensors set to 0: the 6 `box_gate`s, `_perception.bev_pool.map_chan_proj.proj.weight` and `.proj.bias`.
    Both wires are zero-init, so this is the OFF build's inference path by construction. Every other tensor stays bit-equal to the tail.
  * K0 has none of these tensors. Its file is its tail unchanged (bit-equal, same md5).
  * Both arms then go through the same pipeline. Output name: `model_wireoff.pt`.
* **Control before any read:** exactly 8 tensors zeroed in K2DC and 0 in K0, with every other tensor bit-equal. A different count stops the read as a tool defect.
* **Stage A:** the map primary and box AP do not read either wire. So stage A must reproduce E2-TAIL's map points (1e-9) and box points (1e-6),
  as in Addendum 1. If it does not, that is a tool or read defect, not a result.
* **Stage B:** E2's rule as amended by the PI on 2026-11-17 ((b) = not separated worse), seeds 0 and 1, DEV. Every dump records `ckpt_md5` = the file's md5.
* **Order on Thor:** after WISE-BOTH frees Thor (Addendum 3's order is kept). If the PI approves a training run in the meantime, that run goes first.
* **Cost:** about 6 Thor-h of eval plus about 1.5 h of dev-box S6c (including the paired box step). Zero training.

## Committed outcomes

Primary readout: (d). Secondary: the four lateral seed-0 gaps of the table above, read against their E2-TAIL values.

1. **WIRE-OFF passes (d) at both seeds.** The lateral deficit is carried by the planner reading the wires at inference.
   * Next, pre-registered here: **BOX-OFF**, the 6 `box_gate`s alone, same pipeline, to attribute the effect to one wire.
     * BOX-OFF passes (d): the box-token wire carries it.
     * BOX-OFF fails (d): the map-channel wire carries it, alone or with the box wire.
   * The training lever is the attributed wire (retrain without it, or in a constrained form). It gets its own prereg; launching it is the PI's call.
   * If (c) or (e) fails while (d) passes, report the trade. Map channels carry drivable and sidewalk information, so the S6c rate may move. It is not reported as a pass.
2. **WIRE-OFF fails (d), but every lateral gap is below its E2-TAIL value.** The wires carry part of the deficit.
   * BOX-OFF still runs, to attribute that part.
   * The residual rows are named and get their own prereg.
3. **A lateral gap is not below its E2-TAIL value.** The wires at inference do not carry the deficit.
   * It lives in what training did to the weights outside the wires: trunk and map co-adaptation.
   * The lower map dose (the 0.02 share target, `hold_r9e2_k2dc02`) and the planner L2-SP anchor (`hold_r9_l2sp_planner`, unlanded) stay
     the candidate training levers. Both are the PI's decisions; report to the PI.

* **Caveat, all outcomes:** zeroing a wire at inference removes a signal the trained planner co-adapted to.
  * A pass shows that removing it recovers the lateral rows.
  * A fail does not show the wire is harmless during training.
* One training run per arm (`H-ESTIM-SEED-1`): a separated interval answers another draw of EPISODES only.
* This read never selects a config on DEV. Its outcomes pick the next training lever, which is then pre-registered and read on its own.
