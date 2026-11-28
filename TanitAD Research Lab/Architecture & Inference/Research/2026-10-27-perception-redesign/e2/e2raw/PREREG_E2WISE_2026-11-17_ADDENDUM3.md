# PREREG E2-WISE — ADDENDUM 3: PLAN-RESET, the planner at α = 0 (MM 4a3d05, 2026-11-17 ~18:05Z)

**Written after WISE-ENC's family rows and before its verdict file.** No number of the read defined here exists yet.
* At 17:58Z the dev-box S6c of WISE-ENC was scoring K0's seeds; `E2_VERDICT.md` did not exist.
* It adds one read and reorders two. It changes nothing in sections 1 to 6, in Addendum 1, or in Addendum 2's outcomes.

## Why: the measured dose response (MEASURED, T1, DEV, 70 episodes, paired K2DC − K0, episode-cluster bootstrap)

| row, seed 0 | E2-TAIL (α 1) | WISE-PLAN (planner α 0.5) | ratio | WISE-ENC (encoder α 0.5) |
|---|---|---|---|---|
| `LAT_cross_mae_m` | +0.0126 | +0.0061 | 0.48 | +0.0127 |
| `LAT_cross_abs_final_m` | +0.0288 | +0.0137 | 0.48 | +0.0279 |
| `LAT_heading_mae_deg` | +0.1096 | +0.0505 | 0.46 | +0.1036 |
| `LAT_yaw_rate_mae_radps_valid` | +0.0022 | +0.0012 | 0.55 | +0.0018 |
| `LON_dk_min_ttc_s` | −0.3953 | +0.0706 | — | −0.3910 |
| `TAC_traj_lat_correct` | −0.0071 | −0.0088 | — | −0.0075 |

* **The planner shows a dose response.** Halving the planner's drift from the init halved every lateral gap (ratios 0.46 to 0.55) and
  removed the min-TTC deficit. A gap that is linear in the planner's drift extrapolates to about zero at α = 0.
* **The encoder shows none.** WISE-ENC moved neither arm's lateral: seed-0 cross-track K2DC 0.1160 → 0.1153 m, K0 0.1034 → 0.1026 m.
  * Addendum 2's reason for WISE-BOTH (a broken planner/trunk pairing would hurt K2DC more) predicted a larger K2DC loss. It did not happen.
* **`TAC_traj_lat_correct` did not shrink at α 0.5.** It is watched as its own row below.
* A post-hoc pattern is not a result. This addendum turns it into a pre-registered test.

## The read

* **PLAN-RESET:** S_plan (the 20 prefixes of section 2), **α = 0**: every planner tensor that exists in the init with the same shape is
  set to the init. Every other tensor comes from `model_ema_tail5k6k.pt`. Both arms, same S, same α.
  * α = 0 is the endpoint of the dose axis, chosen to test the extrapolation. It is not a DEV-tuned value, and the config it makes is
    never selected on DEV.
* **Tool:** `stack/scripts/weight_interp.py` (20ab9f54), output `model_wise_plan_a000.pt`.
  * Control: `n_interpolated` 324 and `n_copied_in_S` 6, both equal to WISE-PLAN's; perception bit-equal.
  * Each of the 324 interpolated tensors must equal the init bit-for-bit (`torch.lerp` at weight 0). Any other result is a tool defect.
  * The 6 copied tensors are the decoder `box_gate`s, absent from the init. They keep each arm's trained value, as in WISE-PLAN.
    So α = 0 removes the planner's drift everywhere except those 6 gates, and a residual gap may live there.
* **Stages:** A, then B, E2's rule unchanged, seeds 0 and 1, DEV. Stage A must reproduce E2-TAIL's (a) and (b) within 1e-6, as in Addendum 1.
  * Dump guard: `ckpt_md5` = the new file's md5.
* **Cost:** about 6 Thor-h of eval plus about 45 min of dev-box S6c. Zero training.

## Committed outcomes

Primary: (d). Secondary: the seed-0 gaps of the four lateral rows above, read against their α 0.5 values.

1. **PLAN-RESET passes (d) at both seeds.** Planner drift carries the whole lateral deficit.
   * The training lever is a **planner-only L2-SP anchor** toward the init (prereg section 4.1). Build it with its own G-DVB entry and its own prereg.
   * Launching it is the PI's call. The 0.02 share target is not indicated by this read.
2. **PLAN-RESET fails (d), but every lateral gap is below its α 0.5 value.** The planner carries most of the deficit, plus a residual.
   * The L2-SP anchor stays the indicated lever.
   * The residual rows are named and diagnosed under their own prereg.
3. **A lateral gap at α 0 is not below its α 0.5 value.** The response is not monotone, so planner drift is not the simple carrier.
   * The next diagnostic is the coupling between the planner and the modules outside S_plan (`_map_hires`, `_perception`), under its own prereg.
* In all three, (b) stays the E2-TAIL tie (Addendum 1), so PLAN-RESET cannot GO. (c) and (e) are reported as read.
* One training run per arm (`H-ESTIM-SEED-1`).

## Order on Thor

1. PLAN-RESET first: its planner lever has the largest measured effect (Rule Zero, item 5).
2. WISE-BOTH (Addendum 2) second, as committed. Its trigger fired (WISE-ENC fails (d)). It is not dropped, only reordered.
3. If the PI approves the 0.02 arm in the meantime, that training run takes Thor first.
