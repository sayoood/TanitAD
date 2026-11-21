# PREREG E2-WISE — is the lateral deficit carried by PLANNER drift or by TRUNK drift? (MM 4a3d05, 2026-11-17 ~04:45Z)

**Written before E2-TAIL's stage B lateral rows exist.**
* At 04:39Z the Thor roll had produced the rolls, the camera-blind reads and the map-selection traces. No families or distance-keeping
  file and no verdict existed (`runs/E2TAIL_K2DC/stageB/read/` held only the seeded dumps and EVALs).
* Zero training. It runs **only if E2-TAIL reads NO-GO with (d) failing.**
* With an E2-TAIL GO it is void; the GO goes to the PI.

## 1. Why

* **E2-RAW** was NO-GO on (d) only: LATERAL was separated worse at both seeds (`e2/e2raw/README_STAGE_B.md`, 68fe22a8).
* **The two training levers on the table act on different parameters:**
  * the 0.02 share target (`hold_r9e2_k2dc02`, PI-gated) lowers the map term's pull on the **shared trunk** (`core.encoder`);
  * a planner-only L2-SP anchor (not built) would restrain the **planner** and leave the trunk alone.
* **Neither is cheap:** ~20.5 Thor-h, or new code plus ~20.5 Thor-h.
* A zero-training read can say which one targets the defect.

## 2. The weights (MEASURED census, Thor `inputs/param_census.py`, K2DC tail vs init)

* **Init:** `/home/nvidia/r8ns/ckpt/ckpt_ema_48852.pt`, K2DC's and K0's `--init-from`.
* **Interpolation:** for an arm X ∈ {K2DC, K0}, a subset S and α ∈ [0, 1]:
  `θ = θ_init + α (θ_X_tail − θ_X_init)` on every tensor in S that exists in the init with the same shape.
  * Every other tensor is copied from `model_ema_tail5k6k.pt` unchanged: new modules, frozen tensors, everything outside S.
* **Both arms get the same S and α.** The E2 rule still compares K2DC against K0 under identical processing.
* **S_plan (the planner):**
  * `core.decoder`, `core.tactical_trunk`, `core.lat_head`, `core.lon_head`, `core.law_head`;
  * `core.ego_hist`, `core.measurement`, `core.lat_log_prior`, `core.lon_log_prior`;
  * `tac_decoder_v6`, `tac_behaviour_gate_v6`, `tac_goal_head`, `tac_latent_proj`, `phi_tac`, `lat_head_tac`, `lon_head_tac`;
  * `nav_inj`, `nav_to_tac`, `goal_gate`, `scorer`.
  * **Excluded:** `core.encoder`, `_perception`, `_map_hires`, `core.agent_head`, `core.agent_embed`, `core.box_tok_r9`, `r8_drv`,
    and the strategic modules (`core.strategic`, `str_goal_head`, `gstr_*`, `nav_to_str`), which are equal to the init anyway.
* **S_enc (the trunk):** `core.encoder` only.
* **α = 0.5**, fixed here and never tuned on DEV.

## 3. The reads (E2's rule unchanged, seeds 0 and 1, DEV)

1. **WISE-PLAN** (S_plan, α 0.5): stage B only, ≈ 4.4 Thor-h.
   * Stage A's map and box are reused from E2-TAIL. That is admissible only if every tensor the perception dump reads is bit-equal to
     the tail file.
   * The tool proves this per tensor and writes the list. If it cannot prove it, stage A runs first.
2. **WISE-ENC** (S_enc, α 0.5): stage A then B, ≈ 6 Thor-h. Run after WISE-PLAN under the outcomes below.

* Every dump records `ckpt_md5` = the interpolated file's md5, the same guard as E2-TAIL.
* One training run per arm (`H-ESTIM-SEED-1`).
* Four metric families. STRATEGIC is reported as the read reports it.

## 4. Predictions and committed outcomes (Rule Zero: each names the next lever)

Primary readout: (d) under the E2 rule (K2DC − K0, paired, both seeds). Secondary: K2DC's seed-0 `LAT_cross_mae_m`.

1. **WISE-PLAN passes (d)**, and (a), (b), (c) and (e) still pass. The deficit is carried by planner drift.
   * **Next:** the planner-only L2-SP anchor is the training lever. Build it with its own G-DVB entry and own prereg; launching it is the PI's call.
   * A planner-interpolated K2DC is also a zero-training candidate config. Report it to the PI beside the anchor.
2. **WISE-PLAN fails (d).** Planner drift alone does not carry it.
   * **Next:** run WISE-ENC.
   * If **WISE-ENC passes (d)** with (a) still passing, trunk drift carries it. The 0.02 share target (`hold_r9e2_k2dc02`) is the lever
     the PI is asked to approve.
   * If **WISE-ENC fails (d) as well**, neither half alone carries it. Report it, and the 0.02 arm stays the pre-registered next arm.
3. **WISE-PLAN passes (d) but breaks another clause** (c or e). Report the trade. Do not propose the interpolated config; diagnose that clause first.
4. **Interpolation hurts both arms' lateral** (both s0 cross-track means above their tail values) and the K2DC − K0 gap is unchanged:
   α = 0.5 is too far. Report it. This is not a reason to tune α on DEV.

## 5. Controls the tool must pass (before any read)

* α = 1 reproduces the tail file **bit-exactly**: every tensor equal, and the md5 of the state dict is reported.
* α = 0 gives the init exactly on S, and the tail exactly off S.
* A shape-mismatched or absent tensor in S is copied, never interpolated, and listed.
* An unknown prefix in S refuses.
* Deliberate regression: with S emptied, the output must equal the tail file. With the interpolation line deleted, the α = 0.5 test must go RED.

## 6. Cost and authority

* Zero training. WISE-PLAN ≈ 4.4 Thor-h; WISE-ENC ≈ 6 Thor-h (ESTIMATED from the E2-RAW and E2-TAIL stage timings).
* Thor is this stream's resource (PI 2026-11-02). No training spend.
* The 0.02 retrain and any L2-SP training stay the PI's decisions.
