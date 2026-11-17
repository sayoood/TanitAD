# PREREG E2-WISE — ADDENDUM 2: WISE-BOTH, the read after a double (d) failure (MM 4a3d05, 2026-11-17 ~11:45Z)

**Written before any WISE-ENC result.** At 11:43Z WISE-ENC stage A (started 11:29:32Z) had written only K2DC's fit thresholds
(`thr_K2wenc_*.json`) and was dumping K2DC's DEV windows. No EVAL file, no clause value and no family row of WISE-ENC existed.
This addendum adds one read under one trigger. It changes nothing in sections 1 to 6 of the prereg or in Addendum 1.

## Why

* **WISE-PLAN's family rows** (MEASURED, T1, 70 episodes, paired episode-cluster bootstrap; `runs/E2WISE_PLAN/stageB/read/families_s*.json`):
  * pulling the planner halfway to the init halved the seed-0 cross-track gap: +0.0126 separated (E2-TAIL) to +0.0061 [−0.0002, +0.0124] not separated;
  * yaw rate, final cross-track and `TAC_traj_lat_correct` stay separated worse at both seeds. So (d) fails (outcome 2).
* **WISE-ENC has a confound the prereg did not name.** Each arm's planner co-adapted to that arm's drifted trunk.
  * Pulling the trunk back while keeping the planner breaks that pairing.
  * The break is larger for the arm whose trunk moved more, which is K2DC.
  * So a WISE-ENC (d) failure cannot by itself separate "trunk drift does not carry the deficit" from "a mismatched planner hurts K2DC more".
* **WISE-BOTH pulls the planner and the trunk halfway together**, which keeps each arm's pair consistent.
  * It is also the closest zero-training stand-in for a lower dose: a smaller map pull moves the trunk and the planner less, together.

## Trigger

* **Runs only if WISE-ENC fails (d).**
* If WISE-ENC passes (d) with (a) passing, the PI's conditional approval of the 0.02 arm fires and this read is void.
* If WISE-ENC passes (d) but fails (a), this read is void too: it would lose the map gain the same way. That case goes to the PI.

## The read

* **S_both = S_plan ∪ S_enc** (the 20 planner prefixes of section 2 plus `core.encoder`), **α = 0.5**, both arms, fixed here and never tuned on DEV.
* **Tool:** `stack/scripts/weight_interp.py` (landed 20ab9f54). Init `/home/nvidia/r8ns/ckpt/ckpt_ema_48852.pt`, run file `model_ema_tail5k6k.pt`,
  output `model_wise_both_a050.pt`.
* **Control before any read (from the tool's provenance):**
  * `n_interpolated` = 848 = WISE-PLAN's 324 + WISE-ENC's 524, since the two subsets are disjoint;
  * `n_copied_in_S` = 110 = 6 + 104;
  * every tensor outside S_both is bit-equal to the tail file.
  * A different count stops the read as a tool defect.
* **Stages:** stage A (map and box re-read), then stage B (rolls, families, S6c), with E2's rule unchanged, seeds 0 and 1, DEV.
  * Every dump records `ckpt_md5` = the interpolated file's md5, the same guard as E2-TAIL and WISE.
* **Cost:** about 6 Thor-h of eval plus about 45 min of dev-box S6c. Zero training.

## Committed outcomes

Primary readout: (d). Secondary: (a), and both arms' seed-0 `LAT_cross_mae_m` against their tail values.

1. **WISE-BOTH passes (d) with (a) passing.** The deficit is carried by joint drift of trunk and planner.
   * The lever is a lower dose, i.e. the 0.02 share target.
   * This is NOT the PI's condition (WISE-ENC failed), so it goes to the PI with this read.
2. **WISE-BOTH passes (d) but fails (a).** Halfway to the init removes the lateral deficit and the map gain together; the trade is dose-shaped.
   * Report to the PI. No arm is proposed without a dose-response read.
3. **WISE-BOTH fails (d).** Drift of S_plan ∪ S_enc at α 0.5 does not carry the deficit.
   * The next diagnostic targets the modules outside both subsets: `_map_hires`, `_perception`, `core.agent_head`, `core.box_tok_r9`. It needs its own prereg before any number.
   * Report to the PI.

* In all three outcomes the 0.02 arm needs the PI. Nothing here starts training.
* One training run per arm (`H-ESTIM-SEED-1`): a separated interval answers another draw of EPISODES only.
