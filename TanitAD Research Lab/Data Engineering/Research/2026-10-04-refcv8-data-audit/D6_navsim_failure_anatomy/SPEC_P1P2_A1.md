# SPEC_P1P2 AMENDMENT A1 -- the as-launched refcv7 has NO WTA proposals and NO refcv7 scorer: the candidate universe is the 117 fan

Stream D6, 2026-10-04, written BEFORE any P1 or P2 number exists, after SPEC_P1P2.md (sha256 `4eaf655f8e000f5ae06a4b0ba93ff05be04a037064d1c16d108e7c11c1f5f4bd`,
registered by the Master Mind 2026-10-04T12:00:42Z, `raw/SPEC_SHA256.txt`). To be registered (sha256 + time) before the queued GPU job can hold the lock. The GPU lock is held by the
battery's step-50,400 chain and the P1/P2 job only waits, so no number can exist before the Master Mind has seen this file.

## Why (MEASURED, found while checking the export against the launch record, not on any probe data)

SPEC_P1P2 s1 assumed the model emits `out["r7_candidates"]` (117 fan + 64 WTA proposals), `r7_score`, `r7_sel_idx`, `traj_r7`. Those exist only when the model was built with the
refcv7 WTA / DisentangledScorer heads (`refc_v3.py:1327-1355`, `_refcv7_forward` at 2305). The launch record says refcv7-r101-s0 was NOT built with them:

* `D:/refcv7_eval_kit/ckpt/config.json` `argv` (154 tokens) contains no `--refcv7`; `seams/agent_knobs`: `w_r7_wta = 0.0`, `w_r7_scorer = 0.0`;
  the registered term table says "`--w-r7-wta` needs `--refcv7` (no decoder => no `r7_wta`)" (config.json lines 377-385).
* the bridge manifest `param_breakdown` of the loaded model (`.../step30000/bridge_navhard/seam_R7_A1.manifest.json`, equal to config.json) lists core 90,787,546 / phi_tac / str_goal_head /
  gstr_cond / tac_heads / tac_latent_proj / scorer 1,145 / nav_inject / tac_decoder_v6 / tac_behaviour_gate_v6, total 100,468,987 -- no WTA decoder, no `r7_scorer` group.

So the 117 fan (`out["anchor_traj"]`, the set `RefCV3Model.forward` E9-selects from, `refc_v3.py:2247-2298`) is the whole candidate universe.

## Changes (everything else in SPEC_P1P2 stands: token sets and their hashes, clean definitions, controls K1-K5 and K7-K9, P2 arms, thresholds, the HT weights)

* **A1.1 Universe.** `U` = FAN117 = `anchor_traj` (117 members), plus the STOP plan as candidate index 117. Wherever SPEC_P1P2 says U181 / "181" read U117 / "117"; the FAN117, WTA64 split is void
  (no WTA64); `RND_L = n_clean / 117`. The export falls back to `anchor_traj` when `r7_candidates` is absent and records `universe = "FAN117"` in every sidecar line; it still aborts if neither exists.
* **A1.2 Ranking.** Every rank read (`RANK_L`) uses the selector the model actually applies: `sel_score_v3` masked by `reach_keep` (the E9 blended score), rank 1 = the emitted pick. The `r7_score` rank is void.
  Tier C re-scores the best-ranked clean candidate under that same E9 ranking.
* **A1.3 R4 -> R4'.** `R4_L` (the refcv7 scorer's own pick) does not exist. Replaced by `R4'_L` = share of scenes where the DECODER's own pick `sel_idx_base` (the pick the E9 re-selection demotes; what the model
  emits with the E9 goal graft off) is L-clean. The s6 branch "(a) free lever `refcv7_select`" becomes **"(a) E9 re-selection OFF at inference (emit the decoder's own pick) -- no retraining"** with the same
  trigger (`F >= 0.70` AND `R4' >= 0.5 x F`). The other branches, thresholds and the S2 SLOWER bands are unchanged.
* **A1.4 Controls.** K1: `cands_poses[sel_idx]` equals the banked seam plan (unchanged). K2 unchanged. K6: `sel_idx` AND `sel_idx_base` are < 117 on 100 % of scenes and `derived_ok` >= 99.5 % (the "WTA flag" part is void).
  K4: the +30 m lateral mutation of the 117 candidates on Cpass (unchanged magnitude). K3 / K5: STOP is candidate 117.
* **A1.5 Cost.** P1 CPU scoring per scene shrinks (118 instead of 182 proposals per batch, ~0.5 s less); GPU time unchanged (the forward is the same).
* **A1.6 What the amendment cannot hide.** F is now a statement about the 117 anchors only. A generation-side verdict ("the fan holds no clean path") therefore also covers the case that the model has no
  learned proposal head beyond the anchor-residual decoder; any WTA-style lever is a NEW component, not a diagnosis of this checkpoint.

## Instrument status at the time of writing

`code/bridge_fix/` regenerated from the unchanged `.base` files (insert-only; `test_default_identical.py` 23/23 incl. the new as-launched-model check 7c and a mutation control);
`code/d6_fan_score.py`, `code/d6_p1_analyze.py` handle the 117 universe and were exercised end to end on SYNTHETIC fans of 117 candidates built around real banked plans
(exact pick reproduction 0.0, STOP batch == frame, batch == single). The queued job's first act (the K1 reproduction) is the full-scale check of the export on the real model.
