# HANDOVER_SRC design: --t0-state own|logged|speed_logged|path_logged

Wrapper `v8f_t1_eval_reader_rule_rzgfhqs.py` (generator `make_handover_src.py`, parent `..._rzgfhq.py` sha256 prefix 92a8034cc0f7b0d2). Pinned tool v8f_t1_eval.py (sha 61e140ed...) is not touched; line numbers below are of its copy from tree cdbff971 (`handover_scratch/v8f_t1_eval_ref_copy.py`, same sha256).

## Where the first scored tick gets its state (evidence)
- The batch is built by `build_item` (pinned:237) from `st.d` (EGO arrays pose, kin3, p10f, base, a, ch1, v) and `st.nav`; `rollout` (pinned:399) calls it for every tick, collates and keeps `MODEL_KEYS` = INFERENCE_INPUT_KEYS minus act, pose (v8f_mini.py:176, pinned:62).
- Fresh window: `WindowState.__init__` (pinned:166-174) copies the logged cache rows <= t0 into d and nav (rows after t0 NaN).
- Aged window: the pre-roll commits own plans (`commit_plan`, pinned:181; `_finish`, pinned:201) so rows t0-P+1..t0 of d and nav are own: kin3 / p10f / base / a / ch1 / v re-derived from the executed track by `yaw_kin_from_track` (v8f_mini_data.py:350), nav dead-reckoned from t0-P. Rebase at wrapper rzgfhq:892-905.

## State inputs the model reads at a tick (hmax rows ending at t0; tick-0 ages are 0)
| input (batch key) | fresh source | aged source (own mode) | group in this study |
|---|---|---|---|
| p10f [hmax,8] = (v, a, kappa, v^2, v a, v kappa, v^2 kappa, a kappa) | cache | executed track via kin3 | speed (v, a) + yaw rate (kappa = clamp(omega / max(v,2), +-0.3)) |
| base (kd_x at t0, from kin3) | cache | kin3 of own row | follows kin3 |
| pose_now, pose_t, pose_s ([:3] of pose at t0; same row at age 0) | cache pose | own pose | positions + headings |
| nav [hmax,5] (route token, progress channel 3, validity) | cache nav rows | held from t0-P, progress from own arc | own in 3 of 4 modes |
| limit / limit_ok | clip constant (`--limit logged`) or causal bucket of d["v"] rows [j-20, j] (`--limit causal`) | same, from own d["v"] | follows the speed group |
| feats, frame_ok | logged tokens; frame_ok needs finite p10f | same | not changed |
| a_t, a_s | schedule (0 at tick 0) | same | not changed |
Not read by forward_infer (NO_READ): act, pose. The rule reads `v0 = batch["p10f"][:, -1, 0]` (distrule/v8f_t1_eval_rule_mix.py:52), so under `logged` and `speed_logged` the rule's v0 is the handed speed; its geometry (own position, a_m0) stays own.

## Definitions (implemented in `handover_shadow`, applied only at tick 0 of the main call, rows <= t0)
- `logged`: every EGO array and nav = the cache rows. The item is the one a fresh window at t0 builds (identity checked by check 2).
- `speed_logged`: v, a, kin3 v and a, pose column 3 (speed) from the cache; pose x, y, yaw, ch1 (omega), nav as own. kappa recomputed = clamp(own omega / max(logged v, 2)); p10f and base follow from that kin3 by the pinned functions (`T.fk.kin_features`, `T.D.kdx_base`).
- `path_logged`: pose x, y, yaw and ch1 from the cache; v, a, pose column 3, nav as own. kappa = clamp(logged omega / max(own v, 2)).
- Executor: own state, own pose; the scored track is written by the unchanged `exec_logged_frame`. Plans are ego-frame, so a plan made from handed frames is executed from the own pose.
- Later ticks read d rows executed after t0 (own, unchanged). With --replan-steps below hmax the history of tick 1 still holds aged rows <= t0: those are not replaced (the brief says later ticks are unchanged). At m = 10 = hmax nothing aged is read after tick 0.

## Rejected alternatives
1. Replacing the executed track frames (x, y, psi, v columns) before `_finish`: it would change rows the executor continues from and the pre-roll offsets file. Rejected: the pre-roll must stay untouched.
2. Taking kappa wholly from one source in the partial modes: kappa depends on both omega (path group) and v (speed group). Rejected: it would leak the other group; the mixture is recomputed instead. Cost: for the two partial modes kappa and p10f/base are recomputed from stored (rounded) values; the residual against the stored own and logged kappa is logged (`kappa_recompose`).
3. Putting nav into path_logged (route progress is a distance): rejected, nav is a separate input; only `logged` replaces it, so `logged` minus the two partial modes leaves nav plus interaction. A fifth value `nav_logged` is a one-line extension (not added: four values asked).
4. Replacing the speed-bucket input only in `logged`: rejected, with `--limit causal` the bucket is computed from d["v"], so it follows the speed group; with `--limit logged` it is a clip constant and never differs.
5. A rigid transform of logged frames to the own pose: not needed. Plans are ego-frame and the model reads only relative pose geometry at age 0; the logged frames are handed as they are in the cache (world frame).

## Does the executor force a first-segment difference under identical frames? No
Executed integer-step knots of a commit are exactly the plan knots (`hermite_frames`, pinned:85: the even frame is `K[i+1]`); the own previous knot only enters mid frames, headings and speed columns. So for `logged` the first replan segment, in the own frame at t0, must equal the fresh track to rounding (check 2b, bar 1e-3 m) and the tick-0 plan, tau, P equal fresh (check 2a, bar 1e-6; exact on a deterministic device). Rule forms add own-geometry terms (a_m0), so exactness is stated for form R0 only.

## Assumptions (file:line)
- ch1 = yaw rate omega in the cache as in own rows: `_finish` pinned:201-215 writes `d["ch1"] = om`; cache kin_mode yaw (pinned cmd_run:503-515). Verified at run time only through the logged `kappa_recompose` residuals.
- kin3 = (v, a, kappa) and `d["v"]`, `d["a"]` equal kin3 columns 0, 1 (v8f_mini_data.py:350-365; pinned `_finish`).
- cache pose has 4 columns (x, y, yaw, v) like own rows (check_track_matches_grid, v8f_mini_data.py:380).
- `rollout` resolves `build_item` in the module namespace at call time (pinned:411), so `T.build_item` can be replaced and restored (as the wrapper does for rollout, hermite_frames).
- `store.get(clip)` returns the unmodified cache dict (WindowState copies EGO arrays: pinned:166-170); `J["nav"]` slicing as in pinned `make_states` (pinned:427-434).
- Tick-0 rows of the tick file (v0, s_pose, t_pose, plan) show what was handed over; the pinned `TickRecorder.save` carries no wrapper meta, so the mode is recorded in preroll_log_<run>.json (t0_state) and in the meta of rule_log_pre_<run>.npz.
- The own reference item is built a second time at tick 0 for the comparison (counters untouched); CAUSAL_STATS count the handed item once.
