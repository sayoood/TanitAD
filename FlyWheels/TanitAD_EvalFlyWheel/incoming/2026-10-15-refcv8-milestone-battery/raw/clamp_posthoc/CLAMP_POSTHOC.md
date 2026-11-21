# CLAMP_POSTHOC -- the first clamp-ON numbers on real refcv8 windows (AMENDMENT A-R8-CLAMP-1 section 2.3)

Generated 2026-10-18T19:48:38 by `code/clamp_posthoc_r8.py` (CPU only). **TIER:** T1 self-action OPEN LOOP on logged eval139 frames (not driving performance). **ESTIMATOR / QUESTION:** episode-cluster bootstrap over the 139 eval episodes, models FIXED: answers 'another draw of EPISODES' only; NOT another training run, NOT another inference run; B 2000, bootstrap seed 0, 95 % percentile; paired = refcv8 minus the reference on the same windows. Rates carry the same cluster bootstrap (mean of a 0/1 indicator vs zeros). **Single training seed; inference seeds 0 and 1 are the two banked rolls.** The post-hoc clamp is deterministic given the plan, so seeds 0/1 differ only through the plans.  `**W**` = CI lower > 0 on an error metric (separated worse); `**B**` = CI upper < 0 (separated better).

**Evidence class:** MEASURED (ours) on `C:/Users/Admin/refcv8_eval_archive/battery/<tree>/{dump_s*,panel_s*,s6_s*}`; the clamp is the landed `cap_plan_to_limit`, vendored byte-for-byte. No clamp-ON number below is a GPU eval-forward number: K2/K4 are OWED, so every ON cell is a POST-HOC read on the banked plans.

## 0. Headline

* **Controls:** K1 (inert limit: function bit-identity AND full-pipeline reproduction of every banked cell), K3 (mispairing) and the analytic postcondition PASS on all 9 trees x 2 seeds; K2 / K4 are GPU-only and OWED.
* **R1 gap at step 30,000 EMA (S2 window set, n = 4,754 windows / 139 episodes):** the emitted clamp-OFF plan exceeds the fed ceiling on **5.30 % [3.15, 7.67]** of windows (seed 0; seed 1 5.30 % [3.13, 7.67]), median excess 2.1458 m/s, p90 4.4699, max 9.8806; **250 of 252** exempt rows are rows where v0 is already above the ceiling.
* **On every one of the 252 exempt rows the logged ego's OWN 6-s max speed exceeds the fed ceiling too** (252/252); across all windows the logged ego exceeds the fed ceiling on 12.52 % [8.94, 16.26]. At this tree the clamp binds ONLY where the logged-ego ground truth itself violates the label (all-tree counts in the next bullets).
* **Clamp-ON S2 ADE moves by +0.0629 m** (seed 0; 252 of 4754 windows changed, +1.19 m each on average; 236 worsened / 16 improved), 100 % of it on windows where the ground truth exceeds the ceiling (contribution on ceiling-respecting windows: +0.0000 m).
* **Bars at step 30,000 EMA, clamp-ON beside banked clamp-OFF:** BAR-R8-E1 FAILED (OFF FAILED); BAR-R8-E3 PASS (OFF PASS); NR per family ON {"ADE": "FAIL: S2:ade_m, S6:ade_m", "LONGITUDINAL": "FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m", "LATERAL": "FAIL: S2:LAT_heading_mae_deg"} vs OFF {"ADE": "FAIL: S6:ade_m", "LONGITUDINAL": "FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m", "LATERAL": "FAIL: S2:LAT_heading_mae_deg"}.
* **Metric-level NR verdict flips ON vs OFF across the 9 trees:** 15 (step10500ema_unc S2:LON_along_abs_final_m; step10500ema_unc S2:LON_speed_mae_mps; step10500ema_unc S2:ade_m; step15000ema_unc S2:LON_along_abs_final_m; step15000ema_unc S2:LON_speed_mae_mps; step15000ema_unc S2:ade_m; step15000ema_unc S6:ade_m; step20000ema_unc S2:LON_along_abs_final_m; step20000ema_unc S2:LON_speed_mae_mps; step20000ema_unc S2:ade_m; step20000ema_unc S6:ade_m; step30000ema_unc S2:ade_m; step5000ema_unc S2:LON_along_abs_final_m; step5000ema_unc S2:LON_speed_mae_mps; step5000ema_unc S2:ade_m).
* **The landed clamp does not make the ceiling hard where v0 is above it:** after the clamp 4.69 % [2.65, 6.97] of windows (223) are still above the ceiling (median 1.3421 m/s, max 5.7479), because the allowance is max(limit, v0 - brake * t_mid) with brake = 2.5 m/s^2.
* **Across the 9 trees (seed 0):** exempt rate 4.94-5.70 % with the ceiling filter acting; 10.4-14.6 % on the filter_off arm. Windows changed by the clamp (summed over trees) and, of those, windows where the logged ego stays WITHIN the ceiling: s0 S2 1 of 2283; s0 S6 0 of 1817; s1 S2 0 of 2283; s1 S6 0 of 1820.
* **BAR-R8-E1 cell (os - ha0_ext, S2 ADE 0-2 s) at step 30,000 EMA, OFF -> ON:** seed 0: -0.0008 [-0.0149, +0.0136] -> +0.0621 [+0.0267, +0.1022]; seed 1: +0.0015 [-0.0131, +0.0163] -> +0.0634 [+0.0279, +0.1039]. The bar needs CI upper < 0; it FAILED OFF (not separated) and FAILS ON (separated WORSE than the echo).
* **Verdict changes ON vs OFF over all 9 trees:** BAR-R8-E1 ['FAILED->FAILED'], BAR-R8-E3 ['PASS->PASS'] (no tree changes); NR metric cells 15 PASS->FAIL, 0 FAIL->PASS, all on EMA trees (['step10500ema_unc', 'step15000ema_unc', 'step20000ema_unc', 'step30000ema_unc', 'step5000ema_unc']).

## 1. Vendoring (task 1)

* Source: commit `de4e15b`, `stack/tanitad/refs/refcv6_selection.py`; `git rev-parse de4e15b:stack/tanitad/refs/refcv6_selection.py` = `b4e0e80439eda771b972d9556180feff55f4d66c`; body re-hashed = `b4e0e80439eda771b972d9556180feff55f4d66c` (equal), body == `git show` bytes.
* Vendored file `code/_vendored_de4e15b_refcv6_selection.py`: 5-line header + the blob; file git-blob sha `8446be5c1a14b5c3380647cb417f952add1948fc`, sha256 `38689597f33da44cf09fc0a4abc3c306e8cd2e175ee537551d560bfce2e39fcd`, 23973 bytes. Loaded by `importlib` (`spec_from_file_location`); `clamp_posthoc_r8.py` refuses if the body hash drifts.
* Dependencies resolved against the fb47a58 tree (`C:/lgs/Lfb47a58376`, `PYTHONPATH`). Blob comparison de4e15b vs fb47a58:

| file | de4e15b | fb47a58 | identical |
|---|---|---|---|
| `stack/tanitad/models/vocab_v7.py` | `4022cf009916` | `4022cf009916` | yes |
| `stack/tanitad/refs/refc_selector_targets.py` | `f89830d7bb79` | `f89830d7bb79` | yes |
| `stack/tanitad/refs/refcv6_tactical.py` | `11de086266dc` | `11de086266dc` | yes |
| `stack/tanitad/refs/refcv6_max_speed.py` | `4c3d45b37d01` | `4c3d45b37d01` | yes |
| `stack/tanitad/refs/refcv6_selection.py` | `b4e0e80439ed` | `7b2dd69c2bef` | NO -- differs ONLY by ADDED lines (the clamp): removed 0 lines, added 74 lines (fb47a58 -> de4e15b) |
| `stack/tanitad/refs/refc_v3.py` | `335103cbc3f1` | `68ff839c6aa4` | NO -- differs ONLY by ADDED lines (the clamp): removed 0 lines, added 35 lines (fb47a58 -> de4e15b) |

No dependency differs. The two files that differ (`refcv6_selection.py`, `refc_v3.py`) differ ONLY by lines added for the clamp (0 lines removed), so every other line of the fb47a58 forward the batteries ran is the line the clamp would sit inside.

## 2. Exactness from source (task 2) -- file:line

### C1_os_is_the_tensor_the_clamp_acts_on
* **EXACT (conditional on the 3 empirical identities below holding for every tree / seed)**
* reasoning: the clamp (de4e15b) sits between `traj = fan[arange, idx]` and `out["traj"], out["sel_idx"] = traj, idx`; fb47a58 has the same two lines with nothing between them, so the banked `out["traj"]` IS the clamp's input.  The only later writer of out["traj"] is the refcv7 scorer pick (`_refcv7_forward`, gated on `self.refcv7_wta is not None`), which exists only under `cfg.refcv7`; the run argv carries no --refcv7 flag (argv_has_refcv7 = False), and de4e15b refuses plan_vmax_clamp together with refcv7_select.
* fp32: the extras' traj is not bf16- or fp16-representable (checked per tree), i.e. fp32 resident, as the clamp's input would be

### C2_S2_and_S6_are_slots_of_one_plan
* **EXACT (S2 slots [0,1,2,3] of the full plan; S6 slots [1,3,4,5,6,7]; S6 windows are a subset of S2's)**
* caveat: the clamp acts on the FULL 8-slot plan; binding is decided over all 8 intervals and a binding row is re-integrated with cumsum, so even a row binding only in 3-6 s changes slots 0-3 by fp32 rounding.  This read therefore clamps the full plan and slices, never the slice.
* evidence: {"S2_grid": {"slots": [0, 1, 2, 3], "instants_s": [0.5, 1.0, 1.5, 2.0]}, "S6_grid": [1, 3, 4, 5, 6, 7]}

### C3_no_emitted_plan_feeds_a_later_model_input
* **EXACT for all 18 banked dump / panel sets (9 trees x 2 inference seeds): each is a single forward per logged window**
* reasoning: refcv3_arm.py runs ONE forward per logged window from logged frames, the clip's nav token and the measured scalar v0 = pose_last[3] (manifest t1_definition: 'consuming the observed frames, the clip's v7.2 nav token and the MEASURED v0 at t0 and nothing else'; `no_closed_loop_because`: forward has NO action argument, no per-step decode, no state that advances).  `ego_state_fed` is false in the manifest; `ego_actions` is a forward kwarg that refcv3_arm never passes; the proxy passes `ego_poses` = the OBSERVED pose history of the window.  The sidecar only RECORDS out[...] rows.  The next window's inputs come from the dataset index over logged episodes.
* caveat: SeamState (`self._seam`) accumulates across forwards but is fed by the pre-pick blended-score norms, not by the emitted plan; it does not read `out["traj"]`.
* t1_definition: os: ONE forward pass of RefCV3Model at the window origin, consuming the observed frames, the clip's v7.2 nav token and the MEASURED v0 at t0 and nothing else, emitting the whole 6 s path as the model's OWN sel_score_v3-ranked choice among its 117 anchors (the run'

### C4_numerics
* **post-hoc is exact in DEFINITION; BIT-exactness against the eval-forward clamp is UNVERIFIED (K4 owed: CPU fp32 vs GPU kernels may differ in the last ulp)**

Source quotes (line numbers re-found by the script, not typed):

| file | what | line | text |
|---|---|---|---|
| de4e15b refc_v3.py (clamp call site) | E9 pick: traj = fan[arange, idx] | 2655 | `traj = fan[torch.arange(b, device=fan.device), idx]` |
| de4e15b refc_v3.py (clamp call site) | clamp gate (eval forward only) | 2656 | `if bool(getattr(self.cfg, "plan_vmax_clamp", False)) and not self.training:` |
| de4e15b refc_v3.py (clamp call site) | clamp call | 2666 | `traj, _cd = v6sel.cap_plan_to_limit(` |
| de4e15b refc_v3.py (clamp call site) | emitted plan assigned | 2673 | `out["traj"], out["sel_idx"] = traj, idx` |
| de4e15b refc_v3.py (clamp call site) | limit exported by the scene hook | 1968 | `cache["plan_v_lim_ms"] = v_lim.detach()          # the E9 block clamps the EMITTED plan to it` |
| de4e15b refc_v3.py (clamp call site) | limit = bin limit | 1944 | `v_lim = v6ms.limit_ms_of_bin(vmax_1h.argmax(dim=-1)).to(dev)` |
| de4e15b refc_v3.py (clamp call site) | unknown row -> +inf | 1948 | `torch.full_like(v_lim, float("inf")))` |
| de4e15b refc_v3.py (clamp call site) | refcv8 label rule | 1949 | `if str(getattr(self.cfg.refcv8, "speed_input", "") or ""):      # also a speed-only arm (V0)` |
| de4e15b refc_v3.py (clamp call site) | refcv7 scorer pick would replace out[traj] AFTER the clamp (call) | 2681 | `self._refcv7_forward(out, v0, nav_cmd)` |
| de4e15b refc_v3.py (clamp call site) | refcv7 heads only built under cfg.refcv7 | 1475 | `if bool(getattr(cfg, "refcv7", False)):` |
| de4e15b refc_v3.py (clamp call site) | refcv7_select replaces out[traj] | 2828 | `out["traj"] = pick` |
| fb47a58 refc_v3.py (the tree the batteries were rolled with; clamp absent) | E9 pick: traj = fan[arange, idx] | 2634 | `traj = fan[torch.arange(b, device=fan.device), idx]` |
| fb47a58 refc_v3.py (the tree the batteries were rolled with; clamp absent) | emitted plan assigned | 2638 | `out["traj"], out["sel_idx"] = traj, idx` |
| fb47a58 refc_v3.py (the tree the batteries were rolled with; clamp absent) | refcv7 scorer pick call (only if refcv7_wta is built) | 2646 | `self._refcv7_forward(out, v0, nav_cmd)` |
| fb47a58 refc_v3.py (the tree the batteries were rolled with; clamp absent) | refcv7 heads only built under cfg.refcv7 | 1456 | `if bool(getattr(cfg, "refcv7", False)):` |
| fb47a58 refc_v3.py (the tree the batteries were rolled with; clamp absent) | refcv7_select replaces out[traj] | 2793 | `out["traj"] = pick` |
| taniteval/tools/refcv3_arm.py (fb47a58) | window loop (one forward per logged window) | 2067 | `for i, (wi, e_i, t) in enumerate(sel):` |
| taniteval/tools/refcv3_arm.py (fb47a58) | v0 = measured scalar at t0 | 2225 | `v0 = float(pose_last[3])` |
| taniteval/tools/refcv3_arm.py (fb47a58) | v0 broadcast to the batch | 2272 | `v0_t = torch.full((b,), v0, dtype=torch.float32, device=dev)` |
| taniteval/tools/refcv3_arm.py (fb47a58) | the model call (frames, nav_cmd, v0, steps, ego_state) | 2358 | `out = model(fr_b, nav_cmd=nav_t if nav_on else None,` |
| taniteval/tools/refcv3_arm.py (fb47a58) | os = out[traj] row 0 (nav_true) | 2405 | `traj = out["traj"].float()                            # [b, S, 2]` |
| taniteval/tools/refcv3_arm.py (fb47a58) | os slot selection | 2406 | `acc["os"].append(traj[0:1, slots].cpu().numpy())` |
| taniteval/tools/refcv3_arm.py (fb47a58) | plan_full_nav_true = same tensor, every slot | 2522 | `dec.setdefault(f"plan_full_{cname}", []).append(` |
| r7_roll.py Refcv6Proxy (extras sidecar) | per-row extras recorded from out[...] | 202 | `rec[c] = {k: out[k][r].detach().float().cpu().numpy() for k in EXTRA_ROW_KEYS` |
| r7_roll.py Refcv6Proxy (extras sidecar) | extras keys incl. traj | 66 | `EXTRA_ROW_KEYS = ("tacv6_goal_logits", "tacv6_goal_conf", "tacv6_lat_logits",` |
| r7_roll.py Refcv6Proxy (extras sidecar) | v_max_ms / v_max_valid / v0 per window | 213 | `"v_max_ms": float(it["v_max_ms"]), "v_max_valid": float(it["v_max_valid"]),` |
| r7_roll.py Refcv6Proxy (extras sidecar) | v0 per window | 215 | `"v0": float(v0.reshape(-1)[0]) if v0 is not None else None}` |
| refc.py (decoder attributes the clamp reads) | anchor_dt | 1748 | `self.anchor_dt = 0.1` |
| refc.py (decoder attributes the clamp reads) | anchor_horizons | 1769 | `self.anchor_horizons: tuple[int, ...] = tuple(int(k) for k in _h)` |
| refc.py (decoder attributes the clamp reads) | sel.accel_max from sel_accel_max | 1067 | `accel_max=self.sel_accel_max,` |
| refcv6_max_speed.py | limit_ms_of_bin | 214 | `def limit_ms_of_bin(bin_idx: Tensor) -> Tensor:` |
| refcv6_max_speed.py | MaxSpeedOneHotEncoder | 480 | `class MaxSpeedOneHotEncoder(nn.Module):` |

Empirical identity checks over 9 trees x 2 inference seeds ({'panel_os_equals_extras_slots_and_v0_equal_and_fp32_in_all_18_sets': True, 'decisions_plan_full_nav_true_equals_extras_in_all_18_sets_x_2_surfaces': True}):

| tree | seed | S2 n | S2 os vs extras slots (max abs) | S6 n | S6 os vs extras slots (max abs) | decisions plan_full vs extras (S2 / S6) | S6 subset of S2 | traj fp32-resident |
|---|---|---|---|---|---|---|---|---|
| step30000ema_unc | 0 | 4754 | 0.0 | 3668 | 0.0 | 0.0 / 0.0 | True | True |
| step30000ema_unc | 1 | 4754 | 0.0 | 3668 | 0.0 | 0.0 / 0.0 | True | True |
| step20000_unc | 0 | 4754 | 0.0 | 3668 | 0.0 | 0.0 / 0.0 | True | True |
| step20000_unc | 1 | 4754 | 0.0 | 3668 | 0.0 | 0.0 / 0.0 | True | True |
| step20000ema_unc | 0 | 4754 | 0.0 | 3668 | 0.0 | 0.0 / 0.0 | True | True |
| step20000ema_unc | 1 | 4754 | 0.0 | 3668 | 0.0 | 0.0 / 0.0 | True | True |
| step15000_unc | 0 | 4754 | 0.0 | 3668 | 0.0 | 0.0 / 0.0 | True | True |
| step15000_unc | 1 | 4754 | 0.0 | 3668 | 0.0 | 0.0 / 0.0 | True | True |
| step15000ema_unc | 0 | 4754 | 0.0 | 3668 | 0.0 | 0.0 / 0.0 | True | True |
| step15000ema_unc | 1 | 4754 | 0.0 | 3668 | 0.0 | 0.0 / 0.0 | True | True |
| step10500_unc | 0 | 4754 | 0.0 | 3668 | 0.0 | 0.0 / 0.0 | True | True |
| step10500_unc | 1 | 4754 | 0.0 | 3668 | 0.0 | 0.0 / 0.0 | True | True |
| step10500ema_unc | 0 | 4754 | 0.0 | 3668 | 0.0 | 0.0 / 0.0 | True | True |
| step10500ema_unc | 1 | 4754 | 0.0 | 3668 | 0.0 | 0.0 / 0.0 | True | True |
| step5000_unc | 0 | 4754 | 0.0 | 3668 | 0.0 | 0.0 / 0.0 | True | True |
| step5000_unc | 1 | 4754 | 0.0 | 3668 | 0.0 | 0.0 / 0.0 | True | True |
| step5000ema_unc | 0 | 4754 | 0.0 | 3668 | 0.0 | 0.0 / 0.0 | True | True |
| step5000ema_unc | 1 | 4754 | 0.0 | 3668 | 0.0 | 0.0 / 0.0 | True | True |

**Surface verdicts (amendment 2.3):** S2 = EXACT, S6 = EXACT, for all 9 trees and both seeds. No surface is NOT EXACT, so none is withheld.  Residual (NOT a surface defect): bit-exactness of a CPU fp32 post-hoc clamp against the GPU eval-forward clamp is unverified until K4.

## 3. Inputs (task 3)

* **Plan:** `dump_s<seed>/refcv6_extras.npz :: nav_true.traj` [N, 8, 2], full 8-slot 0.5-6 s plan, fp32, ego frame (== `plan_full_nav_true`, max abs 0.0). The clamp acts on all 8 slots; S2 / S6 are slices of the CLAMPED plan.
* **Fed ceiling per window (m/s):** derived with the model's own functions (`MaxSpeedOneHotEncoder` -> `limit_ms_of_bin`, all-zero row -> +inf, refcv8 label rule), from the dump's `v_max_ms` + `v_max_valid` (`r7_roll.py:213`), the road-type step itself: [8.333333, 13.888889, 27.777779, 33.333332] m/s (30 / 50 / 100 / 120 km/h); unknown (+inf) rows: 0; `over_ceiling` rows: 0; fp32 vs bf16 bins equal: True. Units from source: `v_max_ms` in m/s (`SPEED_MAX_STEPS_MS_V6`), `limit_ms_of_bin` returns m/s; +inf = no label (X15).
* **v0 (m/s):** the dump's per-window `v0` (`r7_roll.py:215`; = `float(pose_last[3])`, `refcv3_arm.py:2225`); equals the panel's `v0` bit-for-bit (table above). Range 0.0 .. 36.271 m/s.
* **horizons** [5, 10, 15, 20, 30, 40, 50, 60] (ticks) from `config.json['horizons']`; **anchor_dt** 0.1 (`refc.py:1748`); **brake / sel.accel_max** 2.5 m/s^2 (`config.json['selection']['sel_accel_max']`, `refc.py:1067`; the run argv carries no `--sel-accel-max`). The model object itself was not rebuilt on CPU, so `decoder.sel.accel_max` is read from the run's recorded config, not from a live attribute (stated, not hidden).

## 4. The R1 gap per tree / seed (task 4) -- clamp-OFF emitted plans

Exempt = planned max speed of the emitted plan on the waypoint grid (module `planned_max_speed`) > the fed ceiling, strict. S2 window set = 4,754 windows / 139 episodes. Excess in m/s on exempt rows. 'v0>c' = rows whose v0 is above the ceiling. 'GT>c on exempt' = exempt rows where the logged ego's own 6-s max speed exceeds the ceiling.

| tree | seed | exempt rate [cluster CI] (n) | excess med / p90 / max | v0>c | exempt AND v0>c | GT>c on exempt | binding | residual after clamp (n) med / max | filter_off exempt |
|---|---|---|---|---|---|---|---|---|---|
| step30000ema_unc | 0 | 5.30 % [3.15, 7.67] (252) | 2.1458 / 4.4699 / 9.8806 | 6.75 % | 250 of 252 | 252 of 252 | 5.30 % [3.15, 7.67] (252) | 4.69 % (223) 1.3421 / 5.7479 | 13.02 % [9.88, 16.43] (619) |
| step30000ema_unc | 1 | 5.30 % [3.13, 7.67] (252) | 2.1323 / 4.4656 / 9.3177 | 6.75 % | 250 of 252 | 252 of 252 | 5.28 % [3.13, 7.63] (251) | 4.69 % (223) 1.3421 / 5.7479 | n/a (os-only roll) |
| step20000_unc | 0 | 5.22 % [3.09, 7.57] (248) | 2.1965 / 5.0136 / 9.5256 | 6.75 % | 246 of 248 | 248 of 248 | 5.22 % [3.09, 7.57] (248) | 4.67 % (222) 1.3439 / 5.7479 | 14.56 % [11.20, 18.10] (692) |
| step20000_unc | 1 | 5.26 % [3.13, 7.60] (250) | 2.203 / 4.7432 / 9.2494 | 6.75 % | 248 of 250 | 250 of 250 | 5.24 % [3.11, 7.55] (249) | 4.69 % (223) 1.3421 / 5.7479 | n/a (os-only roll) |
| step20000ema_unc | 0 | 5.38 % [3.20, 7.73] (256) | 2.0809 / 4.5128 / 9.2585 | 6.75 % | 254 of 256 | 256 of 256 | 5.32 % [3.17, 7.67] (253) | 4.69 % (223) 1.3421 / 5.7479 | 11.46 % [8.42, 14.54] (545) |
| step20000ema_unc | 1 | 5.38 % [3.20, 7.74] (256) | 2.0785 / 4.4845 / 8.9759 | 6.75 % | 254 of 256 | 256 of 256 | 5.34 % [3.16, 7.70] (254) | 4.69 % (223) 1.3421 / 5.7479 | n/a (os-only roll) |
| step15000_unc | 0 | 4.94 % [2.86, 7.24] (235) | 2.1546 / 4.4615 / 9.0267 | 6.75 % | 234 of 235 | 235 of 235 | 4.88 % [2.84, 7.14] (232) | 4.61 % (219) 1.3639 / 5.7479 | 10.43 % [7.59, 13.34] (496) |
| step15000_unc | 1 | 4.92 % [2.84, 7.22] (234) | 2.1564 / 4.47 / 8.7522 | 6.75 % | 233 of 234 | 234 of 234 | 4.86 % [2.82, 7.14] (231) | 4.61 % (219) 1.3639 / 5.7479 | n/a (os-only roll) |
| step15000ema_unc | 0 | 5.55 % [3.34, 7.93] (264) | 2.0995 / 4.6647 / 9.3141 | 6.75 % | 261 of 264 | 264 of 264 | 5.49 % [3.29, 7.85] (261) | 4.69 % (223) 1.3421 / 5.7479 | 11.42 % [8.39, 14.49] (543) |
| step15000ema_unc | 1 | 5.57 % [3.41, 7.94] (265) | 2.0794 / 4.5711 / 9.0374 | 6.75 % | 263 of 265 | 265 of 265 | 5.51 % [3.32, 7.87] (262) | 4.71 % (224) 1.3278 / 5.7479 | n/a (os-only roll) |
| step10500_unc | 0 | 5.13 % [3.04, 7.49] (244) | 2.302 / 4.8404 / 7.7817 | 6.75 % | 241 of 244 | 244 of 244 | 5.05 % [2.96, 7.35] (240) | 4.59 % (218) 1.3833 / 5.7479 | 13.65 % [10.36, 17.14] (649) |
| step10500_unc | 1 | 5.13 % [3.04, 7.49] (244) | 2.3022 / 4.9434 / 9.684 | 6.75 % | 241 of 244 | 244 of 244 | 5.05 % [2.96, 7.35] (240) | 4.59 % (218) 1.3833 / 5.7479 | n/a (os-only roll) |
| step10500ema_unc | 0 | 5.64 % [3.46, 8.04] (268) | 2.0944 / 4.5895 / 9.7824 | 6.75 % | 264 of 268 | 268 of 268 | 5.62 % [3.45, 8.02] (267) | 4.71 % (224) 1.3278 / 5.7479 | 11.72 % [8.69, 14.76] (557) |
| step10500ema_unc | 1 | 5.62 % [3.40, 8.00] (267) | 2.0771 / 4.6025 / 9.5651 | 6.75 % | 263 of 267 | 267 of 267 | 5.60 % [3.40, 7.98] (266) | 4.71 % (224) 1.3278 / 5.7479 | n/a (os-only roll) |
| step5000_unc | 0 | 5.57 % [3.41, 7.93] (265) | 2.1476 / 4.9265 / 8.7891 | 6.75 % | 261 of 265 | 264 of 265 | 5.45 % [3.36, 7.73] (259) | 4.69 % (223) 1.3421 / 5.7479 | 13.65 % [10.25, 17.20] (649) |
| step5000_unc | 1 | 5.49 % [3.32, 7.84] (261) | 2.1529 / 4.8905 / 9.0277 | 6.75 % | 258 of 261 | 261 of 261 | 5.38 % [3.26, 7.68] (256) | 4.69 % (223) 1.3421 / 5.7479 | n/a (os-only roll) |
| step5000ema_unc | 0 | 5.70 % [3.50, 8.11] (271) | 2.0098 / 4.4388 / 10.3481 | 6.75 % | 265 of 271 | 271 of 271 | 5.70 % [3.50, 8.11] (271) | 4.71 % (224) 1.3278 / 5.7479 | 12.43 % [9.30, 15.71] (591) |
| step5000ema_unc | 1 | 5.76 % [3.58, 8.17] (274) | 2.0198 / 4.3807 / 10.1612 | 6.75 % | 267 of 274 | 274 of 274 | 5.76 % [3.58, 8.17] (274) | 4.71 % (224) 1.3278 / 5.7479 | n/a (os-only roll) |

Same rates on the S6 window subset (3,668 windows) and on the 0-2 s slots only are in `CLAMP_POSTHOC.json` and the per-tree files (`r1_gap_emitted_plan_s6_subset`, `exempt_rate_strict_s2_slots_0_2s`).

## 5. Controls (task 5)

| tree | K1a function inert-limit bit-identical (s0 / s1) | K1b pipeline reproduces banked cells (cells exact / compared: S2 s0, S6 s0, S2 s1, S6 s1) | P postcondition max violation (m/s) s0 / s1 | K3a rows whose binding changes under the 35-window roll s0 / s1 | K3b (primary, s0) |
|---|---|---|---|---|---|
| step30000ema_unc | True / True | 28/28, 28/28, 28/28, 28/28 (max |mean diff| 0.0e+00) | 1.51e-05 / 1.54e-05 | 1470 / 1469 | S2 True, S6 True |
| step20000_unc | True / True | 28/28, 28/28, 28/28, 28/28 (max |mean diff| 0.0e+00) | 1.55e-05 / 1.44e-05 | 1475 / 1474 | - |
| step20000ema_unc | True / True | 28/28, 28/28, 28/28, 28/28 (max |mean diff| 0.0e+00) | 1.8e-05 / 1.4e-05 | 1469 / 1470 | - |
| step15000_unc | True / True | 28/28, 28/28, 28/28, 28/28 (max |mean diff| 0.0e+00) | 1.22e-05 / 1.3e-05 | 1450 / 1450 | - |
| step15000ema_unc | True / True | 28/28, 28/28, 28/28, 28/28 (max |mean diff| 0.0e+00) | 1.49e-05 / 1.68e-05 | 1477 / 1474 | - |
| step10500_unc | True / True | 28/28, 28/28, 28/28, 28/28 (max |mean diff| 0.0e+00) | 1.6e-05 / 1.3e-05 | 1465 / 1466 | - |
| step10500ema_unc | True / True | 28/28, 28/28, 28/28, 28/28 (max |mean diff| 0.0e+00) | 1.29e-05 / 1.57e-05 | 1477 / 1475 | - |
| step5000_unc | True / True | 28/28, 28/28, 28/28, 28/28 (max |mean diff| 0.0e+00) | 1.66e-05 / 1.43e-05 | 1484 / 1480 | - |
| step5000ema_unc | True / True | 28/28, 28/28, 28/28, 28/28 (max |mean diff| 0.0e+00) | 1.87e-05 / 1.82e-05 | 1474 / 1475 | - |

K1b compares, per surface, the cells of 14 metrics x {os - ha0_ext, os - refcv7-50,400} (delta, lo, hi, n_windows, n_episodes, separated, n_dropped_nonfinite) plus the window means.

K3b (primary, seed 0): rolling the limits by 35 windows changed 4 of 6 S2 NR headline deltas and 1 of 1 S6 ones (pre-registered pass: >= 1 changed).

**Owed (GPU, not run here):** K2 flag-off bit parity of the clamp-carrying eval tree on >= 3 episodes; K4 post-hoc clamp == eval-forward clamp on the same subset. No clamp-ON number above is admissible as a decision number until K4.

## 6. Clamp-ON beside clamp-OFF -- the bars

OFF = the banked `PAIRED_READ_<tree>.json` verdict (the K1 re-read reproduces it exactly). ON floors = the clamp-ON |mean(os, s0) - mean(os, s1)|. The NR reference side (refcv7-50,400, ceiling filter inert, no clamp) is NOT decided here (`D-R8-NR-INSTRUMENT`).

| tree | E1 OFF | E1 ON | E3 OFF | E3 ON | NR ADE OFF -> ON | NR LONG OFF -> ON | NR LAT OFF -> ON |
|---|---|---|---|---|---|---|---|
| step30000ema_unc | FAILED | FAILED | PASS | PASS | FAIL: S6:ade_m -> FAIL: S2:ade_m, S6:ade_m | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m -> FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | FAIL: S2:LAT_heading_mae_deg -> FAIL: S2:LAT_heading_mae_deg |
| step20000_unc | FAILED | FAILED | PASS | PASS | FAIL: S2:ade_m, S6:ade_m -> FAIL: S2:ade_m, S6:ade_m | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m -> FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid -> FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid |
| step20000ema_unc | FAILED | FAILED | PASS | PASS | PASS -> FAIL: S2:ade_m, S6:ade_m | PASS -> FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid -> FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid |
| step15000_unc | FAILED | FAILED | PASS | PASS | FAIL: S2:ade_m, S6:ade_m -> FAIL: S2:ade_m, S6:ade_m | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m -> FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid -> FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid |
| step15000ema_unc | FAILED | FAILED | PASS | PASS | PASS -> FAIL: S2:ade_m, S6:ade_m | PASS -> FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid -> FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid |
| step10500_unc | FAILED | FAILED | PASS | PASS | FAIL: S2:ade_m, S6:ade_m -> FAIL: S2:ade_m, S6:ade_m | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m -> FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid -> FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid |
| step10500ema_unc | FAILED | FAILED | PASS | PASS | PASS -> FAIL: S2:ade_m | PASS -> FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | FAIL: S2:LAT_heading_mae_deg -> FAIL: S2:LAT_heading_mae_deg |
| step5000_unc | FAILED | FAILED | PASS | PASS | FAIL: S2:ade_m, S6:ade_m -> FAIL: S2:ade_m, S6:ade_m | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m -> FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid -> FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid |
| step5000ema_unc | FAILED | FAILED | PASS | PASS | PASS -> FAIL: S2:ade_m | PASS -> FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | PASS -> PASS |

E1 / E3 cells (os - ha0_ext ADE, error metric; the bar needs CI upper < 0 at both seeds) and the paired ON - OFF on refcv8's own ADE:

| tree | surface | seed | OFF banked | ON | paired ON - OFF |
|---|---|---|---|---|---|
| step30000ema_unc | S2 | 0 | -0.0008 [-0.0149, +0.0136] | +0.0621 [+0.0267, +0.1022] | +0.0629 [+0.0316, +0.1017] **W** |
| step30000ema_unc | S2 | 1 | +0.0015 [-0.0131, +0.0163] | +0.0634 [+0.0279, +0.1039] | +0.0619 [+0.0309, +0.1003] **W** |
| step30000ema_unc | S6 | 0 | -0.8146 [-1.1390, -0.5132] **B** | -0.6871 [-1.0261, -0.3526] **B** | +0.1275 [+0.0287, +0.2563] **W** |
| step30000ema_unc | S6 | 1 | -0.7902 [-1.1101, -0.4921] **B** | -0.6719 [-1.0117, -0.3391] **B** | +0.1183 [+0.0209, +0.2456] **W** |
| step20000_unc | S2 | 0 | +0.0146 [+0.0003, +0.0293] | +0.0766 [+0.0426, +0.1156] | +0.0620 [+0.0313, +0.1005] **W** |
| step20000_unc | S2 | 1 | +0.0135 [-0.0006, +0.0283] | +0.0757 [+0.0414, +0.1144] | +0.0621 [+0.0312, +0.1005] **W** |
| step20000_unc | S6 | 0 | -0.8371 [-1.1368, -0.5600] **B** | -0.7199 [-1.0362, -0.4113] **B** | +0.1172 [+0.0217, +0.2463] **W** |
| step20000_unc | S6 | 1 | -0.8342 [-1.1203, -0.5570] **B** | -0.7213 [-1.0352, -0.4136] **B** | +0.1129 [+0.0173, +0.2391] **W** |
| step20000ema_unc | S2 | 0 | -0.0020 [-0.0168, +0.0131] | +0.0609 [+0.0251, +0.1013] | +0.0629 [+0.0319, +0.1016] **W** |
| step20000ema_unc | S2 | 1 | -0.0025 [-0.0169, +0.0125] | +0.0598 [+0.0239, +0.1000] | +0.0623 [+0.0317, +0.1002] **W** |
| step20000ema_unc | S6 | 0 | -0.8957 [-1.2084, -0.6024] **B** | -0.7702 [-1.1053, -0.4392] **B** | +0.1255 [+0.0282, +0.2562] **W** |
| step20000ema_unc | S6 | 1 | -0.8878 [-1.1999, -0.5935] **B** | -0.7662 [-1.1028, -0.4437] **B** | +0.1216 [+0.0241, +0.2499] **W** |
| step15000_unc | S2 | 0 | +0.0452 [+0.0275, +0.0625] | +0.1051 [+0.0683, +0.1462] | +0.0599 [+0.0293, +0.0979] **W** |
| step15000_unc | S2 | 1 | +0.0448 [+0.0276, +0.0614] | +0.1048 [+0.0680, +0.1448] | +0.0600 [+0.0295, +0.0979] **W** |
| step15000_unc | S6 | 0 | -0.6430 [-0.9576, -0.3396] **B** | -0.5141 [-0.8499, -0.1799] **B** | +0.1289 [+0.0324, +0.2584] **W** |
| step15000_unc | S6 | 1 | -0.6473 [-0.9657, -0.3405] **B** | -0.5208 [-0.8611, -0.1885] **B** | +0.1264 [+0.0307, +0.2527] **W** |
| step15000ema_unc | S2 | 0 | +0.0003 [-0.0150, +0.0163] | +0.0627 [+0.0264, +0.1042] | +0.0624 [+0.0315, +0.1011] **W** |
| step15000ema_unc | S2 | 1 | +0.0001 [-0.0148, +0.0161] | +0.0626 [+0.0263, +0.1040] | +0.0625 [+0.0319, +0.1011] **W** |
| step15000ema_unc | S6 | 0 | -0.9159 [-1.2255, -0.6377] **B** | -0.8020 [-1.1273, -0.4823] **B** | +0.1139 [+0.0188, +0.2424] **W** |
| step15000ema_unc | S6 | 1 | -0.9219 [-1.2306, -0.6417] **B** | -0.8052 [-1.1278, -0.4849] **B** | +0.1167 [+0.0212, +0.2448] **W** |
| step10500_unc | S2 | 0 | +0.0710 [+0.0533, +0.0891] | +0.1288 [+0.0923, +0.1695] | +0.0578 [+0.0289, +0.0930] **W** |
| step10500_unc | S2 | 1 | +0.0693 [+0.0514, +0.0877] | +0.1273 [+0.0906, +0.1677] | +0.0580 [+0.0290, +0.0936] **W** |
| step10500_unc | S6 | 0 | -0.6449 [-0.9381, -0.3577] **B** | -0.5352 [-0.8597, -0.2148] **B** | +0.1098 [+0.0191, +0.2308] **W** |
| step10500_unc | S6 | 1 | -0.6447 [-0.9414, -0.3646] **B** | -0.5356 [-0.8713, -0.2161] **B** | +0.1091 [+0.0191, +0.2299] **W** |
| step10500ema_unc | S2 | 0 | -0.0039 [-0.0184, +0.0105] | +0.0593 [+0.0239, +0.0996] | +0.0633 [+0.0320, +0.1021] **W** |
| step10500ema_unc | S2 | 1 | -0.0026 [-0.0168, +0.0121] | +0.0604 [+0.0245, +0.1010] | +0.0630 [+0.0321, +0.1015] **W** |
| step10500ema_unc | S6 | 0 | -0.9957 [-1.2924, -0.7263] **B** | -0.8814 [-1.2010, -0.5709] **B** | +0.1143 [+0.0187, +0.2424] **W** |
| step10500ema_unc | S6 | 1 | -0.9652 [-1.2585, -0.6950] **B** | -0.8518 [-1.1677, -0.5464] **B** | +0.1134 [+0.0185, +0.2424] **W** |
| step5000_unc | S2 | 0 | +0.0735 [+0.0509, +0.0987] | +0.1271 [+0.0881, +0.1705] | +0.0536 [+0.0266, +0.0863] **W** |
| step5000_unc | S2 | 1 | +0.0749 [+0.0511, +0.1001] | +0.1279 [+0.0894, +0.1718] | +0.0531 [+0.0265, +0.0847] **W** |
| step5000_unc | S6 | 0 | -0.5563 [-0.8477, -0.2767] **B** | -0.4639 [-0.7848, -0.1447] **B** | +0.0924 [+0.0081, +0.2024] **W** |
| step5000_unc | S6 | 1 | -0.5314 [-0.8296, -0.2411] **B** | -0.4382 [-0.7633, -0.1050] **B** | +0.0932 [+0.0122, +0.1964] **W** |
| step5000ema_unc | S2 | 0 | -0.0004 [-0.0161, +0.0159] | +0.0635 [+0.0280, +0.1045] | +0.0639 [+0.0323, +0.1029] **W** |
| step5000ema_unc | S2 | 1 | -0.0022 [-0.0171, +0.0134] | +0.0618 [+0.0263, +0.1028] | +0.0640 [+0.0328, +0.1026] **W** |
| step5000ema_unc | S6 | 0 | -0.9876 [-1.2924, -0.7085] **B** | -0.8743 [-1.2084, -0.5591] **B** | +0.1133 [+0.0189, +0.2441] **W** |
| step5000ema_unc | S6 | 1 | -0.9922 [-1.2926, -0.7247] **B** | -0.8797 [-1.2096, -0.5744] **B** | +0.1125 [+0.0173, +0.2421] **W** |

NR headline cells (refcv8 - refcv7-50,400, error metric; FAIL iff CI lower > 0 at either seed):

| tree | headline | OFF s0 | ON s0 | OFF s1 | ON s1 | verdict OFF -> ON |
|---|---|---|---|---|---|---|
| step30000ema_unc | S2:ade_m | +0.0093 [-0.0016, +0.0197] | +0.0723 [+0.0379, +0.1134] **W** | +0.0110 [-0.0005, +0.0220] | +0.0729 [+0.0383, +0.1133] **W** | PASS -> FAIL **CHANGED** |
| step30000ema_unc | S6:ade_m | +0.1658 [+0.0268, +0.3153] **W** | +0.2933 [+0.1132, +0.4805] **W** | +0.1895 [+0.0464, +0.3391] **W** | +0.3077 [+0.1272, +0.4962] **W** | FAIL -> FAIL |
| step30000ema_unc | S2:LON_speed_mae_mps | +0.0108 [-0.0012, +0.0224] | +0.0682 [+0.0353, +0.1074] **W** | +0.0127 [+0.0004, +0.0244] **W** | +0.0689 [+0.0360, +0.1079] **W** | FAIL -> FAIL |
| step30000ema_unc | S2:LON_along_abs_final_m | +0.0241 [-0.0013, +0.0482] | +0.1367 [+0.0708, +0.2158] **W** | +0.0285 [+0.0024, +0.0532] **W** | +0.1385 [+0.0725, +0.2182] **W** | FAIL -> FAIL |
| step30000ema_unc | S2:LAT_heading_mae_deg | +0.0270 [-0.0120, +0.0649] | +0.0270 [-0.0120, +0.0649] | +0.0441 [+0.0001, +0.0879] **W** | +0.0441 [+0.0001, +0.0879] **W** | FAIL -> FAIL |
| step30000ema_unc | S2:LAT_curvature_mae_1pm_valid | +0.0000 [-0.0002, +0.0002] | +0.0000 [-0.0002, +0.0002] | +0.0002 [-0.0001, +0.0004] | +0.0002 [-0.0001, +0.0004] | PASS -> PASS |
| step30000ema_unc | S2:LAT_yaw_rate_mae_radps_valid | +0.0002 [-0.0006, +0.0010] | +0.0002 [-0.0006, +0.0010] | +0.0005 [-0.0004, +0.0014] | +0.0005 [-0.0004, +0.0014] | PASS -> PASS |
| step20000_unc | S2:ade_m | +0.0248 [+0.0139, +0.0359] **W** | +0.0868 [+0.0534, +0.1258] **W** | +0.0231 [+0.0125, +0.0340] **W** | +0.0853 [+0.0520, +0.1241] **W** | FAIL -> FAIL |
| step20000_unc | S6:ade_m | +0.1433 [+0.0166, +0.2685] **W** | +0.2605 [+0.0967, +0.4354] **W** | +0.1455 [+0.0247, +0.2652] **W** | +0.2583 [+0.0939, +0.4329] **W** | FAIL -> FAIL |
| step20000_unc | S2:LON_speed_mae_mps | +0.0170 [+0.0058, +0.0269] **W** | +0.0738 [+0.0422, +0.1116] **W** | +0.0151 [+0.0042, +0.0252] **W** | +0.0720 [+0.0406, +0.1092] **W** | FAIL -> FAIL |
| step20000_unc | S2:LON_along_abs_final_m | +0.0324 [+0.0071, +0.0563] **W** | +0.1433 [+0.0785, +0.2209] **W** | +0.0273 [+0.0027, +0.0510] **W** | +0.1386 [+0.0744, +0.2150] **W** | FAIL -> FAIL |
| step20000_unc | S2:LAT_heading_mae_deg | +0.0840 [+0.0404, +0.1291] **W** | +0.0840 [+0.0404, +0.1291] **W** | +0.0973 [+0.0512, +0.1486] **W** | +0.0973 [+0.0512, +0.1486] **W** | FAIL -> FAIL |
| step20000_unc | S2:LAT_curvature_mae_1pm_valid | +0.0002 [-0.0000, +0.0004] | +0.0002 [-0.0000, +0.0004] | +0.0003 [+0.0001, +0.0005] **W** | +0.0003 [+0.0001, +0.0005] **W** | FAIL -> FAIL |
| step20000_unc | S2:LAT_yaw_rate_mae_radps_valid | +0.0011 [+0.0002, +0.0020] **W** | +0.0011 [+0.0002, +0.0020] **W** | +0.0014 [+0.0005, +0.0024] **W** | +0.0014 [+0.0005, +0.0024] **W** | FAIL -> FAIL |
| step20000ema_unc | S2:ade_m | +0.0082 [-0.0042, +0.0202] | +0.0711 [+0.0363, +0.1129] **W** | +0.0071 [-0.0050, +0.0188] | +0.0694 [+0.0352, +0.1114] **W** | PASS -> FAIL **CHANGED** |
| step20000ema_unc | S6:ade_m | +0.0847 [-0.0568, +0.2250] | +0.2101 [+0.0324, +0.3991] **W** | +0.0918 [-0.0534, +0.2344] | +0.2134 [+0.0369, +0.4077] **W** | PASS -> FAIL **CHANGED** |
| step20000ema_unc | S2:LON_speed_mae_mps | +0.0073 [-0.0062, +0.0200] | +0.0648 [+0.0322, +0.1047] **W** | +0.0062 [-0.0068, +0.0190] | +0.0629 [+0.0303, +0.1027] **W** | PASS -> FAIL **CHANGED** |
| step20000ema_unc | S2:LON_along_abs_final_m | +0.0172 [-0.0113, +0.0429] | +0.1297 [+0.0646, +0.2096] **W** | +0.0152 [-0.0124, +0.0413] | +0.1260 [+0.0612, +0.2058] **W** | PASS -> FAIL **CHANGED** |
| step20000ema_unc | S2:LAT_heading_mae_deg | +0.0424 [+0.0009, +0.0849] **W** | +0.0424 [+0.0009, +0.0849] **W** | +0.0600 [+0.0199, +0.1026] **W** | +0.0600 [+0.0199, +0.1026] **W** | FAIL -> FAIL |
| step20000ema_unc | S2:LAT_curvature_mae_1pm_valid | +0.0001 [-0.0002, +0.0003] | +0.0001 [-0.0002, +0.0003] | +0.0003 [+0.0000, +0.0005] **W** | +0.0003 [+0.0000, +0.0005] **W** | FAIL -> FAIL |
| step20000ema_unc | S2:LAT_yaw_rate_mae_radps_valid | +0.0002 [-0.0005, +0.0011] | +0.0002 [-0.0005, +0.0011] | +0.0005 [-0.0003, +0.0014] | +0.0005 [-0.0003, +0.0014] | PASS -> PASS |
| step15000_unc | S2:ade_m | +0.0554 [+0.0401, +0.0713] **W** | +0.1153 [+0.0793, +0.1577] **W** | +0.0544 [+0.0392, +0.0689] **W** | +0.1144 [+0.0787, +0.1566] **W** | FAIL -> FAIL |
| step15000_unc | S6:ade_m | +0.3374 [+0.1520, +0.5167] **W** | +0.4663 [+0.2564, +0.6739] **W** | +0.3324 [+0.1466, +0.5114] **W** | +0.4588 [+0.2452, +0.6611] **W** | FAIL -> FAIL |
| step15000_unc | S2:LON_speed_mae_mps | +0.0461 [+0.0294, +0.0623] **W** | +0.1015 [+0.0674, +0.1433] **W** | +0.0453 [+0.0292, +0.0608] **W** | +0.1006 [+0.0673, +0.1420] **W** | FAIL -> FAIL |
| step15000_unc | S2:LON_along_abs_final_m | +0.0899 [+0.0559, +0.1243] **W** | +0.1989 [+0.1288, +0.2837] **W** | +0.0888 [+0.0548, +0.1214] **W** | +0.1976 [+0.1288, +0.2812] **W** | FAIL -> FAIL |
| step15000_unc | S2:LAT_heading_mae_deg | +0.1261 [+0.0677, +0.1889] **W** | +0.1261 [+0.0677, +0.1889] **W** | +0.1409 [+0.0790, +0.2076] **W** | +0.1409 [+0.0790, +0.2076] **W** | FAIL -> FAIL |
| step15000_unc | S2:LAT_curvature_mae_1pm_valid | +0.0005 [+0.0002, +0.0008] **W** | +0.0005 [+0.0002, +0.0008] **W** | +0.0007 [+0.0003, +0.0011] **W** | +0.0007 [+0.0003, +0.0011] **W** | FAIL -> FAIL |
| step15000_unc | S2:LAT_yaw_rate_mae_radps_valid | +0.0019 [+0.0008, +0.0031] **W** | +0.0019 [+0.0008, +0.0031] **W** | +0.0023 [+0.0011, +0.0036] **W** | +0.0023 [+0.0011, +0.0036] **W** | FAIL -> FAIL |
| step15000ema_unc | S2:ade_m | +0.0105 [-0.0016, +0.0224] | +0.0729 [+0.0377, +0.1160] **W** | +0.0097 [-0.0019, +0.0218] | +0.0722 [+0.0371, +0.1143] **W** | PASS -> FAIL **CHANGED** |
| step15000ema_unc | S6:ade_m | +0.0645 [-0.0740, +0.1998] | +0.1784 [+0.0033, +0.3631] **W** | +0.0577 [-0.0811, +0.1940] | +0.1745 [-0.0039, +0.3657] | PASS -> FAIL **CHANGED** |
| step15000ema_unc | S2:LON_speed_mae_mps | +0.0076 [-0.0054, +0.0205] | +0.0645 [+0.0308, +0.1046] **W** | +0.0064 [-0.0068, +0.0192] | +0.0634 [+0.0301, +0.1033] **W** | PASS -> FAIL **CHANGED** |
| step15000ema_unc | S2:LON_along_abs_final_m | +0.0183 [-0.0093, +0.0450] | +0.1295 [+0.0615, +0.2111] **W** | +0.0164 [-0.0120, +0.0422] | +0.1278 [+0.0608, +0.2083] **W** | PASS -> FAIL **CHANGED** |
| step15000ema_unc | S2:LAT_heading_mae_deg | +0.0713 [+0.0277, +0.1167] **W** | +0.0713 [+0.0277, +0.1167] **W** | +0.0826 [+0.0382, +0.1281] **W** | +0.0826 [+0.0382, +0.1281] **W** | FAIL -> FAIL |
| step15000ema_unc | S2:LAT_curvature_mae_1pm_valid | +0.0002 [-0.0001, +0.0004] | +0.0002 [-0.0001, +0.0004] | +0.0003 [+0.0001, +0.0006] **W** | +0.0003 [+0.0001, +0.0006] **W** | FAIL -> FAIL |
| step15000ema_unc | S2:LAT_yaw_rate_mae_radps_valid | +0.0007 [-0.0001, +0.0016] | +0.0007 [-0.0001, +0.0016] | +0.0010 [+0.0002, +0.0019] **W** | +0.0010 [+0.0002, +0.0019] **W** | FAIL -> FAIL |
| step10500_unc | S2:ade_m | +0.0812 [+0.0650, +0.0972] **W** | +0.1390 [+0.1032, +0.1797] **W** | +0.0789 [+0.0626, +0.0952] **W** | +0.1369 [+0.1003, +0.1770] **W** | FAIL -> FAIL |
| step10500_unc | S6:ade_m | +0.3355 [+0.1831, +0.4958] **W** | +0.4452 [+0.2666, +0.6459] **W** | +0.3349 [+0.1756, +0.4954] **W** | +0.4440 [+0.2598, +0.6466] **W** | FAIL -> FAIL |
| step10500_unc | S2:LON_speed_mae_mps | +0.0606 [+0.0447, +0.0767] **W** | +0.1142 [+0.0808, +0.1522] **W** | +0.0578 [+0.0421, +0.0742] **W** | +0.1116 [+0.0778, +0.1501] **W** | FAIL -> FAIL |
| step10500_unc | S2:LON_along_abs_final_m | +0.1124 [+0.0788, +0.1461] **W** | +0.2174 [+0.1494, +0.2943] **W** | +0.1066 [+0.0729, +0.1409] **W** | +0.2121 [+0.1430, +0.2913] **W** | FAIL -> FAIL |
| step10500_unc | S2:LAT_heading_mae_deg | +0.2450 [+0.1838, +0.3134] **W** | +0.2450 [+0.1838, +0.3134] **W** | +0.2541 [+0.1850, +0.3312] **W** | +0.2541 [+0.1850, +0.3312] **W** | FAIL -> FAIL |
| step10500_unc | S2:LAT_curvature_mae_1pm_valid | +0.0008 [+0.0004, +0.0012] **W** | +0.0008 [+0.0004, +0.0012] **W** | +0.0009 [+0.0005, +0.0013] **W** | +0.0009 [+0.0005, +0.0013] **W** | FAIL -> FAIL |
| step10500_unc | S2:LAT_yaw_rate_mae_radps_valid | +0.0037 [+0.0025, +0.0050] **W** | +0.0037 [+0.0025, +0.0050] **W** | +0.0038 [+0.0025, +0.0053] **W** | +0.0038 [+0.0025, +0.0053] **W** | FAIL -> FAIL |
| step10500ema_unc | S2:ade_m | +0.0062 [-0.0047, +0.0172] | +0.0695 [+0.0349, +0.1116] **W** | +0.0070 [-0.0040, +0.0185] | +0.0700 [+0.0357, +0.1116] **W** | PASS -> FAIL **CHANGED** |
| step10500ema_unc | S6:ade_m | -0.0153 [-0.1314, +0.1022] | +0.0990 [-0.0619, +0.2803] | +0.0144 [-0.0959, +0.1294] | +0.1278 [-0.0289, +0.3064] | PASS -> PASS |
| step10500ema_unc | S2:LON_speed_mae_mps | +0.0028 [-0.0092, +0.0143] | +0.0606 [+0.0272, +0.1000] **W** | +0.0041 [-0.0075, +0.0155] | +0.0616 [+0.0288, +0.1008] **W** | PASS -> FAIL **CHANGED** |
| step10500ema_unc | S2:LON_along_abs_final_m | +0.0098 [-0.0153, +0.0337] | +0.1226 [+0.0570, +0.2022] **W** | +0.0125 [-0.0118, +0.0367] | +0.1251 [+0.0598, +0.2031] **W** | PASS -> FAIL **CHANGED** |
| step10500ema_unc | S2:LAT_heading_mae_deg | +0.0428 [-0.0006, +0.0868] | +0.0428 [-0.0006, +0.0868] | +0.0500 [+0.0002, +0.0985] **W** | +0.0500 [+0.0002, +0.0985] **W** | FAIL -> FAIL |
| step10500ema_unc | S2:LAT_curvature_mae_1pm_valid | -0.0000 [-0.0002, +0.0002] | -0.0000 [-0.0002, +0.0002] | +0.0001 [-0.0001, +0.0004] | +0.0001 [-0.0001, +0.0004] | PASS -> PASS |
| step10500ema_unc | S2:LAT_yaw_rate_mae_radps_valid | +0.0003 [-0.0005, +0.0011] | +0.0003 [-0.0005, +0.0011] | +0.0005 [-0.0004, +0.0014] | +0.0005 [-0.0004, +0.0014] | PASS -> PASS |
| step5000_unc | S2:ade_m | +0.0837 [+0.0608, +0.1092] **W** | +0.1373 [+0.0981, +0.1823] **W** | +0.0845 [+0.0606, +0.1106] **W** | +0.1375 [+0.0981, +0.1827] **W** | FAIL -> FAIL |
| step5000_unc | S6:ade_m | +0.4241 [+0.2338, +0.6260] **W** | +0.5165 [+0.2909, +0.7535] **W** | +0.4482 [+0.2486, +0.6615] **W** | +0.5414 [+0.3114, +0.7894] **W** | FAIL -> FAIL |
| step5000_unc | S2:LON_speed_mae_mps | +0.0715 [+0.0482, +0.0962] **W** | +0.1202 [+0.0828, +0.1637] **W** | +0.0718 [+0.0476, +0.0975] **W** | +0.1203 [+0.0830, +0.1641] **W** | FAIL -> FAIL |
| step5000_unc | S2:LON_along_abs_final_m | +0.1311 [+0.0832, +0.1821] **W** | +0.2273 [+0.1505, +0.3161] **W** | +0.1317 [+0.0831, +0.1849] **W** | +0.2272 [+0.1504, +0.3164] **W** | FAIL -> FAIL |
| step5000_unc | S2:LAT_heading_mae_deg | +0.3007 [+0.2043, +0.4116] **W** | +0.3007 [+0.2043, +0.4116] **W** | +0.3274 [+0.2323, +0.4322] **W** | +0.3274 [+0.2323, +0.4322] **W** | FAIL -> FAIL |
| step5000_unc | S2:LAT_curvature_mae_1pm_valid | +0.0009 [+0.0005, +0.0014] **W** | +0.0009 [+0.0005, +0.0014] **W** | +0.0011 [+0.0007, +0.0015] **W** | +0.0011 [+0.0007, +0.0015] **W** | FAIL -> FAIL |
| step5000_unc | S2:LAT_yaw_rate_mae_radps_valid | +0.0051 [+0.0034, +0.0069] **W** | +0.0051 [+0.0034, +0.0069] **W** | +0.0057 [+0.0040, +0.0075] **W** | +0.0057 [+0.0040, +0.0075] **W** | FAIL -> FAIL |
| step5000ema_unc | S2:ade_m | +0.0098 [-0.0007, +0.0203] | +0.0737 [+0.0388, +0.1144] **W** | +0.0074 [-0.0024, +0.0176] | +0.0714 [+0.0369, +0.1128] **W** | PASS -> FAIL **CHANGED** |
| step5000ema_unc | S6:ade_m | -0.0072 [-0.1067, +0.0977] | +0.1061 [-0.0382, +0.2685] | -0.0126 [-0.1089, +0.0848] | +0.0999 [-0.0430, +0.2563] | PASS -> PASS |
| step5000ema_unc | S2:LON_speed_mae_mps | +0.0092 [-0.0026, +0.0219] | +0.0681 [+0.0346, +0.1062] **W** | +0.0066 [-0.0044, +0.0180] | +0.0653 [+0.0324, +0.1032] **W** | PASS -> FAIL **CHANGED** |
| step5000ema_unc | S2:LON_along_abs_final_m | +0.0206 [-0.0037, +0.0463] | +0.1353 [+0.0669, +0.2128] **W** | +0.0159 [-0.0063, +0.0391] | +0.1307 [+0.0643, +0.2075] **W** | PASS -> FAIL **CHANGED** |
| step5000ema_unc | S2:LAT_heading_mae_deg | +0.0133 [-0.0161, +0.0443] | +0.0133 [-0.0161, +0.0443] | +0.0157 [-0.0150, +0.0485] | +0.0157 [-0.0150, +0.0485] | PASS -> PASS |
| step5000ema_unc | S2:LAT_curvature_mae_1pm_valid | +0.0000 [-0.0001, +0.0002] | +0.0000 [-0.0001, +0.0002] | +0.0001 [-0.0001, +0.0002] | +0.0001 [-0.0001, +0.0002] | PASS -> PASS |
| step5000ema_unc | S2:LAT_yaw_rate_mae_radps_valid | +0.0002 [-0.0003, +0.0007] | +0.0002 [-0.0003, +0.0007] | +0.0002 [-0.0003, +0.0008] | +0.0002 [-0.0003, +0.0008] | PASS -> PASS |

## 7. The four families at the primary tree (step 30,000 EMA), both seeds, clamp-OFF (banked) beside clamp-ON

Per surface and metric: `os - ha0_ext` and `os - refcv7-50,400` (OFF, ON), and the paired ON - OFF on refcv8's own metric. TACTICAL rows here are the TRAJECTORY-DERIVED decisions (they move with the plan); the tactical HEAD logits are not touched by the clamp. STRATEGIC: not applicable (`--no-strategic`, n = 0).

**S2, inference seed 0** (4754 windows / 139 episodes; 252 windows changed; os mean ADE ON 0.3507 m vs OFF 0.2878 m)

| family | metric | os - ha0_ext OFF | os - ha0_ext ON | os - refcv7 OFF | os - refcv7 ON | paired ON - OFF |
|---|---|---|---|---|---|---|
| ADE | ADE (mean over the surface's instants) (m) | -0.0008 [-0.0149, +0.0136] | +0.0621 [+0.0267, +0.1022] | +0.0093 [-0.0016, +0.0197] | +0.0723 [+0.0379, +0.1134] **W** | +0.0629 [+0.0316, +0.1017] **W** |
| ADE | FDE (final instant) (m) | -0.0156 [-0.0508, +0.0202] | +0.0941 [+0.0234, +0.1747] | +0.0204 [-0.0054, +0.0449] | +0.1300 [+0.0662, +0.2080] **W** | +0.1097 [+0.0520, +0.1823] **W** |
| LONGITUDINAL | speed MAE (m/s) | +0.0148 [+0.0014, +0.0280] | +0.0723 [+0.0381, +0.1101] | +0.0108 [-0.0012, +0.0224] | +0.0682 [+0.0353, +0.1074] **W** | +0.0574 [+0.0284, +0.0936] **W** |
| LONGITUDINAL | along-track MAE (m) | +0.0031 [-0.0091, +0.0156] | +0.0678 [+0.0325, +0.1080] | +0.0117 [+0.0009, +0.0220] **W** | +0.0764 [+0.0410, +0.1184] **W** | +0.0647 [+0.0329, +0.1040] **W** |
| LONGITUDINAL | |along-track| at the final instant (m) | +0.0260 [-0.0007, +0.0533] | +0.1385 [+0.0702, +0.2148] | +0.0241 [-0.0013, +0.0482] | +0.1367 [+0.0708, +0.2158] **W** | +0.1125 [+0.0538, +0.1856] **W** |
| LONGITUDINAL | target-speed accuracy (+-0.5 m/s, fraction of steps) (frac) | -0.0236 [-0.0347, -0.0126] | -0.0598 [-0.0828, -0.0393] | -0.0115 [-0.0216, -0.0011] | -0.0476 [-0.0695, -0.0282] | -0.0362 [-0.0559, -0.0198] |
| LONGITUDINAL | accel MAE (m/s2) | +0.0559 [+0.0363, +0.0751] | +0.0876 [+0.0596, +0.1175] | +0.0024 [-0.0128, +0.0170] | +0.0342 [+0.0110, +0.0595] **W** | +0.0318 [+0.0141, +0.0532] **W** |
| LATERAL | heading MAE (deg) | -0.0673 [-0.1754, +0.0407] | -0.0673 [-0.1754, +0.0407] | +0.0270 [-0.0120, +0.0649] | +0.0270 [-0.0120, +0.0649] | -0.0000 [-0.0000, +0.0000] |
| LATERAL | curvature MAE (masked) (1/m) | -0.0001 [-0.0006, +0.0004] | -0.0001 [-0.0006, +0.0004] | +0.0000 [-0.0002, +0.0002] | +0.0000 [-0.0002, +0.0002] | -0.0000 [-0.0000, +0.0000] |
| LATERAL | yaw-rate MAE (masked) (rad/s) | -0.0046 [-0.0074, -0.0018] **B** | -0.0046 [-0.0074, -0.0018] **B** | +0.0002 [-0.0006, +0.0010] | +0.0002 [-0.0006, +0.0010] | +0.0000 [-0.0000, +0.0000] |
| LATERAL | cross-track MAE (m) | -0.0037 [-0.0141, +0.0063] | -0.0033 [-0.0138, +0.0070] | +0.0012 [-0.0035, +0.0055] | +0.0017 [-0.0030, +0.0059] | +0.0005 [-0.0001, +0.0012] |
| LATERAL | |cross-track| at the final instant (m) | -0.0516 [-0.0799, -0.0245] **B** | -0.0504 [-0.0791, -0.0235] **B** | +0.0040 [-0.0058, +0.0137] | +0.0052 [-0.0045, +0.0149] | +0.0012 [-0.0003, +0.0029] |
| TACTICAL | trajectory-derived lateral decision correct (frac) | +0.0185 [+0.0074, +0.0294] | +0.0185 [+0.0074, +0.0294] | -0.0021 [-0.0065, +0.0023] | -0.0021 [-0.0065, +0.0023] | +0.0000 [+0.0000, +0.0000] |
| TACTICAL | trajectory-derived longitudinal decision correct (frac) | -0.0168 [-0.0317, -0.0019] | -0.0307 [-0.0535, -0.0113] | -0.0036 [-0.0164, +0.0095] | -0.0175 [-0.0397, +0.0002] | -0.0139 [-0.0314, -0.0023] |

**S6, inference seed 0** (3668 windows / 139 episodes; 200 windows changed; os mean ADE ON 2.8608 m vs OFF 2.7333 m)

| family | metric | os - ha0_ext OFF | os - ha0_ext ON | os - refcv7 OFF | os - refcv7 ON | paired ON - OFF |
|---|---|---|---|---|---|---|
| ADE | ADE (mean over the surface's instants) (m) | -0.8146 [-1.1390, -0.5132] **B** | -0.6871 [-1.0261, -0.3526] **B** | +0.1658 [+0.0268, +0.3153] **W** | +0.2933 [+0.1132, +0.4805] **W** | +0.1275 [+0.0287, +0.2563] **W** |
| ADE | FDE (final instant) (m) | -2.3980 [-3.3067, -1.5507] **B** | -2.2591 [-3.1925, -1.3830] **B** | +0.4763 [+0.0876, +0.9005] **W** | +0.6151 [+0.1888, +1.0492] **W** | +0.1388 [-0.0182, +0.3659] |
| LONGITUDINAL | speed MAE (m/s) | -0.0849 [-0.1762, +0.0029] | -0.0514 [-0.1481, +0.0455] | +0.0779 [+0.0166, +0.1424] **W** | +0.1114 [+0.0403, +0.1842] **W** | +0.0335 [+0.0063, +0.0704] **W** |
| LONGITUDINAL | along-track MAE (m) | -0.2882 [-0.5202, -0.0637] **B** | -0.1584 [-0.4128, +0.1080] | +0.1678 [+0.0285, +0.3127] **W** | +0.2976 [+0.1185, +0.4896] **W** | +0.1297 [+0.0299, +0.2605] **W** |
| LONGITUDINAL | |along-track| at the final instant (m) | -0.9716 [-1.6374, -0.3189] **B** | -0.8313 [-1.5267, -0.1448] **B** | +0.4713 [+0.0921, +0.8793] **W** | +0.6116 [+0.1844, +1.0599] **W** | +0.1403 [-0.0182, +0.3693] |
| LONGITUDINAL | target-speed accuracy (+-0.5 m/s, fraction of steps) (frac) | -0.0175 [-0.0381, +0.0044] | -0.0341 [-0.0585, -0.0085] | -0.0294 [-0.0480, -0.0114] | -0.0460 [-0.0694, -0.0245] | -0.0166 [-0.0311, -0.0063] |
| LONGITUDINAL | accel MAE (m/s2) | -0.0356 [-0.0747, +0.0034] | -0.0262 [-0.0652, +0.0122] | +0.0393 [+0.0135, +0.0669] **W** | +0.0487 [+0.0219, +0.0776] **W** | +0.0094 [+0.0028, +0.0181] **W** |
| LATERAL | heading MAE (deg) | -2.5601 [-3.4165, -1.7269] **B** | -2.5601 [-3.4165, -1.7269] **B** | +0.1295 [-0.0405, +0.3237] | +0.1295 [-0.0405, +0.3237] | +0.0000 [-0.0000, +0.0001] |
| LATERAL | curvature MAE (masked) (1/m) | -0.0019 [-0.0030, -0.0008] **B** | -0.0019 [-0.0030, -0.0008] **B** | +0.0000 [-0.0003, +0.0004] | +0.0000 [-0.0003, +0.0004] | +0.0000 [+0.0000, +0.0000] **W** |
| LATERAL | yaw-rate MAE (masked) (rad/s) | -0.0194 [-0.0265, -0.0122] **B** | -0.0194 [-0.0265, -0.0122] **B** | +0.0003 [-0.0010, +0.0019] | +0.0003 [-0.0010, +0.0018] | -0.0000 [-0.0000, +0.0000] |
| LATERAL | cross-track MAE (m) | -0.8059 [-1.0276, -0.5929] **B** | -0.8042 [-1.0269, -0.5910] **B** | +0.0209 [-0.0251, +0.0685] | +0.0225 [-0.0233, +0.0696] | +0.0016 [-0.0025, +0.0072] |
| LATERAL | |cross-track| at the final instant (m) | -2.2144 [-2.8002, -1.6547] **B** | -2.2146 [-2.8010, -1.6564] **B** | +0.0546 [-0.0678, +0.1827] | +0.0544 [-0.0686, +0.1835] | -0.0002 [-0.0075, +0.0068] |
| TACTICAL | trajectory-derived lateral decision correct (frac) | +0.0911 [+0.0679, +0.1134] | +0.0911 [+0.0679, +0.1134] | -0.0038 [-0.0155, +0.0071] | -0.0038 [-0.0155, +0.0071] | +0.0000 [+0.0000, +0.0000] |
| TACTICAL | trajectory-derived longitudinal decision correct (frac) | +0.0294 [-0.0063, +0.0642] | +0.0237 [-0.0120, +0.0573] | -0.0226 [-0.0560, +0.0079] | -0.0284 [-0.0630, +0.0033] | -0.0057 [-0.0155, +0.0014] |

**S2, inference seed 1** (4754 windows / 139 episodes; 251 windows changed; os mean ADE ON 0.3520 m vs OFF 0.2901 m)

| family | metric | os - ha0_ext OFF | os - ha0_ext ON | os - refcv7 OFF | os - refcv7 ON | paired ON - OFF |
|---|---|---|---|---|---|---|
| ADE | ADE (mean over the surface's instants) (m) | +0.0015 [-0.0131, +0.0163] | +0.0634 [+0.0279, +0.1039] | +0.0110 [-0.0005, +0.0220] | +0.0729 [+0.0383, +0.1133] **W** | +0.0619 [+0.0309, +0.1003] **W** |
| ADE | FDE (final instant) (m) | -0.0102 [-0.0457, +0.0258] | +0.0973 [+0.0257, +0.1779] | +0.0245 [-0.0020, +0.0493] | +0.1320 [+0.0672, +0.2097] **W** | +0.1074 [+0.0505, +0.1788] **W** |
| LONGITUDINAL | speed MAE (m/s) | +0.0175 [+0.0043, +0.0304] | +0.0737 [+0.0397, +0.1117] | +0.0127 [+0.0004, +0.0244] **W** | +0.0689 [+0.0360, +0.1079] **W** | +0.0563 [+0.0276, +0.0919] **W** |
| LONGITUDINAL | along-track MAE (m) | +0.0056 [-0.0068, +0.0183] | +0.0691 [+0.0340, +0.1089] | +0.0136 [+0.0026, +0.0240] **W** | +0.0771 [+0.0420, +0.1191] **W** | +0.0635 [+0.0321, +0.1024] **W** |
| LONGITUDINAL | |along-track| at the final instant (m) | +0.0315 [+0.0053, +0.0589] | +0.1415 [+0.0732, +0.2189] | +0.0285 [+0.0024, +0.0532] **W** | +0.1385 [+0.0725, +0.2182] **W** | +0.1100 [+0.0525, +0.1825] **W** |
| LONGITUDINAL | target-speed accuracy (+-0.5 m/s, fraction of steps) (frac) | -0.0250 [-0.0360, -0.0140] | -0.0604 [-0.0836, -0.0398] | -0.0114 [-0.0214, -0.0015] | -0.0468 [-0.0692, -0.0267] | -0.0353 [-0.0548, -0.0192] |
| LONGITUDINAL | accel MAE (m/s2) | +0.0588 [+0.0397, +0.0769] | +0.0897 [+0.0619, +0.1185] | +0.0045 [-0.0107, +0.0186] | +0.0355 [+0.0117, +0.0608] **W** | +0.0309 [+0.0135, +0.0524] **W** |
| LATERAL | heading MAE (deg) | -0.0618 [-0.1733, +0.0481] | -0.0618 [-0.1733, +0.0481] | +0.0441 [+0.0001, +0.0879] **W** | +0.0441 [+0.0001, +0.0879] **W** | -0.0000 [-0.0000, +0.0000] |
| LATERAL | curvature MAE (masked) (1/m) | -0.0001 [-0.0006, +0.0004] | -0.0001 [-0.0006, +0.0004] | +0.0002 [-0.0001, +0.0004] | +0.0002 [-0.0001, +0.0004] | -0.0000 [-0.0000, +0.0000] |
| LATERAL | yaw-rate MAE (masked) (rad/s) | -0.0045 [-0.0073, -0.0017] **B** | -0.0045 [-0.0073, -0.0017] **B** | +0.0005 [-0.0004, +0.0014] | +0.0005 [-0.0004, +0.0014] | +0.0000 [-0.0000, +0.0000] |
| LATERAL | cross-track MAE (m) | -0.0033 [-0.0139, +0.0070] | -0.0028 [-0.0137, +0.0077] | +0.0016 [-0.0033, +0.0062] | +0.0021 [-0.0029, +0.0068] | +0.0005 [-0.0001, +0.0012] |
| LATERAL | |cross-track| at the final instant (m) | -0.0504 [-0.0795, -0.0229] **B** | -0.0492 [-0.0789, -0.0217] **B** | +0.0050 [-0.0059, +0.0153] | +0.0062 [-0.0045, +0.0165] | +0.0012 [-0.0002, +0.0029] |
| TACTICAL | trajectory-derived lateral decision correct (frac) | +0.0164 [+0.0055, +0.0277] | +0.0164 [+0.0055, +0.0277] | -0.0046 [-0.0092, -0.0002] | -0.0046 [-0.0092, -0.0002] | +0.0000 [+0.0000, +0.0000] |
| TACTICAL | trajectory-derived longitudinal decision correct (frac) | -0.0183 [-0.0326, -0.0040] | -0.0330 [-0.0551, -0.0139] | -0.0103 [-0.0227, +0.0027] | -0.0250 [-0.0461, -0.0073] | -0.0147 [-0.0322, -0.0027] |

**S6, inference seed 1** (3668 windows / 139 episodes; 200 windows changed; os mean ADE ON 2.8760 m vs OFF 2.7577 m)

| family | metric | os - ha0_ext OFF | os - ha0_ext ON | os - refcv7 OFF | os - refcv7 ON | paired ON - OFF |
|---|---|---|---|---|---|---|
| ADE | ADE (mean over the surface's instants) (m) | -0.7902 [-1.1101, -0.4921] **B** | -0.6719 [-1.0117, -0.3391] **B** | +0.1895 [+0.0464, +0.3391] **W** | +0.3077 [+0.1272, +0.4962] **W** | +0.1183 [+0.0209, +0.2456] **W** |
| ADE | FDE (final instant) (m) | -2.3467 [-3.2443, -1.4920] **B** | -2.2274 [-3.1518, -1.3540] **B** | +0.5214 [+0.1317, +0.9354] **W** | +0.6407 [+0.2107, +1.0844] **W** | +0.1193 [-0.0356, +0.3356] |
| LONGITUDINAL | speed MAE (m/s) | -0.0797 [-0.1712, +0.0086] | -0.0489 [-0.1469, +0.0473] | +0.0802 [+0.0182, +0.1444] **W** | +0.1110 [+0.0396, +0.1845] **W** | +0.0308 [+0.0043, +0.0675] **W** |
| LONGITUDINAL | along-track MAE (m) | -0.2721 [-0.5066, -0.0496] **B** | -0.1511 [-0.4097, +0.1143] | +0.1848 [+0.0442, +0.3237] **W** | +0.3058 [+0.1323, +0.4951] **W** | +0.1210 [+0.0226, +0.2496] **W** |
| LONGITUDINAL | |along-track| at the final instant (m) | -0.9418 [-1.6092, -0.2884] **B** | -0.8191 [-1.5191, -0.1483] **B** | +0.4998 [+0.1148, +0.8971] **W** | +0.6225 [+0.2034, +1.0677] **W** | +0.1227 [-0.0338, +0.3410] |
| LONGITUDINAL | target-speed accuracy (+-0.5 m/s, fraction of steps) (frac) | -0.0190 [-0.0404, +0.0031] | -0.0357 [-0.0608, -0.0104] | -0.0279 [-0.0469, -0.0088] | -0.0445 [-0.0686, -0.0233] | -0.0167 [-0.0318, -0.0063] |
| LONGITUDINAL | accel MAE (m/s2) | -0.0348 [-0.0736, +0.0038] | -0.0258 [-0.0653, +0.0126] | +0.0385 [+0.0127, +0.0665] **W** | +0.0475 [+0.0204, +0.0759] **W** | +0.0090 [+0.0025, +0.0174] **W** |
| LATERAL | heading MAE (deg) | -2.4835 [-3.3286, -1.6456] **B** | -2.4834 [-3.3286, -1.6455] **B** | +0.1935 [+0.0026, +0.3996] **W** | +0.1935 [+0.0026, +0.3996] **W** | +0.0000 [-0.0000, +0.0001] |
| LATERAL | curvature MAE (masked) (1/m) | -0.0018 [-0.0029, -0.0007] **B** | -0.0018 [-0.0029, -0.0007] **B** | +0.0002 [-0.0001, +0.0006] | +0.0002 [-0.0001, +0.0006] | +0.0000 [+0.0000, +0.0000] **W** |
| LATERAL | yaw-rate MAE (masked) (rad/s) | -0.0190 [-0.0261, -0.0119] **B** | -0.0190 [-0.0261, -0.0119] **B** | +0.0009 [-0.0005, +0.0024] | +0.0009 [-0.0005, +0.0024] | -0.0000 [-0.0000, +0.0000] |
| LATERAL | cross-track MAE (m) | -0.7948 [-1.0128, -0.5841] **B** | -0.7936 [-1.0125, -0.5828] **B** | +0.0261 [-0.0236, +0.0773] | +0.0273 [-0.0226, +0.0790] | +0.0012 [-0.0027, +0.0063] |
| LATERAL | |cross-track| at the final instant (m) | -2.1888 [-2.7663, -1.6315] **B** | -2.1902 [-2.7680, -1.6321] **B** | +0.0662 [-0.0631, +0.2032] | +0.0647 [-0.0633, +0.2020] | -0.0015 [-0.0084, +0.0049] |
| TACTICAL | trajectory-derived lateral decision correct (frac) | +0.0935 [+0.0709, +0.1160] | +0.0935 [+0.0709, +0.1160] | -0.0011 [-0.0120, +0.0093] | -0.0011 [-0.0120, +0.0093] | +0.0000 [+0.0000, +0.0000] |
| TACTICAL | trajectory-derived longitudinal decision correct (frac) | +0.0303 [-0.0055, +0.0625] | +0.0237 [-0.0117, +0.0566] | -0.0188 [-0.0522, +0.0112] | -0.0254 [-0.0592, +0.0052] | -0.0065 [-0.0166, +0.0000] |

## 8. Attribution of the ADE move (exact decomposition) -- every tree, S2 / S6, seed 0 / 1

| tree | surface | seed | mean ADE ON - OFF (m) | windows changed | changed AND GT>ceiling | changed AND GT within ceiling | mean per changed window (m) | contribution from GT>ceiling windows (m) | contribution from GT-within windows (m) | worsened / improved |
|---|---|---|---|---|---|---|---|---|---|---|
| step30000ema_unc | S2 | 0 | +0.0629 | 252 | 252 | 0 | 1.1868 | +0.0629 | +0.0000 | 236 / 16 |
| step30000ema_unc | S2 | 1 | +0.0619 | 251 | 251 | 0 | 1.1725 | +0.0619 | +0.0000 | 233 / 18 |
| step30000ema_unc | S6 | 0 | +0.1275 | 200 | 200 | 0 | 2.3384 | +0.1275 | +0.0000 | 138 / 62 |
| step30000ema_unc | S6 | 1 | +0.1183 | 200 | 200 | 0 | 2.1689 | +0.1183 | +0.0000 | 136 / 64 |
| step20000_unc | S2 | 0 | +0.0620 | 248 | 248 | 0 | 1.1885 | +0.0620 | +0.0000 | 229 / 19 |
| step20000_unc | S2 | 1 | +0.0621 | 249 | 249 | 0 | 1.1864 | +0.0621 | +0.0000 | 232 / 17 |
| step20000_unc | S6 | 0 | +0.1172 | 198 | 198 | 0 | 2.1712 | +0.1172 | +0.0000 | 128 / 70 |
| step20000_unc | S6 | 1 | +0.1129 | 198 | 198 | 0 | 2.091 | +0.1129 | +0.0000 | 127 / 71 |
| step20000ema_unc | S2 | 0 | +0.0629 | 253 | 253 | 0 | 1.1824 | +0.0629 | +0.0000 | 237 / 16 |
| step20000ema_unc | S2 | 1 | +0.0623 | 254 | 254 | 0 | 1.1657 | +0.0623 | +0.0000 | 237 / 17 |
| step20000ema_unc | S6 | 0 | +0.1255 | 203 | 203 | 0 | 2.2674 | +0.1255 | +0.0000 | 138 / 65 |
| step20000ema_unc | S6 | 1 | +0.1216 | 204 | 204 | 0 | 2.1862 | +0.1216 | +0.0000 | 142 / 62 |
| step15000_unc | S2 | 0 | +0.0599 | 232 | 232 | 0 | 1.2267 | +0.0599 | +0.0000 | 220 / 12 |
| step15000_unc | S2 | 1 | +0.0600 | 231 | 231 | 0 | 1.2341 | +0.0600 | +0.0000 | 219 / 12 |
| step15000_unc | S6 | 0 | +0.1289 | 187 | 187 | 0 | 2.5281 | +0.1289 | +0.0000 | 136 / 51 |
| step15000_unc | S6 | 1 | +0.1264 | 186 | 186 | 0 | 2.4932 | +0.1264 | +0.0000 | 136 / 50 |
| step15000ema_unc | S2 | 0 | +0.0624 | 261 | 261 | 0 | 1.1374 | +0.0624 | +0.0000 | 242 / 19 |
| step15000ema_unc | S2 | 1 | +0.0625 | 262 | 262 | 0 | 1.1346 | +0.0625 | +0.0000 | 240 / 22 |
| step15000ema_unc | S6 | 0 | +0.1139 | 208 | 208 | 0 | 2.0092 | +0.1139 | +0.0000 | 132 / 76 |
| step15000ema_unc | S6 | 1 | +0.1167 | 210 | 210 | 0 | 2.0392 | +0.1167 | +0.0000 | 137 / 73 |
| step10500_unc | S2 | 0 | +0.0578 | 240 | 240 | 0 | 1.1456 | +0.0578 | +0.0000 | 217 / 23 |
| step10500_unc | S2 | 1 | +0.0580 | 240 | 240 | 0 | 1.1495 | +0.0580 | +0.0000 | 218 / 22 |
| step10500_unc | S6 | 0 | +0.1098 | 190 | 190 | 0 | 2.1189 | +0.1098 | +0.0000 | 119 / 71 |
| step10500_unc | S6 | 1 | +0.1091 | 190 | 190 | 0 | 2.1062 | +0.1091 | +0.0000 | 123 / 67 |
| step10500ema_unc | S2 | 0 | +0.0633 | 267 | 267 | 0 | 1.1263 | +0.0633 | +0.0000 | 246 / 21 |
| step10500ema_unc | S2 | 1 | +0.0630 | 266 | 266 | 0 | 1.1265 | +0.0630 | +0.0000 | 243 / 23 |
| step10500ema_unc | S6 | 0 | +0.1143 | 211 | 211 | 0 | 1.9865 | +0.1143 | +0.0000 | 132 / 79 |
| step10500ema_unc | S6 | 1 | +0.1134 | 212 | 212 | 0 | 1.9622 | +0.1134 | +0.0000 | 136 / 76 |
| step5000_unc | S2 | 0 | +0.0536 | 259 | 258 | 1 | 0.9845 | +0.0538 | -0.0001 | 234 / 25 |
| step5000_unc | S2 | 1 | +0.0531 | 256 | 256 | 0 | 0.9852 | +0.0531 | +0.0000 | 232 / 24 |
| step5000_unc | S6 | 0 | +0.0924 | 206 | 206 | 0 | 1.6452 | +0.0924 | +0.0000 | 133 / 73 |
| step5000_unc | S6 | 1 | +0.0932 | 204 | 204 | 0 | 1.6758 | +0.0932 | +0.0000 | 138 / 66 |
| step5000ema_unc | S2 | 0 | +0.0639 | 271 | 271 | 0 | 1.1217 | +0.0639 | +0.0000 | 245 / 26 |
| step5000ema_unc | S2 | 1 | +0.0640 | 274 | 274 | 0 | 1.1106 | +0.0640 | +0.0000 | 249 / 25 |
| step5000ema_unc | S6 | 0 | +0.1133 | 214 | 214 | 0 | 1.9425 | +0.1133 | +0.0000 | 127 / 87 |
| step5000ema_unc | S6 | 1 | +0.1125 | 216 | 216 | 0 | 1.9096 | +0.1125 | +0.0000 | 129 / 87 |

## 9. Caveats and what this read does NOT decide

* POST-HOC on banked plans, CPU fp32. K2 and K4 (GPU) are owed, so these are not eval-forward numbers; they decide nothing until K4 passes.
* T1 open loop on logged frames: a clamp that makes a plan slower than the logged ego cannot score better on ADE against that ego. Section 8 shows the clamp changes only windows where the logged ego itself exceeds the fed ceiling, so the ADE read is structurally one-sided; it measures what compliance costs against a ground truth that does not comply, not driving quality.
* Single training seed; the interval answers 'another draw of EPISODES' only -- not another training run, not another inference run (the two inference seeds are shown separately).
* The 5k-30k banked verdicts are NOT moved (amendment section 4); the ON column is an additional read.
* The NR reference side is unchanged (refcv7-50,400 without clamp); `D-R8-NR-INSTRUMENT` is open.

## 10. Exploratory: the braking-allowance frontier (primary tree; NOT amendment-covered, NO verdict)

Tree `step30000ema_unc`, the landed function with only `brake_ms2` varied (landed = 2.5 m/s^2 = `sel.accel_max`). TIER: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance). ESTIMATOR: episode-cluster bootstrap over the 139 eval episodes, models FIXED: answers 'another draw of EPISODES' only; NOT another training run, NOT another inference run. Cross-check (brake 2.5 reproduces the main read's paired ON - OFF ADE exactly): **True**.

| seed | brake (m/s^2) | binding rate (n) | still above ceiling after clamp (n) / max excess m/s | S2 ADE ON - OFF | S2 os - ha0_ext (ON) | S6 ADE ON - OFF | S6 os - ha0_ext (ON) |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 5.30 % (252) | 0.00 % (0) / - | +0.1145 [+0.0527, +0.1902] **W** | +0.1137 [+0.0488, +0.1878] | +0.1910 [+0.0475, +0.3693] **W** | -0.6236 [-0.9840, -0.2667] **B** |
| 0 | 1.0 | 5.03 % (239) | 5.22 % (248) / 6.1229 | +0.0305 [+0.0150, +0.0506] **W** | +0.0297 [+0.0081, +0.0536] | +0.0654 [+0.0061, +0.1558] **W** | -0.7492 [-1.0784, -0.4312] **B** |
| 0 | 2.5 (landed) | 5.30 % (252) | 4.69 % (223) / 5.7479 | +0.0629 [+0.0316, +0.1017] **W** | +0.0621 [+0.0267, +0.1022] | +0.1275 [+0.0287, +0.2563] **W** | -0.6871 [-1.0261, -0.3526] **B** |
| 0 | 5.0 | 5.30 % (252) | 3.22 % (153) / 5.1229 | +0.0870 [+0.0425, +0.1412] **W** | +0.0862 [+0.0386, +0.1408] | +0.1589 [+0.0386, +0.3151] **W** | -0.6557 [-1.0064, -0.3100] **B** |
| 0 | 10.0 | 5.30 % (252) | 1.37 % (65) / 3.8729 | +0.1048 [+0.0493, +0.1709] **W** | +0.1040 [+0.0463, +0.1703] | +0.1789 [+0.0450, +0.3502] **W** | -0.6357 [-0.9888, -0.2812] **B** |
| 1 | 0.0 | 5.30 % (252) | 0.00 % (0) / - | +0.1135 [+0.0518, +0.1884] **W** | +0.1149 [+0.0496, +0.1901] | +0.1815 [+0.0402, +0.3555] **W** | -0.6087 [-0.9736, -0.2541] **B** |
| 1 | 1.0 | 4.99 % (237) | 5.22 % (248) / 6.1229 | +0.0297 [+0.0144, +0.0495] **W** | +0.0312 [+0.0097, +0.0552] | +0.0577 [-0.0003, +0.1481] | -0.7325 [-1.0666, -0.4139] **B** |
| 1 | 2.5 (landed) | 5.28 % (251) | 4.69 % (223) / 5.7479 | +0.0619 [+0.0309, +0.1003] **W** | +0.0634 [+0.0279, +0.1039] | +0.1183 [+0.0209, +0.2456] **W** | -0.6719 [-1.0117, -0.3391] **B** |
| 1 | 5.0 | 5.30 % (252) | 3.22 % (153) / 5.1229 | +0.0860 [+0.0421, +0.1396] **W** | +0.0874 [+0.0393, +0.1418] | +0.1495 [+0.0312, +0.3016] **W** | -0.6407 [-0.9904, -0.2993] **B** |
| 1 | 10.0 | 5.30 % (252) | 1.37 % (65) / 3.8729 | +0.1038 [+0.0485, +0.1697] **W** | +0.1052 [+0.0468, +0.1722] | +0.1694 [+0.0375, +0.3371] **W** | -0.6208 [-0.9794, -0.2724] **B** |

Reading: brake 0 is the STRICT clamp (the literal R1: the plan may never exceed the ceiling, even when v0 already does). For a positive brake the allowance is max(limit, v0 - brake * t_mid): a SMALLER brake leaves an over-ceiling ego more speed (looser), a LARGER brake forces it down to the limit sooner (stricter), and brake -> infinity converges to the strict case. 'still above ceiling' counts rows more than 0.01 m/s over the ceiling after the clamp (fp32 dust of a plan capped AT the ceiling is excluded; the strict > 0 count is in the JSON). The ADE column is a cost against the LOGGED ego, which itself exceeds the ceiling on every binding window (CLAMP_POSTHOC.md section 8); it does not say which point the programme should pick.
