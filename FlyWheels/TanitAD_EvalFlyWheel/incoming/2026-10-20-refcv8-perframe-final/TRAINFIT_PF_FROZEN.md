# TRAINFIT_PF_FROZEN -- the per-frame RETIME table (SHIP-R8 AMENDMENT 3 (ii))

**sha256 to register: `2aacc9ff32d3d2a5a18dda2d72ab5ce6c42d1c1268e9186912c5151bbe4d6d80`**

* file: `C:\Users\Admin\qland\hold_r8pf\stage\FlyWheels\TanitAD_EvalFlyWheel\incoming\2026-10-20-refcv8-perframe-final\raw\retime_trainfit_pf\FROZEN_GAINS_RETIME_train_pf.json` (9374 B); record `FROZEN_GAINS_RETIME_train_pf.sha256` equals the file (re-computed at report time)
* evidence class: MEASURED (fit on our rolls; T1 open loop; single training seed; inference seeds 0 and 1 pooled)
* procedure: H-R8-RETIME-TRAINFIT-1, UNCHANGED fit code `retime_trainfit_r8.py` sha256 `17b05b7ee48f1fe477f95f920e472caecd902e917f324a24aa7d96ab3c2526d4` (the clip-fed freeze ran `9494d8032681a8ffe73005a253a349db5ada305d129c8ae458b353db5212c055`; diff = cmd_forward role string only (raw/retime_trainfit_pf/FIT_CODE_DIFF.txt)); PREREG_TRAINFIT.md sha256 `648cdee6e68f70c2a91f518dbcce2999e846d5de6759a7deaf8a7d07311096f6`
* tree: `C:/Users/Admin/refcv8_pf/trainfit/step30000ema_train139_pf` (EMA-30k, `--nav-feed perframe_full`), seeds pooled [0, 1], prior `ha0_ext_pose`
* fit guard: {"n_fit_clips": 139, "n_overlap_with_eval139": 0, "n_outside_train4572": 0, "fit_clips_fingerprint": "6086d9cc0c90281619eb0bb63736a73e012322b48dddf5b4ed0363008a0ee3ed", "is_registered_A6_sample": true}
* windows by surface: {'S2': 4855, 'S6': 3749}; ceiling-compliant: {'S2': 4211, 'S6': 3301}

## Feed parity of the train rolls (clause (1b) form, asserted on the roll artifacts; independent of the overlay)

* train139 EMA-30k seed 0: **PASS** -- {"n_windows": 4855, "n_episodes": 139, "n_windows_clip_reference": 4855, "n_windows_only_in_pf": 0, "n_windows_only_in_reference": 0, "n_token_differs_from_clip": 979, "n_args_known": 2809, "n_clips_whose_v9_token_changes_within_clip": 47, "n_fail": {"T": 0, "C": 0, "K": 0, "A": 0, "I": 0}} (file sha256 `198e3666ff6e3f31dbe09818b44675b792dd94fcd59311f9ff524270866eef6d`)
* train139 EMA-30k seed 1: **PASS** -- {"n_windows": 4855, "n_episodes": 139, "n_windows_clip_reference": 4855, "n_windows_only_in_pf": 0, "n_windows_only_in_reference": 0, "n_token_differs_from_clip": 979, "n_args_known": 2809, "n_clips_whose_v9_token_changes_within_clip": 47, "n_fail": {"T": 0, "C": 0, "K": 0, "A": 0, "I": 0}} (file sha256 `65d4bf28e104825b080f10dd16d43d044b776c388dcf6260b2da382a70b25bbb`)

## Rolls

* s0: 4855 windows / 139 episodes, device cuda, wall 2930.4 s, pf.ok True, overlay sha256 `98293dd7411270160cb47d070bb1538babe1ff3a0f8ab85da61dcc84129ba72f`, record sha256 `53252fc3389e86cbb9caebf1a11ed4d13c3b97c15c5bf6179b5285d28fb7b338`
* s1: 4855 windows / 139 episodes, device cuda, wall 2384.2 s, pf.ok True, overlay sha256 `98293dd7411270160cb47d070bb1538babe1ff3a0f8ab85da61dcc84129ba72f`, record sha256 `3c3472a962b6765fadcbb44bfad2a90bc4bbefe81cb14c1127199790f48938e1`

## Gains (all windows, `RETIME`) -- per-frame (SCORED once registered) vs clip-fed (BESIDE ONLY)

| surface | instants s | per-frame (this table) | clip-fed b208c9c1... (beside) |
|---|---|---|---|
| S2 | [0.5, 1.0, 1.5, 2.0] | [0.9628, 0.6193, 0.5787, 0.5834] | [0.9674, 0.6193, 0.5511, 0.5305] |
| S6 | [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] | [0.6452, 0.6614, 0.7767, 0.8509, 0.8925, 0.9135] | [0.6394, 0.5970, 0.6546, 0.6995, 0.7223, 0.7454] |

Amendment 1's deployed form (knots S2 0.5-2 s, S6 3-6 s, piecewise-linear, held outside) is DERIVED from this file at forward time by `deploy_gains_pf.py`; no second fit exists.

## Gate

The 48,852 per-frame eval rolls REFUSE to start unless `C:/Users/Admin/qland/hold_r8pf/REGISTERED_PF_TABLE.sha256` exists and equals the sha256 above (the Master Mind writes it after registering this hash in GOALS_AND_CLAIMS).
