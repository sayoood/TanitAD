# STOPORC read: implementation notes and numbers (no verdict; the Master Mind writes it)

Card: `PREREAD_V8F_STOPORC.md`, sha256 `49eb7d7b...c34535` re-hashed before the run (equal) and pinned again in `test_stoporc.py::test_card_hash_unchanged`. Tier: T1-kin first tick, zero training, oracle diagnostic. CPU, threads 2. Tree `F:/TanitAD-scratch/stopbal_read/tree` (`tanitad.__file__` asserted under it).

## Files
* `stoporc_read.py`: forced codes (numpy), controls, estimator calls, outcome rule. Stage `run` banks one npz per arm to scratch (rows before analysis); stage `analyse` reads them and writes `RESULT_STOPORC.json`.
* `test_stoporc.py`: 22 tests, all pass (MEASURED, CPU): layout, forcing counts, lon rule (3 cases), red arms (forcing that forgets the goal entry, touches a lat cell, touches a cons cell by one ulp, picks a wrong lon class, goal mode that also moves lon), mutant exactly 0.0 / not separated, outcome table, no-floor mutant, fail-closed.
* `RESULT_STOPORC.json`: every cell, control and the outcome label.

## Implementation choices (the card is not reinterpreted; these are where it is silent)
* Indices verified against source in the run (assert): N_GOAL 22, N_LAT 5, N_LON 7, N_CONS 26 (60 entries); STOP_POINT = goal index 6 (`D.GOAL22.index`); STOP = LON9 index 2 (`D.LON9.index`); lon part = code[27:34]. GT allowed lon set = `lab_t["lon_al"]`.
* OWN plan = `forward_infer` traj30 (its code = T's own code); IDENT = `forward_diag(OWN code)`; forced plan = `forward_diag(batch, forced code)["traj30"]`. `forward_topk` is not called (not needed; `topk_shim` not attached).
* Rule where no lon class is allowed (the card does not say): the lon part stays T's own and the window is counted. Stop-point windows: 0 of 117. Control set: 18 of 2,234 (14 have STOP allowed, 2,202 take the highest-probability allowed class).
* "Separated" is the CI of `paired_episode_cluster_bootstrap` excluding 0 (either side, for the reaches-O control). Delta = forced minus OWN, point = unrounded mean difference. Floor = |delta(NULL_rep) - delta(NULL_s0)| on the primary cell.
* Windows from the STOPDEC rule (has-label and at least one present speed cell; strat tlstop == STOP_POINT): 117 stop-point windows (25 episodes), 2,234 control windows (139 episodes); equal to the STOPDEC npz on every arm.
* Outcome code returns INVALID-CONTROLS if any control fails; none did.

## Controls (MEASURED, `RESULT_STOPORC.json` `arms.*.controls`, all 7 arms)
* IDENT vs OWN max abs 0.0 m on every arm; own plan vs dump `ol` 0.0 m.
* Control 2: F-GT changes exactly 8 entries per stop-point window (goal entry 6 + the 7 lon cells), 117/117 windows; everything else equal bit for bit (0 entries outside). F-GOAL and F-LON likewise.
* Red arm (mutant F-GT := OWN): every cell delta exactly 0.0, reaches-O separated False, on every arm.

## Numbers (tactical ADE, knots 11-30, m; forced minus OWN, [95% CI], * separated)
| arm | primary: F-GT, 117 stop-point windows | reaches-O: F-GT, 2,234 control windows | F-GOAL stop-point | F-LON stop-point |
|---|---|---|---|---|
| BAL-NULL_s0 | -1.539 [-2.034, -1.036]* | +0.258 [0.186, 0.329]* | -1.037* | -0.882* |
| BAL-NULL_s1 | -1.477 [-2.018, -0.952]* | +0.117 [0.045, 0.187]* | -0.978* | -0.863* |
| BAL-NULL_rep (floor arm) | -1.134 [-1.588, -0.652]* | +0.377 [0.287, 0.466]* | -0.897* | -0.817* |
| BAL-G_s0 / s1 | -1.439* / -1.347* | +0.199* / +0.082* | -0.894* / -0.834* | -0.840* / -0.829* |
| BAL-L_s0 / s1 | -1.560* / -1.238* | +0.242* / +0.100* | -1.087* / -0.804* | -0.792* / -0.646* |
OWN mean tactical ADE on the stop-point windows: NULL_s0 5.021, NULL_s1 4.945, NULL_rep 4.055. Floor = 0.405 m.
Beside (BAL-NULL_s0, F-GT, stop-point): operative ADE -0.052*, tactical FDE -4.028*, tactical speed MAE -0.944*, tactical accel MAE -0.091*, tactical heading MAE +1.433* deg, tactical yaw-rate MAE +2.930* deg/s. TACTICAL family: n = 0 (T's decision is the forced input). STRATEGIC: PAUSED (PI item 25). Estimator answers "another draw of episodes" only.

## Outcome label computed by the code
`O-FOLLOWS` (primary separated, lower ADE, above the floor at both NULL seeds; reaches-O separated at both seeds). The four lever arms, by the same rule, are also gains at both seeds (beside; not part of the label).

## Temporary data
Scratch `F:/TanitAD-scratch/stoporc_build/` (seven per-arm npz banks, logs): re-creatable by `stoporc_read.py --stage run`; deleted at the end of the job (see hand-back).
