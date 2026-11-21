# The one crash seen during the smokes -- reported, not explained (UNVERIFIED cause)

The FIRST launch of the real-data smoke with `--vmax-input --w-goal 0.1 --w-speed-band 0.1
--w-traj 1.0 --r5-prior damp50` (strategic layer ON) exited with **code 139 (native SIGSEGV),
no Python traceback**, during training step 1 -- after the loader, the G-DVB check, the goal-head
fit and the config stamp had all completed. Its last lines were (verbatim, paths shortened):

    [refav1] R3 goal head: 5/22 classes trainable, negatives=measured
    [refav1] PI 2026-09-27 levers stamped -> ...\on_damp50\config.json (['goal_head', 'speed_band_head', 'speed_max_census', 'speed_max_derivation_v6', 'traj_head'])
    [refav1] forward() consumes ['a0', 'goal_w', 'goal_y', 'kappa0', 'lat_label', 'lon_label', 'nav_cmd', 'route_label', 'speed_band', 'speed_band_mask', 'speed_max_ms', 'speed_max_valid', 'str_ext_actions', 'str_ext_targets', 'traj_gt', 'traj_mask', 'v0']
    [refav1] loader emits but forward() does NOT accept: ['nav_valid'] ...
    EXIT=139

It ran CONCURRENTLY with a 4-thread pytest process of mine and other sessions' python jobs
(18.6 GB of 31.8 GB free when checked right after). The IDENTICAL command, re-run alone under
`python -X faulthandler`, completed (rc 0, 2 steps, 104 s) -- `smoke_real_ON_damp50_strategic_on.log`.
Not reproduced; cause UNVERIFIED (a native crash under memory/thread contention is the leading
guess, not a finding). A Thor/pod launch should run under `-X faulthandler` so a recurrence
leaves a native stack.
