# MM note on FAM4 (L_s0 minus BPRIME_s0, four families, T1-kin first tick eval-141; Master Mind, 2026-11-15)

Checked by value against `RESULT_FAM4_L_vs_BPRIME.json`:
* Control (a): all 52 cells of BPRIME_s0 against itself read delta exactly 0.0 with CI [0, 0].
* Control (b): counts are 2,399 windows, 2,365 with `has`, and 139 episodes, the S0-L set.
* 18 cells separate. One training seed per arm and no replicate floor exist for this pair, so "separated" answers *another draw of episodes* only (H-ESTIM-SEED-1). No cell is a lever effect.

**All windows (2,399, 141 episodes):**

| band | L better | L worse |
|---|---|---|
| 0-2 s | ADE -0.023 m [-0.036, -0.014]; FDE; speed -0.054 m/s [-0.078, -0.034]; along-track; acceleration -0.259 m/s^2 [-0.355, -0.168]; yaw rate -0.12 deg/s | |
| 2-6 s | FDE -0.77 m [-1.35, -0.22] | yaw rate +9.6 deg/s [+3.3, +17.0] |

* **TACTICAL:** TAC-LAT accuracy +0.042 [+0.016, +0.068], L better. This cell is missing from the agent's summary.
* **Not separated:** distance-keeping, and the stop-cue AUROCs (+0.010 and +0.011, as in S0-L).
* **Separated, L worse:** the goal-stop false-positive rate, +0.025.

**Stop-point windows (117 windows, 25 episodes, the least stable set):**
* **0-2 s:** L is better on yaw rate and curvature.
* **2-6 s:** L is worse on ADE (+0.82 m), speed (+0.64 m/s), along-track (+0.91 m), heading (+8.9 deg), curvature and yaw rate (+70 deg/s [+26, +121]).
* **TACTICAL:** TAC-LON -0.137 [-0.231, -0.048], L worse.
* ⚠️ **HYPOTHESIS:** the 2-6 s heading, yaw-rate and curvature cells on these windows include steps where the car is at or near standstill, where heading is ill-defined. This is the pattern of RETR-2026-09-26-YAWMASK (yaw-rate cells scoring standstill jitter). Whether `read_lib_stopbal.band_rows` masks tangent-less steps is not checked here. Until it is, those three cells are not quotable as lateral skill.

**Reading:** the LoRA encoder helps the short-horizon (0-2 s) longitudinal cells and the lateral decision. It does not help the stop cue, and it is worse in 2-6 s around stops. STRATEGIC: PAUSED (PI item 25).
