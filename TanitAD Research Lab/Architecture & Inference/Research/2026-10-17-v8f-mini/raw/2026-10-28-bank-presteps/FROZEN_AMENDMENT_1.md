# FROZEN AMENDMENT 1 (bank_presteps) -- written BEFORE any bank array, track array or eval dump was read

Amends `FROZEN_CARDS.md` (sha256 `3202a62fbf7699005c60f7a6ae37fb544e4bf0a0f0564b7a9115ee7bf865fd5b`, file `FROZEN_CARDS.sha256`).
**No card threshold changes.** The cards' numbers (15 % / 3 % / 5x; 53.9 % -> ~78 %; exactly 0 violations; identity bit-for-bit; r >= 0.5 / <= 0.25; 5 of 6) and every reading L1, L2, L4-L8, R1-R6, C1-C3, F2, F3 (except as stated in A2), F5 stand.
At the time of writing this file I had read code, run metadata and the dev-box RAM only; the only numbers I had seen were from SYNTHETIC paths built for the unit tests.

## A1. L3 (the labeller's construction) is REPLACED, because the first version was wrong on a synthetic panel

The version frozen in L3 traversed a Hermite blend with arc fraction `a[k] / a[k_rej]` (a time warp). On a 240-state synthetic off-route panel (lateral 3-8 m, heading error up to 35 deg, own speed 0-14 m/s) it refused 158 / 240 states, 169 of 240 failed the acceleration check even at the longest rung, and a traced case showed why: the warp multiplies the label's speed by `B / S*` (Hermite length over target arc) during the blend and drops it back to 1 at the rejoin, i.e. a speed jump the speed profile never asked for (up to 14 % at large heading error). The defect is in the construction, not in the checker; the checker found it.

**Replacement (Design C, fixed now, before data):**
* Speed profile `v_l` unchanged (L3): starts at the own speed, rate limit `0.99 * 4.0 m/s^2` while unmerged, follows the human unconditionally once equal.
* Label distance `l[k] = sum_i 0.5 (v_l[i-1] + v_l[i]) * 0.2`, k = 0..30: the label MOVES at `v_l` along its own path, so timing is consistent by construction (no warp).
* Path = cubic Hermite from the own pose (position q, tangent = own heading, magnitude = chord) to the human position at row `r' + k_j` (tangent = the central chord of the human polyline across +-0.6 m of arc there, not a one-sided segment direction), then the human polyline from that point. The label sits at path arc `l[k]`.
* Ladder over the target row `k_j in {5, 10, 15, 20, 25, 30}`; the first rung that passes the checker is emitted. A rung is infeasible if the label cannot reach the target within the 30 steps (`l[30] < B`) or if the human polyline is too short for `l[30]`.
* L4's "recovery phase" = from the own state up to `k_e + 1`, where `k_e` is the first step at which the label has arrived at the target (`l[k_e] >= B`). Everything else in L4 (convention, limits, virtual predecessor, hold-heading rule for a zero-length step) is unchanged.
* Identity (L5) is unchanged: bit-equal pose and speed returns the human label; the near-zero-offset continuity check is now a bounded discretisation floor (< 2 cm, not growing with the offset), not an O(eps) bound.
* Synthetic result of the replacement (not a card result): 217 / 240 emitted, 0 checker violations among emitted, the offset-ignoring arm violates the gate on >= 90 % of the panel, 21 / 21 unit tests pass.

## A2. FITGAP: the eval reference for M2b exists only as a trajectory dump, so the ratio is computed on the EXECUTED-CHORD speed on BOTH sides

The only banked eval-141 T1-kin dumps of plain M2b are `dump_B_s0_m10` / `dump_B_s1_m10` ("Plain M2b (B) at m = 10", `RESULT_V8F_T1KIN_HACC.md` line 55; float64 originals in `v8f-mini-hacc/t1kin_hacc_dumps_f64_2026-10-26.tar`). They carry `g` (GT), `cl` (executed), `v0`, `eid`, `ws` but NO per-tick state speed (no ticks file for B), so the registered R1 reading (the tick's `p10f` speed) cannot be formed on the eval side.
* **Primary FITGAP statistic (the one `r` uses):** the M-LAUNCH sensitivity reading R2 of `launch_intent_mlaunch/NOTES.md`: `v_own(tick n) = |K[10n] - K[10n-1]| / 0.2` on the executed track and `v_GT(tick n) = |Kg[10n] - Kg[10n-1]| / 0.2` on the GT track (K = [origin; knots]), applied identically on the eval side (from `cl`, `g`) and the TRAIN side (from the bank record's own pose history rows j and j-1, and the logged grid rows j and j-1). Per-window value = mean over ticks 1 and 2; launch windows and bootstrap as in F2 / F3.
* R1 on the TRAIN side (`p10f` speed against the central-difference GT) is reported BESIDE, and so is its eval-side value where a ticks file exists (DAG arms), to size the R1 / R2 gap.
* **Config mismatch, disclosed:** the B dumps were run with `--limit logged`; the bank rollout used `--limit causal` (`bank.json` ROLLOUT). The effect of that single setting on the launch statistic is measured on the DAG arms that exist in both forms (`v8f-dagger-t1`: DAG s0/s1 causal vs logged) and reported beside; it enters no verdict.
* The F5 verdict stays PARTIAL (2 of 6 checkpoints).

## A3. Bank table adds the previous history row

The compact bank table also stores the own pose of row j-1 (the second-to-last `pose` history row) so the R2 form can be computed; nothing else about the table changes.
