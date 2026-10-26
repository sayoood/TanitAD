# NOTES - t1kf_read (PREREG_V8F_T1_FAMILIES @ 635487e5)

RESULT OF THE RUN: controls C1-C5 PASS, **C6 FAILS** => per the PREREG both families are **REFUSED** (no outcome word). Rows are in the JSON as data only.

## clip_id mapping and its proof
* clip_id = `join_eval141.npz['clip'][clip_index]`, raw frame = `2 * ws` (`v8f_t1_eval.make_states`: `clip = str(J['clip'][ci])`; `cmd_run`: `wins = D.eval_grid_windows(grid, J)` -> `(ci, t0, 0, 0)`;
  `write_dumps`: `clip_index = ci`, `ws = t0`; `v8f_dk_read`: cache step t <-> frame 2t). The dump `eid` equals `clip_index` (checked) and is the bootstrap cluster id, as in every T1 read.
* Proof (C3, label-free): dump v0 vs block speed at all 2,399 joined rows, max |d| = 2.62e-05 m/s (tol 1e-3); the dump's (clip_index, ws) sequence equals the grid enumeration
  (`eval_grid_windows.json`, 141 clips, 2,399 windows, unique); 2,399 distinct block rows; block `ts_rel_s` deviates 0.0 from 0.2*(1..30); block sha256 equals 7a879baa...33073d9.
* States on the grid: LEAD 743, NO_LEAD 552, NOT_STRAIGHT 931, NO_LABEL 173 (sum 2,399). Only LEAD scores a lead row; the rest are counted, never scored as free flow.

## Guards adapted from v8f_dk_read's eval-dir format
* Used unchanged: `load_block` (duplicates), `join_block` (ts grid, speed proof, missing windows), `contract_check` (C4: median |d headway| 0.0222 m vs tol 0.25, n 724), `check_pred` (shape, finite, first-step units).
* NOT applicable, replaced: `saved_ade6` guard (eval_*.npz carries `ade6`; T1 dumps do not) -> C5 (ADE6 delta == banked -0.7680/-0.7336, read 0.0000 off); the "P10 reproduces floors_perwindow" guard
  (T1 `fl` is the executed floor, not the open-loop P10 plan). `ground_truth()` (rebuilt GT) is unused: `g` comes from the dump (identical in all six dumps, checked).
* v8f_dk_read's own grid-order `eid` is not used; the dump `eid` is (same bootstrap draws as t1k_read).

## Ambiguities resolved (PREREG line -> choice)
1. C6 "rolled by one window ... (window i scored with track i+1)" -> `np.roll(cl, -1, axis=0)` vs P10; gating at s0 only ("at s0"), s1 reported.
2. "Each must pass, or the read is REFUSED" / "The family is then reported REFUSED" -> ANY failed control refuses BOTH families (C6 spans both anyway).
3. DK-VIOL "at its registered at-risk denominator" -> `paired_lead_viol(...)['at_risk']` (n_at_risk 595 of 743 LEAD).
4. DK-HW/TG "on LEAD windows where both are finite" -> windows where arm, P10 and human values are all finite; n, n_a_only, n_b_only reported.
5. F "|mean_s0 - mean_s1| of the arm" -> over each seed's own finite windows (TAC: all 2,399; DK-VIOL: share_of_at_risk). Reported only; no outcome reads it.
6. C5 "the banked" numbers -> the causal-bucket headline (decisive config). The logged/strict dumps are checked informationally against t1kin-hacc/read4 and t1kin-strict/read: all four diffs 0.0000.
7. Tactical labels from float64 tracks via `maneuver_kinematics` -> `factor_from_kinematics` (the exact chain in `tactical_from_trajectory`); float32 labelling flips 0 windows (cl, fl, g).
8. Extra internal checks X1 (indicator mean == `tactical_from_trajectory` accuracy), X2 (DK rows == `pair_block`), X3 (six dumps share grid/GT) are NOT PREREG controls; they refuse only on an implementation bug. All PASS.

## C6 - what it shows (information for the amendment decision; not a substitute control)
* s0 rolled - P10: TAC-LAT +0.1296 [0.0937, 0.1672], TAC-LON +0.1121 [0.0667, 0.1563], DK-HW -0.1207 [-0.4702, 0.1931]: not worse. s1 the same direction.
* Why: a one-window roll shifts the path by 0.8 s inside a 6 s horizon, so the manoeuvre class barely moves; P10's executed track is a worse tactical arm than even the rolled HACC
  (TAC-LAT acc 0.5932 P10 vs 0.7228 rolled vs 0.7295 HACC; TAC-LON 0.3714 vs 0.4835 vs 0.5027). Diagnostic (non-gating, in the JSON): rolled - unrolled HACC is TAC-LAT -0.0067 [-0.0154, 0.0025] (ns),
  TAC-LON -0.0192 [-0.0317, -0.0067] (sep), DK-HW +0.1527 [-0.1315, 0.4207] (ns): the instrument sees the roll weakly. Whether C6 should compare against unrolled HACC or use a larger shift is a PI/MM amendment; I did not change it.
* Honesty note: two B=40 debug runs (`smoke/`) preceded the real run; the first exposed `t1k_read.paired`'s hard-coded B and showed C6 failing. Nothing in the rows, bars, controls or outcome rule was changed afterwards;
  only the non-gating C6 diagnostics and a `_status` label on `configs` were added.

## Tier / caveat (stamped in the JSON)
T1-kin self-action OPEN loop, never closed loop. The lead's logged future does not react to the ego: a track that closes on the lead is scored against where the lead WAS.
