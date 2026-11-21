# SHIP-R8 AMENDMENT 4 -- the registered battery read on the H2cn progress stage (Master Mind, 2026-10-26 ~03:52 CET)

PI decision **D-R8-STAGE-H2CN -> (a)**, 2026-10-26 ~03:30 Berlin: *"Yes, run SHIP-R8"*. Sealed BEFORE any SHIP-R8 number for an H2cn
stage exists. Amendments 1-3 stand; this one changes ONLY the scored arm's stage (and the clause-(1) controls that belong to it).
Known before the seal: `H-R9-L1C-H2CN-1` (PREREG_L1C_H2CN 84820fa6; S2 per-surface H2cn - H0 ADE -0.0130 / -0.0136; S6 four-family rows
of H2cn were NEVER computed -- only its S6 braking AUC).

## 1. The stage (frozen, then registered)
* **Fit:** `l1_fit_score.h2n_fit` on the 300 FRESH clips only (`l1c_fit_score.load_fresh`, tree `step48852ema_fresh`, bank
  `F:/l1c/bank/ztac_fresh.npz`), seed 0, once per surface -- S2 (4 outputs, slots 0..3) and S6 (6 outputs, S6 slots) -- exactly the
  fits of `l1c_h2cn_confirm.py`. Target `arc(G) - arc(HP)` (HP = the prior `ha0_ext_pose` at the surface's slots).
* **Freeze + register:** both fitted networks + their standardisation stats + the chosen (wd, epochs) are written to ONE file
  `H2CN_STAGE_FROZEN.pt`; its sha256 is written to a gate file and recorded in GOALS_AND_CLAIMS BEFORE any eval-side stage output
  is computed. The battery read refuses a stage file whose sha256 differs.

## 2. The scored arm `H2CN_DEPLOY8` -- ONE call per window on all 8 model slots (Amendment 2's rule)
* Inputs per eval window: the model plan `plan8` (the battery tree's `nav_true.traj`, as RETIME_DEPLOY8), the prior `prior8`
  (`ha0_ext_pose` at the same 8 slots), the scalars `X` (`PS.feats`) and z_tac (L1's `ztac_eval_s{seed}.npz`, joined on (clip, ws)).
* Progress: slots 0..3 (0.5-2 s) `s_k = arc(prior8[:, :4])_k + f_S2(X, Z)_k`; slots 4..7 (3-6 s)
  `s_k = arc(prior8[:, S6])_j + f_S6(X, Z)_j` for the S6 outputs j = 2..5. Then `s = max(s, 0)` and the running maximum along k
  (progress never decreases); the count of windows the running maximum changes is reported per seed.
* Positions: `RT.eval_polyline(RT.polyline_geometry(plan8), s)` -- the 8-point model plan, linear extrapolation past its end.
* Surfaces read the call at their own instants (S2 = slots 0..3, S6 = slots 1,3,4,5,6,7), exactly as RETIME_DEPLOY8.

## 3. Beside (never scored)
* `H2CN_PERSURFACE` -- L1C's per-surface form (4 / 6 points); **identity control: its S2 H2cn - H0 ADE reproduces -0.0130 / -0.0136
  within 1e-4.**
* `H2CN_NEAR` -- H2cn on slots 0..3 and the registered RETIME_DEPLOY8 progress on slots 4..7 (running maximum as above).
* `RETIME_DEPLOY8` -- the registered arm, re-read through the same code path; **identity control: equals the registered
  `SHIP_R8_step48852ema_pf.json` cells exactly.**

## 4. Clauses -- SHIP-R8 (1)-(7) + (1b), read MECHANICALLY on `H2CN_DEPLOY8`, both inference seeds, paired episode-cluster bootstrap B 2000
* **(1)** replaced by the stage's own controls: the frozen-file sha256 equals the registered one AND the registration precedes the
  first eval-side stage output (timestamps) AND the fit used no eval139 / train4572 clip (fresh-clip disjointness, asserted) AND the
  `H2CN_PERSURFACE` and `RETIME_DEPLOY8` identities above hold.
* **(1b)** FEED PARITY -- unchanged (same rolls): the registered FEED_PARITY files are re-checked by sha256.
* **(2) E1, (3) E3, (4) NR + S6 lateral, (5) TACTICAL, (6) STRATEGIC n/a, (7) PRIOR PARITY** -- verbatim from the registered verdict
  (`ship_r8_verdict_pf.py` / `_ship_r8_verdict_base.py`), with the arm name swapped.
* Compliant windows (A-R8-CLAMP-2) and the compliance row, as registered.
* A missing clause reads INCOMPLETE, never PASS. OVERALL: FAIL if any clause fails; else INCOMPLETE if any is incomplete; else PASS.

## 5. Committed next steps
* **PASS** -> H2cn ships as refcv8's stage (the PI's (a)); registry + GOALS_AND_CLAIMS updated; NavSim only with a separate go.
* **FAIL only on S6 clauses ((3), or (4)/(5) rows at S6) while `H2CN_NEAR` holds them** -> a sealed AMENDMENT 4b making `H2CN_NEAR`
  the arm is the next step (the head was validated at S2 only); reported as a FAIL of `H2CN_DEPLOY8` first.
* **FAIL otherwise** -> the shipped RETIME stage stays; the failing clause names the next lever.
