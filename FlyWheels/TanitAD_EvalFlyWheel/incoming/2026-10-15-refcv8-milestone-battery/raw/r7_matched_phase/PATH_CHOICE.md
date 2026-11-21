# Which instrument path the refcv7 matched-phase batteries run on (2026-10-18)

Pre-registered known-value control (PREREG_R7_MATCHED_PHASE_TURN.md section 2): re-roll refcv7-50,400 smoke-restricted (4 clips, 137 windows, seeds 0 and 1) and
compare against the banked `step50400_a6text`; `os` within 1e-3 m, GT / model-free arms bit-identical.

| path | artifact | verdict |
|---|---|---|
| wrapper, ceiling filter as built (ON) | KNOWN_VALUE_CONTROL.{json,md} | **FAIL** (os differs) |
| native launch-tree `run_battery_r7.py` | KNOWN_VALUE_CONTROL_native.{json,md} | **FAIL**, same numbers -> the wrapper is exonerated |
| wrapper + ceiling filter OFF (`R7M_CEIL_OFF=1`) | KNOWN_VALUE_CONTROL_ceiloff.{json,md} | **PASS** (os max abs 0.0 m, 137 windows, both seeds) |
| wrapper + E9 re-rank ignores `ceil_keep` (`R7M_E9_NO_CEIL=1`) | KNOWN_VALUE_CONTROL_e9noceil.{json,md} | **PASS** (os max abs 0.0 m; filter ON == OFF, 0 selection changes) |

Cause (MEASURED on the control): the banked refcv7-50,400 roll did not apply the max-speed ceiling in E9's final re-rank (its final `sel_idx` equals the
filter-OFF selection in 4,754 / 4,754 windows); the launch tree's `e9_rank` applies `out["ceil_keep"]`.  Both passing paths reproduce the banked plan exactly
and are equivalent on `os`.  The chain runs on **ceiloff** (`R7M_PATH=ceiloff`) because the refcv8 seed-0 dumps already carry the SAME instrument as a
`filter_off` arm, so refcv8-vs-refcv7 contrasts on the matched instrument need no extra GPU: `code/r7m_ceiling_effect.py --contrast`.
Caveat: the existing refcv8 `os` numbers are ceiling-ON; the ceiling moves the sharp-turn slope by +0.002..+0.014 (CEILING_EFFECT_refcv8.md), so read D_ema on
the OFF instrument (seed 0).  Seed-1 refcv8 rolls are os-only (no filter_off arm): the seed-1 D_ema is mixed-instrument.
