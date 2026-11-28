# PREREG_MAPSEL -- AMENDMENT 2 (Master Mind, 2026-10-26 ~01:20 CET; sealed separately; replaces ONLY the K_STAGE control)

## What happened, stated before anything else
The s0 scoring (00:51 CET) read **VOID** on one hard control, `K_STAGE_BASE_staged_equals_banked_S2_pos`. **This amendment is written
AFTER that VOID verdict and AFTER the full s0 scoring ran, so MAPSEL's arm outcomes EXIST on disk** (`result/s0/MAPSEL_RESULT.json`,
renamed to `result/s0_void_kstage/`). The Master Mind has read ONLY the controls block and the head of `VERDICT.md` (VOID reasons +
controls table); **no arm rate, no comparison cell, no verdict clause has been read.** The re-score below is the first reading.

## Diagnosis (MEASURED, 0 GPU, `kstage_diag.py` result quoted)
* `K_STAGE` compared MAPSEL's BASE staged plan against `DEPLOY8_POS…S2_pos` (`stage_banked`). That bank was staged from the BATTERY's
  `S2_plan8` -- the full arm-set draw -- while MAPSEL's BASE is the os-only roll (its `K_A_roll_equals_L1_BASEos` passes: 4,754 / 4,754
  bit-equal to L1's BASE-os). The two are different sampling draws.
* (a) the shipped stage applied to the battery `S2_plan8` reproduces `S2_pos` **exactly** (`array_equal` True);
  (b) the os-only plan differs from the battery plan on **4,630 / 4,754** windows (staged difference on 4,621, max 2.477 m);
  (c) on every window where the two plans are equal, the staged plans are equal.
* => the stage CODE is correct; the control's REFERENCE was the wrong draw. Same mechanism as the BASE-bat / BASE-os trap measured in
  `oracle_attr` (identity 17 m against the battery, 0.0 against L1's `traj_own`).

## A2.1 K_STAGE (replaces the K_STAGE line of PREREG_MAPSEL.md sec. 5)
**K_STAGE (amended)**: the battery `S2_plan8`, placed on the trace index exactly as the arms are (`jS2`, `pres2`) and staged through the
SAME call the arms use (`RS.retime_plan(…, prior_t, C.ALL_T, table, cast=True)`), equals `S2_pos` bit-for-bit on every present S2
window. Together with `K_A` (BASE == L1 BASE-os, bit-equal) this proves the arms are staged by the shipped stage on the right prior.
The old comparison (BASE-os staged vs the battery's banked stage) is still computed and REPORTED as a descriptive line (`n` windows
that differ), never as a control.

Nothing else changes: definitions, arms, populations, metrics, estimator, every other control, C1-C4, the verdict word and the
next-lever template are exactly as sealed in PREREG_MAPSEL.md + AMENDMENT 1.
