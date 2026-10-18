# AMENDMENT A-R7M-1 to `PREREG_R7_MATCHED_PHASE_TURN.md` (7deccde) — the instrument is the CEILING-FILTER-OFF path

**Registered by the Master Mind 2026-10-18 ~04:15 Berlin — the commit time is the record. NO refcv7 milestone number exists:
the first milestone roll (refcv7 step 20,000) started 04:10:25 and is due ~07:10.** Tier: T1 self-action OPEN LOOP on logged
eval139 frames. Evidence class of everything below: MEASURED, artifacts under
`FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/raw/r7_matched_phase/` (lands with that package).

## 1. The registered known-value control FAILED — reported as written

Re-rolling refcv7-50,400 through the refcv8 launch tree on 4 clips / 137 windows did **not** reproduce the banked
`step50400_a6text` plans: `os` max abs **3.676 m (seed 0, 36/137 windows differ)** and **3.474 m (seed 1, 34/137)**, against a
1e-3 m tolerance (`KNOWN_VALUE_CONTROL.md`). The native `run_battery_r7.py` path gives the same numbers
(`KNOWN_VALUE_CONTROL_native.md`), so the refcv8 wrapper is not the cause. **Under the registered rule, every refcv7 milestone
number from that path is VOID — and that path is not run.**

## 2. Why (MEASURED mechanism)

The banked refcv7-50,400 roll was made by a code tree (`C:/Users/Admin/ev7`, since deleted) whose E9 final re-rank did **not**
apply the max-speed ceiling: its final `sel_idx` equals the filter-OFF selection in **4,754 / 4,754** windows. The launch tree's
`e9_rank` applies `ceil_keep`, which changes the selected anchor on ~6–7 % of windows. Two instrument variants reproduce the
banked plans **exactly** (`os` max abs **0.0 m**, both seeds, GT / model-free arms bit-identical): the ceiling filter OFF
(`KNOWN_VALUE_CONTROL_ceiloff.md`) and E9 ignoring `ceil_keep` (`KNOWN_VALUE_CONTROL_e9noceil.md`).

## 3. The amendment

1. **Instrument:** the refcv7 milestone rolls run with the ceiling filter **OFF** (`R7M_PATH=ceiloff`) — the only kind of path
   that reproduces the known value, i.e. the instrument refcv7's banked number was measured on.
2. **`P7(N)` = s7(N) − s7f:** on that instrument at **both** inference seeds (the control passed at both); the outcome rule is
   unchanged.
3. **`D_ema(N)` / `D_raw(N)`:** refcv8's `os` was rolled ceiling-ON. The matched contrast uses refcv8's own **`filter_off.traj`**
   arm (same window, same DDIM draw, filter off), which exists **only at inference seed 0**. ⇒ **`D_ema` / `D_raw` are decided at
   seed 0 on the matched instrument.** The seed-1 value (refcv8 ON vs refcv7 OFF) is reported beside it, stamped
   MIXED-INSTRUMENT, and **cannot override**. This replaces "both inference seeds must agree" **for `D` only**; it is a loss of
   one replicate, stated, not a relaxation of the bar.
4. Bound on the substitution (MEASURED, seed 0, sharp windows, last segment): the ceiling moves refcv8's slope by
   OFF − ON **+0.0080 [+0.0034, +0.0136]** (EMA-5k), **+0.0052 [+0.0018, +0.0095]** (EMA-10.5k), **+0.0035 [−0.0004, +0.0082]**
   (EMA-20k) — `CEILING_EFFECT_refcv8.md`.

## 4. Blast radius beyond this test — named now, re-read owed

**Every banked refcv8-vs-refcv7-50,400 paired row is MIXED-INSTRUMENT**: refcv8 ceiling ON vs refcv7 with an inert ceiling.
That covers **BAR-R8-NR at steps 5,000 / 10,500 / 15,000 / 20,000, raw and EMA**, and the refcv8-vs-refcv7 rows of every
`PAIRED_READ_*`. **BAR-R8-E1 / E3 are NOT affected** (they compare `os` with the echo `ha0_ext` inside the same panel and roll).
Already measured for the turn slope: at EMA-20k seed 0 the gap to refcv7 is **−0.0438 [−0.0750, −0.0180]** ceiling-ON and
**−0.0402 [−0.0715, −0.0139]** on the matched OFF instrument — **the turn gap survives the matched instrument.** The
four-family NR rows (the ceiling is a SPEED filter, so the LONGITUDINAL family is the exposed one) are re-read on the matched
instrument (refcv8 `filter_off.traj` vs refcv7 banked, seed 0) before any further NR verdict is quoted.
