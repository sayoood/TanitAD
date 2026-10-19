# PREREG -- L2 headroom: how much of refcv8's cruise false-brake does the SHIPPED re-timing already absorb?

Written 2026-10-19, BEFORE any RETIME_DEPLOY8 number was computed for this question (only the already-banked
`raw/s6_lon_decomp/S6_LON_DECOMP.json`, i.e. os vs refcv7, and the npz *shapes / contract text* had been read).
Register row: `H-R8-S6-LON-1`. Sizes refcv9 lever L2 (`PLAN_REFCV9.md` section 1 D2, section 2 L2).
Tier: T1 self-action OPEN LOOP on logged eval139 frames (NOT driving performance). Evidence class: MEASURED (ours), CPU only, banked dumps only.

## Arms (EMA-30k, tree `step30000ema_unc`, S6 = 0-6 s, instants 1..6 s, dt 1.0), BOTH inference seeds
| arm | positions | source |
|---|---|---|
| `os` | unmodified refcv8 plan | the panel's `os` == `z['S6_os']` (asserted bit-equal) |
| `DEPLOY8` | the SHIPPED re-timing stage applied to os | `raw/retime_deploy/DEPLOY8_POS_step30000ema_unc_s{0,1}.npz`, `z['S6_pos']` |
| `refcv7` | refcv7-50,400 | the panel's embedded `b_refcv7_50400_s{seed}`, the reference `s6_lon_decomp.py` uses |
Seed `s` of DEPLOY8 is paired with seed `s` of os and of refcv7 (the npz is one file per INFERENCE seed).

## Surfaces
* **PRIMARY (the verdict surface): S6 COMPLIANT windows** = `z['S6_compliant']`, expected **3,251 windows / 138 episodes** (the shipped-bar surface; PI 2026-10-18).
* **SECONDARY: S6 ALL windows** (3,668 / 139) -- the surface `H-R8-S6-LON-1` was measured on, hence the surface the os-reproduction control is run on.
  The same rule is applied there and its label is reported beside the primary one. If the two labels differ the headline says both.

## Definitions: REUSED VERBATIM from `code/s6_lon_decomp.py` (imported, not re-implemented)
* **GT-cruise mask (ADE and along-track)** = `partitions(T)['dvclass'] == 'cruising'`: dv = v6 - v0 from `decisions/gt_future_ext[:,59,3]` and measured `v0`; first matching rule wins
  (start_from_stop | stopping | slow_both | accelerating | decelerating | cruising = |dv| < 1.5 m/s). 1,739 windows / 128 episodes on ALL windows. Intersected with the surface mask.
* **false deceleration / missed deceleration** = `lon_confusion`'s definition: `plan_lon_class(P, v0, dt)` = decel / cruise / accel from the path speed over the LAST second minus v0 (+-1.5 m/s),
  applied identically to GT (`g`) and to every plan. `false_decel` = indicator(plan class == decel) over windows whose GT class is cruise (1,949 on ALL windows);
  `missed_decel` = indicator(plan class != decel) over windows whose GT class is decel (832 on ALL). NOTE the instrument itself uses two different "GT-cruise" sets
  (the dvclass one for ADE / along-track, the plan-class one for false decel); both are kept as the instrument defines them. The false-decel rate on the dvclass-cruise mask is reported as a SUPPLEMENT.
* **signed along-track at 6 s** = `P_x(6 s) - G_x(6 s)` in the t0 ego frame (negative = plan ends short = too slow), mean over the mask.
* **ADE** = `PR.components(...)['ade_m']` via `S6.window_metrics`, per window.

## Estimator
`pair_read_r8.paired_cell` -> `taniteval.ci.paired_episode_cluster_bootstrap`, B 2000, seed 0, windows finite on both sides (outside-mask windows are NaN and dropped), 95 % percentile.
It answers "another draw of EPISODES" ONLY. It is blind to training variance (single training seed per arm, `H-ESTIM-SEED-1`). The two inference seeds are the INFERENCE replicate
(iCEM-style sampling floor); they are shown, never pooled. Shrinkage-ratio CI: the same joint episode-cluster draws as `S6.boot_counts` (percentile over draws with os delta > 0, share of such draws reported).

## Quantities per (surface, seed), restricted to the masks above
1. Cruise ADE: delta(os - refcv7), delta(DEPLOY8 - refcv7), delta(DEPLOY8 - os).
2. Cruise signed along-track at 6 s: mean for os / DEPLOY8 / refcv7 (+ paired deltas).
3. False-decel rate: os / DEPLOY8 / refcv7 (+ paired deltas).
4. Missed-decel rate: os / DEPLOY8 / refcv7 (+ paired deltas).
5. Descriptive absorbed shares (NOT in the rule): ADE `1 - D_dep/D_os` with `D_x = cruiseADE(x) - cruiseADE(refcv7)`; false decel `(fd_os - fd_dep)/(fd_os - fd_r7)`; along-track `(a_os - a_dep)/(a_os - a_r7)`.

## PRECONDITION (a guard, declared in advance so ABSORBS cannot be won vacuously)
`os_regression_present` = at >= 1 inference seed, the os - refcv7 cruise ADE delta is > 0 AND its CI lower bound > 0 on that surface.
If it is FALSE on a surface, the label on that surface is **NOT-APPLICABLE (nothing to absorb on this surface)**, the numbers are still reported, and the other surface carries the verdict.

## OUTCOME RULE (committed; every clause is evaluated at EACH of the two inference seeds and must hold at BOTH)
* **ABSORBS** = (a) DEPLOY8 cruise ADE delta vs refcv7 is NOT separated worse (CI lower bound <= 0) at seed 0 AND at seed 1
  AND (b) false-decel(DEPLOY8) <= 0.5 x false-decel(os) (point means, per seed)
  AND (c) missed-decel(DEPLOY8) is NOT separated worse than os (paired DEPLOY8 - os, CI lower bound <= 0) at seed 0 AND at seed 1.
* **PARTIAL** = the cruise ADE delta shrinks by >= 25 % (`1 - D_dep/D_os >= 0.25`, point, at BOTH seeds) but ABSORBS fails.
* **NONE** = otherwise.

## What each outcome means for L2 (committed)
* ABSORBS: the shipped re-timing already removes the cruise regression; L2's remaining headroom is the *residual* reported (expected ~0) and refcv9 should not spend a lever on it beyond protecting it.
* PARTIAL: L2 is sized at the RESIDUAL cruise ADE delta vs refcv7 and the residual false-decel excess over refcv7 reported in the table, not at the full +0.166 m.
* NONE: L2 is sized at the full measured D2; the re-timing is not the answer to the cruise false-brake.
In every case the L2 sizing figure is DEPLOY8's residual `D_dep` (cruise ADE, vs refcv7), per seed.

## CONTROLS (mandatory; the tool exits non-zero and writes CONTROL_FAILED.json on any failure)
C1 **os reproduces H-R8-S6-LON-1 on ALL windows**, as LITERALS (4 dp): all-window ADE delta +0.1658 [+0.0268, +0.3153]; cruise n 1,739 / 128 ep; cruise ADE contribution +0.2099 and share 1.2657
   (the register's 127 %); cruise bin-mean delta +0.4427; cruise signed along-track os -2.2078 vs refcv7 +0.0602; false decel os 0.2745 vs refcv7 0.0965, delta +0.1780 [+0.1228, +0.2404], n 1,949 / 138;
   missed decel os 0.3450 vs refcv7 0.5481, delta -0.2031 [-0.2707, -0.1435], n 832 / 80; seed 1 cruise share 1.1316, total +0.1895.
C2 **Alignment**: `S6_ws`, `S6_eid` equal the panel's; `S6_os` bit-equal the panel's `os` (both seeds); the npz compliant count is 3,251 / 138; panel GT identical across seeds.
C3 **Independent derivation**: numpy ADE and numpy signed along-track equal the `components` values (< 1e-4).
C4 **Retimed-arm reproduction** (read AFTER this file is sealed): the DEPLOY8 - refcv7 S6 ADE delta I recompute on the compliant and on the all set equals the banked `FORWARD_DEPLOY8_step30000ema_unc.json`
   `seeds.<s>.surfaces.S6.sets.<set>.arms.RETIME_DEPLOY8.NR_minus_refcv7.ade_m` cell (delta, lo, hi, n) field for field.
C5 **Identity arm**: DEPLOY8 := os must give exactly 0 deltas with zero-width CIs and label NONE.
C6 **Decision-function arms with answers fixed by CONSTRUCTION** (run on the real windows and real os / refcv7, DEPLOY8 replaced):
   `oracle` (DEPLOY8 := GT) -> **ABSORBS** (false decel and missed decel are 0 by construction, cruise ADE delta = -ADE_refcv7 < 0);
   `oracle_never_brakes` (GT everywhere except GT-decel windows, where the plan holds v0 constant, so every GT decel is missed) -> **PARTIAL** (ADE and false decel are won, the missed-decel guard must trip);
   `overbrake` (os x 0.5 on every point) -> **NONE**; `identity` -> **NONE**.
C7 **Mutations that MUST go RED** on the C1 literals: (m1) cruise threshold 1.5 -> 1.0 m/s; (m2) the decel threshold -1.5 -> -1.0 m/s in `plan_lon_class`; (m3) os shifted +0.01 m in x.
   Each mutated run must fail C1; the unmutated run must pass it. Also: the PREREG sha256 in `PREREG_L2_HEADROOM.sha256` must match this file's bytes or the tool refuses to run.

## Caveats committed in advance
Single training seed per arm; T1 open loop on logged frames; the interval is an episode draw only; the speed-label input differs between refcv7 (per-clip continuous) and refcv8 (4-class), so every
refcv8-vs-refcv7 row crosses that input change (as in the register row); the DEPLOY8 table was fitted on TRAIN-side data (`retime_trainfit`), not on these eval episodes.
This note does not claim anything about closed-loop driving, and a label here is a statement about THIS stage on THIS checkpoint.
