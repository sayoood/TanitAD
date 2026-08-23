# The speed channel is a REGRESSOR deficit, not a discretisation artifact — unlike route

**Task #44 step 1. MEASURED 2026-07-29**, v4 from-scratch 30k arm, 721 of 881 windows
(81.8 % coverage — `tspeed_5s` carries its own mask). **No GPU, no re-run**: this reads the
per-window dump the eval already produces.

## 1. Why this question

Route was ruled out by a pre-registered paired test: fixing its threshold lifted balanced route
accuracy **0.4242 → 0.5493** yet moved paired `ade_0_2s` by only **+0.0022 [−0.0008, +0.0055]** — a
*precise* null bounding route's share of the oracle-vs-produced gap at **≤ 2.6 %**. The gap is
longitudinal (paired `long_abs_2s` **+0.4260** vs `lat_abs_2s` **+0.0274**), so the speed channels
are what remain.

`vt_band` is **not** independently predicted — `goal_modes.scalars_to_goal` derives it by applying
the labeler's own banding function to the regressed `tspeed_5s`. So the same question that cracked
route applies: **is the damage in the continuous estimate, or in the discrete mapping over it?**

## 2. Result — the regressor, decisively

| quantity | value |
|---|---|
| `tspeed_5s` RMSE | **4.4545 m/s** (~16 km/h) |
| implied band width | **median 1.089 m/s** (min 0.437, max 3.067) |
| **RMSE in band units** | **≈ 4.1 median bands** |
| \|band error\| | mean **2.175** · median **2.000** · p90 **5.000** |
| \|speed error\| | mean 3.306 · median 2.426 · p90 **7.365 m/s** |

**Tolerance curve:** ±1 band **0.4730** · ±2 **0.6630** · ±3 **0.7906** · ±4 **0.8904**.

⇒ **The discretisation is not too fine in any fixable sense — the speed estimate is simply wrong by
about four bands.** Forgiving ±2 bands still only reaches 0.663. There is **no threshold trick here**:
the pre-registered "regressor is the binding limit" reading fired.

## 3. ⭐ The contrast that matters for planning the work

| | route | speed |
|---|---|---|
| nature of the defect | **calibration** — a hard threshold discarded good signal (left precision 0.907 at 23 % recall) | **estimation** — RMSE spans ~4 bands |
| cheap fix exists? | **yes**, one constant, +0.1251 balanced accuracy, free | **no** |
| did it / would it move ADE? | **no** — +0.0022 [−0.0008, +0.0055], ≤ 2.6 % of the gap | **unknown, and now worth testing** |

**Route was cheap and inert. Speed is expensive and is where the longitudinal gap actually lives.**
That is an uncomfortable but useful conclusion: the remaining lever is a real modelling problem, not
a wiring bug — consistent with the programme's standing longitudinal-blindness finding, the harness's
`tracks speed > CV: False`, and `win lives: lateral only`.

## 4. ⚠️ Limits, stated

1. **My reference differs slightly from the harness's.** I compare `band(pred tspeed)` against
   `band(true tspeed label)`; the harness compares against the **oracle `vt_band`**, minted from the
   ego's own future poses. Hence exact 0.1900 here vs 0.1725 reported, within-1 0.4730 vs 0.3837.
   Close enough to establish the mechanism, **not** the same quantity — do not mix the two.
2. **This measures the GOAL channel only.** Route's lesson binds: a channel-metric improvement is not
   evidence of a trajectory improvement. **Any speed-side fix must clear a paired ADE test on the
   same windows before it counts.**
3. **81.8 % coverage** — `tspeed_5s` has its own mask; this is not the full 881-window population.

## 5. What follows

- ⛔ **Do NOT spend effort on re-banding.** It is measured not to be the binding constraint.
- The lever is `tspeed_5s` itself (R² 0.7635, RMSE 4.4545 m/s). It is the **best-fit** of the four
  goal scalars and still this weak, which sets expectations for the others.
- **Before any training is spent**, the cheap prior question is worth answering: how much would a
  *perfect* speed goal buy? That is measurable by feeding oracle `vt_band`/`tspeed` and re-measuring
  paired ADE — an eval-only run that bounds the prize before anyone pays for it.
