# The TURN_L / TURN_R asymmetry is NOT in the penalty -- it is in what the goal term will PAY

`raw/TURN_ASYMMETRY_EXPLAINED.md` -- MEASURED 2026-09-06, T1, **ZERO GPU**, from the banked
decisions sidecars. Instrument `raw/turn_asym.py`; output `raw/turn_asymmetry.txt`.

## The puzzle

`W_KAPPA` penalises `kappa^2`, which **cannot see the sign**. Yet at `W_KAPPA = 15.11245` the
`TURN_L`-goal windows collapse to med|k|max **0.02066** while `TURN_R`-goal windows hold at
**0.08000**. A symmetric penalty produced a signed outcome.

## The answer, in one line

> **The penalty is symmetric; the BENEFIT is not.** The `ccos` goal term pays **+0.71638** (median
> `finecost_cv - finecost_plan`) for turning RIGHT and **approximately ZERO** for turning LEFT
> (**-0.00812** at `wk7`, **+0.00703** at `wk15`). On a `TURN_L` window the planned candidate is
> worth essentially nothing over doing nothing, so **any** curvature price tips it -- which is
> exactly why the left turn dies first and the right turn survives.

| decoded goal | n | med `finecost_cv` | med `finecost_plan` | **med margin (cv - plan)** | med \|k\|max @wk7 |
|---|---|---|---|---|---|
| `LANE_KEEP` | 18 | 0.9242 | -- | **0.00000** | 0.00000 |
| **`TURN_L`** | 9 | **1.7030** | 1.9964 | **-0.00812** | 0.08000 |
| **`TURN_R`** | 13 | 0.7685 | 0.0526 | **+0.71638** | 0.08000 |

(`wk15`: `TURN_L` margin **+0.00703**, `TURN_R` **+0.66446**. `gkappa`, with the turn weight set
to zero: `TURN_L` **-0.14789**, `TURN_R` **+0.76118** -- the gap is there with the penalty
switched off entirely, which is the cleanest proof it is not the penalty's doing.)

## Why this is the real next lever, and it is not `W_KAPPA`

The do-nothing candidate already costs **1.7030** on a median `TURN_L` window against **0.7685** on
a `TURN_R` one, and the planner's own choice costs **more** than doing nothing there. ⇒ **the goal
field for a left turn is not giving the planner anything to aim at.** No setting of a curvature
weight repairs that; it is a defect in the `ccos` goal term's left-turn field, and it is where the
next experiment belongs.

## Controls, all three read their known values

1. ⭐ **`w_kappa_eff_cl` reads the class's own weight**: **7.00000** on every window of the scalar
   `wk7`, and on `gkappa` exactly the by-goal map -- `LANE_KEEP` **15.11245**, `TURN_L` **0.00000**,
   `TURN_R` **0.00000**. This is what proves `goal_lat_cl` is the field the cost actually keyed on,
   rather than a label that merely correlates with it.
2. **`LANE_KEEP` margin is EXACTLY 0.00000** on all three arms -- it must be: a `LANE_KEEP` decode
   forces curvature exactly 0 (MEASURED 244/244 windows), so the plan **is** the constant-velocity
   candidate and the two costs are the same number.
3. **`wk15`'s `w_kappa_eff_cl` column is absent (`nan`)** -- that arm predates the by-goal
   instrumentation, and it is reported absent rather than imputed.

## Scope

40 windows / 8 episodes, ckpt 21,109, T1 (self-action open loop). Medians over 9 / 13 / 18 windows
carry no interval and are quoted with their `n`. ⚠️ `finecost` is a RE-SCORING; the selection
itself uses the base cost, and `baseline_won_frac` is 0.25 on these arms -- so this table explains
the RANKING pressure the goal term applies, not the selection mechanics end to end.
