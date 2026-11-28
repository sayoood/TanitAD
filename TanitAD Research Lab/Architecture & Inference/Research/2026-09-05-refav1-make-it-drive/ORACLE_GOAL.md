# ⛔ REDIRECT: the prize is NOT the goal head. A PERFECT goal makes refav1 4× worse.

**Row:** `D-REFAV1-DRIVE-ORACLE` · **Class:** MEASURED · **Date:** 2026-09-05
**Source:** Thor Stage B, `rec_B1_puregoal.json` / `dump_B1_puregoal`, step 21,109,
142 windows / 71 episodes, `ccos`, weights `(0, 0, 64.297)`

---

## The number the brief asked for

> *"Its `cl_oraclegoal` arm bounds the prize: if a perfect goal clears the controls, Steps
> 1–2 are worth every hour; if it does not, the ceiling is elsewhere — say so immediately
> and redirect."*

| arm | tier | ADE | |
|---|---|---|---|
| `cl` (the shipped planner) | T1 | **1.8147** [1.355, 2.303] | |
| **`cl_oraclegoal`** — goal from the **TRUE FUTURE** | **T0** | **7.4708** [5.621, 9.511] | **4.1× WORSE than `cl`** |
| `ha` | T1 | 0.5635 [0.431, 0.707] | |
| `ha0` | T1 | 0.5057 [0.424, 0.588] | |
| `ha0_ext` | T1 | 0.5370 [0.416, 0.671] | |
| `ol` | ⛔ T0 — WM diagnostic, never driving | 0.4406 [0.332, 0.563] | |

**A perfect goal does not clear the controls. It fails by 14× against the `ha0` floor.**

## ⚠️ But it does NOT bound the prize either — the arm is CONFOUNDED

Two things differ between `cl` and `cl_oraclegoal`, not one.

**1. The oracle arm plans with NO SEED.** From source, `refa_v1.py`:

```
goal_action = None
if goal_field is not None:                     # <- the ORACLE path
    goal_t, goal_source = self._tac_field(goal_field), "supplied"
elif brains is not None:                       # <- the NORMAL path
    goal_t, ga = self._imagine_tactical_goal(...)
    goal_action = {...}                        # <- only set HERE
...
if goal_action is not None:                    # <- so the seed is built
    seed = goal_action["controls"][:cfg.plan_steps][None]
```

⇒ supplying `goal_field` leaves `goal_action = None`, so **no seed enters the pool.**
MEASURED in the dump: `goal_source_cl_oraclegoal` = **`supplied` on 142/142**;
`goal_source_cl` = **`tactical_imagined` on 142/142**. The arms differ in the goal **and**
in the presence of the seed, so **7.4708 cannot be attributed to goal quality.**

**2. Its target is a latent field the planner cannot reach, and it SATURATES trying.**

| | `cl_oraclegoal` | `cl` |
|---|---|---|
| per-window max abs curvature — mean | **0.1741** | — |
| per-window max abs curvature — max | **0.2000 = `kappa_max`** | — |
| windows with curvature > 1e-3 | **100.0 %** | 0 % on LANE_KEEP windows |
| per-window max abs accel — mean | **1.6412** | 0.3029 |
| per-window max abs accel — max | **4.0000 = `a_max`** | 1.9928 |

⇒ the oracle arm **pins both control channels against their limits on essentially every
window**. That is not a planner tracking a good goal; it is an optimiser climbing a cosine
toward a target no control sequence can realise.

## ⭐ What the three experiments say TOGETHER

| the goal it was given | what the planner did | ADE |
|---|---|---|
| `LANE_KEEP` (86.5 % of windows) | curvature **exactly 0.0** — nothing | — |
| a *better* decision rule (more turn goals) | turned more | **worse** (+0.89) |
| the **true future** | saturated `kappa_max` and `a_max` on ~100 % | **7.47** (14× the floor) |

**All three say the same thing: refav1's planner cannot convert a goal into good controls.**
Give it no signal and it does nothing; give it more signal and it gets worse; give it a
perfect signal and it saturates. ⇒ **THE BINDING CONSTRAINT IS THE GOAL→PLAN PATHWAY — the
cost geometry and the optimiser — NOT the goal head's accuracy.**

## ⇒ The redirect

⛔ **`MUST_WE_RETRAIN.md`'s conclusion is SUPERSEDED in its priority, though not in its
measurement.** Its finding stands (the head separates at AUC 0.712 with a 0.297-logit margin
gap, and the trunk and WM need nothing). What changes is the **ranking**: fine-tuning the
goal head — **Step 2 — should NOT be launched next**, because the pathway that would consume
a better goal is itself the defect. A sharper head feeds a mechanism that already fails on a
perfect input.

**What to do instead, cheapest first:**

1. **Re-run the oracle arm UNCONFOUNDED** — supply the oracle goal *and* a seed derived from
   it (or run `cl` with the oracle's decoded token). Until that exists, **the prize is
   unbounded, not bounded**: this arm answers a different question than the one asked.
2. **The `ccos` hold-branch and goal-scale work is now the critical path**, not a parked
   item. Both the LANE_KEEP degeneracy (goal term = float32 rounding) and this saturation
   are the same defect — a goal term whose scale is not commensurate with what the
   controls can achieve.
3. **The lateral-baseline asymmetry** (`RESULT.md` item 4) becomes more attractive, not
   less: it makes turning reachable without routing through the goal term at all.

## Evidence class and scope

**MEASURED** throughout: the ADEs from `rec_B1_puregoal.json` (episode-cluster bootstrap,
n_boot 2000, 71 clusters); the control saturation and `goal_source` codes read directly from
`dump_B1_puregoal/decisions`. The seed-skip is **source-read** and corroborated by the
`goal_source` codes (142/142 `supplied` vs 142/142 `tactical_imagined`).

⚠️ `cl_oraclegoal` is **T0** — true-future-conditioned, a diagnostic, never a driving number
(EVAL_DOCTRINE). It is used here only as a *bound*, and the finding is that it fails even as
a bound. ⚠️ One checkpoint; `H-ESTIM-SEED-1` applies to any arm-vs-arm claim built on it.
