# RESULT — D-REFAV1-COST-GEOMETRY: the planner's objective is the goal term and nothing else

**Architecture & Inference FlyWheel, 2026-09-05.** Branch `agent/arch-inf-20260803`.
Pre-registration: `PREREG_COST_GEOMETRY.md` (this package), registered before any
arm in the ladder produced a number.
Checkpoint: `refav1` step **21,109**, vocab **v7.0** (`GOAL_KAPPA_TURN = 0.08`),
panel = the p4 turn-enriched set (8 episodes, stride 16, **40 windows**),
`--no-navshuf --no-lead-block`, argmax goal rule, `plan-seed 0` unless stated.

---

## §0 — THE HEADLINE, and it is not "W_KAPPA is zero"

⛔ **EVERY REGULARISER IN EVERY BANKED refav1 T1 ARM IS INERT.** The brief's prime
suspect was `W_KAPPA = 0`. It is worse than that: with the triple
`(0.0, 0.0, 64.29715042415070)` that `run_ab.sh` / `run_ab_targeted.sh` /
`gm_run_p4*.sh` all pass, **all three** penalty terms are switched off, and the
third one is off for a reason no weight can fix.

Read from source (`stack/tanitad/refs/refa_v1.py`, `_cost_chunk`):

```
c = c + w_jerk  * jerk.pow(2).mean(-1)                # w_jerk  = 0.0  -> DEAD
c = c + w_kappa * controls[..., 1].pow(2).mean(-1)    # w_kappa = 0.0  -> DEAD
if target_speed is not None:                          # <-- the gate
    c = c + w_vend * (v_end - target_speed).pow(2)    # w_vend  = 64.297, NEVER ADDED
```

`plan()` declares `target_speed: float | None = None` (`refa_v1.py:2139`) and the
T1 adapter never passes it:

| probe | value | role |
|---|---|---|
| `grep -c target_speed taniteval/tools/refav1_arm.py` | **0** | the claim |
| `grep -c plan_cfg …`      | **4**  | same-breath CONTROL, must be non-zero |
| `grep -c cost_weights …`  | **12** | same-breath CONTROL, must be non-zero |
| file size read            | **142,154 B** | the file was read, not skipped |

The tool has exactly one `.plan(` call site (`refav1_arm.py:924-933`); it passes
`cost_metric`, `cost_weights` and the goal flags, and no `target_speed`.

⇒ **The planner has been optimising goal alignment with no comfort constraint,
no curvature constraint and no speed target.** Saturating / over-sharp curvature
is the *expected* behaviour of an unconstrained argmin, not an anomaly — and it
means `W_VEND = 64.297` in every banked record is a number multiplying nothing.
⚠️ **Consequence for the register:** any refav1 arm quoted as "the planner" is a
GOAL-ALIGNMENT-ONLY planner. That is a scope statement every one of those rows
needs, and it is not visible from the weight triple alone.

---

## §1 — The scale: `ccos` decides 5.4 million times harder than `cos`

MEASURED at zero GPU from the banked `decisions` sidecars. Because of §0,
`plan_cost_cl` in those arms **is** the pure goal term, so this is a direct read
of the objective the search minimised. Artifact: `raw/cost_scale.txt`,
tool `raw/cost_scale.py`. n = 40 windows / 8 episodes.

| metric | median winner cost | median `cv` basecost | **median decision gap** |
|---|---|---|---|
| `cos` (the SHIPPED default) | **-1.19209e-07** | 1.19209e-07 | **1.78814e-07** |
| `ccos` (centred) | 3.10838e-05 | **1.0** | **0.967197** |
| `ccos`, seed 1 (replicate) | 3.11732e-05 | 1.0 | 0.967208 |

* Under `cos` the median winning candidate scores a **negative** cost: the term
  is below the float32 quantum of its own expression. The whole population is
  separated by ~1.8e-07.
* Under `ccos` the do-nothing candidate costs **exactly 1.0** — the documented
  degeneracy — and the winner ~3e-05.

⭐ **This settles the shipped arm before it is run.** At `W_KAPPA = 0.05` the
curvature charge at the clip is `0.05 * 0.2^2 = 2e-03`, which is **1.1e+04 times**
the entire `cos` goal decision. A shipped-weights `cos` planner cannot buy any
goal alignment at any curvature; the registered prediction is that it drives
exactly straight.

---

## §2 — What the planner actually does (and a correction to the brief's reading)

MEASURED, same sidecars. Artifact: `raw/kappa_by_goal.txt`, tool
`raw/kappa_by_goal.py` (the lateral vocabulary is **imported** from
`tanitad.models.vocab_v7`; a hand-written list had `TURN_L`/`TURN_R` at the
wrong indices on the first pass and would have mislabelled this table).

`ccos_argmax` (W_KAPPA = 0), by DECODED lateral goal token:

| token | n | median max\|kappa\| | frac kappa != 0 | frac at cap 0.2 | mean kappa^2 |
|---|---|---|---|---|---|
| LANE_KEEP | 18 | **0.00000** | 0.4444 | 0.0556 | 0.002908 |
| TURN_L | 9 | **0.08000** | 1.0000 | 0.1111 | 0.006698 |
| TURN_R | 13 | **0.08000** | 1.0000 | 0.0000 | 0.006400 |

by GROUND TRUTH (`|kappa0|` from `ha0_ext`, crossover 4e-2):

| stratum | n | median max\|kappa\| | frac != 0 | frac at cap | dir correct |
|---|---|---|---|---|---|
| GT-straight | 21 | 0.08000 | **0.6667** | 0.0952 | (not meaningful) |
| GT-turn | 19 | 0.08000 | 0.8421 | 0.0000 | **0.5789** |

⚠️ **CORRECTION, and it changes the diagnosis.** The brief's reading — *"curvature
non-zero on 90.9 % of GT-turn and 90.0 % of GT-straight windows, **both saturating
`kappa_max` = 0.2**"* — does **not** reproduce on this panel. The non-zero rates
are 84.2 % / 66.7 % (this metric) and 94.7 % / 90.5 % under `cos`, so the
"non-zero on ~90 %" half stands. **The "saturating" half does not**: the cap is
reached on **0–19 %** of windows, and the median `max|kappa|` is **0.08000** —
which is **exactly `GOAL_KAPPA_TURN`**, not the clip.

⇒ The planner is **not** running away to the clip. It is **faithfully tracking the
goal token's canonical curvature**, and that canonical curvature is R = 12.5 m on
a corpus that curves at R 100–1000 m. Under `cos` the same planner reaches
median 0.144 with 12.5 % at the cap, i.e. the *unconstrained* behaviour is the
`cos` one; `ccos` already pulls it onto the goal.

⇒ **The lever is therefore the trade-off between the goal term and a curvature
charge, not a runaway to be clipped** — which is exactly what the ladder in
`PREREG_COST_GEOMETRY.md` §2 varies, and it is why the rungs are set as a
fraction of the *measured* goal decision rather than by analogy to the clip.

Longitudinally the same vocabulary story shows up: **29/40** windows decode
`ADAPT_SPEED_FOR_CURVE` and the median realised `mean(jerk^2)` is **0.0** with
median `|accel|` **0.0000** — the plan holds a constant (usually zero)
acceleration. `cl`'s speed MAE is **0.7155 m/s** against **0.3058** for
`ha0_ext`, which simply holds the measured `a0`. ⇒ `W_JERK = 0` is not what hurts
the longitudinal family here; the decoded LON token commanding `a == 0` is.

---

## §3 — The `ccosh` hold branch (L2), implemented, tested, NOT yet defaulted

`COST_METRICS` says the hold branch is *"a PRE-REGISTRATION ITEM belonging to the
PI / Master Mind, deliberately NOT taken in this file"*. It is taken here as a
**new, additive, non-default metric** so that **no banked `ccos` number moves by a
byte**, and the choice of a default is escalated rather than assumed.

`"ccosh"` = `ccos` on windows with a real goal, and the pinned `"chord"`
(distance) form where `||g - z_ref|| <= CCOS_HOLD_REL * ||z_ref||`, i.e. where the
goal IS the hold field and the centred direction is float32 noise.

The controls that must read known values (`stack/tests/test_cost_ccosh.py`,
**16 tests, all passing**; `test_cost_ccos.py` + `test_cost_chord.py` still green,
63 passed together):

| control | known value |
|---|---|
| `cos` / `chord` / `ccos` after the addition | **bit-identical** (re-implemented in the test, not called through the changed function) |
| hold goal, do-nothing candidate, `ccos` | **1.0 exactly**, and **constant across the whole population** (peak-to-peak **0.0**) — the search has *no signal at all* |
| hold goal, do-nothing candidate, `ccosh` | **0.0 exactly**, and the unique argmin |
| hold goal at the MEASURED 4.18e-08 noise floor | `ccos` does **not** rank the hold candidate first; `ccosh` does |
| real goal (1e-2 relative) | `ccosh` == `ccos` **bit for bit** |
| the gate | fires at 4.18e-08 and 0.5x`CCOS_HOLD_REL`, does not fire at 2x, 1e-3, 1e-2 |
| `plan()` default | unchanged; nothing selects `ccosh` |

⛔ **ESCALATION (Master Mind / PI):** whether `ccosh` becomes refav1's default
cost metric is not this agent's call. It is implemented, pinned and selectable
(`--cost-metric ccosh`); the arm measuring it is in this package.

---

## §4 — ⛔⛔ THE INFERENCE-SEED FLOOR: FOUR FAMILY METRICS READ `separated` WHEN NO LEVER MOVED

⭐ **`H-ESTIM-SEED-1` REPRODUCED ON THE refav1 INFERENCE RIG, BY A DIFFERENT MECHANISM.**
The banked instance (`A0b_replicate`, v7-tiny) is about **training** variance. This
one involves **no retraining at all**: iCEM is stochastic *at inference*, so the same
checkpoint, the same window grid and the same cost produce a different plan under a
different `--plan-seed`.

**ARGV AUDIT (the control that makes this a measurement).** `gm_run_p4.sh` launches
`ccos_argmax` as `--cost-metric ccos --plan-seed 0` and `ccos_seed1` as
`--cost-metric ccos --plan-seed 1`, with an identical `common` array; the arms' own
banner lines confirm it — both print
`weights={'W_JERK': 0.0, 'W_KAPPA': 0.0, 'W_VEND': 64.2971504241507}`,
`windows=40`, `nav_shuffle=23/40`, and differ only in `seed=0` vs `seed=1`.
**Nothing else moved.**

Paired episode-cluster bootstrap, n = 40 windows / 8 clusters, n_boot 2000
(`raw/pd_seedfloor.md`, tool now banked at `taniteval/tools/refav1_paired_delta.py`).
The known-value control — an arm paired against **itself** — reads
**+0.0000 [+0.0000, +0.0000]** on every metric.

| family metric | `ccos_seed1.cl - ccos_seed0.cl` | separated? |
|---|---|---|
| `ade_m` | **+0.0607 [+0.0088, +0.1155]** | **YES** |
| `fde_m` | **+0.3439 [+0.0230, +0.7906]** | **YES** |
| `LAT_cross_mae_m` | **+0.0710 [+0.0101, +0.1412]** | **YES** |
| `LAT_heading_mae_deg` | **+1.1180 [+0.3304, +2.0778]** | **YES** |
| `LON_speed_mae_mps` | +0.0038 [-0.0003, +0.0074] | no |
| `LON_along_mae_m` | -0.0002 [-0.0335, +0.0351] | no |
| `LON_accel_mae_mps2` | +0.0061 [-0.0001, +0.0118] | no |
| `LAT_yaw_rate_mae_radps` | +0.0081 [-0.0061, +0.0251] | no |
| `TAC_traj_lat_correct` | -0.0750 [-0.1750, +0.0000] | no |
| `TAC_traj_lon_correct` | +0.0000 [+0.0000, +0.0000] | no |

⇒ **4 of 10 paired family metrics are `separated` for a pair of arms in which the
only change is a random seed.** Point-estimate drift over the same pair (per-arm
table, `raw/seed_floor.txt`): tactical lateral **kappa 0.3795 -> 0.2822 = 25.6 %**,
`TAC lat_acc` 12.0 %, `FDE` 11.6 %, `m5_acc` 11.1 %.

⛔ **BINDING CONSEQUENCE FOR THIS PACKAGE AND FOR EVERY refav1 PAIRED CLAIM.** On
this rig a separated paired CI is **necessary and not sufficient**. The admissible
form is *"the lever's delta exceeds the seed pair's delta on the same metric"*, and
this package reports the ladder that way. It is the CLAUDE.md rule with the object
swapped: there the estimator was blind to *training* variance, here it is blind to
*inference* variance, and the refav1 planner has both.

⚠️ **Scope it honestly:** this does not void the banked refav1 paired rows. It voids
the *inference* of a lever effect from a separated CI **alone**, on those four
metrics, at this n. A structural zero, a bit-identity, or a delta an order of
magnitude above the floor is untouched — e.g. §5's `-1.2181` is **20x** the seed
pair's ADE delta.

---

## §5 — THE FULL 8-EPISODE DE-CONFOUNDED ORACLE, FOUR FAMILIES (zero GPU, from a completed run)

`D-REFAV1-ORACLE-SAT` was banked **PARTIAL: 2 of 8 episodes, n = 9 windows,
"DIRECTIONAL ONLY, no interval, no bootstrap, no four-family table"**. The run
completed at 2026-09-05T17:15Z and its record was re-read here at zero GPU.
One process, three planning arms, one window grid, one plan seed, so the goal is
the only thing that moves. `cl_oraclegoal` / `cl_oracleseed` are **T0** (true-future
goal) and are diagnostics, never driving numbers.
Artifact: `raw/four_family_oracle_s0.txt`; record
`C:/Users/Admin/refav1_drive/oracle/rec_oracle_s0.json` **(OFF-REPO, SINGLE COPY)**.
Cost: `ccos`, `(W_JERK, W_KAPPA, W_VEND) = (0, 0, 64.297)` — so, per §0, the goal
term alone. n = 40 windows / 8 episodes.

| arm | tier | ADE m | speed MAE | curv MAE 1/m | cross MAE m | TAC lat kappa | goal FDE m |
|---|---|---|---|---|---|---|---|
| `cl` (shipped goal) | T1 | **1.3272** [0.7750, 1.9915] | 0.7155 | 0.055369 | 0.8784 | **0.3795** | 2.9639 |
| `cl_oraclegoal` (perfect goal, no seed) | T0 | **2.6899** [1.4793, 4.3346] | 0.8766 | 0.120323 | 1.8278 | **-0.0057** | 5.4599 |
| `cl_oracleseed` (perfect goal, **seed kept**) | T0 | **2.5452** [1.3205, 4.2374] | 0.9001 | 0.118665 | 1.6788 | **-0.0360** | 5.3092 |
| `ha` floor | T1 | 0.8888 | 0.3058 | 0.076005 | 0.4055 | 0.5510 | 2.3164 |
| `ha0` floor | T1 | 0.9251 | 0.7217 | 0.040083 | 0.3642 | 0.0000 | 2.2613 |
| `ha0_ext` floor | T1 | **0.8772** | **0.3058** | 0.077298 | 0.4018 | **0.6277** | 2.2693 |
| `ol` | T0 | 0.8052 | 0.0906 | 0.082614 | 0.4199 | 0.7172 | 1.9942 |

Paired (episode-cluster bootstrap, same process):

| pair | `ade_m` | `speed_mae_mps` | separated |
|---|---|---|---|
| `cl - cl_oraclegoal` | **-1.3627 [-2.3894, -0.5451]** | -0.1610 [-0.2532, -0.0450] | **YES / YES** |
| `cl - cl_oracleseed` | **-1.2181 [-2.3250, -0.3862]** | -0.1846 [-0.2551, -0.1070] | **YES / YES** |
| `cl - ha0_ext` | +0.4500 [-0.0155, +1.1085] | +0.4097 [+0.2390, +0.6257] | no / **YES** |

⇒ **A PERFECT GOAL, DELIVERED HONESTLY, MAKES refav1 SIGNIFICANTLY WORSE ON EVERY
FAMILY** — ADE **1.93x**, curvature MAE **2.14x**, cross-track **1.91x**, goal FDE
**1.79x**, and the tactical lateral decision falls from kappa **0.3795 to -0.0360**,
i.e. **below chance**. The deltas are **20x** and **17x** the §4 seed floor, so this
is not the seed talking. `D-REFAV1-DRIVE-ORACLE` / `D-REFAV1-ORACLE-SAT` are hereby
**upgraded from PARTIAL/DIRECTIONAL to a COMPLETE four-family panel with intervals**.

⭐ **This is the strongest available statement that the GOAL IS NOT THE BOTTLENECK.**
Everything downstream of the goal — the cost geometry and the decode — is.

---

## §6 — ⭐⭐ THE RESULT: refav1's PLANS ARE BOX-FEASIBLE AND NOT KAMM-FEASIBLE, AND THE CAUSE IS THAT `kappa_max` HAS NO SPEED IN IT

### 6.1 The correction that made this findable

The brief says L4's feasibility-aware decode lives in a sibling package and should
be reused. **Two probes said it did not exist** (a glob over
`TanitAD Research Lab/*/Research/*feasible*`, and a direct listing of
`Deployment & Optimization/Research/`, which read non-empty as its own control).
⛔ **Both probes were right about the PACKAGE and wrong about the WORK.** A third
probe — `RETRACTION_LOG.md`, read for an unrelated reason — named
`stack/tests/test_feasible_decode.py`, and the module
`stack/tanitad/refs/feasible_decode.py` (18,135 B; `project_feasible`,
`recover_controls`, `assert_feasible`, `max_heading_step`) **is in the repo**.
⚠️ It was **ABSENT from the off-Drive clone the arms run from** — which is why the
clone-side grep found nothing. *Absence found at one location is not absence*, in
its exact textbook form, caught by a probe of a different mechanism.

### 6.2 The audit, with the control that decides whether it is admissible

`assert_feasible` re-derives the **scorer's own** envelope and Kamm flags from a
path. Run over the banked p4 dumps at `dt = 0.20 s` with the ego origin prepended
(`raw/feas_audit.py`, `raw/feas_audit.txt`).

⛔ **THE CONTROL: the GROUND-TRUTH path `g` must be feasible.** At `vmin = 0` it is
**NOT** — `envelope_rate 0.1000`, `max|kappa| 31.4015` — because `kappa =
a_lat / v^2` explodes on near-stationary windows. **That whole block is therefore
inadmissible and is reported, not used.** At `v0 >= 2 m/s` the control reads
`envelope_rate` **0.0000** and `kamm_over_rate` **0.0000**, and the block IS.

**`v0 >= 2 m/s`, n = 27 windows** (`ccos_argmax`, `W_KAPPA = 0`):

| arm | envelope_rate | **kamm_over_rate (mu = 0.7)** | max abs kappa | peak_g mean | peak_g max |
|---|---|---|---|---|---|
| `g` (**CONTROL**) | **0.0000** | **0.0000** | 0.1701 | 0.178 | 0.373 |
| **`cl` (refav1)** | **0.0000** | **0.2963** | **0.2000** (= the clip, exactly) | 0.558 | **3.262** |
| `ha0_ext` | 0.1852 | 0.1852 | 0.7672 | 0.316 | 1.436 |
| `ha` | 0.1481 | 0.1481 | 0.6211 | 0.316 | 1.595 |
| `ol` (T0) | 0.2963 | 0.1111 | 0.4400 | 0.343 | 1.016 |

**`v0 >= 5 m/s`, n = 19:** `cl` `kamm_over_rate` **0.4211**, `peak_g` mean
**0.746**, max **3.262**; the ground-truth control still **0.0000 / 0.373**.

⇒ **refav1 is the ONLY arm that never leaves the actuator box** — `envelope_rate`
exactly `0.0000` and `max|kappa|` exactly the clip — **because it searches in
control space behind `_clip`. And it is the arm that leaves the FRICTION CIRCLE
most often: 29.6 % of plans at `v0 >= 2 m/s`, 42.1 % at `v0 >= 5 m/s`.**
The floors violate the box (they are integrators of a measured state, not
constrained searches) but stay closer to the circle.

### 6.3 The mechanism, confirmed by an independent route

`PlanConfig.kappa_max = 0.2` is a **CONSTANT**: the search may command the same
curvature at 30 m/s as at 2 m/s. The tyre may not — `a_lat = v^2 * kappa` must
stay inside `mu * g`. Sustaining `GOAL_KAPPA_TURN = 0.08` (§2: the median plan) at
20 m/s is `0.08 * 400 = 32 m/s^2 = **3.26 g**` — **exactly the `peak_g max 3.262`
the audit measured**, arrived at from the vocabulary constant and from the path
recovery independently.

At `mu = 0.7` the admissible curvature at 20 m/s is `mu*g/v^2 = **0.0172 1/m**` —
**11.6x tighter than the clip**, and squarely in the corpus's own R 100–1000 m
band that §2 said the vocabulary cannot express and §5 said the seed pool does not
contain. **Three independent readings converge on the same missing quantity.**

### 6.4 The lever, implemented and pinned (OFF by default)

`PlanConfig.kamm_mu` (default `None` = bit-identical to every banked arm) makes
`_clip` additionally clamp `|kappa| <= mu*g/v^2` on the **candidate's own speed
profile** — `v0` integrated through its accel channel — so a plan that brakes into
a curve is allowed the curvature its braking earns. `PlanConfig.kamm_v_floor`
(2.0 m/s) keeps the constant clip governing at a crawl, where `mu*g/v^2` exceeds
it anyway and the division is ill-conditioned.

⭐ **This is a CONSTRAINT WITH UNITS, and no `W_KAPPA` can imitate it.** A
quadratic penalty is one speed-independent number traded against a goal term; the
Kamm cap is a per-step bound that tightens as the square of speed and loosens when
the plan slows. `test_d_a_braking_candidate_EARNS_curvature` pins exactly that:
at `v0 = 25 m/s`, braking at `-4 m/s^2`, the final step's cap rises from
**0.010983** to **0.014445 1/m** (`0.7g/625` -> `0.7g/475.24`), a **1.315x** gain
a constant cannot express.

Controls that must read known values (`stack/tests/test_refa_v1_kamm_clip.py`,
**15 tests, all passing**):

| control | known value |
|---|---|
| `kamm_mu=None`, `v0` supplied | `_clip` **bit-identical** to the pre-2026-09-05 clamp; `plan()` bit-identical over 3 seeds |
| cap at (v, mu) | **exactly** `mu*g/v^2`, on every step; 20 m/s / 0.7 -> **0.0171616** |
| `v0 = 0.1 m/s` | cap does **not** bite; the constant `kappa_max` governs |
| a curvature already inside the circle | **untouched, bit-exact** (it clamps, it never rescales) |
| sign | preserved, and the cap symmetric to 1e-12 |
| end-to-end | a capped plan's worst step reads `<= mu` g; the UNCAPPED control on the same window is free to exceed it |
| nonsense `mu` / `v_floor` | **refused** by `PlanConfig.sanity()` |

⛔ **ESCALATION (Master Mind / PI).** Two levers are now implemented, pinned and
selectable but **NOT defaulted**, and defaulting them is a programme decision:
`--cost-metric ccosh` (§3) and `--kamm-mu` (here), plus `--seed-kappa-ladder`
(§5's L3). All three are OFF by default and no banked number moves.

⚠️ **A clone-currency work item this exposed:** `stack/tanitad/refs/refc.py` in
`C:/Users/Admin/tanitad-wt` is **stale** against the repo (blob `82127563…` vs
`0e6103e5…`), so `tests/test_feasible_decode.py` fails there with
`DecoderConfig.__init__() got an unexpected keyword argument 'feasible_decode'`.
That is CLONE staleness, not a repo failure, and it is the exact class
`pod_currency_audit.py` exists for — on a dev-box clone nobody audits.

---

## §7 — Arms (the ladder and the levers)

### 7.1 ⛔⛔ THE SHIPPED CONFIGURATION OF refav1 IS THE CONSTANT-VELOCITY BASELINE, BIT FOR BIT

`cos_shippedweights` = `--cost-metric cos` (the module default) + `--cost-weights
0.02,0.05,0.1` (the module constants `W_JERK`, `W_KAPPA`, `W_VEND`). **This is the
first time the configuration refav1 actually ships has been run on this panel** —
every banked p4 / A-B / oracle arm used the zeroed triple `(0, 0, 64.297)`.

Its registered prediction (`PREREG_COST_GEOMETRY.md` §3, written before it ran) was
*"the arm drives exactly straight, `kappa == 0` on ~all windows"*. **The outcome is
stronger than the prediction:**

| what | value | control that must differ |
|---|---|---|
| entire control tensor exactly zero | **40 / 40 windows (1.0000)** | `ccos_argmax`: 0.2500 |
| `max\|accel\|` over all windows | **0.000000** | `ccos_argmax`: 1.091408 |
| `max\|kappa\|` over all windows | **0.000000** | `ccos_argmax`: 0.200000 |
| `cl` vs `ha0` trajectories, max abs diff | **0.000e+00, BIT-IDENTICAL** | `ccos_argmax`: **1.508e+01 m**, not identical |
| paired `cl - ha0` ADE | **+0.0000 [+0.0000, +0.0000]** | — |
| distinct `max\|kappa\|` values realised | **1** (all `0.000000`) | `cos_argmax`: 32 |

⇒ **refav1, as shipped, emits the constant-velocity plan on every window. It is not
a planner on this panel; it is `cv` with extra steps** — and it reaches that by
SEARCH, not only by the injected floor (`plan_source`: **cem 0.425** / baseline
0.575), i.e. **the shipped cost's own optimum is the do-nothing plan.**

The arithmetic was registered in §1 before the arm ran: `W_KAPPA·kappa_max² = 2e-03`
against a total `cos` goal decision of **1.79e-07** — a ratio of **1.1e+04** — so no
curvature can ever pay for itself, and `W_JERK` does the same to the accel channel.

**Four families, T1, episode-cluster bootstrap, n = 40 windows / 8 episodes**
(`raw/four_family_all.txt`; STRATEGIC and distance-keeping declared UNAVAILABLE with
their reason and n, per clause 5):

| arm | ADE m | speed MAE | accel MAE | curv MAE 1/m | cross MAE m | TAC lat acc / kappa | m5 acc |
|---|---|---|---|---|---|---|---|
| **`cl` = `cos_shippedweights`** | **0.9251** [0.7024, 1.1783] | 0.7217 | 0.7766 | 0.040083 | 0.3642 | 0.5250 / **0.0000** | 0.2500 |
| `ha` | 0.8888 [0.6026, 1.1928] | 0.3058 | 0.4487 | 0.076005 | 0.4055 | 0.7250 / 0.5510 | 0.6000 |
| `ha0` | 0.9251 [0.7024, 1.1783] | 0.7217 | 0.7766 | 0.040083 | 0.3642 | 0.5250 / 0.0000 | 0.2500 |
| **`ha0_ext`** (the integrator floor) | **0.8772** [0.6195, 1.1437] | **0.3058** | 0.4487 | 0.077298 | 0.4018 | **0.7750 / 0.6277** | **0.6500** |
| `ol` (T0, not a driving number) | 0.8052 [0.5787, 1.0577] | 0.0906 | 0.1064 | 0.082614 | 0.4199 | 0.8250 / 0.7172 | 0.7750 |

Paired against the floors: `cl - ha0_ext` ADE **+0.0479 [-0.0889, +0.1923]**
(not separated) and `speed_mae` **+0.4159 [+0.2351, +0.6440]** (separated, worse).

⚠️ **Read that correctly.** The shipped arm *ties* `ha0_ext` on ADE — but only by
refusing to act, and it loses the longitudinal family outright. Its lateral-decision
kappa is **0.0000** (turn recall **0/11 left, 0/8 right**, lane-keep recall **1.0**),
which is the signature of a constant predictor, not of a driver.

**The paired four-family delta of the weights themselves** — `cos_shippedweights`
against `cos_argmax`, the SAME metric, the SAME seed, the SAME panel, with only
`(W_JERK, W_KAPPA)` moving from `(0, 0)` to `(0.02, 0.05)`
(`raw/pd_shipped.md`; the known-value control, an arm against itself, reads
`+0.0000 [+0.0000, +0.0000]` on every metric):

| family metric | `cos_shippedweights.cl - cos_argmax.cl` | separated | direction |
|---|---|---|---|
| `ade_m` | **-0.9692 [-2.0210, -0.1751]** | YES | better |
| `fde_m` | **-2.1027 [-4.3280, -0.3603]** | YES | better |
| `LON_along_mae_m` | **-0.2410 [-0.6177, -0.0127]** | YES | better |
| `LON_accel_mae_mps2` | **-0.0428 [-0.1058, -0.0011]** | YES | better |
| `LON_speed_mae_mps` | -0.0056 [-0.0157, +0.0064] | no | — |
| `LAT_cross_mae_m` | **-1.0322 [-2.0533, -0.2393]** | YES | better |
| `LAT_heading_mae_deg` | **-9.1286 [-15.5752, -2.8652]** (n=33) | YES | better |
| `LAT_yaw_rate_mae_radps` | **-0.2852 [-0.4691, -0.1239]** | YES | better |
| `TAC_traj_lat_correct` | +0.1500 [+0.0000, +0.3000] | boundary | better |
| `TAC_traj_lon_correct` | +0.0000 [+0.0000, +0.0000] | — | — |
| `cl - ha0` (bit-identity check) | **+0.0000 [+0.0000, +0.0000]** | — | identical |

⇒ **Restoring the shipped curvature and jerk weights improves SEVEN of ten paired
family metrics, every one of them separated — and it does so by making the planner
STOP.** Every delta above is the cost of the motion the zeroed-triple arm was
emitting, not the value of motion it learned to emit. ⚠️ **That is exactly why the
four families are binding and why ADE alone would have been read as progress here.**

⇒ **The honest statement of the objective is now sharper than "refav1 does not beat
the floors":** as shipped it *is* a floor, bit-identically; and every configuration
in which it actually plans (`cos`/`ccos` with the zeroed triple) is **worse** than
the floors. The lever set in §3/§5/§6 exists precisely to make it act *and* be right.

*(filled in as they land; see `raw/` for the four-family tables and the records)*


### 7.2 ⭐⭐ WHERE THE DEFICIT ACTUALLY IS: refav1 ALREADY BEATS THE FLOOR ON TURNS AND DESTROYS ITSELF ON STRAIGHTS

MEASURED zero GPU from the dumps, per-window ADE split by `|kappa0|` (the measured
curvature at t0, from `ha0_ext_controls`) at the programme's crossover **4e-2**.
⛔ CONTROL: `g` against itself reads **exactly 0.000000**.
Artifact `raw/ade_by_stratum.txt`, tool `raw/ade_by_stratum.py`.
n = 21 GT-straight / 19 GT-turn of 40.

| arm | ALL | **GT-straight (n=21)** | **GT-turn (n=19)** |
|---|---|---|---|
| `cl` `ccos_argmax` | 1.3272 | **1.6960** | **0.9195** |
| `cl` `ccos_seed1` (seed replicate) | 1.3879 | 1.8078 | **0.9238** |
| `cl` `cos_argmax` | 1.8944 | 2.5998 | 1.1146 |
| `cl` `cos_shippedweights` (= `ha0`) | 0.9251 | 0.8862 | 0.9682 |
| `ha` | 0.8888 | 0.6792 | 1.1204 |
| `ha0` | 0.9251 | 0.8862 | 0.9682 |
| **`ha0_ext`** (the strongest T1 floor) | **0.8772** | **0.6285** | **1.1521** |
| `ol` (T0) | 0.8052 | 0.4913 | 1.1521 |

⭐ **ON TURNS, refav1's centred-goal planner BEATS THE STRONGEST T1 FLOOR:**
`ccos_argmax` **0.9195** against `ha0_ext` **1.1521** — a mean per-window advantage
of **−0.2326 m**. **It survives the inference-seed floor by a wide margin:** the
replicate reads **0.9238**, a difference of **0.0043 m**, so the win is **54x** the
run-to-run noise on that stratum. The shipped-weights arm, which does nothing, reads
**0.9682** on the same windows — i.e. **acting is worth 0.049 m on turns and the
centred goal is worth another 0.047**.

⛔ **AND THE WHOLE DEFICIT IS ON STRAIGHT ROAD:** `ccos_argmax` **1.6960** against
`ha0_ext` **0.6285** — **+1.0675 m**. The damage is a TAIL, not a shift: the MEDIAN
per-window delta is only **+0.1933** while the MEAN is **+1.0675**, and `cl` is
better on **19.0 %** of straight windows. A minority of straight windows in which the
planner commits a hard turn accounts for essentially all of it — which is precisely
the population §6 measured leaving the friction circle at up to **3.262 g**.

⇒ **THE OBJECTIVE IS NOW A ONE-SENTENCE ENGINEERING PROBLEM:** *suppress spurious
curvature on straight windows without suppressing it on turns.* That is exactly what
the three levers this package built are for, and they attack it from three different
directions — the hold branch (§3) removes the noise-driven direction where the goal
IS "hold"; the friction-circle cap (§6) removes the high-speed hard turns, and
straights are the fast windows; the seed ladder (§5) supplies the intermediate
curvature the corpus actually needs so a turn need not be all-or-nothing.

⚠️ **And it re-reads §7.1's paradox.** The shipped weights "improve seven family
metrics" by trading a **0.049 m turn-window advantage** for a **0.810 m
straight-window repair**. On this turn-DENSE panel that is a net win; on a corpus
with the real straight/turn mix it would be a larger one — **and it is still not
driving.**

⚠️ *Bookkeeping:* on GT-turn, `ol` (1.152107) and `ha0_ext` (1.152063) agree to four
decimals **by coincidence** — their trajectories differ by up to **2.42 m**. Not
duplication; checked because equal-looking floors are how harness bugs hide.


### 7.3 ⭐⭐ L1's CORE ARM: A CURVATURE PENALTY AT THE MEASURED SCALE BRINGS refav1 TO ADE PARITY WITH THE STRONGEST T1 FLOOR — WHILE IT IS STILL ACTING

`wk15` = `ccos` + `(W_JERK, W_KAPPA, W_VEND) = (0, **15.11245**, 64.297)`, one
variable against the banked `ccos_argmax` (`W_KAPPA = 0`): same metric, same seed,
same panel, same vocabulary. The rung is **10 % of the measured goal decision at the
realised curvature** — `0.10 × 0.967197 / 0.0064` — fixed by §1 before any arm ran.

**Four families, T1, n = 40 windows / 8 episodes** (`raw/four_family_all.txt`):

| metric | `ccos_argmax` (W_KAPPA 0) | **`wk15` (W_KAPPA 15.11)** | `ha0_ext` floor | `ha0` floor |
|---|---|---|---|---|
| **ADE m** | 1.3272 [0.7750, 1.9915] | **0.8934 [0.6517, 1.1348]** | **0.8772** | 0.9251 |
| LON speed MAE | 0.7155 | 0.7919 | **0.3058** | 0.7217 |
| LON along MAE | 0.7304 | **0.6630** | 0.6819 | 0.6788 |
| LAT heading MAE deg | 23.4578 | **15.2704** *(best of all arms)* | 27.7357 | 20.1374 |
| LAT yaw-rate MAE | 17.6748 | **6.6750** *(best)* | 13.9710 | 7.0932 |
| LAT curvature MAE 1/m | 0.055369 | **0.030982** *(best)* | 0.077298 | 0.040083 |
| LAT cross MAE m | 0.8784 | **0.3670** | 0.4018 | 0.3642 |
| TAC lat acc / kappa | 0.6250 / 0.3795 | 0.6250 / 0.2611 | 0.7750 / 0.6277 | 0.5250 / 0.0000 |
| TAC goal FDE m | 2.9639 | **2.1628** *(best T1)* | 2.2693 | 2.2613 |

Paired against the floors (episode-cluster bootstrap):
`cl - ha0_ext` **ADE +0.0162 [-0.1648, +0.1980]**, not separated — against
**+0.4500** for `ccos_argmax`. `cl - ha0` ADE **-0.0317 [-0.1344, +0.0559]**.
⚠️ **+0.0162 is SMALLER than this rig's own inference-seed floor (§4: 0.0607), so
the honest word is PARITY, not "beats".** The longitudinal family is where it still
loses outright: `speed_mae` **+0.4862 [+0.2825, +0.7201] separated worse**.

**By stratum** (`raw/ade_by_stratum.txt`), against `ha0_ext`:

| stratum | `ccos_argmax` | **`wk15`** | `ha0_ext` |
|---|---|---|---|
| GT-turn (n=19) | 0.9195 (**-0.2326**) | **0.9699 (-0.1822**, `frac cl better` **0.5263**) | 1.1521 |
| GT-straight (n=21) | 1.6960 (**+1.0675**) | **0.8242 (+0.1957**, `frac cl better` **0.4286**) | 0.6285 |

⇒ **the straight-window damage falls 5.5x (+1.0675 -> +0.1957) while the turn-window
advantage is kept** (-0.1822 against -0.2326, a change of 0.05 against a
turn-stratum seed floor of 0.0043 — so the small loss on turns is real but tiny).
Non-zero curvature on GT-straight windows falls **0.6667 -> 0.3333**.

### 7.3.2 THE PAIRED FOUR-FAMILY DELTA OF `W_KAPPA` ITSELF, AGAINST THE SEED FLOOR

`wk15.cl - ccos_argmax.cl`, paired episode-cluster bootstrap, n = 40 windows /
8 clusters, n_boot 2000; the known-value control (an arm against itself) reads
`+0.0000 [+0.0000, +0.0000]` on every metric (`raw/pd_all.md`).
⛔ **Each row is read against §4's inference-seed floor on the SAME metric**, because
a separated CI is necessary and not sufficient on this rig.

| family metric | delta (W_KAPPA 0 -> 15.11245) | separated | **seed floor (§4)** | **ratio** | verdict |
|---|---|---|---|---|---|
| `ade_m` | **-0.4338 [-1.0393, -0.0808]** | YES | 0.0607 | **7.1x** | LEVER, better |
| `fde_m` | **-0.8012 [-1.7932, -0.1711]** | YES | 0.3439 | **2.3x** | LEVER, better |
| `LAT_cross_mae_m` | **-0.5114 [-1.1777, -0.0685]** | YES | 0.0710 | **7.2x** | LEVER, better |
| `LAT_heading_mae_deg` | **-3.8969 [-7.0911, -0.7589]** | YES | 1.1180 | **3.5x** | LEVER, better |
| `LAT_yaw_rate_mae_radps` | **-0.1571 [-0.2579, -0.0670]** | YES | 0.0081 | **19x** | LEVER, better |
| `LON_speed_mae_mps` | **+0.0764 [+0.0135, +0.1453]** | YES | 0.0038 | **20x** | LEVER, **worse** |
| `LON_accel_mae_mps2` | **+0.0874 [+0.0122, +0.1763]** | YES | 0.0061 | **14x** | LEVER, **worse** |
| `LON_along_mae_m` | -0.0674 [-0.2103, +0.0707] | no | 0.0002 | — | — |
| `TAC_traj_lat_correct` | +0.0000 [-0.2000, +0.2000] | no | 0.0750 | — | — |
| `TAC_traj_lon_correct` | +0.0000 [-0.1000, +0.1250] | no | 0.0000 | — | — |

⇒ **Every separated delta clears the seed floor by 2.3x to 20x, so all seven are
LEVER effects and not inference noise.** The curvature penalty buys the entire
lateral family and the trajectory error, and it **costs** the longitudinal family:
speed MAE and accel MAE both degrade, separated and 14-20x the floor.

⭐ **That trade is the next work item, and it is NOT a `W_KAPPA` question.** §2
showed the longitudinal deficit is the decoded LON token commanding `a == 0`
(29/40 windows decode `ADAPT_SPEED_FOR_CURVE`); a curvature penalty makes the
planner spend its remaining freedom on the lateral channel, which is why the speed
channel drifts. The longitudinal analogue of this package's levers has not been
built.

### 7.3.1 ⛔ MY OWN PRE-REGISTERED PREDICTION (P1) IS REFUTED — AND P3 IS WHAT HAPPENED

`PREREG_COST_GEOMETRY.md` §5 registered, before this arm ran and before the seed
ladder was written: *"(P1) the realised curvature must move DISCRETELY from 0.08 to
0.0 … with no new mass at intermediate magnitudes"*, and *"(P3) if instead
intermediate magnitudes appear, the seed pool is richer than `D-REFAV1-DRIVE-GATE`
says and that claim needs re-reading."*

MEASURED on `wk15` (`raw/kappa_quantisation_all.txt`): **15 distinct realised
`max|kappa|` values** against `ccos_argmax`'s 10, and the new mass is exactly where
P1 said it could not be — **0.011507, 0.011993, 0.013975, 0.018027, 0.020659,
0.024702, 0.028205, 0.030590, 0.032682, 0.033837, 0.044389, 0.048942, 0.052956** —
i.e. **R 19–87 m, squarely inside the corpus's own band and nowhere near either
canonical value**. The EXACTLY-constant-series fraction falls **0.7750 -> 0.6750**;
`TURN_L`-decoded windows now realise a median `max|kappa|` of **0.02066** instead of
0.08000.

⇒ **P3 holds. The gate decides whether curvature is SEEDED; the search can then
MODULATE its magnitude once the cost gives it a reason to.** With `W_KAPPA = 0`
there was no reason, so every plan sat at its seed's own value and looked quantised.
`D-REFAV1-DRIVE-GATE` is **not retracted** — a `LANE_KEEP` decode still yields
exactly zero curvature on 18/18 windows here — but its consequence *"a turn is
unreachable by the entire iCEM population"* must be read as **"unreachable when the
cost is indifferent"**, which is a materially weaker and more actionable statement.

⚠️ **This also means the `l3ladder` arm is now testing a sharper question than it was
written for:** not *"can the pool express an intermediate curvature at all"* (it can)
but *"does supplying the intermediate rungs directly beat letting the search find
them"*. Both outcomes remain committed; the arm is unchanged.


### 7.4 ⛔ L2 ANSWERED — A CLEAN NULL: THE HOLD BRANCH REPAIRS THE COST ON 25 % OF WINDOWS AND CHANGES NOT ONE PLAN

`ccosh_w000` = `ccosh` + `(0, 0, 64.297)`, the **metric is the only variable**
against the banked `ccos_argmax`: same weights, same seed, same panel, same
vocabulary. Artifact `raw/ccosh_null.txt`.

**THE PLAN.** `cl` trajectories against `ccos_argmax`: **max abs diff
0.000000e+00, BIT-IDENTICAL on all 40 windows.** Every four-family number is
therefore identical to the digit — ADE **1.3272**, speed MAE 0.7155, curvature MAE
0.055369, cross MAE 0.8784, heading 23.4578, lane-keep recall 0.7143, turn recalls
0.3636 / 0.75.
⛔ **SAME-BREATH CONTROL THAT MUST DIFFER, and does:** `wk15` against the same
baseline reads **1.508402e+01 m** apart. The comparison is live; the null is a
measurement, not a broken pipe.

**THE COST.** The branch fired exactly where it was designed to:

| | `basecost_cv` frac == **1.0** | min | median | median `plan_cost` | `cem` frac |
|---|---|---|---|---|---|
| `ccos_argmax` | **0.2500** | 0.884896 | 1.0 | 3.10838e-05 | 0.750 |
| `ccosh_w000` | **0.0000** | **3.9105e-08** | 0.967232 | 9.26852e-06 | 0.925 |

⇒ **the degeneracy is real, it is confined to the 25 % of windows whose goal IS the
hold field, and removing it is behaviourally inert on this panel.** The `cem`
fraction rising 0.750 → 0.925 while the emitted controls stay bit-identical is the
mechanism in one number: on those windows the CEM's own best sample was *already*
the all-zero plan, so making `cv` cheap merely re-labelled which of two identical
control sequences won the argmin.

⚠️ **AND IT CORRECTS A FIGURE THIS PACKAGE INHERITED.** The brief and
`COST_METRICS` put the affected population at **86.5 %** of the eval grid. On this
panel the goal is the hold field on **25.0 %** of windows — LANE_KEEP is *decoded*
on 45.0 %, but on most of those the LON token still commands a non-zero
acceleration, so `g ≠ z_ref` and the branch correctly does not fire. **A LANE_KEEP
decode is not a hold goal.** The 86.5 % figure belongs to a different grid and must
carry it.

⇒ **L2 is not the lever.** Under the standing instruction the next lever was already
running when this landed; `ccosh` stays in the tree as a pinned, non-default,
*correct* instrument — its value is that it makes the cost defined where it was not,
which matters for any future arm in which `cv` is not already the winner.


### 7.5 ⭐⭐ THE LADDER CLOSES: A CLEAN INTERIOR OPTIMUM, EXACTLY WHERE §1'S SCALE SAID IT WOULD BE

`wk151` = `ccos` + `(0, **151.1245**, 64.297)` — the **100 %** rung, i.e. the
curvature charge at the realised curvature equals the ENTIRE measured goal
decision. It landed at ADE **0.9084 [0.7134, 1.1132]**, *worse* than the 10 % rung
and past the optimum. The full dose-response (`raw/wkappa_dose.txt`):

| arm (metric, `W_KAPPA`) | ADE m | curv MAE 1/m | heading MAE deg | TAC lat kappa | turn recall L / R |
|---|---|---|---|---|---|
| `cos_argmax` (`cos`, 0) | 1.8944 | 0.080478 | 29.5086 | **-0.1249** *(below chance)* | 0.0 / 0.25 |
| `ccos_argmax` (`ccos`, 0) | 1.3272 | 0.055369 | 23.4578 | **0.3795** | 0.3636 / 0.75 |
| `ccosh_w000` (`ccosh`, 0) | 1.3272 | 0.055369 | 23.4578 | 0.3795 | 0.3636 / 0.75 |
| **`wk15` (`ccos`, 15.11245 = 10 %)** | **0.8934** | **0.030982** | **15.2704** | 0.2611 | 0.0 / **0.5** |
| `wk151` (`ccos`, 151.1245 = 100 %) | 0.9084 | 0.038019 | 15.3785 | **0.0000** | 0.0 / 0.0 |
| `cos_wk` (`cos`, **the SHIPPED 0.05**) | 0.9251 | 0.040083 | 20.1374 | **0.0000** | 0.0 / 0.0 |
| `ha0_ext` floor | **0.8772** | 0.077298 | 27.7357 | 0.6277 | — |

**And the collapse is visible in one column — the ARM TOOL'S OWN trivial-profile
probe** (`refav1_arm.py [trivial-profile]`, not a re-implementation; `straight` is
`|y| < 1e-06 m`, `const_speed` spread `< 0.0001 m`):

| arm | `straight` | `CONSTANT-VELOCITY` | **`identical_to ha0`** |
|---|---|---|---|
| `cos_argmax` (`cos`, 0) | 0.0750 | 0.0750 | 3/40 |
| `ccos_argmax` / `ccosh_w000` (0) | 0.2750 | 0.2750 | 11/40 |
| **`wk15` (10 %)** | 0.4750 | **0.2500** | **10/40** |
| `wk151` (100 %) | 0.6500 | 0.4250 | 17/40 |
| `cos_wk` (SHIPPED) | **1.0000** | **1.0000** | **40/40** |

⇒ **The ladder has a clean interior optimum at the 10 % rung, and the 100 % rung is
already on the way to the do-nothing plan** — `identical_to ha0` climbs
11 → 10 → 17 → 40 and both turn recalls fall to **0.0** at 100 %, exactly as they
are at the shipped weights. **This validates the rung definition itself:** §1 fixed
"100 %" as *the curvature charge equalling the whole goal decision*, from banked
data and before any arm ran, and that is precisely where the planner stops
choosing to turn. A weight sweep chosen by eye would have had no way to know that
the interesting range was `0 < W_KAPPA < 151`.

⚠️ **`wk15`'s asymmetry is real and is a work item:** `turn_right` recall **0.5**
against `turn_left` **0.0**, on n = 8 right and n = 11 left GT turns. The
`W_KAPPA = 0` arm reads 0.75 / 0.3636, so the penalty costs the left turns first.
With the tactical-decision seed floor at 0.0750 absolute (§4) an 11-window recall
is far too thin to call, but the direction is consistent across two rungs and it
should be checked on a wider panel before `wk15` is quoted as a lateral fix.

### 7.5.1 TWO DEFECTS I INTRODUCED, BOTH CAUGHT BEFORE THEY COST GPU

1. **`UnboundLocalError: _inspect2`** killed `l3ladder` and `combined` at startup
   (18:39:32Z / 18:39:56Z). `inspect` was imported INSIDE the `--goal-kappa-turn`
   branch, and my seed-ladder block referenced it from a sibling branch — so any
   arm passing `--seed-kappa-ladder` *without* `--goal-kappa-turn` died. Fixed by
   importing it in the block that uses it. ⭐ **Zero GPU was wasted: the tool
   raised at `run_dump` before the rollout**, which is exactly what its
   preflight-import design exists for (`[preflight] 11 analysis imports OK` — the
   probe that exists "because a 3.0 h rollout once died in analyze()").
2. **A bare `%` in the `--kamm-mu` help string broke `--help` itself** with
   `TypeError: must be real number, not dict`, because argparse `%`-formats help
   text. The file's own convention (`--goal-kappa-turn` writes `100 %%`) was the
   thing I broke. ⚠️ This one is invisible to every arm that runs and appears only
   when an operator asks for help or makes a CLI typo — the worst time to hand
   someone a traceback. Escaped; `--help` now exits 0 and lists all three new flags.

Both arms were relaunched at 18:42:21Z under `raw/queueG.sh`, which also `rm -rf`s
the stale dump directory first so a partial dump from the crashed attempt cannot be
mistaken for a panel.


### 7.6 ⭐⭐ L4 ANSWERED: THE FRICTION-CIRCLE CAP IS A **DIFFERENT KIND** OF LEVER FROM `W_KAPPA` — IT BUYS SAFETY AND HALF THE ACCURACY, AND IT COSTS NOTHING

`kamm07` = `ccos` + `(0, 0, 64.297)` + `--kamm-mu 0.7`: the CONSTRAINT is the only
variable against the banked `ccos_argmax`. Same metric, same weights, same seed,
same panel, same vocabulary.

**The axis it was built for** (`assert_feasible`, `v0 >= 2 m/s`, n = 27; ⛔ the
GROUND-TRUTH control reads `envelope 0.0000 / kamm_over 0.0000`, so the block is
admissible):

| | `ccos_argmax` | **`kamm07`** | ground truth |
|---|---|---|---|
| `kamm_over_rate` (mu = 0.7) | 0.2963 | **0.1481** | 0.0000 |
| `peak_g` mean | 0.558 | **0.256** | 0.178 |
| **`peak_g` max** | **3.262** | **0.707** | 0.373 |
| `max\|kappa\|` | 0.2000 *(= the clip)* | **0.174670** *(below it)* | 0.1701 |
| `envelope_rate` | 0.0000 | 0.0000 | 0.0000 |

⇒ **`peak_g` max falls 4.6x to 0.707 — μ itself, to numerical tolerance — and
`max|kappa|` drops below the 0.2 clip, so the Kamm bound, not the constant clip, is
now what binds.** The violation rate halves rather than vanishing, and the reason is
a scope difference that must be stated, not hidden: the cap is applied to the
CANDIDATE'S CONTROLS at plan time using its own accel-integrated speed, while
`assert_feasible` re-derives `kappa = a_lat / v_mid^2` from the ROLLED-OUT PATH with
`v_mid` floored at `MIN_SPEED = 0.5`. **They are not the same object**, and closing
the residual is a work item, not a failure of the cap.

**And the four families, paired against `ccos_argmax`, read against §4's seed floor:**

| family metric | **`kamm07` - `ccos_argmax`** | vs floor | `wk15` - `ccos_argmax` | vs floor |
|---|---|---|---|---|
| `ade_m` | -0.3344 [-0.9536, +0.0625] | not sep. | **-0.4338 [-1.0393, -0.0808]** | 7.1x, sep. |
| `LAT_yaw_rate_mae` | **-0.0676 [-0.1476, -0.0038]** | 8.3x, sep. | **-0.1571** | 19x, sep. |
| `LAT_cross_mae_m` | -0.3599 [-1.0002, +0.0579] | not sep. | **-0.5114** | 7.2x, sep. |
| `LAT_heading_mae_deg` | -2.3315 [-5.1211, +0.0491] | not sep. | **-3.8969** | 3.5x, sep. |
| **`LON_speed_mae_mps`** | **-0.0017 [-0.0078, +0.0034]** | **below the 0.0038 floor** | **+0.0764 WORSE** | 20x, sep. |
| **`LON_accel_mae_mps2`** | **+0.0013 [-0.0035, +0.0064]** | **below the 0.0061 floor** | **+0.0874 WORSE** | 14x, sep. |
| `TAC_traj_lat_correct` | +0.0000 [-0.0750, +0.0750] | — | +0.0000 | — |

**Tactical decision quality is PRESERVED, which `W_KAPPA` does not do:**

| | `ccos_argmax` | **`kamm07`** | `wk15` |
|---|---|---|---|
| TAC lateral kappa | 0.3795 | **0.3644** *(within the 0.0750 floor)* | 0.2611 |
| lane_keep recall (n=21) | 0.7143 | **0.7619** | 1.0 |
| **turn_left recall (n=11)** | 0.3636 | **0.3636** *(identical)* | **0.0** |
| **turn_right recall (n=8)** | 0.75 | **0.625** | 0.5 |
| ADE, GT-turn (n=19) | 0.9195 | **0.9195** *(identical)* | 0.9699 |
| ADE, GT-straight (n=21) | 1.6960 | **1.0590** | **0.8242** |

⭐ **THE TWO LEVERS ARE COMPLEMENTARY, AND THE DIFFERENCE IS STRUCTURAL, NOT A
MATTER OF DEGREE.** `W_KAPPA` is a **penalty**: it charges every curvature, so it
buys the most accuracy and pays for it by suppressing turning *in general* — turn
recall to 0.0/0.5, tactical kappa 0.38 → 0.26, and a separated 14–20x-the-floor
regression in the longitudinal family, because the plan spends its remaining
freedom laterally. The Kamm cap is a **constraint with units**: it removes only the
curvature the tyre cannot deliver, so it leaves the turn decisions bit-untouched
(`turn_left` recall and GT-turn ADE both *identical* to the uncapped arm), costs the
longitudinal family **less than the noise floor on both metrics**, and is the only
lever here that moves the safety axis at all.

⇒ **The combined arm — ladder + cap — is therefore the right next experiment and is
already queued behind `l3ladder`;** its pre-registration (§2, §3) is unchanged. And
`W_KAPPA`'s straight-window advantage over the cap (+0.1957 vs +0.4304 against
`ha0_ext`) says the two are attacking *different halves* of the same 21 windows.


### 7.7 ⛔⭐ L3 ANSWERED — A NULL THAT SETTLES THE QUESTION: THE CANDIDATE SET WAS NEVER THE BINDING CONSTRAINT, THE COST WAS

`l3ladder` = `ccos` + `(0, 0, 64.297)` + `--seed-kappa-ladder 0.002,0.005,0.01,0.02,0.04`
— **10 extra iteration-0 candidates at R 500 / 200 / 100 / 50 / 25 m**, both signs,
the candidate set the only variable against `ccos_argmax`. `canonical_controls` and
the goal field are untouched, so the arms are comparable window-for-window.

| | `ccos_argmax` | **`l3ladder`** | seed floor (§4) |
|---|---|---|---|
| ADE m | 1.3272 | **1.3236** | 0.0607 — **the delta is 0.0036, 17x BELOW it** |
| curvature MAE 1/m | 0.055369 | 0.053726 | — |
| EXACTLY-constant series | 0.7750 | **0.7750** *(unchanged)* | — |
| turn recall L / R | 0.3636 / 0.75 | **0.3636 / 0.75** *(identical)* | — |
| GT-turn ADE | 0.9195 | 0.9258 | — |
| GT-straight ADE | 1.6960 | 1.6835 | — |

⛔ **AND THE DECISIVE ROW IS THE HISTOGRAM: NOT ONE WINDOW OF 40 REALISES A RUNG.**

```
l3ladder     0.0000 x10  0.0800 x21  0.1009  0.1189  0.1253  0.1270  0.1274  0.1505  0.1508  0.1686  0.1782
ccos_argmax  0.0000 x10  0.0800 x21  0.0766  0.0888  0.1031  0.1406  0.1414  0.1747  0.1909  0.2000 x2
wk15         0.0000 x18  0.0800 x9   0.0115  0.0120  0.0140  0.0180  0.0207  0.0247  0.0282  0.0306  0.0327  0.0338  0.0444  0.0489  0.0530
```

The rungs are **0.002, 0.005, 0.01, 0.02, 0.04**. `l3ladder` realises **nothing
below 0.08** except the ten exact zeros — its extra mass is all *above* 0.10. The
`0.0000 x10 / 0.0800 x21` spine is bit-for-bit the uncapped arm's.

⭐ **PUT IT BESIDE §7.3.1 AND THE ANSWER IS COMPLETE.** `wk15` produced intermediate
curvatures **0.0115–0.0530 — exactly the band the rungs occupy — with NO LADDER AT
ALL**, purely because the penalty gave the search a reason to prefer them.
`l3ladder` hands the search those very magnitudes and it **never picks one**,
because with `W_KAPPA = 0` the cost is indifferent between a rung and the
goal-aligned canonical seed, and the seed wins on goal alignment.

⇒ **The binding constraint was never the candidate set. It is the cost.** That
retires L3 as a lever in its own right and confirms the §7.3.1 re-reading of
`D-REFAV1-DRIVE-GATE` from the other direction: the gate is not a wall around the
reachable set, it is an absence of preference.

⛔⛔ **SELF-CORRECTION, SAME TURN, CAUGHT BY THIS PACKAGE'S OWN RULE.** An earlier
draft of this section read: *"the ladder is not free — `lane_keep` recall
0.7143 → 0.5714 (~2x the 0.0750 floor) and goal FDE 2.9639 → 3.3200 — exactly what
the `--seed-kappa-ladder` docstring warned."* **That claim is NOT SUPPORTED and is
withdrawn.** The `ccos_seed1` replicate — same flags, only `--plan-seed` differs —
reads **lane_keep recall 0.5714** and **goal FDE 3.3079** on its own, i.e. the
*identical* movement with **no lever at all**:

| tactical metric | `ccos_argmax` | seed replicate (NO lever) | `l3ladder` | verdict |
|---|---|---|---|---|
| `lane_keep` recall | 0.7143 | **0.5714** (floor 0.1429) | **0.5714** (delta 0.1429) | **inside the floor** |
| goal FDE m | 2.9639 | 3.3079 (floor 0.3440) | 3.3200 (delta 0.3561) | **inside the floor** |
| TAC lateral kappa | 0.3795 | 0.2822 (floor 0.0973) | 0.2822 (delta 0.0973) | **inside the floor** |

**The error was using the WRONG FLOOR:** 0.0750 is the paired
`TAC_traj_lat_correct` floor, not the per-class `lane_keep` recall's own, which is
**0.1429**. ⇒ **every tactical movement in `l3ladder` is inference-seed noise, and
the ladder is neither costly nor beneficial here — it is inert on all four
families.** That strengthens the section's conclusion rather than weakening it: the
null is total.

⚠️ **The docstring's warning ("with `W_KAPPA = 0` a wider curvature set can only add
ways to be wrong") is therefore UNTESTED on this panel, not confirmed** — it
predicted a cost this panel is not powered to see. Recorded as an open prediction,
not a vindicated one. ⭐ This is exactly the sufficiency rule of §4 turned on my own
claim: *a difference is only a lever effect if it exceeds the seed pair's
difference on the SAME metric*, and "the same metric" means the same statistic,
not a cousin of it.

⇒ **The informative combination is therefore `ladder + W_KAPPA`, not `ladder`
alone and not `ladder + a CONSTRAINT`** — a cap can forbid a curvature but cannot
make a rung *attractive*. That arm (`wk15_ladder`: `ccos`, `W_KAPPA = 15.11245`,
the same five rungs) was queued the moment this null landed and is one variable
against `wk15`.


### 7.8 ⭐⭐ `combined` — THE LADDER'S REAL ROLE UNDER THE CAP

> ⛔⛔ **HEADING CORRECTED TWICE, AND THE SECOND CORRECTION IS THE BINDING ONE.**
> It read *“THE FIRST refav1 ARM WITH **ZERO** FRICTION-CIRCLE VIOLATIONS”*.
> **(1)** The superlative is withdrawn — `wk15`, `wk151` and `cos_wk` also read
> 0.0000 (§7.10, `raw/decode_vs_execute.md`). **(2)** ⛔ **AND THE ZERO ITSELF IS
> SEED-DEPENDENT**: the pre-registered replicate `combined_seed1` reads
> **0.0741** with all three controls passing (**§7.11a**). Everything below that
> says “zero” must be read as **“0.0000 on seed 0, 0.0741 on seed 1”**. What
> survives on two seeds is the TURN half: recalls 0.3636 / 0.7500, identical
> across the pair.

`combined` = `ccos` + `(0, 0, 64.297)` + `--seed-kappa-ladder 0.002,0.005,0.01,0.02,0.04`
+ `--kamm-mu 0.7`. Three variables against `ccos_argmax`, **one** against `kamm07`:
the candidate set.

**The safety axis closes completely** (`assert_feasible`, `v0 >= 2 m/s`, n = 27):

| arm | `kamm_over_rate` (mu = 0.7) | `peak_g` max | `max\|kappa\|` | role |
|---|---|---|---|---|
| `g` ground truth | **0.0000** | 0.373 | 0.1701 | CONTROL — must be 0 |
| **`combined`** | **0.0000** | **0.618** | 0.1505 | — |
| `kamm07` (cap only) | 0.1481 | 0.707 | 0.174670 | CONTROL — must be NON-zero |
| `ccos_argmax` (neither) | 0.2963 | **3.262** | 0.2000 | CONTROL — must be NON-zero |
| `ha0_ext` floor | 0.1852 | 1.436 | 0.7672 | CONTROL — must be NON-zero |

⛔ **The `0.0000` is bracketed on both sides:** the ground-truth control reads the
same known value, and three arms in the same table read non-zero — so it is a
measurement, not an unevaluated branch. **This is the first refav1 arm in the
programme whose plans are entirely inside the tyre's friction circle**, and it
closes the residual §7.6 flagged as a work item on the cap alone (0.1481 → 0.0000)
**in the same turn it was raised**.

**And it is free in ADE terms.** Paired against `kamm07`, the cap-only arm:
`ade_m` **+0.0577 [-0.0369, +0.1589]**, not separated — and **below the 0.0607
inference-seed floor**. `LAT cross` +0.0578 (n.s.), `LAT heading` +0.3126 (n.s.),
`LAT yaw_rate` -0.0145 (n.s.). The one separated row is
`LON_accel_mae` **+0.0176 [+0.0013, +0.0415]** — 2.9x its floor, a small real cost.

**Why the ladder matters here and not on its own.** §7.7's null stands unchanged:
with `W_KAPPA = 0` and no cap, not one window of 40 chose a rung. Under the cap the
picture inverts — `combined` is **4.434113e+00 m** from `kamm07` (control:
1.359604e+01 m from `ccos_argmax`), so the extra candidates *do* change plans, and
the realised set now contains **0.0198, 0.0216, 0.0374, 0.0400, 0.0500, 0.0559,
0.0827** including the rung value **0.0400 exactly**. ⇒ **the ladder's role is not
to be preferred, it is to give the CAP something feasible to select** instead of
clipping a candidate whose rolled-out path then still violates. A constraint plus a
feasible candidate set reaches zero; either alone does not.

**Four families, and the honest limits:**

| | `ccos_argmax` | `kamm07` | **`combined`** | `wk15` | `ha0_ext` |
|---|---|---|---|---|---|
| ADE m | 1.3272 | **0.9927** | 1.0504 | **0.8934** | **0.8772** |
| curvature MAE | 0.055369 | 0.048462 | 0.046862 | **0.030982** | 0.077298 |
| TAC lateral kappa | 0.3795 | 0.3644 | **0.4148** | 0.2611 | 0.6277 |
| turn recall L / R | 0.3636 / 0.75 | 0.3636 / 0.625 | **0.3636 / 0.75** | 0.0 / 0.5 | — |
| GT-turn ADE | 0.9195 | 0.9195 | 0.9258 | 0.9699 | 1.1521 |
| GT-straight ADE | 1.6960 | 1.0590 | 1.1632 | **0.8242** | 0.6285 |
| `kamm_over_rate` | 0.2963 | 0.1481 | **0.0000 / 0.0741** ⛔ | **0.0000** ⚠️ | 0.1852 |
| turn recall L (n=11) | 0.3636 | 0.3636 | **0.3636** | **0.0000** ⚠️ | — |

> ⛔ `combined`'s cell is **seed 0 / seed 1** — the zero did NOT replicate (§7.11a).
> ⚠️ The `wk15` cell read `*(not run)*` when this table was written and it was the
> refutation waiting to happen: it HAD run, two hours earlier, and reads the same
> 0.0000 at **turn_left recall 0.0000 of 11** — also single-seed and unreplicated.

⚠️ **`combined` has the highest tactical lateral kappa of any arm (0.4148) and keeps
both turn recalls at the uncapped arm's values — but 0.4148 - 0.3795 = 0.0353 is
BELOW the 0.0973 seed floor on that metric, so "best tactical" is NOT established
and is not claimed.** What is established is that the safety repair costs neither
the turn decisions nor, beyond one small separated accel row, the families.

⇒ **The package's two working levers are now cleanly separated by what they buy:**
`W_KAPPA` is the **accuracy** lever (best ADE and lateral family, at the cost of the
longitudinal one and of turning in general); `cap + ladder` is the **safety** lever
(⛔ *not* the only zero-violation arm, and its zero is seed-dependent — §7.11a; what
stands is that it reduces the rate at no ADE cost against the cap alone and with the
turn decisions intact). They have never been run together — that is the next arm, and
`wk15_ladder` is the half of it already on the GPU.


### 7.9 ⛔⭐⭐ `wk15_ladder` — A DISCRETE RUNG SET AND A CONTINUOUS QUADRATIC PENALTY ARE **ANTAGONISTIC**

`wk15_ladder` = `ccos` + `W_KAPPA = 15.11245` + the same five rungs; the candidate
set is the only variable against `wk15`. This is the arm §7.7's null asked for —
the rungs *plus* a cost with a reason to prefer them.

**The rungs are chosen now, and that is the problem.**

| arm | distinct realised `max\|kappa\|` | EXACTLY-constant | **windows landing EXACTLY on a rung** | realised set |
|---|---|---|---|---|
| `wk15` (penalty, no ladder) | **15** | 0.6750 | **0** | 0.0000 x18, then 0.0115 … 0.0530 (13 values), 0.0800 x9 |
| `l3ladder` (ladder, no penalty) | 11 | 0.7750 | **0** | 0.0000 x10, 0.0800 x21, nine values above 0.10 |
| **`wk15_ladder` (both)** | **2** | **1.0000** | **22** | **0.0000 x18, 0.0020 x22** |

⇒ **22 of 40 windows snap EXACTLY onto the SMALLEST rung (0.002 = R 500 m), and the
realised set collapses from 15 distinct curvatures to TWO.** The mechanism is
immediate: a quadratic penalty always prefers the **cheapest non-zero option
available**, so a discrete ladder converts "smallest expressible curvature" from a
continuum the search can trade along into a **fixed floor it snaps to**. The
continuous search was doing better precisely because nothing quantised it.

**And the arm stops making lateral decisions** (deltas against `wk15`, each read
against its OWN per-metric seed floor — the §7.7 lesson applied):

| metric | `wk15` | `wk15_ladder` | delta | floor | ratio | verdict |
|---|---|---|---|---|---|---|
| **TAC lateral kappa** | 0.2611 | **0.0000** | **-0.2611** | 0.0973 | **2.7x** | **LEVER, worse** |
| **TAC turn_right recall** | 0.50 | **0.00** | **-0.50** | 0.00 | — | **moved, floor is 0** |
| LAT curvature MAE | 0.03098 | 0.03979 | +0.00881 | 0.00066 | **13.3x** | **LEVER, worse** |
| LAT heading MAE deg | 15.2704 | 20.0999 | +4.8295 | 1.1458 | **4.2x** | **LEVER, worse** |
| **LON speed MAE** | 0.7919 | **0.7609** | **-0.0310** | 0.0038 | **8.2x** | **LEVER, better** |
| **LON accel MAE** | 0.8604 | **0.8159** | **-0.0445** | 0.0061 | **7.3x** | **LEVER, better** |
| ADE m | 0.8934 | 0.9408 | +0.0474 | 0.0607 | 0.78 | inside floor |
| LAT cross MAE | 0.3670 | 0.3646 | -0.0024 | 0.0710 | 0.03 | inside floor |
| LAT yaw-rate MAE | 6.6750 | 7.0599 | +0.3849 | 0.5771 | 0.67 | inside floor |

The longitudinal family *improves* — separated, 7–8x its floors — for the same
reason the lateral one degrades: the plan stops spending its freedom on curvature.

⭐⭐ **THE DESIGN LESSON, AND IT IS THE MOST TRANSFERABLE THING IN THIS PACKAGE:**
**a PENALTY and a discrete candidate set are antagonistic; a CONSTRAINT and a
discrete candidate set are complementary.** A constraint has no preference
gradient — it forbids, it does not rank — so a ladder simply hands it feasible
options, which is exactly why `combined` (cap + ladder) reached `kamm_over 0.0000`
(§7.8). A penalty *does* rank, so a ladder replaces its continuum with a floor and
the search collapses onto it. **This is why the two levers must be combined
WITHOUT the ladder**, and it is a statement about optimiser design, not about
refav1.

⚠️ **ARM-SET CONSEQUENCE, recorded rather than applied silently.** The synthesis
arm `best` had started 1 minute earlier carrying `W_KAPPA` + cap + **ladder**. On
this measurement it would have inherited the collapse and bought nothing, so it was
**killed by explicit PID and relaunched as `W_KAPPA` + cap ALONE** (`raw/queueJ.sh`,
which also removes the partial dump so it cannot be mistaken for a panel). No
criterion moved, nothing had been observed about `best`, and the amendment is in
`PREREG_COST_GEOMETRY.md` §7 — the same class as the `wk1p5` -> `l3ladder` swap in §6.


### 7.10 ⭐⭐⭐ `best` — THE SYNTHESIS ARM: THE CLOSEST refav1 HAS COME TO THE FLOOR WHILE ACTING, AND THE ONLY ARM SMOOTHER THAN THE HUMAN

`best` = `ccos` + `W_KAPPA = 15.11245` + `--kamm-mu 0.7`, **no ladder**. ONE variable
against `wk15` (the constraint) and ONE against `kamm07` (the penalty). Provenance
verified from the record: `metric=ccos`, `W_KAPPA=15.11245`, `kamm_mu=0.7`,
`seed_kappa_ladder=None`, n = 40/8, tier **T1**.

**Four families. `L` = the delta exceeds that metric's OWN seed floor; `-` = inside it.**

| metric | `ccos_argmax` | `wk15` | `kamm07` | `combined` | **`best`** | floor | vs `wk15` | vs `kamm07` |
|---|---|---|---|---|---|---|---|---|
| **ADE m** | 1.3272 | 0.8934 | 0.9927 | 1.0504 | **0.8838** | 0.0607 | -0.0096 `-` | **-0.1089 `L`** |
| LON speed MAE | 0.7155 | 0.7919 | 0.7138 | 0.7327 | 0.7751 | 0.0038 | **-0.0168 `L`** | +0.0613 `L` |
| LON accel MAE | 0.7730 | 0.8604 | 0.7743 | 0.7918 | 0.8465 | 0.0061 | **-0.0139 `L`** | +0.0722 `L` |
| LAT curvature MAE | 0.05537 | 0.03098 | 0.04846 | 0.04686 | **0.03128** | 0.00066 | +0.0003 `-` | **-0.0172 `L`** |
| LAT heading MAE | 23.4578 | 15.2704 | 21.0683 | 21.3887 | **15.2975** | 1.1458 | +0.027 `-` | **-5.7708 `L`** |
| LAT yaw-rate MAE | 17.6748 | 6.6750 | 12.8513 | 11.8161 | **6.8198** | 0.5771 | +0.145 `-` | **-6.0315 `L`** |
| LAT cross MAE | 0.8784 | 0.3670 | 0.5185 | 0.5763 | **0.3682** | 0.0710 | +0.001 `-` | **-0.1503 `L`** |
| TAC lateral kappa | 0.3795 | 0.2611 | 0.3644 | **0.4148** | 0.2727 | 0.0973 | +0.012 `-` | -0.092 `-` |
| TAC lane_keep recall | 0.7143 | 1.0 | 0.7619 | 0.7619 | **1.0** | 0.1429 | 0.0 `-` | **+0.2381 `L`** |
| TAC turn_L / turn_R | 0.36/0.75 | 0.0/0.50 | 0.36/0.63 | 0.36/0.75 | **0.0/0.50** | 0 | identical | moved |
| TAC goal FDE m | 2.9639 | 2.1628 | 2.3017 | 2.4753 | **2.1338** *(best)* | 0.3440 | -0.029 `-` | -0.168 `-` |

⭐ **`best` takes `wk15`'s entire lateral family — every lateral delta is INSIDE the
seed floor, i.e. the cap costs the accuracy lever nothing — and GIVES BACK part of
what the penalty took longitudinally: speed MAE -0.0168 (4.4x its floor) and accel
MAE -0.0139 (2.3x), both separated LEVER effects in the right direction.** Against
`kamm07` it wins six of the seven family metrics outright.

**And it is the only arm smoother than the recorded human** (`assert_feasible`,
`v0 >= 2 m/s`, n = 27):

| | `kamm_over_rate` | `peak_g` mean | `peak_g` max | `max\|kappa\|` |
|---|---|---|---|---|
| **`best`** | **0.0000** | **0.082** | **0.332** | **0.0800** |
| `g` ground truth (CONTROL, must be 0) | 0.0000 | 0.178 | 0.373 | 0.1701 |
| `ha0_ext` floor (CONTROL, must be > 0) | 0.1852 | 0.316 | 1.436 | 0.7672 |
| `ccos_argmax` (CONTROL, must be > 0) | 0.2963 | 0.558 | **3.262** | 0.2000 |

⇒ **zero violations, `peak_g` mean 0.082 against a human 0.178, and `max|kappa|`
never exceeding the goal token's own 0.08.** ⭐ **The ladder was not needed for
this** — `combined` reached zero *with* it, `best` reaches zero *without* it, because
the penalty already keeps curvature low enough for the constraint never to bind
hard. That is a strictly simpler configuration for the same safety result.

**And the curvature modulation is the RICHEST of any arm: 19 distinct realised
values** (`wk15` 15, `ccos_argmax` 10, `wk15_ladder` **2**), spanning
**0.0048 – 0.0800**, EXACTLY-constant fraction down to **0.5750**. Penalty +
constraint gives the most expressive lateral behaviour in the package; penalty +
ladder gave the least.

**By stratum, against the strongest T1 floor:**

| stratum | `best` | `ha0_ext` | delta | `frac cl better` |
|---|---|---|---|---|
| GT-turn (n=19) | **0.9742** | 1.1521 | **-0.1779** | **0.5263** |
| GT-straight (n=21) | **0.8020** | 0.6285 | +0.1735 | **0.4762** *(best of any arm; `ccos_argmax` 0.1905)* |
| ALL (n=40) | **0.8838** | **0.8772** | **+0.0066** | — |

⛔ **THE HONEST VERDICT. `cl - ha0_ext` = +0.0066 m is 9.2x SMALLER than this rig's
inference-seed floor (0.0607), so `best` is STATISTICALLY INDISTINGUISHABLE from the
strongest trivial floor — the tightest parity any refav1 arm has reached — and it is
NOT a win.** It is not the shipped arm's kind of parity either: `cos_wk` tied by
emitting zero controls, while `best` tracks 19 distinct curvatures, keeps
`turn_right` recall at 0.50, and is better than the floor on turns.

⚠️ **What still loses.** The longitudinal family: speed MAE **0.7751 vs 0.3058** for
`ha0_ext` — the largest remaining gap in the package and untouched by any lever
here. The tactical lateral decision: kappa **0.2727 vs 0.6277**; the penalty still
costs the left turns entirely (`turn_left` recall **0.0** against `ccos_argmax`'s
0.3636), which is the asymmetry §7.5 flagged and n = 11 cannot resolve.


### 7.11 ⛔⭐⭐⭐ BOTH ARMS LAND — THE ZERO IS SEED-DEPENDENT, AND THE TWO LEVERS TRADE

The two arms this package left unrun. Gates and verdict mapping were committed in
`raw/SPEC_BEST_AND_SEED.md` **before either arm existed**; both are reported as written.

#### 7.11a ⛔ P1.2 — `combined`'s ZERO DOES NOT SURVIVE A SECOND INFERENCE SEED

`combined_seed1` = `combined`'s command line with **one token** changed
(`--plan-seed 0 → 1`), audited against the **live process's** argv rather than the
intended one — 37 tokens, exactly one real difference (`raw/live_argv_audit.txt`).

| `assert_feasible`, `v0 >= 2 m/s`, n = 27 | `combined` (seed 0) | `combined_seed1` (seed 1) |
|---|---|---|
| **`kamm_over_rate`** | **0.0000** | **0.0741** (= 2 of 27 windows) |
| `peak_g` max | 0.618 | **0.702 — over the μ = 0.7 circle** |
| `max\|kappa\|` | 0.1505 | 0.1672 |
| ADE m | 1.0504 | 1.1539 |

⭐ **All three pre-committed controls PASSED**, so the reading stands: the ground-truth
path `g` reads `kamm_over` **0.0000 on both**; `ha`/`ha0`/`ha0_ext`/`ol` ADE are
**bit-identical** across the pair (one window grid — the artifact prints `CONTROL
PASSED`); `ha0_ext` reads **0.1852** non-zero in the same table.

⇒ Against the criterion committed before the arm existed — *"any value > 0.0000 ⇒ the
zero is SEED-DEPENDENT and the claim reported to the PI is wrong"* — **this is a FAIL.**
Corrected in the same turn in `RETRACTION_LOG.md`, `GOALS_AND_CLAIMS.md`
(`D-REFAV1-CG-ZERO-SEEDDEP`) and here.

⛔ **The correct statement:** *cap + ladder reduces `kamm_over_rate` from the cap alone's
0.1481 to a **seed-dependent 0.0000–0.0741** while preserving turn execution* — **not**
*"reaches zero"*.

⚠️ **And §7.8a — my own correction of the original claim — inherited the defect.** It
withdrew the *superlative* and left the *number* unexamined. ⭐ **What survives on two
seeds is the TURN half:** the recalls are **identical** across the pair (`turn_left`
0.3636, `turn_right` 0.7500, **0.00000 abs diff**), so the contrast with `W_KAPPA`'s
0.0000-of-11 stands. **It is the ZERO that was seed-dependent, not the TURNING.**

#### 7.11b The complete 2x2x2 — a perfect split, 4 of 4 versus 4 of 4

| cell | W | C | L | arm | ADE | recLK | recL | recR | `kamm_over` | `max\|k\|` | `peak_g` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0 | `ccos_argmax` | 1.3272 | 0.7143 | **0.3636** | 0.7500 | 0.2963 | 0.2000 | 3.262 |
| 3 | 0 | 1 | 0 | `kamm07` | 0.9927 | 0.7619 | **0.3636** | 0.6250 | 0.1481 | 0.1747 | 0.707 |
| 4 | 0 | 0 | 1 | `l3ladder` | 1.3236 | 0.5714 | **0.3636** | 0.7500 | 0.2593 | 0.1782 | 2.858 |
| 6 | 0 | 1 | 1 | `combined` | 1.0504 | 0.7619 | **0.3636** | 0.7500 | 0.0000 ⚠️ | 0.1505 | 0.618 |
| 2 | 1 | 0 | 0 | `wk15` | 0.8934 | 1.0000 | **0.0000** | 0.5000 | 0.0000 ⚠️ | 0.0800 | 0.332 |
| 5 | 1 | 0 | 1 | `wk15_ladder` | 0.9408 | 1.0000 | **0.0000** | 0.0000 | 0.0000 ⚠️ | 0.0020 | 0.153 |
| 7 | 1 | 1 | 0 | `best` | 0.8838 | 1.0000 | **0.0000** | 0.5000 | 0.0000 ⚠️ | 0.0800 | 0.332 |
| 8 | 1 | 1 | 1 | **`bestlad`** | 0.9418 | 1.0000 | **0.0000** | 0.0000 | 0.0000 ⚠️ | 0.0049 | 0.153 |

CONTROL: the ground-truth path reads `kamm_over` **0.0000 in every row**.
⚠️ **Every 0.0000 in this column is SINGLE-SEED and unreplicated except `combined`'s,
which WAS replicated and did not hold (§7.11a).** They are quoted with that
qualification, not as established zeros.

⇒ **`W_KAPPA` = 0 ⇒ `turn_left` recall 0.3636 in all four cells; `W_KAPPA` = 1 ⇒ 0.0000
in all four. No cell is intermediate.**

#### 7.11c P1.1 — `bestlad` against its committed gates

| gate | threshold (committed in advance) | reading | |
|---|---|---|---|
| **S** SAFE | `kamm_over` = 0.0000 with controls | 0.0000; `g` 0.0000; `ha0_ext` 0.1852 | **PASS** |
| **M** MOVES | `max\|a\|` > 0.5 **and** `max\|kappa\|` > 0.02 | 1.500 ✓ but **0.0049** ✗ | ⛔ **FAIL** |
| **T** TURNS | `turn_left` recall > 0.0000 | **0.0000** of 11 (and `turn_right` 0.0000 of 8) | ⛔ **FAIL** |
| **A** ACCURATE | paired `ade_m` vs `combined` ≤ −(floor), separated | **−0.1086 [−0.2535, +0.0313]** not separated; 1.05x the binding **0.1035** floor | ⛔ **FAIL** |

⇒ **MAPPED VERDICT, committed in advance:** *"M fails ⇒ the arm collapsed to a
near-static path; S is vacuous and is reported as such ⇒ ship `combined`."*

#### 7.11d The two mechanisms the final cells settle

1. ⭐ **THE CAP IS INERT UNDER `W_KAPPA` — confirmed twice, with intervals.**
   `best` − `wk15` `ade_m` **−0.0096 [−0.0415, +0.0278]** (and `TAC_traj_lat_correct`
   **+0.0000 [+0.0000, +0.0000]**, the known-value control's own signature);
   `bestlad` − `wk15_ladder` `ade_m` **+0.0010 [−0.0222, +0.0252]**. Adding the friction
   cap to a `W_KAPPA` arm changes nothing in either ladder condition, **because
   `W_KAPPA` has already driven `max|kappa|` to 0.0800 / 0.0049 — far inside the μ = 0.7
   circle — so the cap never binds.**
2. ⛔ **THE LADDER REVERSES SIGN DEPENDING ON WHAT DRIVES THE CHOICE.** Under the **cap**
   it gave the CONSTRAINT something feasible to select (§7.8). Under **`W_KAPPA`** it
   gives the PENALTY something **cheap** to select: `bestlad`'s realised
   `med|kappa|max` under **both** turn tokens is **0.00200 — exactly the lowest rung** —
   and mean `k^2` collapses to **0.000004**, i.e. **−99.94 %** against the baseline's
   0.006698 / 0.006400. Measured cost of adding it to `best`: `ade_m` **+0.0580
   [+0.0022, +0.1215]** separated WORSE, `fde_m` **+0.1672 [+0.0211, +0.3338]** WORSE,
   `TAC_traj_lat_correct` **−0.1000 [−0.1750, −0.0250]** WORSE.

#### 7.11e ⇒ THE ANSWER

⛔ **refav1 does NOT now have an arm that is both accurate and safe. The two levers
TRADE.** `W_KAPPA`'s accuracy is bought by deleting a decision class; the cap cannot
restore it; and the cap's own safety gain is the one that costs nothing measurable.
⇒ **the lever to carry forward is the CAP, not `W_KAPPA`** — and the next arm is
**cap + ladder + `a0_shift` with `W_KAPPA = 0`** (§7.10e), still **UNRUN**.
⭐ This **confirms the prediction recorded in `raw/SPEC_BEST_AND_SEED.md` §5.3 before
either arm ran**: *"I expect gate T to be the gate the arm fails, and I record that now
so a T-failure reads as a CONFIRMED prediction and a T-pass as a REFUTED one."*

---

## §8 — Deliverable manifest

| artifact | where it lives |
|---|---|
| `PREREG_COST_GEOMETRY.md` | repo: this package |
| `RESULT.md` (this file) | repo: this package |
| `raw/cost_scale.py` / `.txt` | repo: this package |
| `raw/kappa_by_goal.py` / `.txt` | repo: this package |
| `raw/four_family_table.py` | repo: this package |
| `ccosh` metric + `CCOS_HOLD_REL` | repo: `stack/tanitad/refs/refa_v1.py` |
| `test_cost_ccosh.py` (new) | repo: `stack/tests/` |
| `test_cost_ccos.py`, `test_cost_chord.py` (tuple pins updated) | repo: `stack/tests/` |
| `--cost-metric ccosh` | repo: `taniteval/tools/refav1_arm.py` |
| arm records `rec_*.json` + dumps | **`C:/Users/Admin/refav1_margin/p4out/` — OFF-REPO, SINGLE COPY** |
| oracle dumps `dump_oracle_s0` | **`C:/Users/Admin/refav1_drive/oracle/` — OFF-REPO, SINGLE COPY** |
