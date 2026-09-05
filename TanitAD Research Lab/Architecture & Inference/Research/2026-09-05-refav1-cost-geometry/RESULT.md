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

⇒ **The honest statement of the objective is now sharper than "refav1 does not beat
the floors":** as shipped it *is* a floor, bit-identically; and every configuration
in which it actually plans (`cos`/`ccos` with the zeroed triple) is **worse** than
the floors. The lever set in §3/§5/§6 exists precisely to make it act *and* be right.

*(filled in as they land; see `raw/` for the four-family tables and the records)*

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
