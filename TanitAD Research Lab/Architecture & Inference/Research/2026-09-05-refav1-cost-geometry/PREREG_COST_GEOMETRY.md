# PRE-REGISTRATION — D-REFAV1-COST-GEOMETRY: is the curvature penalty the binding lateral term?

**Registered 2026-09-05, BEFORE any arm in the ladder had produced a number.**
Package: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-cost-geometry/`
Agent: Architecture & Inference FlyWheel. Branch `agent/arch-inf-20260803`.

## 0. The finding that made this arm inevitable (MEASURED, zero GPU, banked before the ladder ran)

⛔ **EVERY REGULARISER IN THE BANKED p4 / oracle ARMS IS INERT — NOT ONLY `W_KAPPA`.**

`run_ab.sh` / `run_ab_targeted.sh` / `gm_run_p4*.sh` all pass
`--cost-weights 0.0,0.0,64.29715042415070`. Read against
`stack/tanitad/refs/refa_v1.py:2488-2493`:

```
c = c + w_jerk  * jerk.pow(2).mean(-1)            # w_jerk  = 0.0   -> DEAD
c = c + w_kappa * controls[..., 1].pow(2).mean(-1) # w_kappa = 0.0   -> DEAD
if target_speed is not None:                       # <- the gate
    c = c + w_vend * (v_end - target_speed).pow(2) # w_vend  = 64.297, NEVER EVALUATED
```

`target_speed` defaults to `None` in `plan()` (`refa_v1.py:2139`) and
**`taniteval/tools/refav1_arm.py` contains the string `target_speed` ZERO times**
(142,154 bytes; same-breath controls that MUST read non-zero: `plan_cfg` = **4**,
`cost_weights` = **12**). The tool's single `.plan(` call site (`:924-933`) passes
`cost_metric` / `cost_weights` / goal flags and no `target_speed`.

⇒ **In every banked refav1 T1 arm the planner's objective is the GOAL TERM AND
NOTHING ELSE.** There is no comfort term, no curvature term, and no speed term.
The saturating / over-sharp curvature is then the *expected* behaviour of an
unconstrained argmin, not an anomaly.

## 1. The scale, MEASURED from the banked `decisions` sidecars (zero GPU)

`plan_cost_cl` in those arms IS the pure goal term (§0). n = 40 windows / 8 episodes,
`dump_ccos_argmax`, `dump_cos_argmax` (artifact: `raw/cost_scale.txt`).

| metric | median winner cost | median `cv` basecost | **median decision gap `cv - winner`** |
|---|---|---|---|
| `cos` (SHIPPED)  | **-1.19209e-07** (negative: float32 cancellation) | 1.19209e-07 | **1.78814e-07** |
| `ccos` (centred) | 3.10838e-05 | 1.0 | **0.967197** |

⇒ the centred metric's decision is **5.4e+06x** larger than the shipped one's.
Under `cos` the goal term is *below the float32 quantum of its own expression*.

Realised curvature, same dumps: `ccos` median `max|kappa|` = **0.08** with
**5.0 %** at the `kappa_max = 0.2` cap; `cos` median **0.144437** with **12.5 %**
at the cap; median `mean(kappa^2)` = **0.0064** (`ccos`) / **0.0061074** (`cos`).
⚠️ **0.08 is exactly `GOAL_KAPPA_TURN`.** The planner is not saturating the clip —
it is *faithfully tracking the goal token's canonical curvature*, which is R = 12.5 m
against a corpus that curves at R 100-1000 m.
⚠️ This does **not** reproduce the "both saturating `kappa_max` = 0.2" reading in
`2026-09-05-refav1-make-it-drive/RESULT.md`; that reading is either a different panel
or a different statistic and is flagged for reconciliation, not overturned here.

## 2. The arms

All share the p4 turn-enriched panel (8 episodes, stride 16, **40 windows**,
`--no-navshuf --no-lead-block`, ckpt step 21,109, vocab **v7.0 / L1-0.08**,
`plan-seed 0`, argmax goal rule). Every arm's `cost` block is banked in its record.

| arm | metric | `(W_JERK, W_KAPPA, W_VEND)` | status |
|---|---|---|---|
| `cos_argmax` (A0-cos)   | `cos`  | (0, **0**, 64.297) | BANKED |
| `ccos_argmax` (A0-ccos) | `ccos` | (0, **0**, 64.297) | BANKED |
| `ccos_seed1`            | `ccos` | (0, **0**, 64.297) | BANKED — **inference-seed replicate** (plan-seed 1) |
| **`cos_shippedweights`** | `cos` | **(0.02, 0.05, 0.10)** = the module constants | THIS TURN |
| **`wk1p5`**  | `ccos` | (0, **1.511245**, 64.297)  | THIS TURN |
| **`wk15`**   | `ccos` | (0, **15.11245**, 64.297)  | THIS TURN |
| **`wk151`**  | `ccos` | (0, **151.1245**, 64.297)  | THIS TURN |

**The ladder's rungs are not tuned on the outcome.** They are fixed by the §1
measurement alone: `W_KAPPA = f * (median goal gap) / (median mean(kappa^2))
= f * 0.967197 / 0.0064 = f * 151.1245` for f = **1 %, 10 %, 100 %** — the
curvature charge as a stated fraction of the goal decision the planner is
actually buying. No ADE, no family metric and no floor comparison entered the
choice, and the ladder is reported in full whatever it says.

`cos_shippedweights` is the one-variable partner of `cos_argmax` and is also,
for the first time in this programme, **the configuration refav1 actually ships**
(default metric + module constants).

## 3. BOTH OUTCOMES, COMMITTED IN ADVANCE

**Ladder (ccos, W_KAPPA 0 -> 1.51 -> 15.1 -> 151.1):**
* **(a)** realised `mean(kappa^2)` falls monotonically with `W_KAPPA` **and** the
  `cl` arm's lateral family (curvature MAE, cross-track) improves **and** ADE moves
  toward / past `ha0_ext` ⇒ the curvature penalty **is** the binding lateral term;
  the zeroed triple is the defect and every arm banked through `run_ab*.sh` needs
  re-reading with its weights quoted.
* **(b)** `mean(kappa^2)` falls but the families do **not** improve ⇒ the penalty
  controls the *magnitude* and not the *direction*; the binding constraint is the
  goal's direction (L2: the `ccos` hold branch) or the seed pool (L3). A real
  outcome, not a failed experiment.
* **(c)** `mean(kappa^2)` does not fall ⇒ the weight never reaches the search
  (a plumbing defect); the runtime assertion at `refav1_arm.py:945-948`
  (`plan() used cost_weights=...`) must then have passed while the search ignored
  them, and that becomes the next work item.

**`cos_shippedweights`:** at the §1 scale the shipped `0.05 * kappa^2` charges
**2e-3 at the cap** against a goal decision of **1.79e-07** — a ratio of
**1.1e+04**. The prediction registered here is that the arm drives **exactly
straight** (`kappa == 0` on ~all windows) and that its ADE is therefore a
*straight-line* number, not a driving one. If it instead turns, the §1 reading of
the `cos` term is wrong and must be retracted.

## 4. What would make the result inadmissible

* Any arm quoted without its `(W_JERK, W_KAPPA, W_VEND)` triple and its metric.
* A cross-vocabulary comparison (all arms here are v7.0 / `GOAL_KAPPA_TURN = 0.08`).
* A `separated` paired-bootstrap CI read as a lever effect **without** the
  run-to-run floor: `ccos_seed1` is the inference-seed replicate of `ccos_argmax`
  and any ladder difference smaller than that pair's difference is noise
  (`CLAUDE.md`, THIRD VARIANCE / `H-ESTIM-SEED-1`).
* ADE reported alone. Four families or the eval is incomplete.

---

## 5. ADDENDUM (2026-09-05T17:35Z) — A SECOND PREDICTION, REGISTERED BEFORE ANY LADDER ARM LANDED

**MEASURED first, at zero GPU, from the banked `ccos_argmax` / `ccos_seed1` dumps
(n = 40 windows each):** the winning plan's curvature series is **EXACTLY
CONSTANT over the whole 2 s horizon on 31 of 40 windows (77.5 %)** — and of
those, **10 are exactly `0.000000` and 21 are exactly `0.080000`**. Both figures
reproduce bit-for-bit on the seed replicate. Under `cos` the same statistic is
**15.0 %** constant across **32** distinct `max|kappa|` values.

`0.000000` is `LANE_KEEP`'s canonical profile and `0.080000` is
`GOAL_KAPPA_TURN`. ⇒ **on 77.5 % of windows the "planned" trajectory IS the
decoded token's canonical control profile, verbatim.** This confirms
`D-REFAV1-DRIVE-GATE`'s mechanism on REAL windows rather than forced-token
probes: `colored_noise` is zero-mean along time, no injected baseline carries
curvature, so a SUSTAINED curvature can only enter through the seed pool, and
the seed pool offers `{0, ±0.08}`.

**THE PREDICTION.** A curvature penalty cannot synthesise a candidate the search
was never given. So, as `W_KAPPA` rises:

* **(P1)** the realised curvature must move **discretely from 0.08 to 0.0** —
  the `0.080000` bar drains into the `0.000000` bar — **with no new mass at
  intermediate magnitudes** (nothing near the corpus's R 100–1000 m, i.e.
  `|kappa| ~ 0.001–0.01`);
* **(P2)** therefore the best `W_KAPPA` can buy is *the better of two bad
  options per window*, and if the ladder's lateral family improves it improves
  by **suppressing wrong turns**, not by **executing right ones**;
* **(P3)** if instead intermediate magnitudes appear, the seed pool is richer
  than `D-REFAV1-DRIVE-GATE` says and that claim needs re-reading.

⇒ **Under (P1)+(P2), L1 is CAPPED BY L3 (the seed pool) and the escalation order
in the brief is confirmed by measurement rather than by assumption.** This is
registered here so that it cannot be claimed after the fact.

---

## 6. ARM-SET AMENDMENT (2026-09-05T17:40Z) — BEFORE ANY LADDER ARM LANDED

⛔ **Recorded here rather than applied silently.** At 17:40Z — with `cos_shippedweights`
at 5/8 episodes, `wk15` at 1/8 and **no ladder record written yet** — the **1 %
rung (`W_KAPPA = 1.511245`) was DROPPED** and the GPU slot reassigned to an **L3
arm**: `ccos` + (0, 0, 64.297) + `--seed-kappa-ladder 0.002,0.005,0.01,0.02,0.04`
(R = 500 / 200 / 100 / 50 / 25 m; both signs added by `plan()`, so +10
iteration-0 candidates).

**The reason is §5's measurement, banked before this decision:** the winning plan
is the decoded token's canonical profile — an EXACTLY constant curvature series —
on **31 of 40** windows, at exactly `0.000000` or `0.080000`. Against a candidate
set that is effectively `{0, ±0.08}`, a curvature charge worth **1 %** of the goal
decision cannot change a discrete choice, so that rung buys ~nothing while the
seed-pool arm tests (P3) directly. The **10 % and 100 % rungs are unchanged**, so
the dose-response the ladder was registered to measure is intact
(`W_KAPPA` ∈ {0 (banked), 15.11245, 151.1245}).

⚠️ **This is an allocation change, not a criterion change.** No outcome
definition in §3, and no prediction in §5, is altered; nothing had been observed
about any ladder arm when it was made. `wk1p5` remains a legitimate arm and can
be run later; its absence is recorded, not hidden.

**The L3 arm is one variable against the banked `ccos_argmax`** — same metric,
same weights, same seed, same panel — because `canonical_controls` and the goal
field are untouched. It is therefore NOT a vocabulary change and is comparable
window-for-window; a `--goal-kappa-levels` arm would not be.
