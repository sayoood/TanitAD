# PRE-REGISTRATION — the pseudo-simulation ARM PANEL

**Written 2026-07-27, BEFORE any arm beyond the three already-published ones was scored.**
Frozen at the moment of writing; the panel report (`PSEUDOSIM_ARM_PANEL.md`) quotes this section
verbatim and does not edit it. Host: **pod2** (A40).

---

## 1. What is being run

Every arm this program can reach, scored on `taniteval.pseudosim` — the bounded pre-generated
perturbation grid (7 heading × 3 longitudinal = 21 points, lateral **refused in code**), on the
canonical 40 val episodes at stride 8, `traffic_mode = log_replay_nonreactive`.

Ranking quantity: the composite **`PSS_recovery_progress`**. ⚠️ **NOT `ade_0_2s`** — MEASURED twice
on 2026-07-26/27 that the ADE-optimal pick collides 4.7× more often than the rule-optimal pick
(3.36 % vs 0.71 %, separated), and PUBLISHED that L2/ADE vs closed-loop Driving Score is
ρ = −0.36, p = 0.43 while Ego Progress is ρ = 0.83.

Estimator: `taniteval.ci.episode_cluster_bootstrap` / `paired_episode_cluster_bootstrap`, **B = 2000**,
unit = **val episode**, paired arms on **identical (episode, anchor, grid-point) rows**.
`overlapping_holdout_se` is refused and is named in `_refused_estimator` on every node.

---

## 2. The fidelity gates that must fire BEFORE any new arm is quoted

Both are reproductions, not new claims. If either fails, the panel reports the failure instead of a
ranking.

| # | gate | pass condition | what I write if it fails |
|:--:|---|---|---|
| **G1** | **Instrument sensitivity.** `v4_oracle − v4_blind` PSS, paired. | **SEPARATED** and positive. Published reference: **+0.1882 [+0.1240, +0.2557]**. | ⛔ The instrument cannot tell seeing from not-seeing on this host. **No arm score is admissible.** The panel is abandoned and the run goes back on the bench. |
| **G2** | **Fidelity of the port to pod2.** `v4_oracle − cv_holdv0` PSS and recovery. | PSS **n.s.** (published −0.0034 [−0.0138, +0.0078]) **and** recovery **separated negative** (published −0.0168 [−0.0332, −0.0008]). | ⚠️ pod2 does not reproduce the eval-pod run. Every new number is quarantined as UNVERIFIED until the discrepancy is explained; the panel reports the discrepancy as its headline. |
| **G3** | **The envelope assertion.** `assert_grid_in_envelope` on the shipped grid. | `frac_steps_any = 0.0`, `frac_windows_any_step_out_of_envelope = 0.0`, verdict `MEASUREMENT`. | The run does not start (it is a hard `EnvelopeViolation`). |
| **G4** | **Validation in the FAILING direction.** A deliberately out-of-envelope grid. | `GridSpec(dyaw_deg=(12.0012,))` **raises** `EnvelopeViolation` **with zero planner calls**, while `12.0` is accepted. | The assertion is decoration and nothing on this surface is a measurement. |
| **G5** | **The standing-still adversary.** A planner that does not move. | `recovery` is **NaN (excluded)**, never 1.0, and the composite does not rank it above a moving arm. | The §5.2 fix has regressed; no `recovery` number is quotable. |

---

## 3. ⭐ The pre-registered discriminative-power criterion — stated before the panel exists

**The brief requires me to say, in advance, what result would make me conclude that the instrument
is NOT discriminative enough to rank arms. Here it is, and it is a real possibility, not a formality.**

> **D-NULL fires iff, after G1 passes, EVERY pairwise paired PSS contrast among the LEARNED arms
> and CV straddles zero** — i.e. no arm is separated from any other arm.
>
> If **D-NULL** fires I write: **"this instrument separates sighted from blind, but it does NOT
> resolve our arms from each other or from constant velocity. The panel produces NO ranking."**
> I will report the point estimates and their intervals, state the ordering is unsupported, and
> refuse to write a leaderboard. A panel where every arm lands inside every other arm's interval is
> a real and reportable outcome and it will be reported at full prominence, in the headline.
>
> **D-PARTIAL**: some contrasts separate, others do not. Then I publish the **separated** contrasts
> only, and state explicitly which orderings are unsupported. No transitive chaining of n.s.
> contrasts into a ranking.
>
> **D-RANK**: a total order in which each consecutive pair is separated. Only then is the word
> "ranking" used without qualification.

⚠️ **A secondary discriminative failure mode, also pre-registered:** if `recovery` fails its own
between-arm-spread gate (`RANGE_MIN = 0.05`) once more arms are in the pool, the composite collapses
onto `ego_progress` alone. If that happens I say so and I do **not** silently keep calling the
result a "recovery" composite. Note the published spread cleared by **6 %** on three arms
(0.0530 vs 0.05) — this is a live risk, not a hypothetical.

---

## 4. ⭐ The claim I am going in to falsify — v1's admissibility

The 2026-07-27 pseudo-simulation report, §9 item 8, states:

> *"The v1 (`flagship4b-speedjerk-30k`) arm was not scored — its plan step decodes conditioned on the
> future action sequence (`rollout_decode(…, fa, …)`), which does not exist for a perturbed state, so
> it is not a planner in the pseudo-simulation sense."*

**HYPOTHESIS (mine, pre-committed): that claim is true of ONE surface and false of the model.**
`taniteval/closedloop.py:317-318` and `:245-246` run a **state-only** plan step on the same
checkpoint —
`strategic_policy(states, nav) → ctx`, `tactical_policy(states, ctx) → waypoints` — with **no future
actions anywhere**. If that path loads and runs on the v1 checkpoint, **v1 IS scoreable** and the
program's most-compared arm enters the panel.

Both outcomes committed now:

| outcome | what I write |
|---|---|
| **V1-IN** | the state-only tactical-policy surface loads and plans ⇒ v1 is scored, and the prior report's §9.8 is **narrowed to the `rollout_decode` surface** rather than left standing as a property of the model. |
| **V1-OUT** | the surface does not exist / does not load on this checkpoint ⇒ §9.8 is **confirmed**, v1 stays out, and I say which of the two mechanisms blocked it. |

⚠️ **And the consequence I am committing to in advance, because it will be uncomfortable if V1-IN:**
v1's headline **`ade_0_2s` 0.4271** is produced by `taniteval/rollout.py`, which feeds the
**expert's true future actions** (`actions_source="expert_future"`, the module's own
`honest_metric_name` is **`wm_fidelity_ade_2s`**). Its **planner** surface is a different, weaker
object — the same file records the tactical waypoint heads at **3.38 m** against the operative's
0.628. **So if v1 scores badly here that is NOT a contradiction of 0.4271 — it is the two surfaces
being different objects, and I will say so rather than reporting a "regression".**

---

## 5. Goal provenance — matched deliberately, and the confound stated

`v4_oracle` uses an **ORACLE** goal (route / route_graded / vt_band minted from the ego's own future).
For the panel to compare like with like, every learned arm is offered the **same class** of goal:

| arm family | oracle goal | deployable control also run |
|---|---|---|
| flagship v4 | `goal_modes` oracle (route, route_graded, vt_band) | — (not run, time) |
| flagship v1 / no-speed | `refb_labels.nav_command` — GT nav from future heading | `nav = follow` (constant), the historical deploy path |
| REF-C (small/base/XL) | `refc_eval.resolve_nav(nav_mode="oracle")` | `nav_mode="produced"` — the model's own route head |

⚠️ **Every oracle-goal number is an UPPER BOUND and is not deployable.** They are marked as such.
⚠️ **The goal objects are NOT identical across families** (v4 gets a 3-field goal incl. a speed band;
v1 and REF-C get a 4-way nav command). This is a **stated limitation of the panel**, not something to
be papered over: the v4-vs-v1 contrast conflates the planner head with the goal interface.

---

## 6. Tier, stated in advance

The whole panel is **tier 2**: oracle-goal (upper bound), **non-reactive log replay**, no collision
gate (no cuboids in the val cache ⇒ `no_collision` / `ttc` emitted as `None` with a reason,
**never a constant**), and `comfort` **dropped by the gate, not retuned**
(at `dt = 0.1` an 8 mm third-difference trips the jerk bound; retuning after seeing who fails is
metric-shopping). The composite is `PSS_recovery_progress` and **is not a Driving Score**.

The **envelope proof (G3/G4) is tier 1** — deterministic, model-free, CPU, seconds.

---

## 7. What is out of scope, declared now

* **flagship-v2corpus** — training on **pod1**, which this brief forbids touching. If no checkpoint is
  reachable without touching pod1, it is reported as **NOT RUN**, with the reason.
* **REF-A / REF-B** — not requested; REF-A additionally consumes frozen features, not raw frames, so
  the warp cannot be applied to its input in this harness. Stated, not attempted.
* **A doubly-blind control** (blind to image *and* goal). The published blind control keeps the oracle
  goal and is confounded for `recovery`; that confound is inherited, disclosed, and not fixed here.
