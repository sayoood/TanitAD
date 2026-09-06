# PREREG — D-NAVROUTE-1: is the planner following the COMMANDED route, its OWN predicted route, or neither?

**Written** 2026-09-06, Arch+Inference. **Arm** refcv4b `ckpt_40284_FINAL.pt` (step 40,284).
**Tier** T1 (self-action OPEN loop, per the 2026-09-02 ruling). **Estimator** paired
episode-cluster bootstrap over the 141 eval episodes (`taniteval/ci.py`), 2,000 resamples,
seed 0. ⛔ Never `overlapping_holdout_se`.

⛔ **WRITTEN BEFORE ANY ARM WAS SCORED.** The banked `nav_compliance` verdicts were read
first (they motivate the design and are quoted in §1), but no number below has been computed.

---

## 0. The PI's hypothesis, and why our own conclusion does not settle it

> *"I have the feeling it is following the nav route head [rather than the command]."*

Our banked conclusion was that the nav edge is **presence-gated** (97.7 % presence /
2.3 % content). ⛔ That does **not** imply the PI's claim. His claim is that the planner
**is** steered — by the route head's vision-derived signal — and that the strategic layer
therefore **injects its own error and can override a correct command**.

Half the experiment is banked: `os_navflip` (nav inverted on all commanded windows) moved
the planner **+0.0022 m [-0.0006, +0.0052], NOT separated**. This registration specifies
the **complement**: intervene on the strategic goal path and measure whether the plan moves.

## 1. ⛔ THE TWO CANDIDATE "ROUTE HEADS" ARE DIFFERENT OBJECTS — the design turns on this

MEASURED from source (2026-09-06):

| # | object | shape / type | site | reaches the plan? |
|---|---|---|---|---|
| **R1** | core `route_head` -> `route_logits` | `nn.Linear(feat, N_ROUTE=3)`, a **3-way categorical** (left/straight/right) | `refc.py` `route_head`; `N_ROUTE = 3  # route-heading aux` | ⛔ **NO** — `route_prior = log_softmax(route_logits) if cfg.graft_route else None`, and the arm's own banked config records **`"graft_route": false`**. `route_to_anchor` is only constructed `if self.sel.graft_route`. |
| **R2** | cascade `str_goal_head` -> `g_str` | `nn.Linear(d_ctx, 3)` -> `cat([unit_bearing(2), tanh(dist_pref)(1)])`, a **GEOMETRIC** goal | `refc_v3.py` `str_goal_head`, `g_str` | ⚠️ yes, but only through a **zero-init FiLM** (`gstr_film`) into `z_tac`, and via `target_latent` into the decoder. |

⭐ **A probability like "RIGHT 0.68 -> 0.73" can only come from R1** — R2 is a unit bearing plus a
tanh scalar and has no softmax. So the PI's video evidence points at **R1, the head the arm's own
config disconnects.** That must be tested, not assumed, because it is the crux of his hypothesis.

⚠️ **The banked `nav_compliance` panel already reports** (`refcv4b_navflip.json`,
`/refcv3/strategic/nav_compliance/verdicts`, n=4,823 windows / 141 episodes):
`plan` **FOLLOWS_NAV** (rate 0.6165; drop **-0.1023 [0.0641, 0.1437]** under shuffle,
**-0.1307 [0.0812, 0.1801]** under zero), `anchor` **FOLLOWS_NAV**, and `gstr` **NAV_BLIND**
(delta exactly **0.0, CI [0,0]** under both interventions). This registration must therefore
also be able to conclude **the plan follows the COMMAND**, which the brief's three outcomes
omit. It is registered as outcome (d).

## 2. Readouts (all pre-specified; no post-hoc thresholds)

Per window, from the banked `navflip_dump/decisions/*.npz` (141 episodes):

* **PLAN direction** `dir(plan)` — from `plan_full_nav_true [S=8, 2]`, the terminal lateral
  offset `y_T` in the ego frame at the 6 s horizon. `LEFT` if `y_T > +tau_y`, `RIGHT` if
  `y_T < -tau_y`, else `STRAIGHT`.
* **R1 direction** `dir(route)` — `route_pred_nav_true` in {0,1,2} = (left, straight, right),
  the mapping pinned by `refb_labels._ROUTE_TO_NAV` = {0:1, 1:0, 2:2} against
  `NAV_COMMANDS = ("follow","left","right","straight")`.
* **R2 direction** `dir(gstr)` — from `gstr_nav_true [3] = (cos, sin, dist_pref)`; `LEFT` if
  `sin > +tau_g`, `RIGHT` if `sin < -tau_g`, else `STRAIGHT`.
* **COMMAND direction** `dir(nav)` — `nav_cmd` in {0 follow, 1 left, 2 right}, restricted to
  `nav_valid`.

**Deadbands, fixed in advance and reported with the result:** `tau_y = 1.0 m`
(a lateral metre at 6 s is the smallest offset that is not plan noise) and `tau_g = 0.10`
(≈ 5.7 deg of bearing). ⛔ Neither is tuned on the outcome. A **sensitivity row** at
`tau_y in {0.5, 1.0, 2.0}` and `tau_g in {0.05, 0.10, 0.20}` ships with the result so a
threshold-driven conclusion is visible.

## 3. The contingency table (the brief's explicit request)

Restricted to **DISAGREEMENT windows** — those where the candidate head and the command name
different directions and at least one is non-STRAIGHT. Two tables, one per candidate head:

|  | plan == head | plan == command | plan == neither |
|---|---|---|---|
| **R1 (core route head) vs command** | n | n | n |
| **R2 (`g_str`) vs command** | n | n | n |

⛔ Reported as **counts with `n`**, never as a bare scalar. Rows where head and command agree
are counted separately and are **not** informative for this question.

**Primary statistic:** `follow_gap = P(plan == head) - P(plan == command)` on disagreement
windows, with a **paired episode-cluster bootstrap** CI over the 141 episodes.

## 4. ⭐ The INTERVENTION (the complement `os_navflip` never ran)

⛔ Observational agreement is not causation — a head and the plan can agree because both read
the same vision. The causal half perturbs **R2 only**, holding nav fixed, in **ONE process on
ONE surface** (Thor, `tanitad-train` venv), via a **forward hook** on the `g_str` consumer.
No model file is edited.

| arm | intervention | what it asks |
|---|---|---|
| `os` | none (reference) | -- |
| `os_gstrzero` | `g_str := 0` before `gstr_embed` | does removing the strategic goal move the plan? |
| `os_gstrflip` | negate the lateral bearing component (`sin := -sin`) | does *reversing* it move the plan? |
| `os_gstrshuf` | permute `g_str` across windows within the episode | the anti-echo twin of `os_navshuf` |
| **`os_replicate`** | ⛔ **same flags, same seed, re-run** | the rig's own **run-to-run noise floor** |

⛔ **`os_replicate` is mandatory**, not optional. A separated CI from a one-seed arm is
necessary and not sufficient — the measured replicate false-positive rate is **6/42 = 14.3 %**.
Every separated claim below is read against this arm, and the report names **which variance**
it answers (episode draw / training run / inference run).
⚠️ refcv4b's selection is an **argmax over a fixed anchor bank**, not a sampling planner, so the
inference-seed variance is expected to be structurally zero here; `os_replicate` measures that
rather than assuming it.

**Primary metric:** the **plan displacement** `||plan_intervened - plan_os||` at the 6 s
terminal point (metres), per window, paired bootstrap. Plus the four metric families
(§6), never ADE alone.

## 5. ⛔ THE OUTCOMES, COMMITTED IN ADVANCE

* **(a) FOLLOWS THE ROUTE HEAD.** `follow_gap > 0` with a **separated** CI on disagreement
  windows, **AND** a `g_str` intervention that moves the plan by more than the `os_replicate`
  floor. ⇒ the strategic layer steers, and can override a correct command. Quantify by how much.
* **(b) FOLLOWS NEITHER (both inert).** No separated movement under **either** the banked nav
  interventions **or** the `g_str` interventions. ⇒ the plan is effectively **open-loop with
  respect to route**, and we say so plainly.
* **(c) INCONCLUSIVE.** Disagreement `n` too small to separate, or the intervention effect is
  inside the `os_replicate` floor.
* **(d) FOLLOWS THE COMMAND** *(the complement the brief's framing omits; registered because
  the banked `nav_compliance` verdicts point here)*. `follow_gap < 0` separated, and/or the nav
  interventions move the plan while the `g_str` interventions do not.

⛔ Whichever obtains is reported **as written**. A refutation of the PI's hypothesis is a
waypoint, not the deliverable: the turn must also name the next lever.

## 6. Reporting rules that bind this result

1. **Four metric families, never pooled** — longitudinal (speed / headway), lateral (heading,
   curvature, yaw-rate, cross-track), tactical (`lat_pred_*` / `lon_pred_*` vs label, anchor
   selection), strategic (this panel). A family that cannot be computed is reported **per
   family with its reason and `n`**, never silently dropped.
2. Every number carries **T-tier** (T1) and its **estimator**.
3. ⛔ `oracle_sel`, `anchor_acc`, `sel_agrees_oracle` are **INVALID on refcv4b** and are not
   quoted, even though the dump carries them.
4. ⛔ The `|dyaw| > 0.15` turn gate is never used (the human fails it 3/9).
5. A separated CI names **which variance** it answers.
