# The goal head's argmax is a HARD GATE on whether refav1 can turn at all

**Row:** `D-REFAV1-DRIVE-GATE` · **Class:** MEASURED (source read, every link quoted file:line)
**Date:** 2026-09-05 · **Agent:** Architecture & Inference FlyWheel
**Artifact:** this file; verification script `tools/assert_gate.py`, output `raw/gate_assert.json`

---

## The claim

> **When refav1's goal head decodes `LANE_KEEP`, every candidate in the entire iCEM
> population has curvature identically zero. The planner cannot turn — not "prefers not
> to", *cannot*.**
>
> ⇒ the goal head's turn recall (**20.5 %**) is a **HARD CEILING** on the planner's turn
> rate, and no cost re-weighting can lift it.

This is a **structural** statement about the search space, not a statement about a
learned preference. It is established by reading source, and it is asserted numerically
by `tools/assert_gate.py`.

---

## The chain — every link from source

| # | link | file:line | what it says |
|---|---|---|---|
| 1 | the goal is decided by **argmax** | `stack/tanitad/refs/refa_v1.py:1933` | `lat_i = self.lat_head(intent).argmax(-1).tolist()` |
| 2 | `LANE_KEEP` ⇒ **zero curvature** | `refa_v1.py:377-382` | `k` is written only when `lat` starts with `TURN_` / `LANE_CHANGE_` / `NUDGE_`. `LANE_KEEP` matches none, so `k` stays the `torch.zeros(op_steps)` of line 352. |
| 3 | that control profile **is the seed** | `refa_v1.py:2187-2188` | `seed = goal_action["controls"][:cfg.plan_steps][None]` |
| 4 | the seed pool is the **only** entry point | `refa_v1_plan.py:240-241` | `if it == 0 and seed_pool is not None: samples = torch.cat([samples, _clip(seed_pool…)], 0)` — and its mean seeds the CEM |
| 5 | **no baseline carries curvature** | `refa_v1_plan.py:178-186` | `cv` and `hold_v0` are `torch.zeros(H, 2)`; `decel_1.5` writes **column 0 only** (`dec[:, 0] = -min(1.5, a_max)`). Curvature column ≡ 0 in all three. |
| 6 | **noise cannot supply sustained curvature** | `refa_v1_plan.py:157` | `out = out - out.mean(dim=-1, keepdim=True)` — `colored_noise` is **zero-mean along time** by construction |

⇒ curvature enters the candidate population through **exactly one door** (link 4), that door
is opened by **exactly one decision** (link 1), and that decision is closed on
**79.5 % of ground-truth turn windows**.

⭐ **The file already knew.** `refa_v1.py:2176-2185` states it outright:

> *"`icem_plan`'s coloured noise is zero-mean over time … so a SUSTAINED curvature or
> acceleration is unreachable unless some candidate carries it — and no baseline carries
> curvature."*

and records the measurement that proved it: *"MEASURED 2026-09-02 on a random-init tiny
model: without this the planner returned hold_v0 on 24/24 windows against a TURN goal at
BOTH residual-init scales."* The gate was documented as a **fix** (the seed was added so a
TURN goal could be executed); what was not carried forward is that the same mechanism is a
**ceiling** when the goal is wrong.

---

## What this retro-explains (and it explains all of it)

| banked observation | source | explanation under the gate |
|---|---|---|
| `cv` wins the argmin on **50.7 %** of windows at **all 59** weight settings | `D-REFAV1-SURFACE-SCREEN` | re-weighting cannot select a behaviour that is **not in the candidate set**. Half the grid was unreachable because the answer was never proposed. |
| `W_JERK` inert on **24/26** candidates; nine settings over 1e13 identical | `D-REFAV1-SURFACE-PAIRED` | the jerk of a family of zero-curvature straight lines is degenerate; the term has nothing to separate. |
| world model **sound** (TURN_L +0.502, TURN_R −0.387, separated) | `D-REFAV1-CCOS-ARMS` | consistent: the WM can *evaluate* turns correctly, it is never *offered* one. |
| direction correct **77.8 %** when it does commit | `H-REFAV1-SURFACE-1` | consistent: a well-aimed head, gated shut most of the time. |

⚠️ **This reframes the cost-surface sweep.** The 59-setting weight sweep was not
under-powered and not wrongly weighted — it was **searching a space from which the answer
was structurally absent**. That is not a refutation of the sweep; it is the reason its
negative result was the correct answer to the wrong question.

---

## Why the fix is NOT "change the argmax"

The banked decision-rule sweep (`raw/sweep_stride40.json`, 29 settings) shows prior
correction / commit thresholds buy recall **by trading precision**:

| setting | SCORE turn recall | SCORE direction acc | SCORE false-turn-on-straight |
|---|---|---|---|
| `argmax` (current) | 0.194 | 0.750 | 0.050 |
| `prior:label tau=0.75` | 0.323 | 0.650 | 0.087 |
| `prior:self tau=1.0` | 0.290 | 0.556 | 0.125 |
| **CONTROL uniform-random** | **0.484** | **0.600** | **0.550** |

⛔ **The random control wins the naive objective.** `kap_correct_turn_rate` reads **0.280**
for uniform-random on ALL — *higher than every real setting* (best real: 0.300 FIT / 0.210
SCORE). A rule that turns constantly scores well on recall×direction. ⇒ **`correct_turn_rate`
is not admissible as the objective**; any recall gain must be read against the
false-turn-on-straight rate in the same breath. This is the CLAUDE.md probe rule (a control
that must read a known value) catching a metric, not a bug.

---

## The fix the architecture itself prescribes

`refa_v1.py:2176` names the doctrine: **"GPC: proposes, never disposes"** — the seed
*"join[s] the iteration-0 seed pool like a proposal mode and win[s] only on modelled cost."*

⇒ **Seed the pool with the top-k goal tokens, not only the argmax.**

* the argmax candidate **remains** in the pool ⇒ on modelled cost the outcome can only
  **tie or improve**; nothing is forced;
* turning becomes **reachable** on every window ⇒ the 20.5 % ceiling is removed;
* the **cost function decides**, so no false-positive turns are manufactured — which is
  exactly the failure mode the random control exposes above;
* it needs **no retraining**: the weights are untouched, only the candidate set widens.

That is the Step-1 intervention, and it is a change to the *proposal* mechanism rather than
to the *decision* rule — which is why it escapes the recall/precision trade the sweep found.

---

## Evidence class and scope

* **MEASURED (source read)** for the six chain links — each is a quoted `file:line` that can
  be re-read; `tools/assert_gate.py` asserts links 2, 5 and 6 numerically on the real
  functions rather than by inspection.
* **INHERITED** for the banked observations in the retro-explanation table (rows
  `D-REFAV1-SURFACE-*`, `D-REFAV1-CCOS-ARMS`), re-read from their artifacts, not re-derived.
* ⚠️ **Scope:** this is a statement about `refa_v1.plan` with `inject_baselines` on and no
  imitation `proposal` supplied — the configuration the refav1 arms actually run. A build
  that supplies a curvature-carrying `proposal` has a second door and this ceiling does not
  bind it.
