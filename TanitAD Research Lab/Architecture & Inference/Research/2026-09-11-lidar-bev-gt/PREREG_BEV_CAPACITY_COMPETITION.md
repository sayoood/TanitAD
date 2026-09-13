# PRE-REGISTRATION — `E-BEVLIDAR-CAP-1`: does a LiDAR-supervised BEV head compete for trunk capacity, and can we see it BEFORE the GPU-days?

**Date** 2026-09-11 (Europe/Berlin) · **Author** Architecture & Inference FlyWheel ·
**Branch** `agent/arch-inf-20260803`
**Status** design only — **no arm launched, 0 GPU spent.** Written BEFORE any outcome number exists.
**Hypothesis id** `E-BEVLIDAR-CAP-1`.

---

## 0. Why this exists, in one paragraph

The programme has measured auxiliary-task capacity competition **twice**, on two different rigs:

| # | fact | class · source |
|---|---|---|
| 1 | `--agents head` (DiffusionDrive's detector) **FAILED its bar on two seeds** at a **17 M** trunk / 500 steps. The **deranged-join** arm reproduced the whole degradation (`shuf − off` headway **−0.4576**, TTC **−1.9334**, separated worse) while **`head − shuf` was NOT separated on any distance-keeping metric** ⇒ the cost is the **AUXILIARY TASK**, not the agent information. | **MEASURED** — `D-P1-AGENTCOND-1`, `D-P1-AGENTCOND-1-SEED1` |
| 2 | WP-D's **182,616-param** polar aux head, on the **108 M** REF-C trunk at 4,000 steps, left a surviving **LATERAL** regression on 3 of 9 metrics (heading 3.3×, curvature 6.0×, yaw-rate 3.5× the replicate floor) after the replicate correction voided the ADE and longitudinal cells. | **MEASURED** — `E-BEV-AUX-1` (`F2`, as corrected 2026-09-10) |
| 3 | On the REF-C planner rig a **one-seed** pair reads `separated` on **5 of 9** metrics with **zero levers moved** — a **55.6 %** false-positive rate, against the 14.3 % `CLAUDE.md` records for the v7-tiny rig. | **MEASURED** — `E-BEV-AUX-1` / `H-ESTIM-SEED-1` |

⇒ **Fact 3 is why this pre-registration funds replicate arms before it funds anything else.** A
separated interval on this rig is not evidence of a lever; it is the rig's baseline behaviour.

⭐ **And fact 1's discriminating control is the design of this experiment.** The thing that
relocated the cost from "the information" to "the task" was a **zero-information arm with the same
parameter count and the same gradient path**. That arm is the primary instrument here, not an
afterthought.

---

## 1. The claim under test

> **`E-BEVLIDAR-CAP-1`** — A **LiDAR-supervised BEV occupancy head** attached to REF-C's trunk
> degrades the planner's four metric families through **auxiliary-task capacity competition**, and
> that competition is **detectable within the first 500 optimiser steps** from the
> **cosine between the trajectory gradient and the auxiliary gradient on the SHARED trunk
> parameters** — before any planner metric separates.

⭐ Two halves, and they fail independently. The claim is **not** "the BEV head is bad". It is
"**competition, if it happens, is cheap to see early**" — which is what converts a 4.7 h-per-arm
gate into a 20-minute one.

---

## 2. ⛔ The premise the brief handed me is HALF TRUE, and the measurement says which half

The brief states: *"a BEV encoder is a **larger** auxiliary task, so that risk rises."*

**MEASURED** (`code/p4_head_sizing.py`, `raw/head_sizing.json`; modules built and counted with
`sum(p.numel() …)`, not estimated):

| candidate | params | % of the 108,257,502 trunk | × the WP-D aux head (182,616) |
|---|---:|---:|---:|
| ⭐ **A — lift-splat, polar 24×20** | **190,258** | **0.18 %** | **1.0×** |
| A — lift-splat, Cartesian 120×64 wide | 726,978 | 0.67 % | 4.0× |
| B — cross-attn, polar 48×40, d192, L3 | 2,284,802 | 2.11 % | 12.5× |
| B — cross-attn, polar 24×20, d256, L4 | 4,517,634 | 4.17 % | 24.7× |
| B — cross-attn, Cartesian 120×64, d256, L4 | 6,360,834 | 5.88 % | 34.8× |
| B — cross-attn, Cartesian 128×128, d256, L6 | 10,695,938 | 9.88 % | 58.6× |

⇒ **The premise holds for the CROSS-ATTENTION route (12.5–58.6×) and FAILS for LIFT-SPLAT at the
polar geometry, which is 1.0× — 190,258 against 182,616, a 4.2 % difference.** The lift is one
1×1 convolution; all the geometry is a **precomputed index with zero learned parameters**, built
from the cylindrical ray model and the rig extrinsics.

⇒ **Arm A is therefore not a bigger bet than the one the programme has already placed and paid
for**, and the ladder starts there. ⛔ Nothing about the parameter ceiling is binding at any rung:
the largest candidate totals **118,953,440**, leaving **181,046,560** of headroom under 300 M. The
binding constraint is competition, not the ceiling, and this document is about the former.

---

## 3. The arms — 5, and two of them exist only to measure noise

All arms share one launch prefix; the only tokens that differ are named here and **audited
mechanically** (the 60-token argv diff `D-P1-AGENTCOND-1` used).

| arm | lever | what it is for |
|---|---|---|
| `C0` | aux OFF | the control |
| `C0b` | aux OFF, **same seed**, `--out` only | ⛔ **the replicate floor.** Without it, a 55.6 % false-positive rate makes every other row unreadable |
| `C0c` | aux OFF, **seed 1** | the seed component of the floor |
| `C1` | aux ON, LiDAR BEV target | the lever |
| `C2` | aux ON, **target shuffled across the batch** | ⭐ **the discriminating control.** Same parameters, same gradient path, **zero information** |

⚠️ `C2` is shuffled across the batch and **never** within a frame: shuffling within a frame changes
the target's marginal statistics as well as its information, and then the arm is not a control.

⛔ A **detached** arm (`C3`) is explicitly **NOT** in this ladder. WP-D's `F3` twin was declared
void, and a detached head answers *"does the optimizer state alone cost anything"*, which is not
the question. If `C2 − C0` separates and `C1 − C2` does not, `C3` becomes the next lever and gets
its own pre-registration.

---

## 4. ⭐ The early detector, and the control that must read a known value

At every step, for the shared trunk parameters θ_trunk:

    g_traj = ∂L_traj/∂θ_trunk        g_aux = ∂L_bev/∂θ_trunk
    conflict(t) = cos(g_traj, g_aux)

Logged to `metrics.jsonl` every step. Cost: one extra backward over the trunk — **no extra arm, no
extra GPU-day.**

⛔ **THREE CONTROLS, EACH WITH AN ANALYTIC TARGET, AND THE PROBE REFUSES TO RUN IF ANY MISSES.**
A cross-check re-derived from the thing it checks measures determinism, not correctness; these are
derived independently of the probe:

| control | must read | why it is not a tautology |
|---|---|---|
| `cos(g_traj, g_traj)` | **exactly +1.0** | an identity, independent of any training |
| `cos(g_traj, g_aux)` with the head **detached** from the trunk | **exactly 0.0** (g_aux is the zero vector on θ_trunk; report `NaN` → refuse, never 0 by accident) | isolates "shares gradient" from "is present" |
| `cos(g_traj, −g_traj)` | **exactly −1.0** | fixes the sign convention, which is the `R-2026-09-08-wpa-mirror` family: a sign nobody asserts |

⭐ **And a MUTATION arm that must go RED:** re-introduce the real historical defect — scale the aux
loss by 30× so it dominates (the MEASURED `E-DEC-18` mechanism, aux ~0.3–0.9 against an objective
~0.03) — and the conflict detector must fire. A detector that cannot be made to fire has not been
shown to work.

---

## 5. Committed criteria — BOTH outcomes, fixed here, in advance

Estimator for every interval: **paired episode-cluster bootstrap** over the val episodes
(`taniteval/taniteval/ci.py`), 10,000 resamples. ⛔ `overlapping_holdout_se` is never called.
Planner bars are **T1** (action-closed loop) and carry **all four metric families**; the
representation bar is a frozen-feature probe and carries **no tier**.

Let `F_m = max(|C0b − C0|_m, |C0c − C0|_m)` be the per-metric replicate floor, measured **in the
same panel**.

**`E-BEVLIDAR-CAP-1` is SUPPORTED iff BOTH:**

* **B1 — competition is real.** `C2 − C0` is separated in the unfavourable direction on
  ≥ 1 of the four families by **≥ 3 × F_m**, on **two seeds**.
* **B2 — the detector saw it first.** The median `conflict(t)` over steps 1–500 of `C2` is
  **< −0.05**, and its sign is stable (≥ 80 % of the first 500 steps negative), while `C0`'s
  self-cosine control reads +1.0 to within 1e-6.

**REFUTED iff** B1 holds and B2 does not (competition is real but invisible early — the detector is
worthless and the expensive gate stays), **or** B1 fails on both seeds (no competition at this head
size — and then the ladder proceeds to the cross-attention rungs, where §2 says the risk is 12–59×).

⛔ **A third, separate verdict that must be reported whatever B1/B2 do:**
`C1 − C2` on the four families and on the frozen-feature BEV AP. That is *"does the LiDAR
information pay for itself"*, and it is the question the PI's directive is actually about.
Reporting B1/B2 without it would answer a methods question and call it a result.

### Failure twins, each naming its next lever

| twin | fires when | next lever |
|---|---|---|
| `G1` | `C2 − C0` separated worse, `C1 − C2` not separated | capacity, not content ⇒ **shrink to arm A (190,258 params) or stage the head after the planner converges** |
| `G2` | `C1 − C2` separated BETTER but `C1 − C0` still worse | the information pays, the task costs more ⇒ **freeze-trunk / adapter route** |
| `G3` | nothing separates at 3 × F_m | **UNDERPOWERED, not negative** — report the floor and the achieved `n`, and state what n would be needed |
| `G4` | B2's controls miss their analytic targets | the detector is broken; **fix the instrument before reading any arm** |

---

## 6. ⛔ Scope, stated so nobody quotes it wider than it is

* This is a **REF-C / refcv5** claim. The 17 M v7-tiny result and this one are **different rigs**
  and must never be quoted interchangeably.
* The BEV target is **LiDAR-built**, which makes it a **different object** from WP-D's
  `obstacle.offline`-built target: it carries static structure (walls, kerbs, vegetation) that
  `obstacle.offline` does not contain at all, and a **measured** visibility state rather than a
  derived lower bound.
* ⛔ **LiDAR is a LABEL. It never reaches an inference path.** The removability proof WP-D
  established — shared parameters bit-identical, planner output dict bit-identical with the aux
  logits the only extra key, head constructed LAST and called LAST — is a **precondition** here,
  not a result, and is re-proven per arm.
