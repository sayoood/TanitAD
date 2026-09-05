# SPEC — `H-PROJ-CONTACT-1`: can a COLLIDING trajectory be made UNREPRESENTABLE?

**Arch+Inference FlyWheel, 2026-09-05.** ⛔ **PRE-REGISTERED. Written and staged BEFORE
the projection existed and BEFORE any number it produces was computed.** Both outcomes
below are committed here; whichever the instrument returns is the result.

---

## 1. The claim under test

`stack/tanitad/refs/feasible_decode.py` makes an **over-friction** path unrepresentable by
projecting in CONTROL space onto the box + Kamm disc and re-integrating with the scorer's
exact forward map — **96.87 %** of the fan-to-vocabulary friction gap at a T1 cost of
**+0.0012 m** (`…/2026-09-05-feasible-decode/`). The penalty route bought **~2.7 %** of the
same gap at **ADE +0.0362 m separated worse** (`…/2026-09-05-veto-only-fan-safety/RESULT.md`).

⇒ **`H-PROJ-CONTACT-1`: the same construction applied to CONTACT drives the emitted fan's
collision rate to a STRUCTURAL zero, at a displacement cost below the replicate floor.**

## 2. The object, named before it is read

| | |
|---|---|
| fan bank | `…/2026-09-05-veto-only-fan-safety/raw/fan_bank_base_240w.npz`, md5 `ccecf0ade75631e315e6b9ae78d83478` |
| tensor | `fan2 [240, 128, 5, 2]` — the 2 s prefix on the 0.5 s grid, origin prepended |
| agents | `lead5 [240, 5, 2]`, `has_lead [240]` (**65** lead windows) — the val40 B1 agent join, banked with the fan |
| predicate | `tanitad.rl.rewards._collision` MOVING-LEAD branch, md5 `bab0469bbd013b71270caef779d23772` — **swept in the RELATIVE frame**, `r = ego 1.0 + obs 1.0 = 2.0 m`, steps `s = 1..4` |
| defect | `f_contact.sum() = 764 / 30,720 = 2.4870 %`, **19 / 240** windows, worst window **104 / 128** — REPRODUCED from the banked tensor before the SPEC was written |
| GT | `out/gt_240w.npz`, extracted at 0 GPU from the same corpus + the same `wi`; join asserted against the bank (`max|v0 diff|`, `max|lead5 diff|`, `eid`, `has_lead`) |

⛔ **Every rate below is reported in BOTH forms — `fan_*` (the generator) and `sel_*` (the
selector's pick).** A `sel_*` number may never stand alone as evidence of model quality
(RETRACTION #31: `sel_contact` read 0.0000 while the generator emitted 764 colliders).

## 3. The instrument

`stack/tanitad/refs/contact_projection.py` — **a control-space retraction**, the same
family as `feasible_decode`, never a position-space edit (`P2-C4`'s `off_reach` +0.2311 is
the warning). It recovers `(s0, h0, a[k], lat[k])` with `feasible_decode.recover_controls`,
scales the speed profile by `sigma` while holding curvature `kappa = lat / v_mid^2` fixed
(so `lat -> sigma^2 * lat`), re-integrates with the exact forward map, and returns the
**largest** `sigma` whose re-derived path clears the contact disc by the margin.

Three properties it must have, asserted rather than assumed:
* `sigma = 1` is the identity (up to the documented fp round-trip residual);
* the retraction can never make a friction-feasible path infeasible (`|a| <= sigma|a|`,
  `kappa` unchanged, `peak_g` non-increasing) — **asserted on the output**, not derived;
* the search runs on **every** candidate, not only the colliding ones, so control 3 tests
  the arithmetic and not a guard (RETRACTION #30).

## 4. ⛔ COMMITTED OUTCOMES — both written before the number exists

| verdict | condition |
|---|---|
| ⭐ **SUPPORTED** | `fan_contact` **2.4870 % -> 0.0000 %** as a STRUCTURAL zero (re-derived from the output tensor by `rewards._collision`, not by the projection's own bookkeeping) **AND** the oracle-in-fan ADE degradation is **< 0.0163 m** (the replicate floor) at a margin that reaches the zero |
| **REFUTED** | contact does not reach a structural zero at any margin (⇒ the constraint is **not expressible in this control space**) **OR** oracle-in-fan ADE degrades **>= 0.0163 m** (⇒ a real trade that must be PRICED, not shipped) |

⚠️ **A third outcome is admissible and is declared here so it cannot be invented later:
PARTIAL — a structural zero over the AVOIDABLE set with a named, measured UNAVOIDABLE
residual.** A window in which the lead enters the ego's 2 m disc even when the ego does not
move at all (`sigma = 0`) has **no** collision-free member in this family — that is physics,
not an instrument defect. If such windows exist, their count and their identity are reported
and the verdict is stated over the avoidable set, with the unavoidable set handed to §7.

## 5. ⛔ THE THREE CONTROLS — each must read a KNOWN value, or the panel is VOID

| # | control | known value it must read |
|---|---|---|
| C1 | projection **disabled** (`enabled=False`) | the fan returns **bit-identically** (`max|diff| == 0.0` exactly, and the returned object is the input object) |
| C2 | a window with **no agents** (`has_lead == False`, 175 windows) | **untouched** — `max|diff| == 0.0` exactly |
| C3 | ⭐ the **19** windows carrying all 764 colliders are **EXACTLY** the windows that move | a collider-free lead window that moves by more than the round-trip residual **VOIDS the panel** — it would mean the projection is touching something other than collision |
| C4 | **round-trip** — an already-feasible, already-clear path at `sigma = 1` | returns to `< 1e-5 m` (the arithmetic control that C1 cannot give, `feasible_decode`'s own doctrine) |
| C5 | **friction non-regression** — `peak_g` of the output | `<= peak_g` of the input, candidate-wise, everywhere |

## 6. The frontier that must be reported (not a single margin defended)

Margin `m` inflates the projection's clearance radius to `r + m`. Its cost curve, per `m`:

* `fan_contact` / `sel_contact` at the scorer's own `r` (the structural zero);
* `fan_contact` under a **perturbed lead track** (the agent-prediction-error robustness the
  design doc's item 3 demands: the lead displaced by `delta` in the closing direction);
* **oracle-in-fan ADE** and **selected-path ADE** against the extracted human GT;
* mean per-candidate displacement `|p' - p|`;
* `fan_peak_g_mean`, `fan_envelope`, `fan_kamm_over` (must not rise) and **`fan_off_reach`**
  (the failure mode this projection is expected to BUY — `feasible_decode`'s scoping
  document names trading one for the other and calling it a win as the failure to avoid).

## 7. ⭐ WHAT EACH OUTCOME MEANS FOR THE RL ARM — committed in advance

* **SUPPORTED** ⇒ RL post-training for collision avoidance is **unnecessary**, exactly as it
  turned out to be for friction. The freed effort goes to what a projection provably
  **cannot** constrain: a projection can only DELETE members of the candidate set, so it can
  never fix a defect of *absence* or of *ranking*. Those are named in the RESULT.
* **REFUTED / PARTIAL** ⇒ the residual is the RL objective, stated as the exact geometry that
  resisted projection.

⛔ Either way the turn continues past the verdict.

**Evidence class of everything this SPEC will produce: MEASURED (ours; rule-based, 0 GPU).
Tier: T0/T1-adjacent — the fan the deployed model EMITS on the eval clips. Never a driving
claim.**
