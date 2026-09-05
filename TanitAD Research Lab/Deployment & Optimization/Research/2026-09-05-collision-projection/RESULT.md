# RESULT — `H-PROJ-CONTACT-1`: **SUPPORTED.** A colliding trajectory is now unrepresentable, at zero measurable cost.

**Arch+Inference FlyWheel, 2026-09-05. 0 GPU.** Pre-registration: `SPEC.md`, committed as
`a6002d9` **before** the projection existed. Instrument:
`stack/tanitad/refs/contact_projection.py` (+ 28 tests). Panels: `raw/panel*.py`, readouts
`raw/contact_projection_panel{,2,3}.json`.
Evidence class of every number below: **MEASURED (ours; rule-based, no learned parts)**.
Tier: **T0 / T1-adjacent** — the fan the deployed model EMITS on the eval clips. Never a driving claim.

---

## 0. The one line

> **Yes — a colliding trajectory can be made unrepresentable, and on this fan it costs
> nothing measurable: `fan_contact` 3.428 % → a structural 0.000000 % at every margin from
> 0 m to 20 m, with oracle-in-fan ADE and selected-path ADE moving by EXACTLY 0.0000 m
> (CI [0, 0]) against a 0.0163 m replicate floor — and `fan_off_reach` IMPROVING 0.1077 →
> 0.0991 rather than paying for it.**

⚠️ **And the honest half of that sentence:** it costs nothing *because the colliding
candidates were never the ones carrying the fan's value.* §4 measures that instead of
asserting it, and §7 says what it does and does not license.

---

## 1. The defect, reproduced — and it is 37.8 % bigger than banked

| | count | rate | windows | worst window |
|---|---|---|---|---|
| banked `f_contact` (POINT test) | **764** | 2.4870 % | 19 / 240 | 104 / 128 |
| re-derived, CURRENT **swept** `_collision` | **1053** | **3.4277 %** | **19 / 240** | **115 / 128** |
| difference | **+289** | | same 19 rows | |

⭐ **+37.83 % — and `D-SWEPT-1` published +37.8 % from an independent measurement.** The bank
was written against a clone frozen before commit `9765634`, so **every contact number banked
on 2026-09-05 is the point-test LOWER BOUND**, exactly as `test_collision_swept_segment.py`
warned. The 19 collider windows are identical under both predicates
(`[1, 2, 21, 22, 44, 48, 51, 62, 105, 106, 128, 131, 132, 154, 196, 197, 204, 226, 230]`).

⛔ **And `sel_contact` is 0.0000 in both.** The selector's argmax never picks a collider — see
§5, which is where that number stops being reassuring.

## 2. ⛔ THE CONTROLS — all six read their known values

| # | control | known value | reading | |
|---|---|---|---|---|
| **C1** | projection **disabled** | returns the input OBJECT, `max|diff| == 0.0` | `is_same_object: True`, `0.0` | ✅ |
| **C2** | a window with **no agent** (175 of 240) | `max|diff| == 0.0` exactly | **`0.0`** | ✅ |
| **C3** | ⭐ **only the collider windows move** | moved set == collider set | **identical, 19 == 19**, both differences empty | ✅ |
| **C3b** | only the collider **candidates** move | moved set == `{clearance < r_need}` | **1057 == 1057**, exact | ✅ |
| **C4** | **round-trip** (the arithmetic control C1 cannot give) | `< 1e-5 m` | **7.41e-14 m** over **29,663** untouched candidates | ✅ |
| **C5** | friction non-regression | `peak_g(out) <= peak_g(in)` everywhere | max increase **0.0 exactly**, 0 candidates increased | ✅ |

**C3b is the one that needed thinking about, and the thinking is the finding.** Four
candidates moved that the SCORER does not call colliders. Their input clearances are
**2.0002, 2.0072, 2.0083, 2.0149 m** — every one inside the projection's own 2 cm numerical
margin (`r_need = 2.0200`). ⇒ the control's correct known value is *"the moved set equals the
set violating the PROJECTION'S clearance"*, not the scorer's `r`; stated that way it reads
**exactly**. ⚠️ Naming the scorer's `r` there would have been a control that is **wrong about
its own object** — the failure C3 exists to catch, committed by C3 itself. The first pass of
C4 made the same error in the other direction (it scoped "untouched" to "clear by the scorer"
and read **2.18 m**); both are recorded in the readout rather than quietly fixed.

⭐ **C1 is deliberately reported as WEAK.** It short-circuits, so the arithmetic never runs and
it proves nothing about the projection — `feasible_decode`'s own doctrine. C4 is the arithmetic's
control, and it is computed on a **different tensor** (`info["uniform"]`, the search-everywhere
re-integration) so that the shipped identity-passthrough cannot fake it.

## 3. The frontier: contact vs ADE vs margin

**Structural zero at every margin tested.** `min_clearance_m` tracks `r_need` to 4 decimals, so
the margin binds rather than being asserted.

| margin `m` | `r_need` | `fan_contact` | `sel_contact` | min clearance | n moved | mean disp (moved) | `σ_min` | ΔoracleADE | ΔselADE | `fan_off_reach` | `fan_peak_g` | `fan_kamm` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **base** | — | **0.034277** | 0.0000 | 0.000 | — | — | — | — | — | 0.10775 | 4.1131 | 0.8406 |
| 0.00 | 2.0200 | **0.000000** | 0.0000 | 2.0201 | 1057 | 4.466 m | 0.129 | **+0.0000** | **+0.0000** | 0.09906 | 3.9629 | 0.8350 |
| 0.25 | 2.2725 | 0.000000 | 0.0000 | 2.2725 | 1094 | 4.508 m | 0.121 | +0.0000 | +0.0000 | 0.09883 | 3.9573 | 0.8338 |
| 0.50 | 2.5250 | 0.000000 | 0.0000 | 2.5251 | 1129 | 4.556 m | 0.113 | +0.0000 | +0.0000 | 0.09827 | 3.9517 | 0.8331 |
| 1.00 | 3.0300 | 0.000000 | 0.0000 | 3.0301 | 1196 | 4.618 m | 0.098 | +0.0000 | +0.0000 | 0.09724 | 3.9416 | 0.8314 |
| 2.00 | 4.0400 | 0.000000 | 0.0000 | 4.0400 | 1346 | 4.849 m | 0.074 | +0.0000 | +0.0000 | 0.09734 | 3.9181 | 0.8277 |
| 3.00 | 5.0500 | 0.000000 | 0.0000 | — | 1543 | — | — | **+0.0441** | **+0.0593** | 0.10000 | — | — |
| 8.00 | — | 0.000000 | 0.0000 | — | 2129 | — | — | +0.0016 | +0.0025 | 0.09110 | — | — |
| 20.0 | — | 0.000000 | 0.0000 | — | 4217 | — | — | +0.1671 | +0.1703 | 0.08986 | — | — |

Every ΔADE is a **paired episode-cluster bootstrap** over the 40 val episodes
(240 windows / 121 episodes, `n_boot` 4000, seed 11); at `m ≤ 2` the interval is `[0, 0]`
because the delta is **structurally** zero (§4), not merely small.

**The knee is at `m = 3 m`**, where the *lateral* axis first fires (128 candidates = one whole
window) and the first real ADE cost appears. **`m ≤ 2.0 m` is free.** `envelope` is unchanged
throughout by design — the contact arm runs with the friction clamp OFF so the effect is
attributable to contact alone.

### 3.1 The margin's meaning, and the failure the first version had

A projection margin `m` is **exact immunity to an agent-POSITION error of `m`**, and the
readout verifies it rather than assuming it:

| projection margin | contact at `r+0.5` | at `r+1.0` | at `r+2.0` | lead track 0.5 s **early** | 0.5 s **late** |
|---|---|---|---|---|---|
| `m = 0.00` | 0.036654 | 0.038737 | 0.043717 | **0.028646** | 0.000033 |
| `m = 0.50` | **0.000000** | 0.038737 | 0.043717 | 0.015495 | 0.000000 |
| `m = 1.00` | 0.000000 | **0.000000** | 0.043717 | 0.013737 | 0.000000 |
| `m = 2.00` | 0.000000 | 0.000000 | **0.000000** | **0.006087** | 0.000000 |

The time-shift columns are the part a radius cannot buy: a lead **0.5 s earlier than
predicted** re-introduces contact at 2.86 % under `m = 0`, falling to 0.61 % at `m = 2`. ⇒ the
margin is a **prediction-error budget**, and the design doc's item 3 is now a table.

⛔⛔ **THE BUG THIS FRONTIER FOUND, AND THE FIX.** The first version searched a single
`r_need`, and where no member cleared it returned the input UNCHANGED and labelled it
`UNAVOIDABLE`. MEASURED: raising the margin from 4 m to 6 m took `fan_contact` at the
scorer's own 2 m radius **from 0.000000 back up to 0.007389 (227 candidates)** —
**asking for more safety delivered less**, because the guarantee was *"clear at `r_need` OR
untouched"*. The fix is a **margin ladder**: retry at decreasing margins and report
`margin_achieved` per candidate, so the guarantee becomes the one a caller wants —
*clear at the scorer's radius unless NO member of the family clears it, and clear at the
requested margin wherever that is possible*. With the ladder, `fan_contact` is `0.000000`
and `unavoidable` is `0` at **every** margin out to 20 m; the price shows up honestly as
`frac_colliders_at_full_margin` (1.000 at `m ≤ 3`, 0.474 at `m = 20`).

## 4. ⛔ WHY ΔADE IS EXACTLY ZERO — measured, because "exactly zero" is a claim that needs a mechanism

| | |
|---|---|
| windows whose **oracle** (min-ADE) candidate collides | **0 / 240** |
| windows whose **selected** candidate collides | **0 / 240** |
| windows whose oracle or selected candidate was **moved** | **0 / 240** |
| `max |ADE_out − ADE_in|` over the 29,663 **untouched** candidates | **0.000e+00** |

⇒ the zero is **structural**, not a rounding of something small: the projection never touches
the candidate that any ADE reading looks at. ⚠️ **Therefore "+0.0000 m" must never be quoted as
"the projection is free."** It is free *on this fan, for this checkpoint, on these readings*
because the colliding candidates were not carrying the fan's usefulness. What they were
carrying is this:

| | base | projected |
|---|---|---|
| ADE of the **moved** candidates (n = 1057) | **11.49 m** | **6.53 m** |
| of which got **closer** to the human | — | **1047 / 1057 (99.1 %)** |
| oracle ADE on the 19 collider windows | 0.1675 | 0.1675 |
| oracle ADE if the colliders were simply **DELETED** (a perfect filter's ceiling) | 0.1675 | — |
| mean ADE over the whole fan | 4.7415 | **4.5709** (−0.1706, **separated**) |

⭐ **The colliding candidates were 11.5 m from the human — garbage, not a trade-off.** Retracting
them moves them 5 m closer to the human, and the fan's mean ADE improves with a separated
interval. A perfect filter would have scored identically on oracle ADE **while deleting 1057
candidates**; the projection keeps them and makes them better.

## 5. ⛔ `sel_contact = 0.0000` IS NOT A PROPERTY OF THE GENERATOR — here is the number that says so

RETRACTION #31 in its quantitative form. On the 65 lead windows:

| readout | value | reading |
|---|---|---|
| `sel_contact` (the argmax) | **0.0000** | the deployed plan is clear TODAY |
| `top1` … `top8` contact | **0.0000** | so is the whole head of the ranking |
| **`top16` contact** | **0.00962** | colliders begin at rank 16 |
| **`top32` contact** | **0.03317** | ⛔ **DiffusionDriveV2's selector top-k is 32** — 3.3 % of it collides, in **6** windows |
| `top64` contact | 0.07212 | 13 windows |
| softmax mass on the colliding set (`sel_score_v3`) | 2.08e-05 mean, **1.19e-03 max** | small, and **not zero** |

⇒ **the selector's immunity is a measured property of one checkpoint's argmax over 128
candidates, not a guarantee.** Any change that re-ranks — an RL stage, a different
temperature, top-k or stochastic selection, a new selector head, the λ sweep already
scheduled — can start picking from a set that is 3.3 % colliding at `k = 32`. After the
projection, that set is 0.0 % colliding **for every ranking, forever**.

## 6. The composed decode — friction AND contact unrepresentable on the same tensor

| arm | `contact` | `envelope` | `kamm_over` | `peak_g` mean / **max** | `off_reach` | oracle ADE (paired vs base) | sel ADE |
|---|---|---|---|---|---|---|---|
| base | 0.034277 | 0.8877 | 0.8406 | 4.1131 / **18.394** | 0.1077 | 0.2306 | 0.4871 |
| friction only (`feasible_decode`) | **0.036230** ⚠️ | **0.0000** | **0.0000** | 0.5953 / 0.693 | **0.3333** | 0.2138 (−0.0168 [−0.0275, −0.0078] **sep**) | 0.4822 |
| contact only (this module) | **0.000000** | 0.8877 | 0.8350 | 3.9629 / 18.394 | **0.0991** | 0.2306 (+0.0000 [0, 0]) | 0.4871 |
| ⭐ **composed: friction → contact** | **0.000000** | **0.000000** | **0.000000** | **0.5802 / 0.693** | 0.3117 | **0.2138 (−0.0168 [−0.0275, −0.0078] sep)** | 0.4822 (−0.0049 [−0.0116, +0.0001]) |

⭐ **All three violations are simultaneously zero on one tensor, and oracle-in-fan ADE
IMPROVES by 0.0168 m with a separated interval** — the opposite sign from the veto arm's
+0.0362 m. The improvement is present in the friction stage and is not destroyed by the
contact stage, which is the composition property C5 proves: the σ-retraction is friction
non-increasing, so the order friction→contact is safe and no third pass is needed.

⛔ **Two things must be said beside that, not after it.**
1. **`off_reach` triples, and it is the FRICTION stage that does it** (0.1077 → 0.3333 alone;
   adding contact brings it *down* to 0.3117). ⚠️ This is exactly the trade `feasible_decode`'s
   scoping document named in advance, and it is **not** solved. It is a work item, not a
   footnote; the composed decode is not shippable until it is priced or fixed.
2. **The contact stage on its own IMPROVES `off_reach`** (0.1077 → 0.0991), and the mechanism
   is measured, not guessed: the base fan had **2151** candidates ABOVE the S2 reachability
   band and 1159 below; retracting moved **277 INTO** the band and only **10 out**, leaving the
   below-band count **unchanged at 1159**. refcv3's colliding candidates are its over-fast
   ones, so slowing them repairs two defects with one operation.
3. ⚠️ **`friction_only` makes contact WORSE** (0.034277 → 0.036230). A friction projection is
   not a safety projection; applying one alone and calling the fan safer would have been
   wrong by a measured 5.7 %.

⚠️ **H-ESTIM-SEED-1 does not apply to these intervals, and here is why.** That rule says a
separated CI from a one-seed arm cannot establish that a LEVER moved a metric, because the
episode-cluster bootstrap is blind to training variance. Every arm here is a **deterministic
transform of one fixed tensor from one checkpoint** — there is no second training run in the
comparison, and the only randomness is the episode draw, which is exactly what the estimator
models. No replicate arm is required, and none would mean anything.

## 7. Two things this found that nobody was looking for

### 7.1 ⛔ The contact predicate never sweeps its FIRST segment

`rewards._collision`'s moving-lead branch slices `rel = lead[..., 1:, :] - traj[..., 1:, :]`
and sweeps *that* polyline. The comment justifies dropping the **point** at `s = 0` ("the ego
is at its own origin") and that justification is sound — but the swept fix (`9765634`)
inherited the same slice, so **the relative segment from `t = 0` to `t = 0.5 s` is never
tested.** That is the `D-SWEPT-1` corridor again, at the step where the ego's displacement is
largest.

* **Structural demonstration** (`test_the_scorers_predicate_never_sweeps_the_FIRST_segment`,
  a deliberate-regression control): a candidate that drives **straight over a car parked
  1.5 m dead ahead at 10 m/s** reads **CLEAR**, with a reported clearance of **3.5 m**.
* **Measured incidence on this fan: 2 candidates (+0.2 %) in 1 window**, 0 in the selected
  path, 0 already in contact at `t = 0`. ⇒ **the hole is real and, here, small.** It is
  reported at its true size rather than at the size the fixture makes it look.
* **The projection closes it by construction** at no extra cost: with `skip_first=False` the
  full-sweep contact rate is **0.000000**, 0 unavoidable, 1059 moved (vs 1057),
  ΔoracleADE **+0.0000**, `off_reach` 0.0990. ⭐ **A projection can enforce a predicate
  STRICTER than the scorer's. A reward cannot, without retraining.**

### 7.2 ⭐ The clearability theorem — and it is what decides the RL question

> **If the agent track stays outside the ego's disc for the whole window while the ego does
> not move, then `σ → 0` is a collision-free member, and the search must find one.**

Checked, not argued: **65 / 65** lead windows are rest-clear, and **0 / 30,720** candidates
violate it under either predicate. **Corollary:** the *only* unclearable contacts are those
in which **the agent's own motion enters a stationary ego's disc** — an agent driving into a
parked car. No candidate set can be constrained out of that, and no planner can plan out of
it; it is a perception / prediction problem, not a generation one. The `UNAVOIDABLE` class is
implemented, tested against a constructed fixture, and **empty on this corpus**.

⚠️ The lateral axis fired **0 times at `m ≤ 2`** and 128 times at `m = 3`. Its scope limit is
measured too: the initial heading is inherited from the candidate and `κ ≤ 0.2` gives a 5 m
turn radius, so over a 2 s horizon a lateral evade cannot rescue a head-on case that the
longitudinal axis cannot. Constructed head-on fixtures confirm it. **Longitudinal retraction
is not merely the first axis — on this geometry it is very nearly the only one that works.**

## 8. ⭐ P4 — WHAT THIS MEANS FOR THE RL ARM, NAMED

**`H-PROJ-CONTACT-1` is SUPPORTED ⇒ RL post-training for collision avoidance is unnecessary,
exactly as it turned out to be for friction.** The veto arm bought ~2.7 % of the safety gap at
`ade_m` **+0.0362 m separated worse**; the projection buys **100.0 %** of the contact gap at
**+0.0000 m**. There is nothing left for an RL collision term to earn.

⛔ **But "unnecessary" is a claim about ONE objective, and the effort does not evaporate — it
moves. Here is exactly where, and each target is named because a projection provably cannot
reach it:**

**A projection can only ever DELETE or RETRACT members of the candidate set.** It cannot add a
member the vocabulary never contained, and it cannot change which member is chosen. That
partitions the remaining defects cleanly:

| defect class | can a projection fix it? | the number, MEASURED today | who owns it |
|---|---|---|---|
| **Emitting something unsafe** | ⭐ **YES, structurally** | contact 3.428 % → 0.000000 %; friction envelope 0.8877 → 0.0000 | **done — `contact_projection` + `feasible_decode`** |
| ⭐ **CHOOSING badly within a safe set** | ⛔ **NO — by construction** | oracle-in-fan **0.2306 m** vs selected **0.4871 m** = a **2.11×** selection gap, and it is **EXACTLY unchanged** by the projection (Δ 0.0000, CI [0,0]). Even inside the selector's own top-32 the gap is **0.2320 vs 0.4871 = 2.10×** | ⭐ **THE RL ARM** |
| **Not CONTAINING the right answer** | ⛔ NO | refav1: `κ ∈ {0, 0.08}` vs corpus median `|κ|` 0.00085, **38.7 %** of turns expressible; **29/40** windows decode a manoeuvre whose canonical control is `a == 0` | vocabulary / representability, **not RL** |
| **The AGENT's behaviour** | ⛔ NO — proven in §7.2 | the unavoidable class is exactly "agent enters a stationary ego's disc"; 0 on this corpus | perception / prediction |
| **Agent-track COVERAGE** | ⛔ NO | the projection is inert on **175 / 240** windows with no agent track, and reports `NO_AGENT` rather than "safe" | the `obstacle.offline` join |

⭐⭐ **AND THE STRONGEST CONSEQUENCE IS NOT THAT RL IS UNNECESSARY — IT IS THAT THE PROJECTION
MAKES THE RL ARM SAFE TO RUN.** The measured reason the veto arm failed is that it *paid ADE
for safety*: it pushed probability mass off the violating region and dragged the selected path
with it (`LON_accel_mae` +0.3349 on a base of 0.6806). On a **projected** fan that trade is not
available — every candidate is already collision-free and friction-feasible, so a reward that
is **purely selection quality** cannot buy safety with ADE, because there is no unsafe
candidate left to move mass away from. ⇒ **the RL brief changes from "make the fan safer" to
"close the 2.11× gap between the fan's best member and the one the model picks", on a
candidate set where the safety axis is a constant.** That is a well-posed objective, it is the
programme's largest measured unexploited lever, and the veto arm's own failure mode is
structurally excluded from it.

⛔ **The residual handed to the RL brief is therefore not a collision residual — it is a
SELECTION residual, and it is 0.2565 m per window (0.4871 − 0.2306), 2.11×, on 240 windows /
121 episodes, unchanged by everything a projection can do.**

## 9. What must happen next — escalations, not suggestions

1. ⛔ **`off_reach` 0.1077 → 0.3117 in the composed decode is unresolved and is the FRICTION
   stage's.** It must be priced or fixed before the composed decode ships. The contact stage
   is clean on its own (0.1077 → 0.0991) and can ship independently today.
2. ⛔ **Every contact number banked before commit `9765634` is a point-test lower bound**, and
   the correction is **+37.8 %** at the fan level (764 → 1053). `MODEL_REGISTRY.md` /
   `GOALS_AND_CLAIMS.md` rows quoting `fan_contact` need the stamp.
3. ⚠️ **The first-segment hole (§7.1) is in the SCORER, not in this module.** Closing it
   changes every banked contact number, so it needs a decision, not a patch;
   `test_contact_projection.py` asserts the defect so a silent fix breaks the suite loudly.
4. ⭐ **Wire `contact_projection` into `refc_v3`'s decode behind a flag**, with `off_reach`
   reported beside `contact` in the same table, per the module's own contract.
5. **v7f has no fan-safety number at all.** The principle transfers as a design, not as a
   module; the check is whether v7f's fan has the same gap between what its vocabulary can
   drive and what its decode emits.

## 10. Deliverable manifest

| artifact | where it lives | what it is |
|---|---|---|
| `stack/tanitad/refs/contact_projection.py` | **repo** (staged) | the projection: control-space σ-retraction + lateral axis + margin ladder |
| `stack/tests/test_contact_projection.py` | **repo** (staged) | 28 tests, incl. 2 deliberate-regression controls; `62 passed` with the 3 neighbouring suites |
| `…/2026-09-05-collision-projection/SPEC.md` | **repo** (committed `a6002d9`) | the pre-registration |
| `…/2026-09-05-collision-projection/RESULT.md` | **repo** (staged) | this file |
| `…/raw/panel.py`, `panel2.py`, `panel3.py`, `diag.py` | **repo** (staged) | the four readouts, runnable at 0 GPU |
| `…/raw/extract_gt.py` | **repo** (staged) | the human-GT extraction + its join control against the bank |
| `…/raw/contact_projection_panel{,2,3}.json` | **repo** (staged) | every number in this file |
| `…/raw/gt_240w.npz` | **repo** (staged) | GT for the 240 banked windows (join control PASS: `max|v0 diff| 0.0`, `max|lead5 diff| 0.0`, `eid` exact, `has_lead` exact) |
| `…/raw/*.log` | **repo** (staged) | the run logs |
| `…/raw/patch_ladder.py` | **repo** (staged) | the margin-ladder rewrite, kept because the bug it fixes is the finding in §3.1 |

⚠️ Nothing is stranded. The work directory `C:\Users\Admin\collproj` is a scratch clone; every
deliverable above was copied to the repo and **md5-verified after the copy** (the G: mount
threw `Errno 22` mid-read twice during this session).
