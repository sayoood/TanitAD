# RESULT — RL AGAINST THE GENERATOR'S COLLISIONS

**status: BANKED INCREMENTALLY WHILE THE ARMS RUN.** §0–§4 are complete and their artifacts are
in `raw/`. §5 (the `coll200` arms against both floors) and §6 (T1, four families) are filled in as
each lands; this header names what is missing so a killed session leaves no silent gap.

Package: `TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-rl-generator-collisions/`
SPEC (pre-registered, both outcomes committed before any arm): `SPEC.md`
Tier: **T0** on the emitted fan for every `fan_*` number — a generator diagnostic, **never a
driving claim**. Evidence class: **MEASURED (ours)** unless stated.

---

## 0. The headline, in one line

**The PI is right: refcv3's generator emits colliding trajectories — 37.8 % more of them than the
brief stated — and a perfectly collision-free fan is available for 0.196 m of amortised
displacement and ZERO cost on the driven path. Neither the veto nor the feasible decode delivers
it, and the reason is not the reward's direction but its SIGNAL DENSITY: 92 % of windows carry no
collision gradient at all.**

---

## 1. ⛔ Two corrections that had to land before anything could be committed

### 1.1 The brief's motivating gain does not exist in any banked artifact

The brief stated *"the veto arm shifted `top32_contact` 0.03299 → 0.02691 = −0.00729 … the one
gain."* **MEASURED** in `…/2026-09-05-veto-only-fan-safety/raw/veto_verdict_veto200.json` and
`…/veto_verdict_veto2k.json`:

| arm | `top32_contact` base | after s0 | after s1 | Δ s0 |
|---|---|---|---|---|
| `veto200` | 0.03298611 | 0.03298611 | 0.03298611 | **+0.000000, CI [0, 0]** |
| `veto2k` | 0.03298611 | 0.03298611 | 0.03298611 | **+0.000000, CI [0, 0]** |

`fan_contact` moved the **wrong way** at both doses: **+0.000391** (200 steps), **+0.001693**
(2 000 steps), neither separated.

The strings `0.02691`, `0.0269` and `0.00729` appear **nowhere** in that package's `RESULT.md`,
searched in bare **and comma-grouped** form. `0.0269` occurs once in
`raw/fan_rerank_veto200s0.json`, whose `*__contact` entries are **all 0.0 for every selection
rule** — those are **selected-path** metrics, so that file cannot be the source of a *fan* number.

⇒ **No RL arm has ever moved the generator's collision rate.** Root-cause class: **a number
quoted without its arm and its artifact path**. Registered as `D-RL-TOP32-NOMOVE-1`.
⭐ The PI's underlying point is untouched, and §2 makes it larger.

### 1.2 The banked collision flag uses a superseded definition — every prior `contact` number is 37.8 % low

`rewards._collision` is **SWEPT in the relative frame** (a segment crossing the 2 m disc counts
even when neither endpoint is inside). `fan_bank_base_240w.npz`'s `f_contact` predates it.

**POSITIVELY IDENTIFIED, not inferred** (control C1b, `raw/p2_feasible_vs_contact.json`): the
per-step **POINT** test reproduces the banked flag with **0 disagreements over 30,720
candidates**, while every structural flag (`kamm_over`, `envelope`, `off_reach`, `infeasible`)
reproduces **exactly** under the current scorer and only `contact` differs, **one-sidedly** (289
extra, 0 fewer).

| | POINT (banked, superseded) | **SWEPT (current)** |
|---|---|---|
| `fan_contact`, all 240 windows | 0.024870 (764 / 30,720) | **0.034277 (1,053 / 30,720)** |
| `fan_contact`, 65 lead windows | 0.095673 | **0.126562** |

⭐ **Corroborated by an independent route**: `_swept_hit` landed in commit `9765634`, which
`…/veto-only-fan-safety/raw/chain_suite.sh` records as arriving **after** the s0/s1 arms were
frozen. ⇒ **State the collision definition beside any `contact` number, or it is not quotable** —
the same family as the `control_units` trap. Registered as `D-RL-CONTACT-DEFN-1`.

⚠️ **Verified for this panel**: the run clone `/c/Users/Admin/refcv4b_repo` carries `rewards.py`,
`fan_safety.py` and `posttrain.py` **md5-identical to the Drive**, and its `rewards.py` contains
`_swept_hit` (3 hits, against a same-breath control that reads 1). **These arms run and report
under the swept definition.**

---

## 2. ⭐ The defect, quantified (the PI's point, made larger)

**MEASURED, 0 GPU**, current swept scorer, `raw/p2_feasible_vs_contact.json`:

| | value |
|---|---|
| `fan_contact`, all 240 windows | **0.034277** (1,053 / 30,720) |
| `fan_contact`, 65 lead-bearing windows | **0.126562** |
| windows carrying ≥ 1 collider | **19 / 240 = 7.92 %** (29.2 % of lead windows) |
| worst window | **104 of 128 candidates collide (81.2 %)** |
| worst 5 windows | **54.7 % of all colliders** |
| `sel_contact` | **0.0000 — a STRUCTURAL zero** |

⛔ **`sel_contact = 0` measures the SELECTOR, not the model.** The selector picks a non-colliding
candidate in **240 / 240** windows. A model that depends on a downstream filter is not the model
to ship — which is precisely the PI's argument, and the reason the primary endpoint is
`fan_contact`. Registered as `D-RL-GEN-COLLIDES-1`.

---

## 3. ⭐⭐ WHY nothing has moved it: the collision gradient is absent in 92 % of windows

The optimiser is GRPO-style — the advantage is **group-relative across one window's candidates**.
A window in which *every* candidate collides, or *none* does, contributes an **identically zero**
collision advantage.

| | count | share |
|---|---|---|
| windows with ≥1 collider **and** ≥1 non-collider (can carry a collision advantage) | **19 / 240** | **7.92 %** |
| windows contributing exactly zero collision gradient | 221 / 240 | **92.08 %** |
| episodes containing any such window | **14 / 121** | 11.6 % |

⭐ **CORROBORATED BY TWO FURTHER, INDEPENDENT PROBES** — and this matters, because the `ls-tree`
lesson is that repeated samples through one channel are one sample. The figure above is an
**eval-side window count**; the trainer's own **train-side counter** is a different mechanism:

`arm_summary.json::counters.components_fired` reads **`collision = 41`** against **`comfort` 200,
`feasibility` 200, `progress` 200** over 200 optimizer steps — **20.5 % vs 100 %** — and it reads
**41 in `ctrl_null` (collision weight 0.0) and 41 in `coll200` (collision weight 1.0), identically**.
⇒ The sparsity is a property of the **data**, not of the arm.

Registered as `D-RL-COLL-SPARSE-1`. This is the **pre-registered** explanation for any null in
§5, committed in `SPEC.md` §2 before the arms ran.

---

## 4. P2 and P3 — the two levers the brief proposed, both measured at 0 GPU

### 4.1 The feasible decode fixes feasibility completely and does NOT fix collisions

`project_feasible` defaults (`mu = 0.7`, `clamp_entry = False`), **both sides re-scored with the
current swept scorer**, paired episode-cluster bootstrap. Controls: **C1a** structural flags
reproduce the bank exactly ✅ · **C1b** bank definition identified ✅ · **C2** `enabled=False`
bit-identical (0.000e+00) ✅ · **C3** round-trip fixed point (2.13e-14) ✅ · **C4** the projection
ran (27,431 / 30,720 = **89.29 %** of candidates moved) ✅.

| metric (240 windows) | OFF | ON | Δ [CI] | separated |
|---|---|---|---|---|
| **`fan_contact`** | 0.034277 | 0.036230 | **+0.001953** [−0.000034, +0.003992] | no |
| `top32_contact` | 0.008984 | 0.013672 | **+0.004687** [+0.001260, +0.009333] | ⛔ **yes, WORSE** |
| `fan_kamm_over` | 0.840625 | **0.000000** | −0.840625 | yes |
| `fan_envelope` | 0.887728 | **0.000000** | −0.887728 | yes |
| `fan_infeasible` | 0.891960 | 0.333301 | −0.558659 | yes |
| `fan_unsafe` | 0.110319 | 0.098210 | −0.012109 | yes, better |

**Mechanism:** infeasible candidates were flying into geometry no vehicle can reach — frequently
*away* from the lead. Making them realizable pulls them back into the reachable set, which is
where the lead is. The residual `infeasible` 0.333 is `off_reach`, untargeted at
`clamp_entry = False`.

⇒ ⛔ **"Run the RL collision stage on top of the projection" cannot by itself be the fix.** The
projection is a clean **substrate** — it removes the feasibility confound entirely — but
collisions are **orthogonal** and it mildly worsens them. Registered as
`D-RL-FEASDECODE-CONTACT-1`.

### 4.2 `progress` binds in isolation and NOT in the composed reward

Tie-safe **AUC** = P(score of a collider > score of a non-collider), ties 0.5. **0.5 = no
information; > 0.5 = the term ranks colliders higher, i.e. it BINDS.**

| term | AUC (unprojected) | AUC (projected) | reading |
|---|---|---|---|
| **`progress`** | **0.8947** [0.8514, 0.9354] sep | **0.7206** [0.6541, 0.7907] sep | ⛔ **BINDS strongly** |
| `feasibility` | 0.5084 [0.3550, 0.6606] | 0.5192 [0.5063, 0.5336] sep | marginal |
| `headway` | 0.1406 sep | 0.1487 sep | already disprefers colliders |
| `comfort` | 0.0261 sep | 0.0788 sep | already disprefers colliders |
| **`_composed_default`** | **0.0000** | **0.0000** | ⭐ **already perfect** |
| `_composed_default` − `progress` | 0.0000 | 0.0000 | unchanged by removing progress |

⚠️ **A first implementation FAILED both controls and would have shipped a false positive.** Using
`argsort(argsort(·))` ordinal ranks, the **constant** control read **+0.6464** instead of 0 and the
**oracle** read **+0.0810** instead of −1 — because the flag is ~97 % ties and `np.argsort` is
stable, so a constant score received ranks 0…K−1 in index order. It would have reported `progress`
at +0.49 from a broken probe. Both controls **PASS exactly** after the switch to AUC. *(This is
the CLAUDE.md probe-panel rule earning its place a fifth time.)*

⭐ **THE ANSWER: `progress` binds in isolation (AUC 0.89) but is completely dominated in the
composed reward** — with `collision` at 1.00 no colliding candidate ever outscores a
non-colliding one (**AUC 0.0000, a structural zero**), and **deleting `progress` changes that by
exactly nothing**. ⇒ **Repairing `progress` is NOT the lever.** The reward's *ranking* is already
correct; what is missing is *signal density* (§3). Registered as `D-RL-PROGRESS-COMPOSED-1`.

---

## 5. ⭐⭐ THE PRICE OF A COLLISION-FREE FAN — measured without training anything

`fan_contact` is a property of the candidate **set**, not of the ranking, so no re-weighting can
change it: the generator has to **move** the paths. Contact is with a time-aligned **lead** and
92.2 % of the programme's measured deficit is along-track, so the minimal physically meaningful
fix is *"travel less far along your own path"* — the path scaled toward the ego origin by
`s ∈ [0, 1]`, shape preserved.

Controls: **Q1** `s = 1.0` reproduces the fan's contact exactly (0 disagreements) ✅ · **Q2**
`s = 0.0` clears contact for **every** candidate, so no window is unfixable-by-braking ✅ ·
**Q3** 9 non-monotone re-entries out of 1,053 × 101 grid evaluations, reported not hidden.

| | value |
|---|---|
| colliding candidates | **1,053** |
| fixable by braking alone | **1,053 (100.0 %)** |
| displacement to clear, per collider | mean **5.73 m**, median 4.84 m, p90 11.69 m |
| retreat fraction needed | mean **42 %** of path length |
| **amortised over every emitted candidate** | **0.196 m** [0.077, 0.346] |
| ⇒ resulting `fan_contact` | **0.034277 → 0.000000** |
| ⭐ **cost on the DRIVEN path** | **0.000 m — the selector already picks a non-collider in 240/240 windows** |

⭐⭐ **THE TRADE IS FAVOURABLE AND THE BRIEF'S CLOSING QUESTION IS ANSWERED AT THE CEILING: a
structurally collision-free fan costs 0.196 m of amortised fan displacement and NOTHING on the
path the car actually drives.** That is the target any collision objective is aiming at, and it is
an **upper bound on the cost** (a lateral evasion could be cheaper) and a **lower bound on the
achievable rate** (0.000000) for this family of fixes.

⚠️ **Stated honestly:** 0.196 m is **fan-wide mean displacement**, not a T1 ADE delta on the
selected path — different objects, and the two must never be compared directly. It is quoted as
the size of the geometric change required, which is what "at what ADE cost" is asking at the
generator level. ⚠️ Braking does **not** repair feasibility (`fan_infeasible` 0.891960 →
0.891960): only 3.4 % of candidates move and ~89 % were already infeasible, so this is a
**collision** fix and not a feasibility one. The two defects are orthogonal, exactly as §4.1 found
from the other side.

Registered as `D-RL-COLL-PRICE-1`. Artifact: `raw/p5_collision_price.json`.

### 5.1 ⭐ Successor S2 is worth building — and the brief's proposed grading is the wrong one

`rewards._collision` returns **−1 or 0**. Inside a GRPO group every colliding candidate therefore
gets the **same** value: the term can say *"these are bad"* but never *"this one is worse"*, so it
supplies **no direction within the colliding set**. Control **G1** measures rather than assumes
this: within-colliders **std = 0.000e+00**, one distinct value (−1.0). ✅

Severity, though, varies a great deal:

| grading candidate | spread among colliders | usable? |
|---|---|---|
| **penetration depth** into the 2 m disc | p10 **0.190 m** → p90 **1.536 m** = **8.09×**; mean 0.842, std 0.495 | ⭐ **yes** |
| `min_ttc_s` | **p0 = p25 = p50 = 0.5000 s** — saturated at the 0.5 s grid floor for ≥ 75 % of colliders | ⛔ **no** |

And the quantity a group-relative advantage can actually use — the **within-window** spread — is
**non-degenerate in 19 / 19 mixed windows** (mean range **1.47 m**, mean std 0.43 m).

⇒ **S2 is worth building, graded by PENETRATION DEPTH, not by time-to-contact.** The brief
proposed *"weight it by time-to-contact rather than a binary"*; MEASURED, a TTC-weighted term
would be **nearly as flat as the binary it replaces**, because TTC on the 0.5 s grid is quantised
to its floor for three quarters of colliders. This refines the successor rather than adopting it.

⭐ **An internal cross-check that two independent derivations agree exactly:** **289 of 1,053
colliders (27.4 %) are SWEPT-ONLY** — no sampled point lies inside the disc, they cross between
samples — and **1,053 − 764 = 289** is precisely the swept-vs-point gap §1.2 measured by a
completely different route. Registered as `D-RL-GRADED-TERM-1`.
Artifact: `raw/p6_graded_term_headroom.json`.

---

## 6. P4 — the collision lever at 200 steps, against BOTH floors

⏳ **PARTIAL — `coll200` s0 has landed; the replicate (s1), `ctrl0` and the definition-matched
null are still running. ⛔ NOTHING IN §6.1 IS QUOTABLE YET** — `H-ESTIM-SEED-1` is explicit that
a separated one-seed CI is **necessary and not sufficient**, and on this rig a *zero-lever*
replicate separated 3 of 18 metrics with nothing moved.

### 6.1 `coll200` seed 0 — the first time any arm has moved the generator's collision rate

`veto_rate_mean` **0.0000** exactly (the veto really is off), `weights_changed` True,
`final_loss` **0.02588** vs `ctrl_null`'s **0.04459**, 318.5 s, `components_fired.collision`
**41 / 200**, `sel_idx_agreement_with_base` **1.000** (the selector picked the identical
candidate in every window, so the fan moved underneath an unchanged selection).

| metric | before | after | Δ s0 [CI] | sep |
|---|---|---|---|---|
| ⭐ **`fan_contact`** (PRIMARY) | 0.13411458 | 0.13259549 | **−0.001693** [−0.003646, −0.000260] | **yes** |
| `top32_contact` | 0.03298611 | 0.03211806 | −0.001042 [−0.003125, **+0.000000**] | marginal (CI touches 0) |
| `top8_contact`, `sel_contact` | 0.0 | 0.0 | +0.000000 | structural zero |
| ⛔ `fan_peak_g_mean` (g) | 4.180896 | 4.211516 | **+0.025303** | **yes — WORSE** |
| ⛔ `sel_peak_g` (g) | 0.194649 | 0.222833 | **+0.023960** | **yes — WORSE** |
| ⛔ `fan_infeasible` | 0.894531 | 0.897070 | **+0.002209** | **yes — WORSE** |
| `fan_unsafe` (contact ∨ ttc) | 0.428385 | 0.429036 | +0.000781 | no |
| `R3` (sel-ADE 2 s, T0) | — | — | +0.006438 [−0.000141, +0.012912] | no |

⇒ On one seed the collision reward **does** reduce the generator's collision rate — and buys it
with a **separated** worsening of friction on both the fan **and the driven path**. That is the
veto arm's trade mirrored: there, a constraint channel made the fan blander and the selected path
more aggressive; here, a collision reward makes the fan safer on contact and more aggressive on
friction. `fan_unsafe` — the aggregate that contains contact — **did not improve**.

### 6.2 ⛔ THE BANKED ZERO-INFORMATION FLOOR IS NOT COMPARABLE, AND THE ARMS' OWN READOUTS SAID SO

The panel's first act was a definition-match check between the lever and the banked null, and it
**failed on exactly the family it should**:

| BEFORE readout | `coll200` | banked `ctrl_null` | |
|---|---|---|---|
| `fan_contact` | **0.1341145833** | **0.0974392361** | ratio **1.3764** |
| `fan_peak_g_mean` | 4.180895572900772 | 4.180895572900772 | **bitwise identical** |

The **contact** family moved and the **friction** family did not — precisely the signature of a
collision-definition change — and **1.3764 reproduces the independent SWEPT/POINT ratio 1.3782**
measured in §1.2 on a *different* window set (36 lead windows of 120 vs 65 of 240). Two
populations, two routes, three significant figures.

⇒ The banked `s0/ctrl_null` was scored under the **POINT** definition; a POINT-definition drift is
**~38 % too small** to floor a SWEPT-definition lever. The verdict is therefore computed with
`--null-dir` pointed at **`s1/ctrl_null`, which this panel runs under the CURRENT code** —
definition-matched and dose-matched (200 steps). The seed differs from the lever's, which is
acceptable because the null is a **direction + magnitude** check and never a paired contrast; the
veto package stated the same caveat for its own single-seed null.

⭐ This is the `H-VETO-FAN-1` **G2 guard** (*"every arm's BEFORE readout bitwise identical"*)
earning its place: it is normally a formality, and here it caught a silent cross-definition
comparison that would have produced a floor 38 % too permissive on the package's primary endpoint.

### 6.3 ⛔ THE REPLICATE LANDED — AND THE ARM FAILS ITS COMMITTED SUCCESS CRITERION

`coll200` s1: `veto_rate_mean` **0.0000** exactly, 326.3 s, `final_loss` **0.11635** (vs s0's
0.02588 — a 4.5× difference, so the two runs are genuinely different training trajectories),
`components_fired.collision` **44 / 200**, `sel_idx_agreement_with_base` 0.975.
**G2 guard between the two seeds: BEFORE readouts bitwise identical on `fan_contact`,
`fan_peak_g_mean` and `fan_infeasible`.** ✅

| metric | Δ s0 | sep | Δ s1 | sep | same sign | reading |
|---|---|---|---|---|---|---|
| **`fan_contact`** (PRIMARY) | −0.001693 | **yes** | −0.001693 | **no** | yes | ⛔ **NOT separated at both seeds** |
| `top32_contact` | −0.001042 | yes | **+0.000000** | yes | **no** | ⛔ sign disagrees |
| `fan_unsafe` | +0.000781 | no | −0.003776 | yes | **no** | ⛔ sign disagrees |
| ⛔ `fan_peak_g_mean` (g) | +0.025303 | yes | **+0.040678** | **yes** | **yes** | **QUOTABLE REGRESSION** |
| ⛔ `fan_infeasible` | +0.002209 | yes | **+0.002489** | **yes** | **yes** | **QUOTABLE REGRESSION** |
| `sel_peak_g` (g) | +0.023960 | yes | +0.008523 | no | yes | not replicated |
| `top32_infeasible` | +0.000857 | no | +0.012790 | yes | yes | not replicated |

> ⛔ **FAILURE against the committed SUCCESS text.** SPEC §6 required `fan_contact` to decrease
> **separated at BOTH seeds**, same sign, above both floors. It is separated on **s0 only**.
> ⭐ **The gain is not replicated; the COSTS are.** `fan_peak_g_mean` and `fan_infeasible` clear
> the bar the collision gain does not: separated at both seeds, same sign, and larger on the
> second seed. A report quoting only s0's `fan_contact` would be true and misleading.

⚠️ **One coincidence, stated rather than built on.** Both seeds report an episode-clustered delta
of **−0.001692708333** to twelve decimal places, while their AFTER states genuinely differ
(`611/4608` vs `613/4608`) and their full-population changes are **−0.001519** and **−0.001085** —
a factor of 1.4 apart. The clustered means coinciding exactly is an artefact of the cluster
weighting over n = 30 episodes, **not** evidence of a stable effect. The full-population numbers
are the more trustworthy read and they do **not** agree.

### 6.4 ⭐ P5 — CONTINUING, not stopping: S2 is built and its controls pass

The arm failed, so per SPEC §6 the named successors run. **S2 (a penetration-graded collision
term) is implemented and controlled** in `raw/s2_graded_collision.py`:

```
pen = (r - d_min_swept).clamp_min(0)                      # r = 2.0 m, the contact radius
out = 0                              where pen == 0
out = -(0.5 + 0.5 * min(1, pen / r)) where pen  > 0       # range [-1, 0], neutral 0
```

⭐ **The normaliser is the contact radius itself — a GEOMETRIC constant, not one fitted to the
observed depths.** The deepest possible incursion into a disc of radius `r` is `r`, so `pen/r` is
in [0, 1] by construction. A severity scale tuned on the scored data is precisely the probe-panel
error, and this avoids it entirely.

| control | result |
|---|---|
| **S1 SUPPORT-IDENTITY** — must punish exactly the same candidates as the stock term | **0 disagreements / 1,053** ✅ |
| **S2 NOT-FLAT** — the entire point | graded std **0.125746** vs binary **0.000e+00** ✅ |
| **S3 CLEAN-ZERO** — non-colliders read the neutral value exactly | max abs **0.000e+00** ✅ |
| **S4 IN-RANGE** — same contract as the stock component | [−0.9995, 0.0], colliders ≤ −0.50 ✅ |
| **S5 MONOTONE** — deeper never scores higher | rank corr **−1.000000000** ✅ |

⭐ **And the swept geometry is load-bearing, not cosmetic: swept penetration is > 0 for all 1,053
colliders, while the per-SAMPLE depth is > 0 for only 764 — grading on sampled points alone would
leave 289 colliders flat at the floor and reintroduce the very flatness S2 exists to remove.**
That 289 is the third independent appearance of the swept-vs-point gap (§1.2, §5.1, here).

### 6.5 ⭐ THE ESCALATION (24bd5e8) — answered, accepted, and acted on

M39 (`c9ab82c`) takes `fan_contact` to a **structural zero at +0.0000 m and zero GPU** with a
contact-stage projection, and the escalation asks **one** thing of this stream: *"supply the
population its `fan_contact` base is computed over, so the two results can be read on ONE
object."* It is right to ask, and here it is.

| number | population | fraction | definition |
|---|---|---|---|
| **0.134114583333** (this package's arms) | **36 lead-bearing windows** of the RL rig's 120-window eval readout × 128 candidates = **4,608** | **618 / 4,608** | SWEPT |
| 0.034277 (banked fan, **all** windows) | 240 windows × 128 = 30,720 | 1,053 / 30,720 | SWEPT |
| **0.126562** (banked fan, **lead-only**) | **65 lead-bearing windows** × 128 = 8,320 | 1,053 / 8,320 | SWEPT |

Source for the first row is the artifact itself, not an inference:
`arm_summary.json::fan_safety_n["fan_contact"] = 36`.

⇒ **The escalation's objection is exactly correct and the reconciliation is the lead-only
restriction: 0.134115 (36 lead windows) against 0.126562 (65 lead windows) — both lead-only,
both SWEPT, 5.97 % apart, consistent with two window draws from one corpus.** `0.034277` is the
all-windows rate and was never comparable to an RL-rig figure. Both streams' relative movements
are therefore readable on one object, and the absolute rates are now reconciled rather than
merely flagged as incomparable.

**Accepted:** `fan_contact` is retired as the RL primary endpoint. It is solved by construction,
and §6.3 is the third instance of the same trade signature. **Acted on:** the T1 four-family
evals — ~40 GPU-minutes to price this arm against a retired endpoint — were **cancelled**
(`raw/chain_after_coll4.sh` records the decision and the reason). The arms in flight finish, as
the escalation asks, because they establish the rig's noise floor.

⭐ **Two independent confirmations between the streams, worth recording because they were
derived separately:** M39 reports the point→swept correction as **764 → 1,053, +37.83 %**; §1.2
of this package measured **+37.8 %** from the opposite direction (identifying the banked
definition rather than the corrected one). And M39's own contact-vs-friction split — *"M35
measured the friction projection making contact slightly worse"* — is **§4.1 of this package**,
measured before either result existed.

### 6.6 ⭐ THE FIRST-SEGMENT BLIND SPOT: real, and MEASURED to be immaterial here

M39 names a defect it does not size: `rewards._collision` builds
`rel = lead[..., 1:, :] - traj[..., 1:, :]`, dropping index 0, so **the t0→t1 segment is never
swept** and a plan driving through a car in the first step can read CLEAR. The size decides
whether a *structural zero on this metric* is a zero on the road, so it is measured
(`raw/p7_first_segment_blindspot.json`).

Controls: **P2** the reimplementation restricted to segments 1–3 reproduces the stock flag with
**0 disagreements** (so any difference is attributable to the first segment and nothing else) ✅ ·
**P1** no candidate loses its flag under a wider sweep ✅ · **P3** the t0 condition never varies
by candidate within a window ✅.

| | value |
|---|---|
| stock colliders | 1,053 (`fan_contact` 0.034277) |
| with the t0→t1 segment swept | 1,055 (`fan_contact` 0.034342) |
| **newly detected** | **2 = +0.19 % of the stock count** |
| **windows gaining a collider** | **0** (of 19 collider-bearing) |
| leads already within 2 m at t0 (a SCENE property, never charged to the plan) | **0 / 65** |

⇒ **The defect is REAL and, on this corpus, IMMATERIAL — it does not undermine M39's structural
zero.** Reporting "the defect is real" without its size would have been alarming and wrong.
⚠️ **But it is immaterial only HERE**: this is a lead-following corpus where no lead is within
2 m at t0. On a corpus with close cut-ins or a parked obstacle — exactly M39's *"car parked
1.5 m ahead"* — it is not, so the fix is still worth making.

⚠️ **And making it exposed a `one_variable` error of my own, caught by the controls.** I first
folded the first-segment fix straight into the S2 graded component; **S1 and S3 immediately
failed**, because the change moved the term's **SUPPORT** (which candidates are punished) as well
as its **GRADING** (how much) — two levers in one arm, and an unattributable result if it moved.
It is now a separate switch, `sweep_first_segment`, **default OFF**, so S2's arm runs
support-identical to the stock term. Both modes pass all five controls, and the fix-ON mode
gains **exactly 2** candidates — independently matching p7's count from a separately written
probe.

### 6.7 Still pending

**One-variable, asserted mechanically** (`raw/patch_add_coll_arms.py`, `ONE_VARIABLE=PASS`):
`coll200` differs from the already-banked `ctrl_null` in **the collision weight alone,
0.0 → 1.0** — same `w_anchor` 1.0, lr 1e-5, 200 steps, `two_scalar`, `use_gt_bar=False`,
`veto_enabled=False`.

Training-side facts already banked for `coll200` s0: `veto_rate_mean` **0.0** exactly (the veto
really is off), `final_loss` **0.02588** vs `ctrl_null`'s **0.04459** (the reward really is
different), `components_fired.collision` **41 / 200**.

⚠️ **A caveat the audit itself raises, recorded now rather than after the numbers:**
`reward_audit.verdict` is **INCONCLUSIVE** for this arm, with every degenerate-policy score at
0.0. That is correct behaviour, not a bug — the `collision` component's documented degenerate is
**"stand still forever (never collides)"**, and a collision-only reward assigns 0 to every
degenerate policy that does not collide, so the audit cannot discriminate. The only thing
preventing the stand-still solution is the **`w_anchor` 1.0 trust region**. ⇒ **the four-family
LONGITUDINAL read is not optional here — it is the check on the degenerate**, and any
`fan_contact` win accompanied by a speed collapse is a reward hack, not a result.

---

## 7. Deliverable manifest

| artifact | where it lives | what it carries |
|---|---|---|
| `SPEC.md` | repo (this package) | pre-registration, both outcomes, the blocking floor check — banked **before** any arm ran |
| `raw/p2_feasible_vs_contact.py` / `.json` | repo | §1.2, §2, §4.1 — decode ON/OFF + the 4 controls |
| `raw/p3_progress_binds.py` / `.json` | repo | §4.2 — AUC panel + the K1/K2 controls |
| `raw/p5_collision_price.py` / `.json` | repo | §5 — the price of a collision-free fan + Q1/Q2/Q3 |
| `raw/patch_add_coll_arms.py` | repo | the `coll200`/`coll2k` arms + the ONE_VARIABLE assertion |
| `raw/run_coll_arms.sh` | repo | the arm panel, priority-ordered, with a stale-driver verify gate |
| `raw/insert_claims_rows.py` | repo | the `GOALS_AND_CLAIMS.md` rows + per-id content verification |
| `stack/scripts/rl_refcv3_min.py` | repo (modified) | `ARMS` now carries `coll200` and `coll2k` |
| `Project Steering/GOALS_AND_CLAIMS.md` | repo (modified) | 7 new rows, inserted and content-verified |
| arm outputs | **`C:/Users/Admin/veto_run/run/s{0,1}/{coll200,ctrl0,ctrl_null}/`** (dev box, off-Drive) | ⚠️ **checkpoints are 428 MB each and are NOT banked to the repo**; the summaries and readouts are what the analysis consumes |
