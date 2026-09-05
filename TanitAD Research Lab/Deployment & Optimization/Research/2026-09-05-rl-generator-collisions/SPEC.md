# SPEC — RL AGAINST THE GENERATOR'S COLLISIONS

**Pre-registered 2026-09-05, BEFORE any arm ran. Both outcomes committed below.**
Package: `TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-rl-generator-collisions/`
Agent: Arch+Inference FlyWheel · branch `agent/arch-inf-20260803`

---

## 0. Why this SPEC exists, and the two corrections it starts from

The PI's instruction is the objective:

> *"our model is creating trajectories with collision, so RL must improve this and it can be
> measured, by punishing trajectories with collisions the quality of output trajectories must
> improve, it is not about assessing the gt trajectory."*

He is right, and the object is the **GENERATOR'S OUTPUT DISTRIBUTION**, not the selected path.
`sel_contact` is **0.0000** and is a **structural zero** — the selector already filters the
garbage, so `sel_contact` **cannot improve and is inadmissible as a primary metric**. Two
corrections to the brief that motivated this work are established below **before** anything is
committed, because both change what may be quoted.

### ⛔ CORRECTION 1 — the brief's motivating gain does not exist in the banked artifacts

The brief states: *"RL ALREADY moved it: the veto arm shifted `top32_contact` 0.03299 → 0.02691
= −0.00729, recorded in the package as 'yes — the one gain'."*

**MEASURED** in `…/2026-09-05-veto-only-fan-safety/raw/veto_verdict_veto200.json` and
`…/veto_verdict_veto2k.json`:

| arm | `top32_contact` base | after s0 | after s1 | delta s0 |
|---|---|---|---|---|
| `veto200` | 0.03298611 | 0.03298611 | 0.03298611 | **+0.000000, CI [0, 0]** |
| `veto2k` | 0.03298611 | 0.03298611 | 0.03298611 | **+0.000000, CI [0, 0]** |

and `fan_contact` moved the **WRONG WAY** at both doses — `+0.000391` (200 steps),
`+0.001693` (2 000 steps), neither separated.

The strings `0.02691`, `0.0269` and `0.00729` appear **nowhere** in that package's `RESULT.md`
(searched in bare and comma-grouped form, per the comma-format rule). `0.0269` occurs once in
`raw/fan_rerank_veto200s0.json`, whose `*__contact` entries are **all 0.0** for every selection
rule — they are **selected-path** metrics, so that file cannot be the source of a *fan* number.

⇒ **No RL arm has yet moved the generator's collision rate.** The premise that it had is
withdrawn. The PI's point is untouched and in fact strengthened: the defect is real and nothing
has yet fixed it.

### ⛔ CORRECTION 2 — the banked collision flag uses a SUPERSEDED definition

`rewards._collision` is **SWEPT in the relative frame** (a segment crossing the 2 m disc counts
even when neither endpoint is inside); the comment dating that change is 2026-09-05. The banked
`f_contact` in `fan_bank_base_240w.npz` predates it.

**MEASURED** (`raw/p2_feasible_vs_contact.json`, control C1b): the per-step **POINT** test
reproduces the banked flag with **0 disagreements over 30,720 candidates** — a positive
identification of which definition produced the bank, not an observation that two numbers differ.
Every structural flag (`kamm_over`, `envelope`, `off_reach`, `infeasible`) reproduces exactly
under the current scorer; only `contact` differs, and one-sidedly (289 extra, 0 fewer).

| | POINT (banked, superseded) | SWEPT (current) |
|---|---|---|
| `fan_contact`, all 240 windows | 0.024870 (764 / 30,720) | **0.034277 (1,053 / 30,720)** |
| `fan_contact`, 65 lead windows | 0.095673 | **0.126562** |

⇒ **The generator emits 37.8 % MORE colliding candidates than the brief's headline states.**
Every number in this SPEC uses the **current swept** definition and says so.

---

## 1. P1 — the primary metric, and the BLOCKING floor check

**Primary: `fan_contact`** — the fraction of EMITTED candidates that collide, over the
lead-bearing population. Secondary (ranked-mass form): **`top32_contact`**.
⛔ Not `sel_contact`. ⛔ Not ADE.

The metric is bounded below by 0, so **the largest possible improvement IS the base value**. A
metric whose base sits below its own separation floor cannot report a win even if one occurs
(`mass_rank_contact` was `UNDETECTABLE-DOWNWARD` for exactly this reason). The check is therefore
blocking and is run **here, before any arm**.

Floors are the replicate-floor CIs banked in `veto_verdict_veto200.json` on the RL rig's own
eval population, which is the population the arm will report on.

⚠️ **Population, corrected in-place before any arm reported.** A first draft of this table wrote
*"72 lead windows of 120"*, inferred from the base value's denominator. The arm summaries carry
the count explicitly — `arm_summary.json::fan_safety_n["fan_contact"] = **36**` — so the
population is **36 lead-bearing windows of 120**, and `fan_contact`'s base is **449 / 4,608**
(36 × 128) while `top32_contact`'s is **38 / 1,152** (36 × 32). Both fractions are unchanged
(72/9,216 and 449/4,608 are the same ratio), so **no base value, floor or ratio in the table
below moves** — but the `n` is a fact the artifact states and the draft had inferred, which is
the trap this programme calls *"a number carries its arm and its artifact path"*. Recorded
rather than silently rewritten.

| metric | base (RL rig, n = 36 lead windows) | separation floor (replicate CI half-width) | base / floor | **admissible?** |
|---|---|---|---|---|
| **`fan_contact`** | **0.0974392** (449 / 4,608) | **0.0024740** | **39.4×** | ⭐ **YES — PRIMARY** |
| `top32_contact` | 0.0329861 (38 / 1,152) | 0.0 (degenerate; quantum 1/1,152 = 8.68e-4) | 38 quanta | **YES — secondary, floor must be re-measured** |
| `top8_contact` | **0.0000000** | 0.0 | — | ⛔ **NO — structural zero** |
| `sel_contact` | **0.0000000** | 0.0 | — | ⛔ **NO — structural zero** |
| `mass_rank_contact` | 0.0000347 | 0.0 | below floor | ⛔ **NO — UNDETECTABLE-DOWNWARD** |

⚠️ `top32_contact`'s replicate floor is **degenerate (exactly 0 with CI [0,0])** because the
replicate did not change the top-32 membership at all. A floor of exactly zero makes any
non-zero move "clear" it, which is an artefact of quantisation, not sensitivity. It is therefore
carried as a **secondary** metric and its floor is **re-measured from this panel's own replicate**,
never reused from the veto package.

⛔ **`fan_contact` is the only collision metric on this rig that is both non-degenerate and
above its floor. It is the primary endpoint.**

---

## 2. ⭐⭐ THE MECHANISM — why no RL arm has ever moved it

**MEASURED, 0 GPU** (`raw/p3_progress_binds.json`, `raw/p2_feasible_vs_contact.json`):

The optimiser is **GRPO-style**: the advantage is group-relative **across the candidates of one
window**. A window in which *every* candidate collides, or *none* does, contributes an
**identically zero** collision advantage.

| | count | share |
|---|---|---|
| windows with ≥1 collider **and** ≥1 non-collider (the only ones that can carry a collision advantage) | **19 / 240** | **7.92 %** |
| windows with 0 colliders (advantage identically 0) | 221 / 240 | **92.08 %** |
| episodes containing any such window | **14 / 121** | 11.6 % |
| among lead-bearing windows only | 19 / 65 | 29.2 % |

⇒ **92.08 % of every training batch contributes exactly zero collision gradient.** The signal is
not weak — it is *absent* almost everywhere and concentrated in a handful of windows (the worst
carries 104 of 128 candidates colliding, and the worst 5 windows carry 54.7 % of all colliders).

⭐ **CORROBORATED BY A SECOND, INDEPENDENT PROBE — the trainer's own counter.** This matters
because the `ls-tree` lesson is that repeated samples through ONE channel are one sample; the
figure above is an EVAL-side window count, so a TRAIN-side counter is a genuinely different
mechanism. `…/veto_run/run/s0/ctrl_null/arm_summary.json::counters.components_fired` reads
**`collision = 41`** against **`comfort = 200`, `feasibility = 200`, `progress = 200`** over 200
optimizer steps. ⇒ On the *training* clips the collision component fires on **41 / 200 = 20.5 %**
of steps while the dense terms fire on **100 %**. Two different populations, two different
denominators, one conclusion: **the collision channel is sparse and every other term is dense.**

**This is the pre-registered explanation for both the null result the veto arms produced and for
any null this panel produces.** It is committed here so that a null is diagnostic rather than
mysterious, and it names the successor (§6, S3) in advance.

---

## 3. P3 — does `progress` bind? (answered BEFORE the arm, 0 GPU)

Tie-safe **AUC** = P(score of a collider > score of a non-collider), ties 0.5.
**0.5 = no information · >0.5 = the term ranks COLLIDERS HIGHER, i.e. it BINDS.**
Controls: **K1** a constant term must read **0.5000 exactly**; **K2** the oracle (−contact) must
read **0.0000 exactly**. ⚠️ A first implementation using `argsort(argsort(·))` ordinal ranks
**FAILED both controls** (K1 read +0.6464, K2 read +0.0810) because the flag is ~97 % ties and
`np.argsort` is stable; it would have reported `progress` at +0.49 from a broken probe. Both
controls **PASS exactly** in the banked run.

| term | AUC vs contact (unprojected) | AUC (projected) | reading |
|---|---|---|---|
| **`progress`** | **0.8947** [0.8514, 0.9354] sep | **0.7206** [0.6541, 0.7907] sep | ⛔ **BINDS, strongly** |
| `feasibility` | 0.5084 [0.3550, 0.6606] | 0.5192 [0.5063, 0.5336] sep | marginal |
| `headway` | 0.1406 sep | 0.1487 sep | already disprefers colliders |
| `comfort` | 0.0261 sep | 0.0788 sep | already disprefers colliders |
| `collision` | 0.0000 | 0.0000 | the flag itself |
| **`_composed_default`** (the shipped weights) | **0.0000** | **0.0000** | ⭐ **already perfect** |
| `_composed_default` minus `progress` | 0.0000 | 0.0000 | unchanged by removing progress |

⭐ **THE ANSWER: `progress` binds in ISOLATION (AUC 0.89) but does NOT bind in the COMPOSED
reward.** With `collision` at weight 1.00 the composed reward ranks colliders last **perfectly —
AUC 0.0000, a structural zero** — and deleting `progress` changes that by **exactly nothing**.

⇒ **Repairing `progress` is NOT the lever for the collision objective**, and this panel takes the
brief's second branch: *demonstrate the collision objective works without touching it*. The
ranking is already correct; what is missing is **signal density** (§2), not ranking.

---

## 4. P2 — the feasible decode, ON vs OFF (answered BEFORE the arm, 0 GPU)

`stack/tanitad/refs/feasible_decode.py::project_feasible`, defaults (`mu=0.7`,
`clamp_entry=False`), applied to the banked fan and **re-scored with the current swept scorer on
both sides**. Controls: **C1a** structural flags reproduce the bank exactly ✅ · **C1b** bank
definition positively identified ✅ · **C2** `enabled=False` returns a bit-identical object
(max|diff| = 0.000e+00) ✅ · **C3** round-trip fixed point (2.13e-14 < 1e-5) ✅ · **C4** the
projection actually ran (27,431 / 30,720 = 89.29 % of candidates moved) ✅.

| metric (240 windows) | decode OFF | decode ON | delta [CI] | separated |
|---|---|---|---|---|
| **`fan_contact`** | 0.034277 | 0.036230 | **+0.001953** [−0.000034, +0.003992] | no |
| `top32_contact` | 0.008984 | 0.013672 | **+0.004687** [+0.001260, +0.009333] | ⛔ **yes, WORSE** |
| `fan_kamm_over` | 0.840625 | **0.000000** | −0.840625 | yes |
| `fan_envelope` | 0.887728 | **0.000000** | −0.887728 | yes |
| `fan_infeasible` | 0.891960 | 0.333301 | −0.558659 | yes |
| `fan_unsafe` (contact ∨ ttc) | 0.110319 | 0.098210 | −0.012109 | yes, better |

(lead-window population: `fan_contact` 0.126562 → 0.133774, ns; `top32_contact` 0.033173 →
0.050481, **separated worse**.)

⭐ **THE ANSWER: the feasible decode does NOT reduce the generator's collision rate — it slightly
INCREASES it, separably so on the ranked-mass form — while eliminating envelope and Kamm
violations ENTIRELY (both to exactly 0.000000).** The residual `infeasible` 0.333 is `off_reach`,
which the projection does not target at `clamp_entry=False`.

**Mechanism:** infeasible candidates were flying into geometry no vehicle can reach — frequently
*away* from the lead. Making them realizable pulls them back into the reachable set, which is
where the lead is.

⇒ **The brief's P2 premise is half right.** The projection *does* repair the upstream feasibility
breakage completely, so it is a clean substrate that removes the feasibility confound — but
**collisions are an orthogonal defect that the projection mildly worsens**, and the collision
objective must be attacked directly. ⛔ It follows that "run the RL stage on top of the
projection" cannot by itself be the fix, and a panel that only did that would report the
projection's feasibility win and miss that collisions went up.

---

## 5. THE ARM — one variable, and the contrast is already banked

⭐ **No arm in the existing panel has ever run a collision-punishing reward in isolation.**
`ARMS` in `stack/scripts/rl_refcv3_min.py` contains `rl` (all five weights — confounded),
`reg_echo`, `ctrl0` (lr = 0), `ctrl_const` / `veto200` / `veto2k` (all weights **0.0**, veto on)
and `ctrl_null` (all weights **0.0**, veto **off**). The PI's instruction — punish collisions —
has **never been run as a lever**.

**New arm `coll200`:** `weights={"collision": 1.0}` and **every other weight 0.0**, `w_anchor` 1.0,
lr 1e-5, 200 steps, `two_scalar` noise, `use_gt_bar=False`, **`veto_enabled=False`**.

⛔ **`one_variable` is satisfied exactly**: `ctrl_null` is `{k: 0.0}`, `w_anchor` 1.0, lr 1e-5,
200 steps, `two_scalar`, `use_gt_bar=False`, `veto_enabled=False`. `coll200` differs from it in
**the collision weight alone, 0.0 → 1.0**. Nothing else moves.

| arm | role |
|---|---|
| `coll200` seed 0, seed 1 | the lever; two seeds give the **replicate floor** |
| `ctrl_null` seed 0 (banked) + seed 1 | the **zero-information** floor, dose-matched |
| `ctrl0` (lr = 0) | ⛔ the **only admissible zero-lever floor** — a zeroed *loss* still moves every tensor because AdamW normalises by the gradient's own scale |

---

## 6. ⛔ BOTH OUTCOMES, COMMITTED IN ADVANCE

**SUCCESS** requires **all three**:
1. `fan_contact` **decreases**, separated at **both** seeds, **same sign**, and the smaller |Δ|
   exceeds **both** the replicate floor and the `ctrl0` zero-lever floor;
2. the `ctrl_null` zero-information arm does **not** drift `fan_contact` in the same direction by
   a comparable magnitude (if it does, the drift is the explanation, not the reward);
3. **ADE at T1 does not regress beyond the replicate floor.** ⛔ ADE is reported against its floor
   **whatever it does** — a collision win bought with an ADE regression is reported as a trade,
   never as a clean win.

**FAILURE** is anything else, and is reported as failure. In particular:
- `fan_contact` unchanged or worse ⇒ **the reward is not the missing piece**, and §2's
  signal-density measurement is the pre-registered explanation.
- `fan_contact` better on one seed only ⇒ **not reproducible across training runs** (`H-ESTIM-SEED-1`).

⛔ **A separated CI is NECESSARY, NOT SUFFICIENT** — `A0b_replicate` produced "separated"
differences on 3 of 18 metrics with **zero levers moved** (~17 % false-positive rate). Every
claim here is read against the replicate and `ctrl0` floors.

### P5 — named successors, in order, if the arm fails
1. **S1 — widen the collision term's horizon** beyond the 5-point / 2 s prefix.
2. **S2 — weight the collision term by time-to-contact** rather than the current binary −1/0
   (a binary term gives no gradient direction *within* the colliding set).
3. ⭐ **S3 — the collider curriculum.** §2 measures the failure mode as **concentrated**: 7.92 %
   of windows carry all of it and 92.08 % contribute exactly zero collision advantage. Mine the
   mixed windows into a training-time curriculum so the collision term is actually in the batch.
   **This is the successor §2 predicts will be needed, and it is named here before the arm runs.**

---

## 7. Estimator, tier, and what may be quoted

* **Tier: T0** on the emitted fan for every `fan_*` metric — a **WM/generator diagnostic, never a
  driving claim**. The ADE guard is **T1** (self-action open loop; never called closed loop).
* **Estimator: paired episode-cluster bootstrap** (cluster = clip), `taniteval/ci.py` via
  `rl_refcv3_min.paired_delta`. ⛔ Never `overlapping_holdout_se`.
* Every eval reports the **four metric families**, not ADE alone.
* Evidence class on every number: `MEASURED (ours + artifact path)`.
