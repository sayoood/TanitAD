# PRE-REGISTRATION — does the budget freed by the anchor band buy the off-fan operator?

**Date:** 2026-08-04 (Europe/Berlin) · **Stream:** `arch-inf` · **Branch:** `agent/arch-inf-20260803`
**GPU cost: 0.** Dev-box CPU, banked fans only. ⛔ No pod touched, nothing pushed.

**This file is content-pinned with `git hash-object` BEFORE any statistic below exists,**
and the blob id is re-verified at the end of the pass. Every threshold, predicate and
outcome branch in §4–§7 is fixed here.

---

## 1. The two inputs, and why they are composable at all

| | A — off-fan refinement | B — anchor-band decode filter |
|---|---|---|
| doc | `Research/2026-08-04-planner-hierarchy-sota/PLANNER_HIERARCHY_SOTA.md` (`7d8ed27`) | `Implementation/incoming/2026-08-04-fan-width/FAN_WIDTH.md` (`be2da04`) |
| claim | HAD's 5-point radial grid on the **already-selected** trajectory, per-window oracle λ: `ade_sel` 0.4728 → **0.3138** (base), 0.4714 → **0.3064** (XL) | filter **anchors** through the reachability band ⇒ **92/256**, **46/128**, **23/64**; selection index identical on **881/881** |
| status | **CEILING with an ORACLE λ** | **bit-exact identity**, not a tolerance |
| n | 881 canonical windows / 40 val episodes | same |
| estimator | paired episode-cluster bootstrap | same |

Both authors flagged the composition **UNMEASURED** and costed it at ~10 min / 0 GPU.
This document registers it.

⚠️ **PRECONDITION ALREADY FOUND FALSE AND HANDLED (declared before the run).** The two
streams did **not** read the same file. A read
`…/2026-08-03-s1-climbout/raw/fan_emitted_refc-{base,xl}-30k.pt`; B read
`taniteval/results/fan_refc-{base,xl}-30k.pt`. MEASURED on XL before writing this prereg:
`fan` **maxabs 0.01373**, `logits` **maxabs 0.01234**, `gt` maxabs 7.63e-6 — **not
bit-identical** — while `v0` is bit-identical, `eid` identical, and the **selection index
agrees on 881/881** (`sel_A == argmax(logits_B)`), which is why both papers publish the same
0.4714 / 0.4728. ⇒ **The composition is therefore run on ONE bank (B's, the one `N_suff` and
the anchor buffers were measured against) and repeated on A's bank as `C-BANK`.** Composing
two numbers computed on two different dumps would have been a category error and is refused.

---

## 2. Data, estimator, parity

| | |
|---|---|
| banks (PRIMARY) | `taniteval/results/fan_refc-{base,xl}-30k.pt`; `…/2026-07-22-refc-small-30k/fan_refc-small-30k.pt` |
| bank (`C-BANK`) | `…/2026-08-03-s1-climbout/raw/fan_emitted_refc-{base,xl}-30k.pt` |
| anchors | `…/2026-08-04-fan-width/raw/refc_anchor_vocab.pt` (`decoder.anchors`, step 29,999) |
| lead block | `…/Benchmarks & Eval/…/2026-08-04-distance-keeping-arms/raw/val40_lead_block.npz`, row-aligned, **no re-inference**. Canonical split **LEAD 270 / NO_LEAD 551 / NO_LABEL 60** |
| n | **881 windows / 40 episodes**, all arms, all rungs |
| band | `tanitad.refs.refc_select.reachability_mask`, `accel_max = 2.5`, `horizon_s = 2.0` — **NOT tuned here**, taken from `flagship_v15` exactly as B did |
| operator | HAD (arXiv 2604.03581) λ ∈ {0.92, 0.96, **1.00**, 1.04, 1.08} × δ ∈ {−6, −3, **0**, +3, +6}° — **verbatim from A's code**, not re-parameterised |
| estimator | `taniteval.ci.paired_episode_cluster_bootstrap`, unit = **episode**, `n_boot = 2000`, `seed = 0` |
| ⛔ | `overlapping_holdout_se` is **never called** |
| 🔒 | parity untouched — no episode is re-selected; `Keys.txt` is never read |

---

## 3. Arms

`N_max` = 256 / 128 / 64 · `N_suff` = 92 / 46 / 23 (XL / base / small, from B).

| tag | decodes | what it is |
|---|---|---|
| `A0_full` | `N_max` | **the incumbent** — argmax over the full decoded fan |
| `A1_band` | `N_suff` | **B alone** — anchor-band FPS prefix, same argmax rule |
| `A3_full_plus_L` | `N_max` | **A alone** — radial λ on `A0`'s selected trajectory, per-window oracle λ |
| `A2_band_plus_L` | `N_suff` | **THE COMPOSITION** — radial λ on `A1`'s selected trajectory |
| `A4_band_plus_L_inband` | `N_suff` | the composition with λ variants that **leave the reachability band declared inadmissible** |
| `A5_band_plus_LA` | `N_suff` | the composition with the **full** HAD grid (λ × δ), oracle over 25 |
| `C_lat` | `N_suff` | **matched-DoF lateral control** — 5 angular offsets, oracle δ, same DoF as the 5 λ |

⚠️ `A2`–`A5` and `C_lat` are **ORACLES** (the ground-truth future picks λ / δ). They are
**ceilings, not results**, and are only ever compared ceiling-to-ceiling. Findability is a
**sibling stream's** question (E-EXP-2, λ\* from latents) and is **not** duplicated here.

---

## 4. Controls, each with its **DIRECTION** predicate (not merely a separation predicate)

| control | predicate fixed here | fires when |
|---|---|---|
| `C-identity` | λ = 1.00, δ = 0 must reproduce the selected trajectory **bit-identically** (`np.array_equal`) | any mismatch ⇒ **STOP, instrument void** |
| `C-upper-bound` | every oracle arm must be **≤ its own base** on **every one of the 881 windows** — `n_windows_worse == 0`, exactly | any window worse ⇒ the "oracle" is not an upper bound ⇒ **STOP** |
| `C-composition-exact` | `A2` per-window realised ADE must equal `A3`'s **exactly**, since `A1`'s selection is bit-identical to `A0`'s | inexact ⇒ **B's bit-exactness does NOT survive A's operator** (outcome O-C) |
| `C-sel-identity` | `A1`'s selection index == `A0`'s on 881/881, **re-derived here**, not inherited | < 1.0000 ⇒ B is not reproduced on this bank ⇒ report and stop composing |
| `C-band-admissibility` | fraction of oracle-λ refined trajectories **outside** the band, and the **direction** of the violation (λ > 1 accelerating vs λ < 1 decelerating) | reported unconditionally; feeds O-B |
| `C-rotation-is-band-neutral` | δ rotation preserves `‖wp_last‖` ⇒ preserves `candidate_mean_speed` ⇒ preserves band membership; asserted numerically | any change ⇒ the band/grid interaction is larger than modelled |
| `C-BANK` | the A-ceiling on `fan_emitted_*` must land within **±0.010 m** of the primary bank's | outside ⇒ the ceiling is bank-specific and neither paper's number is portable |
| ⛔ `C-shuffled` | **deliberately NOT used.** Permute-then-argmax over a candidate axis is a uniform random pick and is **vacuous by construction** here — the same reason B refused it | — |

⚠️ **ρ hygiene, binding.** If any correlation over the candidate axis is reported it is
reported **restricted to reachable survivors** (0.6125 XL / 0.5272 base), never the
full-axis 0.907. Three streams have tripped on this. This pass registers that it will
report **no new full-axis ρ at all**.

---

## 5. THE DECODE LEDGER — the primary deliverable, stated in decodes, not prose

Fixed here so it cannot be reverse-fitted. For each arm the ledger reports, per window:

1. **`n_decodes`** — decoder forward passes over candidates.
2. **`n_refine_evals`** — closed-form geometric transforms of an already-decoded
   trajectory (scale and/or rotate a `(4, 2)` array). **These are NOT decodes** and the
   ledger must not silently convert between the two currencies.
3. **`n_scored`** — how many *trajectories* a scorer must rank. ⚠️ The shipped selector's
   score is an **anchor-indexed logit** `[W, N]`; it is registered here as a **prediction**
   that it therefore **cannot score an off-fan point**, and that prediction is checked
   against the bank's own tensors.
4. **total vs the incumbent 256 / 128 / 64.**

⚠️ **Latency is explicitly NOT the win.** B measured 2.78× fewer decodes worth only
**6.5 %** end-to-end on Thor (decoder = 28.2 % of the frame, fixed-cost dominated). The
ledger is framed as **budget reallocation**, and any latency reading is refused in advance.

---

## 6. P2 — the four families, per family, never pooled

Reported for `A0`, `A1`, `A2`, `A4` on every arm.

| family | what is reported | instrument |
|---|---|---|
| **LONGITUDINAL** | target-speed (`speed_abs_err`, `speed_signed_err` **with no verdict**, a bias is a direction), `along_abs_err`, **and distance-keeping**: min-headway, min time-gap, min-TTC with `n_closing` | `taniteval.four_families`, `taniteval.lead_metrics` |
| **LATERAL** | `cross_abs_err`, `heading_abs_err`, **`curvature_abs_err`**, **`yaw_rate_abs_err`** | `taniteval.four_families` |
| **TACTICAL** | goal/anchor **selection** half — measured. Manoeuvre-**decision** half — `refc_rerank.dump` stores no decoded manoeuvre logits ⇒ **n = 0, a WORK ITEM, not a pass** | — |
| **STRATEGIC** | no route/goal label in a fan bank and the decode ran `nav_mode='follow_constant'` ⇒ **n = 0 of 881, a WORK ITEM, not a pass** | — |

`Δt` comes from `four_families.infer_dt` (**0.5 s** on `wp_steps = [5,10,15,20]`), never
hard-coded — a hard-coded 0.1 s inflates speed 5× and accel 25× (R-2026-08-03-c).

⭐ **The one powered safety read, registered in advance.** Speed-stratified, `min_stratum_n = 30`.
On val40 the **15+ m/s band is POWERED, n = 86** and shows a **1.82 s** mean min time-gap
against **3.15–3.51 s** mid-band. The genuinely starved band is **10–15 m/s (n = 8)**.
**If the operator moves the 15+ band's time-gap in EITHER direction, it goes in the headline.**
Threshold: a paired separated change of **any** magnitude on the 15+ band is headline-worthy;
a non-separated change is reported as such with its `n`.

---

## 7. OUTCOMES — both branches committed in advance

| # | branch | fires when | what I will write |
|---|---|---|---|
| **O-A** | ✅ **THE COMPOSITION IS FREE AND THE OPERATOR SURVIVES** | `C-composition-exact` holds (`A2 ≡ A3` per-window exactly) **and** `A1 − A2` is separated with Δ ≥ **+0.10 m** | The freed budget buys the operator at **strictly lower** total decode cost, and A's ceiling is untouched by B's cut. **The composition adds nothing to A's number — what it adds is that A is affordable.** Report exactly that; do not inflate it. |
| **O-B** | ⚠️ **MATERIALLY DOUBLE-COUNTED** | `A4` (band-admissible refinement) recovers **< 50 %** of `A2 − A1` | B's guarantee does **not** extend to A's output: a large share of A's ceiling is bought by leaving the physical band B enforces. Re-scope the joint recommendation and say so in the headline. |
| **O-B′** | ✅ **NOT double-counted** | `A4` recovers **≥ 50 %** of `A2 − A1` | The two operators are close to orthogonal; the joint recommendation stands. |
| **O-C** | ⛔ **BIT-EXACTNESS BREAKS** | `A2 ≠ A3` per-window | Report as a **refutation** of the composability premise, name the mechanism, and withdraw the joint recommendation. |
| **O-D** | ⚠️ **SAFETY REGRESSION** | the 15+ m/s band's mean min time-gap moves **separated-down** under `A2` or `A4` | The ADE ceiling is partly bought with headway. That goes in the headline **above** the ADE number. |
| **O-E** | ⛔ **CURRENCY MISMATCH** | the bank's `logits` are `[W, N]` anchor-indexed (i.e. no trajectory-conditioned scorer exists) | State plainly that freed **decodes** cannot be spent on the operator without a **scorer for off-fan points**, which no amount of freed decode budget provides. This is a structural finding, not a hedge. |

⚠️ **The null is a result.** If the honest answer is *"the composition adds nothing beyond A
alone"*, that is what gets written, in those words, and it is **not** reframed. Seven levers
have been killed on banked fans in this programme without a GPU-day; that is the standard.

---

## 8. What this pass will NOT do — declared before it starts

1. ⛔ **No findability probe.** λ\* from latents is a **sibling stream** (E-EXP-2). Not duplicated.
2. ⛔ **No retraining, no training pod touched, nothing pushed.**
3. ⛔ **No Thor latency run.** B already measured it; the ledger is in decodes, and the
   latency deflation (6.5 %) is inherited from B and labelled `INHERITED`.
4. ⛔ **No closed-loop claim.** Open-loop only; this programme has measured 0.45 → 1.69 m.
5. ⛔ **The band's own parameters are not tuned** (`accel_max = 2.5`, `horizon_s = 2.0`).
6. **TACTICAL manoeuvre-decision and STRATEGIC stay `n = 0`** — work items, not passes.

---

## 9. Deliverables

`TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-04-budget-composition/`
— `BUDGET_COMPOSITION.md`, this prereg, `code/budget_composition.py`,
`raw/budget_composition_*.json`, `raw/decode_ledger.json`, `raw/prereg_pin.json`.
**Staged, never committed, never pushed**; verified with `git ls-files --cached`, and
mirrored off-Drive.
