# PRE-REGISTRATION — state-conditioned anchor sets for the v5 fan

**Written 2026-07-27, BEFORE a single anchor set was fitted.** Frozen at commit-time of this
file. Anything measured after this point is adjudicated against the bars below and nothing
else.

**Honest ordering note.** One measurement in this stream was taken *before* this file was
written: §S1, the descriptive question *"is the existing fan already `v0`-conditioned?"*. It is
a property of an artifact already committed to the repo, it adjudicates nothing, and it moved
no bar. Every **adjudicated** number — ceiling and realised pick — comes after this file.

---

## 1. The question, split in two because they have different answers

1. **THE CEILING.** Does a `v0`-conditioned anchor set lower `oracle_in_fan` — the best
   achievable pick — relative to a fixed anchor set of the **same per-window size**?
2. **THE REALISED PICK.** Does it lower what a *realisable rule* actually achieves?

⚠️ **A better ceiling the selector cannot reach is not a win.** Both are reported or the
result is unreadable.

## 2. Why the already-published REFUTE does not answer this

`…/2026-07-27-fan-clip-local/FAN_CLIP_LOCAL.md` measured a longitudinal **admissibility
filter** (the brief's priority-1 item) and returned **REFUTE**: at `a_max = 2.5` it deletes
**85.4 %** of the fan and moves the as-trained selector by **+0.0020 m [−0.0016, +0.0076]**,
not separated.

**That result stands, is reproduced in §S0 of this stream, and is NOT the same experiment.**

| | filter | conditioned anchor set |
|---|---|---|
| operation | **removes** candidates | **reallocates** candidates |
| effect on `oracle_in_fan` | can only **worsen** (measured: 0.1640 → 0.1683) | **can improve** |
| per-window proposal count | falls 256 → 37.5 | stays **N** |

A filter cannot raise a ceiling — it is a subset operation. **Only reallocation can.** The
filter's REFUTE therefore constrains the *realised* half of this question and says nothing
about the *ceiling* half. Both are re-asked here.

## 3. Bars — committed in advance, NOT moved afterwards

Inherited from `Project Steering/V5_PLAN.md` §5 and V5 §3.3, both re-derived from raw JSON in
§S0 before use:

| bar | value | provenance |
|---|---:|---|
| **CONFIRM** | realised pick **< 0.4907** | Bar A's in-sample ceiling of *any* re-scoring of this fan |
| **STRONG** | realised pick **< 0.4271** | v1 deployed, `flagship4b-speedjerk-30k` |
| current best training-free realised pick | 0.5645 | C2, v1's WM, v4's fan |
| current as-trained (v4 fan) | 0.8563 | |
| current as-trained (REF-C-XL fan) | 0.4714 | |
| current ceiling (v4 fan / REF-C-XL fan) | 0.2505 / 0.1640 | |

**Outcomes, committed:**

- **CONFIRM** — a conditioned anchor set moves the realised pick below **0.4907**
  (**STRONG** below 0.4271).
- **PARTIAL** — the **ceiling** moves materially (separated) but the realised pick does not
  ⇒ named a *proposal* result carrying a **selection gap**, and the gap is quantified.
- **REFUTE** — neither moves ⇒ the fan was **not** the binding constraint either, and the
  honest conclusion is that the 2 s open-loop selection surface is exhausted. **Reported as
  cleanly as a win. No re-scoping.**

## 4. ⚠️ An analytic bound I commit to BEFORE measuring, because it constrains the answer

Every realisable training-free rule available on a counterfactual anchor set is
**nearest-anchor-to-a-reference-trajectory** (C1 = constant velocity, C2 = one world-model
roll, re-quantisation of a deployed planner's answer). For such a rule, as the anchor set
densifies around the reference, the pick converges **to the reference**. Therefore:

> **A nearest-to-reference rule's error is floored at the reference's own error.** C2's
> reference is v1's world model, whose own ADE is **0.4271**. ⇒ **C2 on ANY anchor set,
> however good, can at best TIE the STRONG bar and can never beat it.**

⇒ The realised-pick question for this family is *exactly*: **does conditioning cut the
quantisation tax (measured at 0.5645 − 0.4271 = 0.1374 m on v4's fan) by more than
(0.5645 − 0.4907) / 0.1374 = 53.7 %?** That is the number this stream must produce.

## 5. What I CANNOT do, stated now rather than discovered later

**The as-trained selector cannot be evaluated on a counterfactual anchor set.** Its head is
trained against the specific anchor indices of the fan it shipped with; a new anchor set
requires retraining, which is a GPU-week and is exactly what this hours-scale ladder exists to
gate. **I will not fake it.** The realised half is therefore answered for the *training-free
transferable* family — which is the family that produced the program's best zero-training
result (C2, 0.5645, separated-better than the as-trained selector). Any claim about a
*retrained* selector is out of scope and will be labelled so.

## 6. Design — fitted OUT-OF-FOLD, episode-disjoint

- **Corpus.** The canonical **881-window / 40-episode** val deployment already committed to
  the repo. **No episode is re-selected**; parity (`e438721ae894`, skip-hash `f09e44db`) is
  untouched because no episode is ever opened.
- **Folds.** 5 **episode-disjoint** folds (8 val episodes held out each). Anchors, bucket
  edges and cluster weights are fitted on the train fold ONLY and applied to the held-out
  fold. In-sample is reported alongside out-of-fold; **only out-of-fold adjudicates.**
- **Anchor sets**, k-means over train-fold GT trajectories in the 4×2 waypoint space:
  - `A_fixed(N)` — one set of N anchors, used in every window.
  - `A_cond(N, B)` — B `v0`-quantile buckets (edges from the train fold), N anchors per
    bucket; a window is offered **its bucket's N anchors**.
- **Two matched comparisons, both reported:**
  - **equal per-window proposals** — `A_cond(N,B)` vs `A_fixed(N)` (the deployment-relevant
    comparison; both offer N);
  - **equal total storage** — `A_cond(N,B)` vs `A_fixed(B·N)` (the anti-privilege control;
    conditioning must not win merely by holding more anchors).
- **Estimator.** Paired episode-cluster bootstrap, `taniteval/ci.py`, **B = 2000**, unit =
  episode. `overlapping_holdout_se` is never called.

## 7. ⚠️ What makes each rule return a FAILING value — and the proof that it can

Required by the brief; this program has shipped vacuous diagnostics.

| rule | FAILS when | proof it can fail |
|---|---|---|
| **ceiling** | `oracle(A_cond) − oracle(A_fixed) ≥ 0` | ▸ **negative control NC1**: bucket windows by a **shuffled** `v0` (permuted across windows). Conditioning on a variable carrying no state information *must* return Δ ≈ 0 or worse. ▸ **NC2**: bucket by pure Gaussian noise. Both are run at every N. |
| **realised** | pick ≥ 0.4907 | v4's fan already returns 0.5645 and REF-C-XL's as-trained 0.4714 — the scale straddles the bar in both directions, so the bar is not trivially passed or trivially failed. |
| **premise (§S1)** | fan speed slope on `v0` ≈ +1 | GT's own slope is **+1.0003** on the identical windows and is computed by the identical code path — so the instrument demonstrably *can* report a conditioned fan. |
| **fidelity** | oracle-reference rule ≠ ceiling | positive control: `pick_nearest_to(GT)` must reproduce `oracle_in_fan` to floating point. |

## 8. Confidentiality and hygiene

🔒 PhysicalAI-AV is gated-confidential: **no clip UUIDs, no raw content** in any artifact.
Episode identifiers appear only as opaque integers already present in committed dumps.
Deliverables are **staged, never committed, never pushed**.
