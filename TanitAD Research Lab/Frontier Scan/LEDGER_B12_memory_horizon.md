<title>LEDGER B12 — memory, long context, state, temporal abstraction</title>

# LEDGER B12 — Memory · long context · state · temporal abstraction

⛔ **APPEND-ONLY.**

## Our position in this track

**2026-08-31.** The programme needs **two horizons at once**: tactical ~**6 s** and strategic ~**30 s**.
Today's MEASURED deployment constraint makes that a hard problem rather than a tuning question:
**rollout latency is LINEAR in K with no amortisation** — 1.08 ms/step at 7.1 M and 3.0–3.4 ms/step
at 37.8 M, flat within 6 % across K=1…120 — so a **flat strategic rollout (K≈300) does not fit at
any scale we measured**, and the tactical 6 s horizon fits only at tiny scale (70 ms of a 100 ms
budget at 7.1 M; **203 ms — over budget — at 37.8 M**).

⇒ **The strategic layer cannot be a longer rollout of the tactical one.** It needs either temporal
abstraction or O(1) state. That is what this track is for.

---

## Entry 2026-08-31-01 — three mechanisms, and one is aimed exactly at our problem

| paper | mechanism | fit to our problem |
|---|---|---|
| ⭐⭐ `2606.21775` **Variable-Length Latent World Models** | predicts future latents conditioned on **variable-length action sequences**, so *"the same predictor [can] evaluate action plans over different horizons"* | **This is our two-horizon problem stated as an architecture.** One predictor, many horizons — instead of a tactical model and a separate strategic model, or one model rolled 300 times. |
| ⭐ `2606.09828` **Latent Spatial Memory for Video World Models** | latent spatial memory over chunks; reported **10.57× faster** end-to-end generation and **55× less GPU memory** than RGB point-cloud methods | If the memory is *latent* rather than pixel/point-cloud, the cost model changes. ⚠️ **RELAYED numbers — banked, unread.** |
| `2605.01694` **Latent State Design under Sufficiency Constraints** | hierarchical decomposition across temporal scales; high-level latent plans guide lower-level transitions | Formal framing for the strategic→tactical interface, which we currently define by hand. |

⭐ **Combination with our own findings — this is where the transfer happens.** Today's Opponent
package found a **1.7 M-parameter Δt=4 "jump" model** recovering **1.02× GT motion magnitude** where
single-pass models capture *"less than half"*. `2606.21775` generalises that: **variable Δt instead
of a fixed one.** Two independent lines now say the answer to long horizons is **temporal
abstraction, not a bigger model or a longer chain** — and our own latency measurement says the
longer chain is not available to us anyway.

⚠️ **Risks, both halves.** (a) All three are **abstract-level**; no number here may decide a GPU-day.
(b) A variable-length predictor multiplies the training-time action-conditioning surface — and our
**measured** action/scene ratio at h=1 is already **0.004**. **Adding horizons to a predictor that
barely reads its action at one step could make the defect worse, not better.** ⇒ **The anti-echo
arms (backlog P-1/P-2) should land BEFORE any variable-horizon arm**, or the result is
uninterpretable.

**Proposed experiment:** read `2606.21775` in full; if its predictor is a drop-in for a fixed-h
predictor, register a v7-tiny arm at h ∈ {1, 4, 16} composed vs variable-length. **Pre-committed
read:** variable-length wins only if it beats *composed h=1* at matched compute — beating a single
long chain is not the comparison, because we already know the chain is the expensive option.

`Next: full-text 2606.21775; sequence it AFTER the anti-echo arms.`


---

## Entry 2026-09-01-01 - VLWM: one-step rollout is a TRAIN/PLANNING OBJECTIVE MISMATCH, not just error accumulation

**Source:** `arXiv 2606.21775`, "Beyond the Next Step: Variable-Length Latent World Models for
Long-Horizon Planning" (Du, Zhang, Wang, Wang; 2026-06-19).
**Evidence class:** PUBLISHED lib 2606.21775 **abstract-only - DECLARED.**
**Discharges the pass-1 debt "B12 full-text 2606.21775" at abstract level; full text still owed.**

> *"existing latent world models typically rely on one-step prediction and must be recursively rolled
> out for long-horizon planning, which leads to compounding errors and a mismatch between training
> objectives and downstream planning tasks."*

Proposes predicting future latents conditioned on **variable-length action sequences**.
**Claim: +13 % average over the SOTA LeWM**, larger gains where planning is extended.

⛔ **TENSION WITH A LIVE P0 ROW.** `LAB_BACKLOG` row 5 proposes dropping the h>=2 predictor heads for
composed h=1. Our evidence (H-PROOF-6d: heads never trained, 1e-3 init; H-PROOF-7: rolled h=1 predicts
to 6 steps) establishes the heads are **vestigial under the objectives we currently run** - which is
NOT the claim that multi-step prediction is the wrong design. VLWM argues the opposite and reports a
number.

⚠️ **Both can be true.** *Vestigial-because-untrained* != *useless in principle*. The asymmetry decides
the action: **dropping the heads is cheap; re-adding them after the v7 geometry is frozen is not.**
⇒ Lab recommendation (escalated, not imposed): execute row 5's simplification but **keep the head
slots in the config at 1e-3 init**, ESTIMATED cost ~zero, so the VLWM arm stays reachable.

`Next in this track: full text - extract how variable-length conditioning is trained, and whether it
needs a different data layout than our fixed-stride windows.`


---

## Entry 2026-09-02-01 — VLWM full text: FiLM conditioning is named as the mechanism that hides actions

`PUBLISHED lib 2606.21775 · FULL TEXT READ 2026-09-02 (abstract-only 2026-09-01)`

### ⭐⭐ The finding the abstract could not show — and it belongs to gate problem P-2

> *"**Architecture-locked action conditioning.** Because actions are fused inside transformer layers, if we
> provide multiple actions, they will be compressed into shared hidden representations, **which makes it
> difficult to disentangle their individual contributions to the predicted dynamics**. Consequently, the
> architecture itself dictates how many actions can be supplied."*

They name the interfaces explicitly: *"dedicated conditioning mechanisms (e.g., **FiLM layers** or
cross-attention modules)"*. VLWM replaces them with **action-as-token** — *"treats each action embedding as
a standard transformer token and interleaves action and state tokens within a single sequence"* — which
*"eliminates the need for a predefined action-conditioning interface"*.

⛔ **This re-opens a branch we closed.** Our P-2 investigation ELIMINATED zero-init FiLM as a cause on the
grounds that **"it trained"** (MM-E18, converged). VLWM's claim is compatible with that observation and
still damning: **FiLM can train perfectly well and still entangle the action's contribution.**
⇒ **"It trained" was never evidence that the conditioning interface is innocent.** P-2 branch (c)
*representation* re-opens with a named mechanism — and with Delta-JEPA's endpoint-leak (LEDGER_A2) it now
has **two** independent published mechanisms where a week ago it had none.

### ⭐ Independent corroboration of backlog L-1 (horizon curriculum)

> *"training with large prediction horizons from the outset is highly unstable. Long-horizon prediction
> tasks are substantially more difficult, often resulting in noisy gradients, optimization failure, and
> representation collapse."*

Fixed by a **cumulative-uniform curriculum** — stage j samples k uniformly from 1 to j — which they compare
against direct full-range training, finding *"the curriculum is essential for stable variable-length
training"*. ⇒ **A second independent line for L-1** (curriculum, not a smaller scalar clip) against our
k=60 divergence at clip 1.0 (gnorm median 5.71 to 2.1e9).

### Other load-bearing details

- **Backbone-agnostic:** *"the encoder f can be a JEPA-style, DINO-style, or any other latent encoder,
  and prior fixed-horizon world models are recovered as special cases of p(k)."* ⇒ composes with our trunk.
- **Planning:** partitions the horizon into contiguous chunks and evaluates the rollout as *"a few chunked
  variable-length latent jumps"* instead of a long autoregressive rollout. ⭐ **Directly relevant to P-7**
  (latency linear in K): fewer, larger jumps is a latency argument as much as an accuracy one.
- Planning is CEM-based latent MPC — the same framework as LeWM and V-JEPA-2, so comparisons are fair.

### ⚠️ An honest threat to H1, recorded as one

> *"By combining multiple horizons within a single rollout, VLWM further enables **hierarchical planning
> without requiring separate high-level and low-level policies**."*

Against HWM they argue *"HWM requires an additional high-level predictor, whereas our approach is
end-to-end and does not rely on auxiliary modules."*

⛔ **This is the clearest published challenge to the programme's hierarchy thesis logged so far.** It is an
**argument**, not an ablation against a hierarchy — but it must be filed as a challenge, never as support.
It belongs in the H1b design (backlog row 18): a variable-horizon single predictor is a **legitimate third
arm** in the hierarchy-vs-flat comparison, and arguably the strongest flat baseline available to us.

### Consequence for live rows

- **FS-2 STRENGTHENED** — keep the h>=2 head slots at 1e-3 init while executing backlog row 5. The
  asymmetry argument now has a mechanism and a number (13 % over LeWM) behind it.
- **L-1 corroborated** by a second independent primary.
- **backlog row 18 (H1b)** gains a required third arm.
