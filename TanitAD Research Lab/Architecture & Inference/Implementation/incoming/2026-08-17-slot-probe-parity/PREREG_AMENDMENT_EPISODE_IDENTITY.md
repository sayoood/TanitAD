# ⛔ PRE-REGISTRATION AMENDMENT — the control list does not cover EPISODE IDENTITY

**Amends:** `…/incoming/2026-08-16-agent-slot-decoder/AGENT_SLOT_DECODER.md` **§1.4 / §4.3**
**Date:** 2026-08-17 · **Author:** slot-probe-parity · **Branch:** `agent/arch-inf-20260803`

> ⚠️ **WHEN THIS WAS WRITTEN, AND WHY THAT IS THE WHOLE POINT.**
> This amendment was made **while the 130-clip corpus was still downloading**, and **NO arm had
> been fitted on it.** The only fits that existed at the time were (a) the pipeline null control on
> **random latents**, and (b) two re-runs of the **2026-08-16** cache used to prove the edited
> instrument still reproduces that run bit-for-bit. **The outcome this amendment could affect had
> not been observed.** An amendment made before the result is legitimate; the identical change made
> after would not be, and would have to be labelled a post-hoc control. Recorded here rather than
> folded silently into the method so a reader can check that ordering rather than take my word.

---

## 1. The defect in the pre-registration

§1.4 / §4.3 register four controls — **C-CONST**, **C-SHUF**, **C-TOK**, **C-V5F**. Between them
they cover *"the head measured nothing"*, *"the head echoes a corpus prior"*, *"the readout grid is
the bottleneck"*, and *"O2/O3/O4 bought nothing over v5f"*.

**None of them covers a head that has learned to recognise WHICH EPISODE it is looking at.**

### 1.1 Why that is a live threat on this data — MEASURED, not hypothesised

The GT in-corridor lead gap is far more stable within an episode than across the corpus
(`raw/results_*.json → c_shuf_discriminability`, label-side only, computed before any head existed):

| | 2026-08-16 sample | this run's sample |
|---|---|---|
| **within-episode** GT gap SD | 4.249 m | **3.855 m** |
| **between-episode** GT gap SD | 6.972 m | 6.239 m |

⇒ A head that emits *"this episode's typical gap"* is already close to right, **without ever
locating an agent.** Episode identity is trivially available to it: the encoder sees the road, the
weather, the vehicle ahead's colour, the time of day. This is not an exotic failure mode — it is the
cheapest thing a 3.2 M-parameter head can learn from a frozen appearance latent.

**MEASURED, and it settles that this is not a theoretical worry:** on the 2026-08-16 windows, a
predictor that knows *only* the episode's own leave-one-out mean gap scores **3.899 m**, against
**7.283 m** for the global constant. ⇒ **Episode identity is worth 3.4 m of headline on that data.**

### 1.2 ⛔ AND C-SHUF IS STRUCTURALLY BLIND TO IT

C-SHUF permutes the memory **within an episode**. Every window it swaps in comes from the *same*
episode, and therefore implies the *same* episode mean. **An episode-identity reader scores
IDENTICALLY under C-SHUF** — the control reports "no echo" while an echo is precisely what is
happening. This is the C13 family: a guard structurally unable to detect the thing it would be
cited for.

---

## 2. The amendment — two controls, and they are NOT interchangeable

| control | what it does | what a null on it rules out |
|---|---|---|
| **C-EPMEAN** *(new)* | the **leave-one-out mean GT gap of the window's own eval episode**. An **ORACLE** — it reads eval labels — so it is a **CEILING on the episode-identity strategy**, never a legitimate baseline. | **HOW MUCH** of an arm's score episode recognition explains. An arm that beats C-CONST but **not** C-EPMEAN has shown nothing about agents. |
| **C-SHUF-XEP** *(new)* | the trained head, weights unchanged, memory taken from a **DIFFERENT episode** (episode blocks cycled by one; deterministic). Destroys episode identity **and** window identity. | **WHETHER THE HEAD READS ITS INPUT AT ALL.** Δ ≈ 0 ⇒ pure prior, the input is unused. |
| **C-SHUF** *(existing)* | memory permuted **within** the episode. Destroys window identity, **preserves** episode identity. | whether anything **varying inside an episode** is used. ⚠️ Blind to episode identity. |

⭐ **The decomposition is the point, and it is why one control could not replace the other:**

* C-SHUF Δ ≈ 0 **and** C-SHUF-XEP Δ ≈ 0 ⇒ the head reads **nothing**; it is a prior.
* C-SHUF Δ ≈ 0 **but** C-SHUF-XEP Δ < 0 (arm better) ⇒ the head **does** read its input, but only at
  **EPISODE granularity** — scene recognition, not agent perception. **C-EPMEAN then says how much
  of the headline that accounts for.**
* C-SHUF Δ < 0 ⇒ the head uses information that varies **window to window**, which is the only
  regime in which an agent-perception claim is even available.

### 2.1 The derived test

> **K5 — an arm that beats C-CONST but NOT C-EPMEAN has shown nothing about agents.**

⛔ **K5 is NOT pre-registered, and it NEVER gates the KEEP decision.** KEEP stays exactly as
registered: K1 ∧ K2 ∧ K3. K5 is an **attribution** test, and it only has to be read **when K1
passes**.

---

## 3. ⚠️ WHAT THIS DOES **NOT** DO — the 2026-08-16 result is not retroactively threatened

**The gap bites only on a POSITIVE result.** The 2026-08-16 run returned the head **WORSE than a
constant** (K1 **+9.84 m** @9000, every interval separated). Episode-identity leakage can only make
a head look *better* than it is; it cannot manufacture a head that loses to a constant by 9.84 m.

⭐ **And the new control confirms that directly rather than by argument.** Re-scored on that run's
own banked cache with C-EPMEAN attached:

| | value |
|---|---|
| arm `cells` @9000 | **17.124 m** |
| C-CONST | 7.283 m |
| **C-EPMEAN** (the episode-identity ceiling) | **3.899 m** |
| Δ arm − C-EPMEAN | **+13.224 [8.846, 17.314]** separated ⛔ |

The 2026-08-16 head was **13.2 m worse** than a pure episode-identity predictor. It was not
exploiting episode identity; it was not exploiting anything. ⇒ **D1 as reported on 2026-08-16
stands, and this amendment is not a retraction of it.**

⚠️ **Where it bites is the run this amendment precedes.** Better-powered, lead-enriched data is
exactly the condition under which a head could finally clear C-CONST — and a head clearing C-CONST
is exactly the result whose attribution C-SHUF cannot settle. The control is being installed
immediately before the experiment that could need it.

---

## 4. Instrument equivalence — proved twice, around the change

`sp2_probe.py` was edited to add these controls. Both **before** and **after** the edit it was
re-run on the 2026-08-16 banked cache at that run's own settings, and reproduces it exactly
(`raw/results_REGRESSION_vs_20260816.json`, `raw/results_REGRESSION_post_amendment.json`):

| | published 2026-08-16 | pre-edit re-run | **post-amendment re-run** |
|---|---|---|---|
| Δ `cells` − C-CONST | +9.8401 [5.6855, 13.7524] | +9.8401 [5.6855, 13.7524] | **+9.8401 [5.6855, 13.7524]** |
| Δ `cells` − C-SHUF | −0.0569 [−2.0867, 1.7501] | −0.0569 [−2.0867, 1.7501] | **−0.0569 [−2.0867, 1.7501]** |
| median abs err | 17.22278 m | 17.22278 m | **17.22278 m** |

The training loss trace is identical line for line in all three (step 0 `80.3456`, 200 `10.7594`,
400 `12.8752`, 600 `10.8874`). ⇒ **The added controls are inert on the existing scoring path.**

---

## 5. Escalation

⛔ **This belongs in the pre-registration itself, not only in this stream's package.** §1.4's
control list should gain C-EPMEAN and C-SHUF-XEP, and §4.3's C-SHUF entry should carry the
"preserves episode identity" caveat. **I have not edited `AGENT_SLOT_DECODER.md`** — it is another
stream's document and amending it is an integrator decision, which is why this file exists and is
named in the report's escalations.
