# WP-M — REF-C's prior port is wired, helps generically, and carries almost no scene-specific content

**TanitAD_TrainingFlyWheel · 2026-09-06 · Tier T0, NON-PARITY pilot.**
Evidence class **MEASURED (ours)**. `n = 1,360` windows · 34 episodes · K = 128 anchors ·
checkpoint `refc-base-30k` · verified join.
Code `code/wpm_port_influence.py` · raw `raw/wpm_port_influence.json`, `raw/wpm_full2.log`.
Tests the mechanism of `DIALOGUE_07` §1–§2 on a real trained model at **zero training cost**.

---

## 0. Method — the guidance delta without touching internals

`decoder.maneuver_to_anchor` is a **zero-initialised** `Linear(5 → 128)` that adds to the
selection logits — structurally the same port the Energy Bridge writes through. Run the
same window twice:

```
  S0 : the port ZEROED    == exactly what a zero-init arm produces at step 0
  S1 : the port as trained
  Δ  = S1 − S0            the per-anchor guidance delta, by construction
```

Setting `energy = −Δ` at `β = 1` reproduces `S1` **identically** (asserted in the run), so
the banked instrument `tanitad.instruments.cot_influence` measures the real model with no
special-casing. The port's trained L2 is **6.3675**, so there is a real effect to measure.

---

## 1. The port is NOT inert — §2's first premise holds

```
  ZERO-PORT IDENTITY   flip 0.0000   kl 0.000000   geo 0.0000 m    <- exact, not approximate
  the TRAINED port     flip 0.2324   kl 0.187789   geo 0.2022 m
```

A trained zero-init port changes the selected anchor in **23.2 %** of windows and moves the
chosen trajectory by **0.20 m**. Because the zero-port identity is exact, that effect is
attributable to the port and to nothing else. **The channel TanitLang writes through works.**

---

## 2. ⛔ The KL screen said the opposite — and the KL screen is the wrong axis

```
  real            kl 0.187789
  norm-matched    kl 0.406544     <- a RANDOM prior at the same per-window spread
  permuted prior  kl 0.193963     <- a REAL prior, attached to the wrong window
  PASSES BOTH: False
```

REF-C's genuinely trained prior **loses the KL contest to random noise by more than 2×.**

⚠️ This is not evidence the port is empty; it is evidence that **magnitude cannot answer
this question**. At matched spread a random direction generically diverges further than a
structured one, so KL cannot separate *moved meaningfully* from *moved hard*. My own
`semantic_null_screen` imported the VLADriveBench failure mode correctly and
**operationalised it on the wrong quantity** — their finding is about the *behavioural
effect*, not the size of the nudge.

⇒ `outcome_null_screen` added: every arm re-ranks, takes its argmax, and is scored by the
**metric it actually achieved**. `semantic_null_screen` is retained and re-scoped in its
docstring — **it can show a prior is inert; it can never show a prior is empty.**

---

## 3. ⭐ On the outcome axis the verdict flips — and then the decomposition matters more

```
  base (no port)      ADE 0.6636 m
  real prior          ADE 0.6521 m
  permuted prior      ADE 0.6534 m     <- a REAL prior, WRONG window
  norm-matched null   ADE 0.7636 m     <- intervenes as hard, says nothing
```

| comparison | margin | reading |
|---|---:|---|
| real − norm-matched | **+0.1114 m** | a *real-prior-shaped* signal is far better than a random one of the same magnitude |
| real − permuted | **+0.0012 m** | ⛔ **the window-specific content is 1.2 mm** |
| base − real | **−0.0115 m** | the port's total benefit, CI95 **[−0.0417, +0.0184]**, overlaps 0 |
| base − permuted | **−0.0102 m** | a WRONG-window prior captures **89 %** of that benefit |

> ⛔⛔ **The printed verdict — "beats both nulls, the channel carries information" — is
> true and misleading, and must not be quoted alone.** The port beats a *random* prior
> decisively (+0.1114 m). It beats a prior *from a different window* by **+0.0012 m**.
> **≈89 % of everything the port achieves is achieved by a prior belonging to some other
> scene.**

⇒ the mechanism carries a **generic bias** — something about the *shape* of a real
manoeuvre prior helps re-ranking regardless of provenance — and **almost no scene-specific
information**.

⚠️ **And the window-specific part is not separable at this n.** The port's *total* effect
already has a CI of [−0.0417, +0.0184] straddling zero; the window-specific component is an
order of magnitude smaller than that interval's half-width. **+0.0012 m is a point estimate
with no evidence of being non-zero**, and it is reported here as a bound, not a finding.

---

## 4. ⭐⭐ The hard design constraint — the most useful number this produced

```
  sd_a[Δ]           1.457     the port's spread across the 128 anchors
  sd[S0]           17.907     the base scorer's spread
  ratio              ~8 %
  Spearman(Δ, S0)   0.096     nearly orthogonal to the scorer
```

`consistency(energy, sel)` reads mean rank **64.397 of 128** against a chance of **64.5**,
top-1 **0.0000**.

⛔ **That is NOT a finding that REF-C is inconsistent.** The metric asks whether the energy
*alone* ranks the chosen anchor first, and a small additive prior cannot do that by
arithmetic, whatever it knows. Reporting it as a REF-C defect would be the same scope error
this campaign has produced repeatedly — so it is reported as a **calibration**:

> **`CON_rank` reads chance whenever `β·E` is small relative to `sd[S0]`.** §2's consistency
> half is therefore **only measurable if the reasoner is given real authority over the
> decision** — `β` large enough that `β·sd_a[E] ≈ sd[S0]`, or an energy that *replaces*
> rather than *nudges* the scorer. **A timid zero-init port that stays timid produces an
> unfalsifiable consistency claim**, which is precisely the failure mode §2 exists to prevent.

⭐ **Spearman 0.096 is the encouraging half**: the trained port is nearly *orthogonal* to
the base scorer, i.e. it contributes **new ordering rather than an echo**. That is the
property the Energy Bridge needs, and it is already present in the existing mechanism.

---

## 5. What this changes in the design

| | |
|---|---|
| ✅ **§2's channel** | **Validated.** A zero-init additive port measurably re-ranks (23.2 % flips), is exactly inert at init, and is orthogonal to the base scorer. |
| ⚠️ **§2's influence metric** | **Corrected.** `INF_kl` is necessary and not sufficient and can rank a real prior *below* noise. The deciding screen is `outcome_null_screen`. |
| ⛔ **§2's consistency metric** | **Conditional on scale.** Unmeasurable below roughly `β·sd_a[E] ≈ sd[S0]`; the design must state its intended β and check this ratio *before* claiming consistency. |
| ⛔ **The bar TanitLang must clear** | **A wrong-window prior is the real baseline, not the no-port arm.** It captures 89 % of the existing port's benefit. Any reasoner claim must beat the **permuted** control, not merely the base. |
| ⭐ **Why the reasoner is motivated** | The existing prior carries *manoeuvre-class* information only (5 classes → 128 anchors) and that turns out to be nearly scene-independent. The headroom is untouched: the fan holds an anchor at **0.2315 m** while the model ships **0.65 m**. |

---

## 6. Scope

* **T0, NON-PARITY pilot, one checkpoint, one inference seed.** No metric family, no driving
  claim. Not a T1 number.
* **One port.** `refc-base-30k` carries only `maneuver_to_anchor`; `lat_to_anchor` /
  `lon_to_anchor` / `lan_gate` are refcv3+ and absent here. The result is about *this* port's
  content, not about the port *family*.
* ⚠️ **The permuted-prior margin (+0.0012 m) has no interval.** The run did not retain the
  per-window arrays needed for a paired bootstrap on that specific contrast; the bound in §3
  is argued from the total effect's CI. A dedicated re-run with the contrast bootstrapped is
  the clean form and is cheap.
* **The ADE effect of the port is not separated from zero** in either direction
  (CI [−0.0417, +0.0184]). "The port helps" is a point estimate, not a result.


---

# !!! CORRECTION (2026-09-06, same night) -- SECTION 3's "89 % generic" IS RETRACTED

**Superseded by `RESULT_WPN_PORT_CONTRASTS.md`.**

Section 3 concluded that a prior "from a different window" captures ~89 % of the port's
benefit, and inferred that the port carries almost no scene-specific information. **The
control was not a control.** `roll_control(energy, shift=1)` mispairs by INDEX, and these
rows are accumulated episode by episode:

> MEASURED: **roll-1 paired a window with the SAME EPISODE in 97.5 % of rows** -- usually
> the adjacent timestep. A prior from 0.1 s earlier in the same clip is very nearly the
> RIGHT prior, so the control was ~97.5 % a no-op.

With an honest **cross-episode** control (partner drawn from a different episode,
rejection-sampled per row with an assertion that no same-episode pair survives):

| paired contrast | delta (m) | CI95 | |
|---|---:|---|---|
| real - cross-episode  **SCENE-SPECIFIC** | **-0.0254** | **[-0.0425, -0.0093]** | **SEPARATED** |
| real - roll-1  *(the retracted number)* | -0.0012 | [-0.0098, +0.0068] | overlaps 0 |
| real - base | -0.0113 | [-0.0416, +0.0175] | overlaps 0 |
| cross-episode - base | +0.0141 | [-0.0097, +0.0363] | overlaps 0 |
| norm-matched - base | +0.0838 | [+0.0600, +0.1087] | SEPARATED |

=> **the port's effect DOES depend on the prior matching the scene**, separated at 0.0254 m.
The retracted reading had it backwards.

!! What still stands from Section 3: **"the port improves on no prior at all" remains
UNESTABLISHED** (`real - base` overlaps zero). And most of the separated gap comes from a
mismatched prior being HARMFUL rather than a matched one being helpful -- see WP-N Section 3.
