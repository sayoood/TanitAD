# SPEC — refav1 frozen-trunk PERCEPTION PROBE (H-REFAV1-PERCEPT-1)

**Written 2026-09-06, BEFORE the full-bank probe was scored.** The feature bank
(cheap, reversible, no free parameters) was already building; no probe number on
the 193-clip bank existed when this file was written. The only numbers seen
beforehand are the 30-clip smoke, which is recorded below as EXPLICITLY
UNDERPOWERED and is not the basis of any bar.

## The question the PI asked

> *"Can we decode environment information from the world model to assess its
> environment understanding capability by training a perception head based on the
> frozen trunk and visualize its results?"*

## Why it is load-bearing — the discriminating logic

refav1's lateral half is settled. What remains is **entirely longitudinal**, and
longitudinal driving *is* distance-keeping to a lead agent. Two facts collide:

* a **perfect goal makes the planner 2.03x WORSE**, and the search optimises
  *better* while driving worse ⇒ the **cost is misspecified**;
* but **a cost is defined IN THE LATENT**.

⇒ If the frozen trunk does not encode where the lead vehicle is, then **no cost
in that space CAN express "keep distance"** — the objective is
**UNREPRESENTABLE**, not mistuned. Different defect, different fix, and the two
are not distinguishable from any planner-side measurement. This probe separates
them.

```yaml
hypothesis: H-REFAV1-PERCEPT-1   # NEW; H-REFAV1-COLLAPSE-2b is the OPEN row it discharges
                                 # ("decodability and T1 not yet read on this checkpoint")
one_variable: the FEATURE SOURCE   # constant / raw pixels / frozen DINOv3 / refav1 field
held_constant: [corpus, clips, frames, split seed, pooling grid, PCA budget,
                final d, lambda grid, lambda selection rule, estimator, n_boot]
controls: [constant_only, raw_pixel_floor, frozen_dinov3_reference,
           within_clip_time_shuffle, alignment_offset_sweep]
splits:
  fit:   60% of CLIPS (episode-disjoint, seeded)
  inner: clip-grouped 5-fold CV carved from FIT only -> selects lambda
  score: 40% of CLIPS -- scored, NEVER tuned on
function_class: ridge (LINEAR); RFF/RBF kernel ridge for the nonlinear read
estimator: paired episode-cluster bootstrap over SCORED clips, 2000 draws
```

## Arms, and the known value each control must read

| arm | what it is | what it must read |
|---|---|---|
| `constant` | predicts the FIT-split mean | **EXACTLY +0.000000.** It *is* the denominator's predictor. If it is not exactly 0 the metric is mis-implemented and nothing else on the panel is readable. |
| `pix` | raw decoded frame, same pipeline | the **FLOOR**. A learned representation that does not beat raw pixels **has added nothing**. |
| `dino` | frozen DINOv3 ViT-L/16 patch tokens | the **EXTERNAL REFERENCE** — the encoder without any refav1 training. |
| `field` | refav1 `_last_state` — **the latent the cost lives in** | the arm under test. |
| `shuffle_within_clip` | targets permuted WITHIN each clip | **~0.** `gap` is strongly autocorrelated within a clip, so a probe can score merely by RECOGNISING THE CLIP. Structure surviving this shuffle is clip identity, not perception. |
| alignment sweep | target shifted -2..+2 rows | **best offset must be 0**, or the feature/label alignment is wrong and every number is meaningless. |

## Metric

`R2_skill = 1 - SSE_model / SSE_fitmean` on the SCORED split, so the constant arm
reads exactly 0 by construction. Reported alongside it, per cell:
`r2_within_clip` (clip-mean-centred) — the honest *"does it track the lead as it
MOVES"* number, which a between-clip-only effect cannot fake.

## Committed outcomes — BOTH, in advance

**PASS (trunk DECODES the lead).** `field` beats BOTH `pix` and `constant` on
`lead_gap_m`, with the **paired** field−pix delta's 95 % CI excluding zero, AND
`shuffle_within_clip` ~0, AND the alignment sweep peaking at offset 0.
⇒ **The latent is adequate; refav1's longitudinal defect is in the PLANNER/COST.**
The fix is cost/search-side and the representation is exonerated.

**FAIL (trunk does NOT decode the lead).** `field` fails to beat the raw-pixel
floor on `lead_gap_m` (paired CI spans or is below zero).
⇒ **The objective is UNREPRESENTABLE in this latent** and the fix is upstream of
the planner entirely — no cost re-weighting can recover it.

⚠️ **Third admissible outcome, named in advance so it cannot be reported as
either of the above:** `field` beats `pix` but NOT `dino`. That is neither
exoneration nor indictment of the *encoder*; it says refav1's **trained adapter
destroyed** lead information the frozen encoder had, which is a THIRD defect with
its own fix (adapter/objective), and it must be reported as such.

⛔ **A separated CI is necessary, not sufficient** (`H-ESTIM-SEED-1`). This
bootstrap answers *"would another draw of EPISODES say this?"* — not *"another
training run"*, not *"another inference run"*. Here the trunk is **frozen** and
the head is a **closed-form ridge solve**, so there is no training or inference
stochasticity to price: the episode draw and the SPLIT SEED are the only
randomness, and the split seed is swept. This is stated in the result, not
assumed away.

## What this delivers regardless of outcome

The **lead track**, joined to our own episodes — which unblocks the
**distance-keeping** metric the four-families rule requires and refav1 has never
reported (the eval currently reads `UNAVAILABLE, n = 0`).

## Underpowered smoke (recorded so it cannot be mistaken for the result)

30 clips / 12 scored clips, n=124 rows on the capped-gap cell. `constant` read
**+0.000000** on all five targets and the within-clip shuffle read
**+0.0001..+0.0005** — i.e. the *instrument* checked out. Every arm value from
that run is inadmissible as a result and no bar above was set from it.
