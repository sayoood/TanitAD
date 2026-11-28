# E-CR is MIS-SPECIFIED for our architecture — and the reason is a real finding

**2026-07-29. MEASURED on v1 `flagship4b-speedjerk-30k` step 29,999, `val40cache`, 4 episodes /
48 windows.** Artifacts: `ecr_sweep.py`, `ecr_control.py`, `zhat_vs_ztrue.py` (banked here; live on
pod3). ⛔ **C61 REMAINS OPEN.**

## The verdict in one line

**`step_readout` cannot decode displacement from TRUE latents. Therefore CR built on decoded
displacement measures how far you pushed the readout out of its domain — not the cost of recursion.**

## The three-arm control that settled it

All three arms decoded by the SAME `grounding.step["op"]`, on the SAME windows, same actions:

| arm | pair fed to the readout | k=4 | k=8 | k=16 | k=20 |
|---|---|---|---|---|---|
| **A rollout** (the canary) | `(z_hat_{j-1}, z_hat_j)` — self-consistent PREDICTED | **0.04491** | **0.08582** | **0.29171** | **0.53253** |
| **B teacher-forced** | `(z_true_{j-1}, z_hat_j)` — MIXED | 0.06160 | 0.13581 | 0.44242 | 0.70582 |
| ⛔ **C oracle** | `(z_true_{j-1}, z_true_j)` — self-consistent TRUE | **1.51451** | **2.94486** | **5.46839** | **6.76310** |

⭐ **Arm C — which contains NO PREDICTION AT ALL, only ground-truth latents — is 12–34× WORSE than
the model's own recursive rollout.** A readout that were a general "latent pair → displacement"
function would score arm C near zero. It scores 6.76 m at 2 s.

⇒ The readout is **not** a general decoder. It is tuned to the predictor's own output distribution.
Arm B sits between A and C exactly as a half-out-of-domain input should.

## Why — the latent geometry, MEASURED

| quantity | value |
|---|---|
| `cos(z_hat, z_true_next)` | **0.98377** |
| `cos(z_hat, z_last_ctx)` | **0.99872** |
| `cos(z_true_next, z_last_ctx)` | **0.97980** |
| `cos(encode(x), encode_window(x)[:, -1])` | 0.999995 *(same space — ruled out as a bug)* |

Consecutive latents are **0.98–0.999 cosine-similar**: the space is highly concentrated and one frame
of motion is a *tiny* vector against the embedding's magnitude. The readout must extract metric
displacement from that tiny difference, so it is exquisitely sensitive to the difference's exact
distribution.

- `z_hat − z_last_ctx` is a **small, learned-consistent** displacement — the readout's training domain.
- `z_true_next − z_last_ctx` is a **different** small vector, dominated by real frame-to-frame content
  (lighting, texture, scene change) that the predictor deliberately does **not** model.

⭐ Note also `cos(z_hat, z_last_ctx) = 0.99872 > cos(z_hat, z_true_next) = 0.98377`: **the 1-step
prediction is closer to "stay put" than to the true next latent.** The predictor is not trying to
reproduce the encoder's next output; it produces a state the *readout* can decode.

## Consequence — the SkyJEPA transfer fails at its premise, for us

CR_k assumes rollout and teacher-forced arms are **exchangeable inputs to the same decoder**. Ours
are not. The published instrument is sound; **our architecture violates its precondition**, which is
exactly the class of thing CLAUDE.md's "the metric design transfers, the magnitudes do not" warning
was pointing at — and it turns out even the design does not transfer unmodified.

⛔ **Do NOT report the earlier CR values (0.729 / 0.632 / 0.659 / 0.755) as evidence about
compounding.** They are an artifact of readout-domain mismatch.

## The redesign that IS well-posed — E-CR v2

Move CR off the decoder entirely and onto **latent error**, which both arms can carry:

```
e_k = 1 − cos(z_hat_k, z_true_k)        (or ||z_hat_k − z_true_k|| / ||z_true_k||)
CR_k = e_k,rollout / e_k,teacher-forced
```

- **rollout arm**: `z_hat_k` from a window advanced with its own predictions.
- **teacher-forced arm**: `z_hat_k` from a window advanced with true latents — i.e. a genuine
  1-step-from-truth prediction at every k.

Both are the predictor's *own* outputs compared against the encoder's, so no decoder is involved and
no distribution is crossed. This measures precisely what C61 asks: **does feeding predictions
forward degrade the prediction itself?**

⚠️ Carry the caveat: a latent-space CR answers the *world-model* question but no longer speaks
directly to metres of ADE. The link from latent error to trajectory error runs through the readout,
which we have just shown is non-linear and domain-sensitive. **Do not convert one into the other.**

## Status of the pre-registration

`PREREG_deep_research_2026-07-29.md` registered two outcomes (CR ≈ 1 ⇒ H-TASK; CR > 1 ⇒
H-COMPOUND). **Neither fired.** The result landed outside both, which the prereg did not anticipate
and which I am recording rather than forcing into the nearest registered box.

**Still in force:** C61's retraction stands; the 1.03 → 1.91 exponent rise may not justify an
architecture change; **E-ROLL, rollout-recovery training and the Koopman lever remain BLOCKED** —
now because E-CR has no admissible number, not because it returned flat.

## Evidence class

| claim | class |
|---|---|
| the three-arm table | **MEASURED** — n=48 windows / 4 eps, no interval (a >10× effect needs none to be directional) |
| the four cosine values | **MEASURED** — 8 windows / 2 eps |
| `encode ≡ encode_window` | **MEASURED** |
| "the readout is tuned to the predictor's output distribution" | **INFERRED** from A/B/C ordering — consistent with every measurement, not independently proven |
| E-CR v2 will be well-posed | **HYPOTHESIS** — it removes the decoder, which is the demonstrated confound |
