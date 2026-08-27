# Design note — the representational arm (requested by the PI, 2026-08-27)

**Author** Master Mind · **Status** FOR PI REVIEW — no launch before approval ·
**Decides** which auxiliary the v7 line adds to attack the measured encoder defect.

---

## 1. The defect, as measured (not as theorised)

E-DEC-63 + F1–F4 (2026-08-26/27, arm `rdw8p30k`, 80 held-out clips, all controls
reading known values):

1. **Raw 32×80 grey pixels at t predict the latent's own future change beyond what
   the latent carries** — marginal over `z_t` = **+0.0096 (t 5.11, SURVIVES)**;
   robust to width (pixels+noise5120: +0.0131, t 9.10).
2. **Our predictor carries none of it** (−0.0023, null) — it sits at the ceiling of
   the current latent.
3. **The encoder's token field not only lacks the content, it DISPLACES it** —
   tokens-alone marginal **−0.0066 (t −3.95)**: drift-redundant token structure
   crowds the pixel signal out of the probe basis.
4. The component is **not photometric** (3-dim luminance control null; F4 spread
   across Δz PCs 0–7) — it behaves like scene structure.

⇒ **The encoder discards pixel-level content that is predictive of its own
dynamics, and fills the latent with drift-redundant structure instead.** This is
E-DEC-7's ego+noise optimum caught in the act, with a number on it.

## 2. What is already ruled out — do not re-propose these

| option | verdict | evidence |
|---|---|---|
| Re-weight the distillation term (O7 w-sweep) | ⛔ **REFUTED** | E-DEC-11: w=10 rejected on rank (3.31/3.59); w=50 reaches `n_agents` +0.2214 **by destroying participation (2.57/3.13) and ego** — fails the pre-registered kill-gate. Non-monotonic on single seeds |
| Freeze the encoder to protect content | ⛔ **REFUTED (D1)** | E-DEC-64: drift 0.3905 but nrmse +14.6 % — DEGENERATE band |
| Widen the readout (4×4 → 4×10/8×20 cells) **as the fix for THIS defect** | ⛔ **mis-aimed** | F3: the **pre-readout token field** (4×10-pooled) already lacks/displaces the content — pooling later cannot recover what the tokens never carried. *(Readout widening stays a live change for the localisation ceiling — a different defect.)* |
| Any term whose target is model-generated | ⛔ class-refuted | E-DEC-7; ten failures, O13's matched pair +192.4 % worse |
| More predictor capacity / horizon tricks | ⛔ wrong axis | E-DEC-63: predictor at ceiling; H-PROOF-5: content-improved arms did not predict better |

## 3. The two candidate designs

Both use an **external, data-side target** (the one property every failed term
lacked, and the one property the two things that ever injected content — DINOv3
distillation, pixel evidence — share).

### R1 — current-frame reconstruction auxiliary (`O14-rec`)

Small decoder from the token field to **32×80 grey pixels at t**; loss `w_o14 ·
L1`. The MAE-style classic.

- **For:** external target; directly forces pixel retention; cheapest to reason about.
- **Against:** retains pixel content **indiscriminately** — the JEPA critique
  (appearance/texture flooding) is a real risk to the participation gate, and the
  defect we measured is about the *predictive* component specifically, which
  reconstruction does not privilege. It optimises a superset of what we want.

### R2 — future-observation prediction (`O14-fut`) ⭐ recommended

Small head from `(z_t, conditioning)` to **32×80 grey pixels at t+k** (k = 4, the
measured horizon); loss `w_o14 · L1`. Optionally the pixel *difference* frame
`pix_{t+k} − pix_t` as the target variant.

- **For:** external **and temporal** — it asks the latent to retain exactly the
  content class E-DEC-63 caught it discarding: *what at time t predicts the
  observation at t+k*. It cannot be satisfied by static appearance (the constant
  part of the frame is free via `pix_t`-independence… and the difference-frame
  variant removes it outright). It is the measured finding turned into a loss.
- **Against:** future pixels are partly unpredictable (other agents' motion) — the
  head will regress toward blur; that is acceptable (the gradient still rewards
  keeping whatever IS predictive) but makes the raw loss value uninformative, so
  **the loss curve must never be read as the result** — only the probes below are.
- ⚠️ **Information-hygiene check (the 2026-08-03 rule):** the target is future
  OBSERVATION — data, not a model product and not an inference-time input; the
  conditioning stays whatever the arm already uses. No leak path.

## 4. ⭐ The success metric is already built — E-DEC-63's own probe

**A representational fix, if real, ABSORBS the pixel component into `z`:** the A3
pixel-marginal must fall from **+0.0096 (t 5.11)** toward **0 (inside the null)**
— because a latent that carries the content leaves pixels nothing to add. That is
a sharp, cheap, pre-registrable primary read, on an instrument whose controls are
already validated. No new estimator, no new defect surface.

## 5. Pre-registration sketch (to be formalised as PREREG_O14 before any launch)

```yaml
hypothesis: E-DEC-67 (to register on approval)
rig: v7-tiny ladder, 2k-step matched pairs first (E-DEC-9 pattern), Thor after A7
arms: incumbent (w=0) · R2 at w ∈ {0.1, 1.0} · R1 at w=1.0 (one arm, as the
      contrast that shows WHETHER the temporal form matters)
one_variable_per_pair: the auxiliary and its weight; all else byte-identical
deliberate_regression: R2 trained with TIME-SHUFFLED pixel targets — the
      absorption gate MUST NOT move. If it does, the gate is broken, not the arm.
primary_read: E-DEC-63 A3 pixel-marginal (must fall toward null)
gates_in_order (TanitAD_ValidateAIDesign):
  G-RANK   participation (sigma^2, val-side) not below incumbent
  G-DECODE n_agents / n_free_cols / occ vs pixel floor AND constant AND incumbent
  drift    must not rise above the incumbent's band
  then     meanpred nrmse (prediction must not degrade > 10% — the D1 lesson)
outcomes:
  ABSORBED:   marginal -> null, gates hold        => scale the aux into v7
  FLOODED:    marginal falls but G-RANK/ego fail  => R2 at lower w, or reject
  INERT:      marginal unchanged                  => the aux does not reach the
              encoder (check gradient path before concluding anything)
  DR-FIRES:   shuffled-target arm moves the gate  => instrument bug, read nothing
cost: ~4 tiny arms x ~20 min + probe reruns; ZERO full-scale GPU before gates
```

## 6. Decision requested

1. **Approve R2 (future-observation prediction) as the primary candidate**, with R1
   as the in-panel contrast — or direct otherwise.
2. The `w` grid {0.1, 1.0} and grey-32×80 target are my defaults; both are cheap to
   revise now and expensive to revise after arms exist.
3. On approval I formalise PREREG_O14, implement `O14-fut`/`O14-rec` behind flags
   (default 0 = bit-identical, test-pinned like `--cond-param`), and run the tiny
   ladder — **no 30k arm before the gates pass.**
