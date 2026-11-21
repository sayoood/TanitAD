# PREREG_V8F_KHYP: several longitudinal hypotheses through O, then pick one (written 2026-10-25, before any KHYP data)

**Status: pre-registered.** Zero model training. Eval-only on the BANKED M2b s0 / s1 checkpoints, on the dev box's CPU, reading the eval cache that `PREREG_V8F_SFRESH` fetches and HBLEND's training-window cache. It runs under `PI-V8F-2026-10-25b` and the PI's 2026-10-25 go on the proposed ideas, *"our priority is to proof very quickly dominance"*.

## Why (MEASURED / INHERITED, named)

* The longitudinal decision is the weakest tactical output (MEASURED): lon macro-F1 0.473 / 0.532 and kappa 0.33–0.37, against lateral 0.84.
* An oracle per-window speed scale is worth 0.20–0.39 m (`H-V8F-LON-1`, **[I]** from the register; not re-verified here). This is the "multimodal decision" defect.
* O was teacher-forced on GT codes whose longitudinal part is a normalised multi-hot (`_code_gt`, `al(lab["lon_al"], N_LON)`). **A one-hot longitudinal code is in O's training distribution.**

⇒ Enumerating the longitudinal class through O gives K = 7 plans per window with no training, and asks two questions:
* How much is a perfect choice among them worth?
* How much of that can a deployable selector capture?

## The arm

For each window, T's own code with its longitudinal slice (indices `N_GOAL + N_LAT` to `+ N_LON`, 7 classes) replaced by the one-hot of class k, for k = 0..6. Goal, lateral and constraint slices stay T's own. O runs through `forward_diag(code_override=…)`, the eval's own diagnostic path, to give $\tau_k$.

**Selectors** (each pre-declared; inputs are model outputs only, so they are vision-only and contain no situation-classifier output):

| selector | rule | tier |
|---|---|---|
| **ORACLE** | $k^\ast = \arg\min_k \mathrm{ADE6}(\tau_k)$ against the GT | T0 diagnostic: headroom, ⛔ never a result |
| **SEL-ARGMAX** | $k$ = T's longitudinal argmax | T1 |
| **SEL-MIX** | $\sum_k p_k \tau_k$, with $p$ = T's longitudinal softmax | T1 |
| **SEL-LEARN** | a multinomial logistic regression (L2 = 1.0, fixed; standardised features; no tuning on eval) predicting $k^\ast$, then $\tau_{\hat k}$. Features: T's longitudinal softmax; per hypothesis, the plan's final speed, minimum and maximum acceleration, arc length, and its violation of T's own predicted constraints (CONSISTENCY(b) per rule). Fitted on HBLEND's training windows (9,582 windows, 141 clips), per seed | T1 |

## Controls

1. With $k$ = T's argmax and a one-hot that equals T's code where T's longitudinal softmax is one-hot, the plan equals `forward_infer`'s exactly (Δ = 0.0).
2. **Reproduction:** B's own-code ADE6 on CPU equals the banked value (tolerance stated; > 1e-3 m stops the read).
3. **Deliberate regression:** SEL-LEARN fitted on labels permuted across windows must not separate from SEL-ARGMAX.
4. ORACLE ≤ every selector on every window, by construction (checked).

## The reading (T1-class open loop, eval-141, paired episode-cluster bootstrap, B = 2,000; F = 0.1207 m; the eval's own `evaluate()` and `families()`)

| read | definition | bar |
|---|---|---|
| **HEADROOM** (T0, descriptive) | ADE6 ORACLE − B | reported; it sizes the lever |
| **PRIMARY** | ADE6 SEL-LEARN − B | **≤ −F = −0.1207 m and separated, at BOTH seeds** |
| **SECONDARY** | speed MAE and accel MAE, SEL-LEARN − B | not separated-worse at either seed |
| beside | SEL-ARGMAX and SEL-MIX − B; all four families; CONSISTENCY(a); SEL-LEARN stacked with H's banked weights (descriptive); the selector's accuracy against $k^\ast$ | descriptive |

| outcome | condition | next (committed) |
|---|---|---|
| **MULTIMODAL PAYS** | PRIMARY and SECONDARY at both seeds | a trained K-hypothesis head (T emits K codes, O plans each, a learned scorer selects) becomes the next box-2 arm, with this result as its prior. The stacked HS / H + SEL-LEARN is scored against P10 and FLAT on the dominance row |
| **HEADROOM, NO SELECTOR** | ORACLE − B ≤ −2F, but PRIMARY fails | the alternatives exist but the cheap features cannot pick them. Next: a scorer that reads the scene (a trained head; box 2) |
| **NO HEADROOM** | ORACLE − B > −2F | enumerating the longitudinal class does not move the plan enough; O does not use the longitudinal code strongly. Then `H-V8F-LON-1`'s speed-scale lever is not reachable through the code, and the next arm targets O's speed profile directly |

⛔ No class set, feature, regularisation, selector or bar changes after any KHYP number is seen.
