<title>PROPOSED_HYPOTHESES - SOTA pass: latent drift over the rollout (theme 1 of 4)</title>

# Proposed register rows — theme 1 (drift), 2026-09-03

Register-ready rows for `GOALS_AND_CLAIMS.md`; **the Master Mind applies them, the Research Lab does not edit the register.** One variable each, both outcomes committed, controls that must read known values, cost. Priors are PUBLISHED-PRIMARY from the PDFs banked for `2026-09-03-sota-drift/RESULT.md`.

Recommended order: **LAB-DRIFT-1 (0 GPU, instrument) → LAB-DRIFT-2 (one v7-tiny arm) → LAB-DRIFT-3 (deferred)**.

---

## LAB-DRIFT-1 — the field's drift on our banked arms (instrument; L-17 / L-18)

```yaml
hypothesis:
  id: LAB-DRIFT-1
  status: PROPOSED (Research Lab 2026-09-03; Master Mind applies)
  claim: >
    The persistence-normalised rollout divergence D(h) = E||zhat_{t+h} - z_{t+h}||^2 / E||z_t - z_{t+h}||^2
    orders our banked arms DIFFERENTLY from the E-DEC-59 statistic r(dz | z_t): the one-variable freeze arm,
    which has the lowest E-DEC-59 drift (0.3905) and the worst nrmse (+14.6%), has the HIGHEST D(h).
  prior: >
    PUBLISHED-PRIMARY lib:2608.24044 (JEPA-x): the field's drift is rollout-vs-encoded-future divergence,
    persistence-normalised, mean over h=1..20; a state-regression head raises decodability (R2 0.978 -> 0.991)
    without moving drift (0.373 vs 0.361) - decodability, forecastability and control are separable.
    lib:2607.16314 caution: unnormalised rollout similarity rewards slow latents.
  one_variable: the drift INSTRUMENT (E-DEC-59 statistic vs D(h)) on the same checkpoints and the same windows;
    nothing is retrained
  arms: postrain30k, postrain30k_seed1, postrain30k_freeze, splitp30k, emao14_30k, k4_30k (banked; md5s in
    MODEL_REGISTRY.md section 13.9/13.10), 40 held-out val episodes
  reads:
    - (a) trained predictor, autoregressive rollout on recorded actions, D(h) for h=1..8
      (h>1 requires the rollout path; the h>=2 heads are untrained on the current recipe, MM-E14 - report h=1
      separately from h>1)
    - (b) JEPA-x forecastability of each ENCODER: fresh 3-layer MLP predictor (512, context 3, residual,
      30k steps) fitted on disjoint episodes, then D(h) for h=1..20
    - (c) the E-DEC-59 statistic on the same windows (for the rename)
  success: >
    outcome A - D(h) orders the arms like held-out nrmse (freeze worst) with episode-cluster bootstrap CIs
    separating freeze from the trainable line -> E-DEC-59 was the outlier; L-18 renames it "increment
    predictability", P3 is re-based on D(h).
  failure: >
    outcome B - D(h) orders the arms like E-DEC-59 (freeze best) while nrmse says the opposite -> the JEPA-x
    dissociation holds on our line; P3 is a diagnostic and leaves the v7f gate.
  controls:
    - persistence must read exactly 1.0 (by construction; print it)
    - constant-only prediction must read the constant value (D = ||mean - z_{t+h}||^2 / persistence; print it)
    - shuffled-action rollout must be reported next to the recorded-action rollout on every arm
    - printed n (windows), d (latent dim), h grid on every row; split fixed before the read
  cost: 0 GPU-h of training for (a) and (c); (b) is six small MLP fits on banked latents (minutes each on the 4060
    or CPU); no pod contact
  tier: T0
  ties_to: [P3, E-DEC-59, E-DEC-64, E-DEC-66, E-DEC-67, L-17, L-18, section 13.9, section 13.10]
```

## LAB-DRIFT-2 — training-only cross-prediction to ego kinematics (JEPA-x lever) as ONE v7-tiny variable

```yaml
hypothesis:
  id: LAB-DRIFT-2
  status: PROPOSED (conditional on LAB-DRIFT-1 delivering the instrument, and on the Master Mind confirming the
    labels-may-use-ego rule for training-time targets - INHERITED, not re-read in this pass)
  claim: >
    A cross-predictive head - the visual latent predicts the future ego-kinematic state (a, kappa) and a
    kinematic stream predicts the future visual latent, both training-only, inference vision-only - lowers D(h)
    at parity nrmse where a plain regression head (O14-style) does not.
  prior: >
    PUBLISHED-PRIMARY lib:2608.24044: multi-task drift 0.361 -> 0.104, control 53.6 -> 78.2%; single-task
    Two-Room 0.503 -> 0.221, Block 0.507 -> 0.269, Push-T 0.399 -> 0.258, Reacher 0.313 -> 0.233; CROSS-ONLY
    0.101 vs SHARE-ONLY 0.309 vs ALIGN-ONLY 0.158; REGRESS control 0.373 (no drift gain despite R2 0.991);
    action-conditioned next-state/increment heads 0.386/0.359 (no gain).
  one_variable: the cross-prediction head ON vs OFF on the postrain30k recipe at 30k parity; a third arm with
    the REGRESS-style head only (the published null) is the deliberate-regression control
  arms: [postrain30k (control), postrain30k_cross (treatment), postrain30k_regress (published-null control)]
  primary_read: D(h) from LAB-DRIFT-1 read (a) at h=1 and h=4 (rollout path required in all arms - o5_k >= 4);
    secondary: held-out nrmse, cos_ctr, decodability of (a, kappa) with constant + raw-pixel floor, anchored
    action read (LAB-ACT-1 definition), E-DEC-59 statistic for the record
  success: >
    D(h=4) treatment < control with the episode-cluster CI excluding 0, nrmse within +2% of control, and the
    REGRESS arm showing NO D(h) gain (as published) while its (a, kappa) decodability rises -> the JEPA-x
    dissociation reproduces on our corpus and the lever is registered for v7f.
  failure: >
    no D(h) change in the treatment arm, or a change that the REGRESS arm reproduces -> the lever does not
    transfer (our targets are not "privileged physical state" in the JEPA-x sense); P3 stays a diagnostic and no
    further drift arm is registered.
  controls:
    - constant-only 0; shuffled-action rollout reported; persistence 1.0
    - decode-target rule: (a, kappa) only, v excluded (Ledger decode-target rule, INHERITED)
    - inference path verified vision-only (no kinematic input at eval; assert in the harness)
    - copy_detector CLEAN on all three arms
  cost: two v7-tiny (~19M) 30k arms on the 4060 at bs 1-4 (treatment + regress control), each ~ the parity
    arm's wall-clock plus a small head; 0 pod contact
  tier: T0 (T1 secondary if the treatment passes)
  ties_to: [P3, E-DEC-67 (O14 null = published REGRESS null), LAB-DRIFT-1, Ledger decode-target rule]
```

## LAB-DRIFT-3 — parallel action-prefix prediction instead of recursion (deferred)

```yaml
hypothesis:
  id: LAB-DRIFT-3
  status: PROPOSED, DEFERRED (an interface change, not a flag; only after LAB-DRIFT-1/2 and the K>=2 rollout exist)
  claim: >
    Predicting all future latents in parallel from z_t and action prefixes (no recursion) lowers the open-loop
    divergence slope and the planning latency at equal one-step quality.
  prior: >
    PUBLISHED-PRIMARY lib:2606.26217 (Fast-LeWM): lower and slower-growing open-loop latent error on four tasks;
    success 85.8 -> 90.5 (92.0 with self-consistency); dynamics time 31.4 -> 8.0 s; Table 4 dense prefix
    supervision 98/88/96/80 vs terminal-only 96/80/90/72 vs recursive Long-Action LeWM 76/70/80/58.
    lib:2608.29029 (Flow-JEPA, whole-trajectory flow matching): success 86 -> 92 clean, 67 -> 86 noisy, no drift
    metric, 8 Euler steps at 5.0 ms each.
  one_variable: the prediction interface (recursive K-step vs prefix-parallel) at equal predictor size
  primary_read: D(h) slope over h=1..8 (LAB-DRIFT-1 instrument); secondary: T1 latency per plan, T1 ADE
  success: slope lower with CI excluding 0 at nrmse parity; latency not worse
  failure: no slope change -> recursion is not our divergence mechanism; the interface stays
  controls: [persistence 1.0, constant-only, shuffled-action, copy_detector]
  cost: one v7-tiny arm plus predictor-interface work (days of engineering, UNPRICED); deferred
  tier: T0 / T1
```

---

### Not proposed, with reasons

- **EMA on/off as a drift arm**: no published prior for a drift effect; our own read (§13.10) already shows EMA buys prediction, not drift. Keep EMA for prediction.
- **Larger o5_k**: the field's optimum is K = 2 (sim) / 6 (real) with a proven accuracy–robustness trade-off; our k = 4 was inert and k 8 → 60 lowered the ratio. No arm.
- **Depth regularisation (`2607.16314`)**: single-run evidence, needs a depth target, and its rollout gain is on a similarity that rewards slow latents; not before LAB-DRIFT-1.
- **Sub-JEPA subspace SIGReg**: its anti-drift claim is qualitative (Fig. 5); its success gains are on toy environments; the encoder-side change belongs to theme 2.
