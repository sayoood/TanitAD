<title>PROPOSED_HYPOTHESES - SOTA pass: prediction / action sensitivity (theme 3 of 4)</title>

# Proposed register rows — theme 3 (action sensitivity), 2026-09-03

Register-ready rows for `GOALS_AND_CLAIMS.md`. **The Research Lab does not edit the register; the Master Mind applies these.** Each row has one variable, a success and a failure outcome committed in advance, the controls that must read known values, and a cost. Priors are PUBLISHED-PRIMARY from the PDFs banked for `2026-09-03-sota-action/RESULT.md` (library keys given). Every proposed read carries the constant-only control and the shuffled-action control (constitution §6.2); every estimator is the episode-cluster bootstrap over held-out episodes with the split fixed before the read.

Ordering recommended to the Master Mind: **LAB-ACT-1 → LAB-ACT-3 (both 0 GPU, same day) → LAB-ACT-2 (one v7-tiny arm) → LAB-ACT-4 (conditional, T1)**.

---

## LAB-ACT-1 — post-hoc action-mean centring on the banked census arms (0 GPU)

```yaml
hypothesis:
  id: LAB-ACT-1
  status: PROPOSED (Research Lab 2026-09-03; Master Mind applies)
  claim: >
    The action pathway of our 30k predictors is MASKED by the common mode (the scene-predictable part of the
    action's effect), not dead: subtracting the action-mean prediction exposes a live action-dependent residual.
  prior: >
    PUBLISHED-PRIMARY lib:2608.06706 (Dueling World Models): post-hoc centring on FROZEN RePo/TIA models raises the
    action-delta probe from -0.002..+0.052 to 0.19..0.52; trained-in vs post-hoc parity on DMC
    0.533/0.328/0.186 vs 0.528/0.323/0.192; the standard (non-anchored) channel collapses 1.28 -> 0.002 while the
    validation loss improves.
  one_variable: >
    the action-channel READ only: raw displacement d_raw = ||zhat(z,a) - z|| versus anchored displacement
    d_anc = ||zhat(z,a) - mean_k zhat(z,a_k)|| with K=16 actions drawn from the corpus action marginal
    (AD-JEPA Monte-Carlo mean). Nothing is retrained; same checkpoints, same windows, same encoder.
  arms: all 32 census arms of MODEL_REGISTRY.md section 13.0c (incl. postrain30k, postrain30k_freeze, splitp30k,
    rdw8p30k, emao14_30k, k4_30k), 40 held-out val episodes, h=1 (the only trained horizon, MM-E14)
  primary_read: >
    s_anc = d_anc(a_true) / d_anc(a_ROLL) per window (median over windows, episode-cluster bootstrap 95% CI),
    reported next to the raw MM-E10 ratio on the SAME windows and in action-side units
    (2026-09-02 scene-matched criterion), never as the raw scene/action quotient.
  success: >
    s_anc >= 3x the raw read on >= 1 arm with the CI excluding 1.0, shuffled-action control at 1.0 within CI,
    constant-only at 0.0000 -> the channel is masked; MM-E17/E18 is re-read as "masked, not dead";
    the first v7-tiny arm is the trained-in AD-JEPA offset head (supersedes LAB-ACT-2).
  failure: >
    s_anc within the CI of the raw read on ALL arms -> the channel is dead (FiLM deafness is a dead channel);
    LAB-ACT-2 is the next arm and the K>=2 rollout is its precondition.
  controls:
    - constant-only prediction must read 0.0000 on both d_raw and d_anc
    - shuffled-action (ROLL) must read 1.0 +/- CI on s_anc
    - K=16 vs K=64 action draws must agree within CI (estimator stability)
    - copy_detector re-run on every arm (must stay CLEAN)
    - printed n (windows) and d (latent dim) on every row
  cost: 0 GPU-h training; ~1-2 h of 4060/CPU inference over 32 arms x 40 episodes x K=16; no pod contact
  tier: T0
  ties_to: [GS-8 (anchored actdiv, running in parallel), MM-E10, MM-E17, MM-E18, E-DEC-33, section 13.0c]
```

## LAB-ACT-2 — ActSWM hinge + frozen random action readout as ONE v7-tiny variable

```yaml
hypothesis:
  id: LAB-ACT-2
  status: PROPOSED (conditional on LAB-ACT-1 failure; Master Mind applies)
  claim: >
    An explicit zero-action separation term on the PREDICTED rollout raises action sensitivity on the trainable
    v7-tiny line at parity prediction quality, where every prediction lever (k, O14, EMA, tau-ramp) left it at ~0.
  prior: >
    PUBLISHED-PRIMARY lib:2607.26712 (ActSWM) Table 5: step-31 cosine gap (GT - zero-action) 0.002 -> 0.592 with the
    frozen random readout alone, -> 0.760 with hinge + readout, at cosine-to-truth 0.923 (vs 0.972 without the term);
    Minecraft tasks 10/20 -> 19/20 (Table 8). Loss: max(0, cos(zhat_gt, zhat_0) - (1 - m)), lambda 0.5, m 0.3, K 12;
    readout lambda 1.0 on [z_t, z_{t+1}] for encoded AND predicted transitions.
  one_variable: the ActSWM term ON vs OFF (hinge + frozen readout as one switch), on the postrain30k recipe at
    30k parity, with o5_k >= 2 in BOTH arms (precondition: MM-E14 - the h>=2 heads are untrained on the current recipe)
  arms: [postrain30k_k2 (control), postrain30k_k2_actswm (treatment)]; seed 1 of each if the first read is within noise
  primary_read: anchored displacement s_anc (LAB-ACT-1 definition) at h=1 and at h=K; secondary: held-out nrmse,
    cos_ctr, E-DEC-59 drift (for the record only), T1 ADE closed-loop and hold-action
  success: >
    s_anc treatment >= 3x control (CI excluding 1.0), with nrmse within +2% of control, shuffled-action control at 1.0,
    and the deliberate-regression arm (hinge computed on shuffled zero-action pairs) NOT raising s_anc.
  failure: >
    s_anc unchanged (CI overlapping control) -> a predictor-side term cannot open the FiLM channel on a
    realised-motion action; P2(b) (a command channel) regains rank and the comma2k19 geometry decision is the blocker.
  controls:
    - constant-only 0.0000; shuffled-action 1.0 on s_anc
    - deliberate-regression arm: hinge on shuffled pairs must read as control
    - nrmse parity band +2% (the term must not buy sensitivity with prediction quality)
    - copy_detector CLEAN; echo 0.0000
    - GS-1 note: the readout is endpoint-shaped ([z_t, z_{t+1}]); report the Delta-z variant as a secondary read
  cost: one v7-tiny (~19M) 30k arm on the 4060 at bs 1-4 plus its control; per-step cost ~1.5-2x the parity arm
    (one extra zero-action K-step rollout per batch) - UNVERIFIED until measured on the first 200 steps
  tier: T0 for the read; T1 for the secondary
  ties_to: [P1, P2, GS-1, GS-8, MM-E14, 2026-09-01 curriculum recommendation (K>=2)]
```

## LAB-ACT-3 — checkpoint selection by the sensitivity read, not by nrmse (0 GPU)

```yaml
hypothesis:
  id: LAB-ACT-3
  status: PROPOSED (Research Lab 2026-09-03; Master Mind applies)
  claim: >
    Selecting the T1 candidate checkpoint by held-out nrmse picks an action-collapsed checkpoint; selecting by the
    anchored sensitivity read picks one that does better at T1 (closed-loop ADE, hold-action) at equal nrmse.
  prior: >
    PUBLISHED-PRIMARY lib:2608.06706: validation-loss checkpoint selection picked action-collapsed checkpoints
    17/36 times (Freeway) while the loss kept improving; lib:2606.09028 Table 2: rho(-L_pred, success) 0.4983 vs
    rho(-D_TT, success) 0.8130.
  one_variable: the checkpoint-selection criterion (min nrmse vs max s_anc) within ONE training run's banked
    checkpoints (emao14_30k or postrain30k, whichever has >= 6 banked intermediate checkpoints - UNVERIFIED count)
  primary_read: T1 ADE closed-loop and hold-action of the two selected checkpoints, paired over the 40 episodes
  success: the s_anc-selected checkpoint beats the nrmse-selected one on T1 ADE with the paired CI excluding 0,
    or ties on T1 while differing on s_anc by >= 3x (then the selection rule changes at no cost)
  failure: no T1 difference and s_anc equal within CI across checkpoints -> nrmse selection stands; the field's
    17/36 does not transfer to our line
  controls:
    - both checkpoints pass copy_detector
    - the T1 harness is unchanged (same episodes, same hold-action baseline)
    - if fewer than 6 checkpoints exist the row is NOT run (say so)
  cost: 0 GPU-h training; T1 re-eval of two checkpoints on the 4060
  tier: T1
  ties_to: [section 13.10 T1 block, LAB-ACT-1, MM-E13]
```

## LAB-ACT-4 — ACID planning-cost augmentation at T1 (conditional)

```yaml
hypothesis:
  id: LAB-ACT-4
  status: PROPOSED, CONDITIONAL on the T1 harness planning by a latent cost (UNVERIFIED in this pass)
  claim: >
    Adding an inverse-dynamics consistency term to the planning cost - with the world model untouched - raises
    closed-loop performance of a banked predictor, because it penalises action sequences whose imagined
    transitions do not carry them.
  prior: >
    PUBLISHED-PRIMARY lib:2607.02403 (ACID): Le-WM Cube 70 -> 74, Reacher 76 -> 88, Push-T 96 -> 100; PLDM 58 -> 68,
    76 -> 90, 72 -> 76 (Table 1); DINO-WM Rope chamfer 1.38 -> 0.56 (Table 2); lambda robust 0.005-0.1;
    +8.9-39% planning latency. IDM: flow-matching prefix-suffix transformer (4 layers, width 192), 200K steps on the
    frozen WM's latents, 1 Euler step.
  one_variable: the planning cost c = c_g + w_a c_a with w_a = lambda sigma_g / sigma_a, lambda in {0 (control), 0.02}
  arms: one banked predictor (emao14_30k) x two cost settings; the IDM is trained ONCE on its banked latents
  primary_read: T1 ADE closed-loop and hold-action, paired over episodes; secondary: fraction of planned actions
    within the corpus action marginal (ACID's OOD guard)
  success: T1 ADE improves with the paired CI excluding 0 at lambda 0.02, with the hold-action baseline unchanged
  failure: no change or a loss -> a planning-time consistency cost cannot substitute for an action-sensitive
    predictor on our line (consistent with ACID's own limitation that the IDM must identify the action from
    consecutive observations - which a context-collapsed predictor does not produce)
  controls:
    - lambda 0 reproduces the banked T1 numbers exactly (harness identity)
    - IDM accuracy on TRUE transitions reported (n, d printed) before any planning read
    - constant-action plan as the floor
  cost: IDM training on the 4060 (hours; 200K small-transformer steps on banked latents) + two T1 evals; 0 pod contact
  tier: T1
  ties_to: [section 13.10 T1 block, GS-1 (the IDM reads [z_t, z_{t+1}])]
```

---

### Not proposed, with reasons

- **LatentAlign anneal (backlog P-1)**: the primary reports no action-sensitivity metric (EPDMS only) and uses ego status at inference; keep P-1 as written but pre-register it on the anchored read, not on EPDMS.
- **UWM-JEPA counterfactual targets**: needs a simulator; not available on our corpus.
- **LDAD / SMWM / EB-JEPA IDM as arms**: encoder-shaping — inapplicable to refav1 (frozen trunk) and, on v7-tiny, they are the theme-2 adapter question (see `2026-09-03-sota-encoder/PROPOSED_HYPOTHESES.md` LAB-ENC-1) rather than an action-theme arm.
