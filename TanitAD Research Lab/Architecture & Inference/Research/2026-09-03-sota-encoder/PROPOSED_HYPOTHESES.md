<title>PROPOSED_HYPOTHESES - SOTA pass: encoder quality (theme 2 of 4)</title>

# Proposed register rows — theme 2 (encoder quality), 2026-09-03

Register-ready rows for `GOALS_AND_CLAIMS.md`; **the Master Mind applies them, the Research Lab does not edit the register.** One variable each, both outcomes committed, controls that must read known values, cost. Priors are PUBLISHED-PRIMARY from the PDFs banked for `2026-09-03-sota-encoder/RESULT.md`.

Recommended order: **LAB-ENC-1 (0 trunk compute) → LAB-ENC-2 (0 GPU if intermediate checkpoints exist) → LAB-ENC-3 (v7f arm, deferred)**.

---

## LAB-ENC-1 — can a frozen DINOv3 trunk be shaped through an adapter? (the inverse-dynamics multiplier test)

```yaml
hypothesis:
  id: LAB-ENC-1
  status: PROPOSED (Research Lab 2026-09-03; Master Mind applies)
  claim: >
    An inverse-dynamics objective trained through a small per-token adapter on FROZEN DINOv3 features raises the
    linear decodability of realised ego motion (a, kappa) from the adapter's dz - i.e. the frozen trunk carries
    temporal-predictive structure that a frozen-compatible shaping objective can multiply.
  prior: >
    PUBLISHED-PRIMARY lib:2606.07687 Table 1 (frozen features + ID objective, MLP probe, task-OOD, 3 seeds):
    V-JEPA 2 ViT-L 0.40 -> 0.85, Web-DINO ViT-L -0.01 -> 0.16, SigLIP2 0.05 -> 0.17; ID is a multiplier on
    temporal-predictive structure; k-step ID peaks at k=4. lib:2602.18639: a 196x384 -> 196x32 adapter on frozen
    DINOv2 trained with the transition model raises PointMaze success under shift 0.48 -> 0.78 (iBOT 0.72).
    lib:2606.31232 Table 3: the decode target must be the commanded/kinematic quantity, not the realised
    end-effector displacement (81.33 vs 64.93).
  one_variable: the adapter objective - inverse dynamics (k=4, predicting (a, kappa) from [adapter(z_t),
    adapter(z_{t+4})] via dz) ON vs an adapter trained with the plain prediction loss only (control) - on the SAME
    frozen features; the trunk is never touched
  arms:
    - refav1 banked fp8 DINOv3 features (INHERITED that they exist and are enumerable) x {ID adapter, control adapter}
    - V-JEPA 2 features from the E-DEC-68 static-decodability rig (INHERITED that they exist) x the same two
  primary_read: >
    cross-fitted ridge decodability of (a, kappa) from the adapter dz (the Delta-JEPA Table 5 transition probe),
    with the constant-only control (must read 0.0000), the raw-pixel floor, PCA basis and lambda fitted on the fit
    split only, split by clip, printed n/d; paired LOEO over clips; R2 gain = ID adapter - control adapter.
  success: >
    DINOv3 gain >= +0.17 R2 (the published Web-DINO gain) with the paired CI excluding 0 -> the frozen trunk CAN be
    shaped through an adapter; GS-2 is narrowed in the register ("nothing to shape IN the trunk"), and an
    adapter-LDAD arm on refav1 is registered (theme 3 family (i) becomes reachable at 0 trunk compute).
  failure: >
    DINOv3 gain within CI of 0 while V-JEPA 2 gains -> DINOv3 is in the Web-DINO regime (no temporal-predictive
    structure to multiply); refav1's role is the static-decodability baseline, v7 stays trainable, and no
    frozen-trunk shaping arm is registered. If NEITHER gains, the probe or the adapter is at fault (check the
    V-JEPA 2 published 0.85 first) - the row is INCONCLUSIVE, not a failure.
  controls:
    - constant-only 0.0000; raw-pixel floor printed; shuffled-time pairs (t, t') must read the constant value
    - the endpoint-leak audit (GS-1): report the concat [z_t, z_{t+4}] probe next to the dz probe
    - decode target (a, kappa) only; v excluded (Ledger decode-target rule, INHERITED)
    - seeds: 3 adapter inits; report the spread
  cost: 0 trunk compute; adapter training on banked features (minutes-hours on the 4060 or CPU); V-JEPA 2 feature
    extraction only if not banked (4060, hours) - UNVERIFIED; no pod contact
  tier: T0
  ties_to: [GS-2 (REVISED), GS-1, GS-9, E-DEC-68, P-4, LAB-ACT-1]
```

## LAB-ENC-2 — the observer effect on our own trunk: static vs dynamic decodability along training

```yaml
hypothesis:
  id: LAB-ENC-2
  status: PROPOSED (Research Lab 2026-09-03; conditional on >= 4 banked intermediate checkpoints of one trainable
    arm - UNVERIFIED; else run on the 2k / 30k pair only and say so)
  claim: >
    As the tiny trunk trains, linear decodability of STATIC scene content (n_agents) rises while decodability of
    DYNAMIC / metric quantities (a, kappa, lead range) falls or stalls - the published "static kept, dynamic
    erased" signature of invasive adaptation.
  prior: >
    PUBLISHED-PRIMARY lib:2602.12218: fine-tuning erases Speed/Radius while preserving Mass; correlation on a
    kinematic invariant 0.94 -> -0.03; deep blocks B5-B10 move most (CKA < 0.2). Ours (MEASURED, section 13.0d):
    splitp30k n_agents +0.3881 vs lead_range_m -0.1611 and d_ego -0.0399 at 30k.
  one_variable: training step (the checkpoint) of ONE arm (rdw8p30k or postrain30k); probes, targets, split and
    controls fixed once
  primary_read: cross-fitted ridge decodability (constant-only 0.0000, raw-pixel floor, PCA/lambda on the fit split,
    split by clip, printed n/d) of n_agents (static) and of (a, kappa, lead_range_m) (dynamic/metric) from the frozen
    trunk output at each checkpoint; paired LOEO CIs
  success: >
    n_agents decodability rises monotonically while >= 2 of the 3 dynamic targets fall or stay within CI of their
    2k value -> the observer effect holds on our trunk; the v7f prereg registers a "frozen trunk + adapter" or
    "full + distillation" arm rather than LoRA-style partial unfreeze (lib:2603.24581 Table 5: LoRA worst).
  failure: >
    dynamic targets rise with the static one -> our trunk is not in the erasing regime; E-DEC-63's mechanism is
    something else and no trunk-side arm is registered from this row.
  controls: [constant-only 0.0000, raw-pixel floor, shuffled-label probe at the constant value, printed n/d]
  cost: 0 GPU-h training; probe fits on banked checkpoints (CPU/4060 minutes per checkpoint)
  tier: T0
  ties_to: [E-DEC-63, section 13.0d, L2, GS-4]
```

## LAB-ENC-3 — distillation INTO the trunk as a continuing loss vs frozen-feature concatenation (v7f arm, deferred)

```yaml
hypothesis:
  id: LAB-ENC-3
  status: PROPOSED, DEFERRED (after LAB-ENC-1/2; a v7-tiny arm)
  claim: >
    A continuing DINOv3-feature distillation loss on the trainable tiny trunk keeps static decodability while the
    prediction objective shapes the dynamics - beating both "frozen distilled init" (E-DEC-60: init is not the
    lever) and "concatenate frozen features".
  prior: >
    PUBLISHED-PRIMARY lib:2603.24581 Table 4: distillation into the backbone 89.3 vs concatenating frozen geometric
    features 88.0 vs none 88.3 EPDMS (driving); Table 5: Base full 89.3, Small 86.3, Small-LoRA 84.7, Base-LoRA 68.5.
    Ours (MEASURED): E-DEC-60 - two arms sharing distill_init.pt differ in drift by 0.47 (the init alone does nothing).
  one_variable: the distillation loss (continuing, weight w) ON vs OFF on the postrain30k recipe at 30k parity
  primary_read: L2 panel (n_agents, lead_range_m, d_ego, a, kappa) with all controls; held-out nrmse; anchored action
    read (LAB-ACT-1 definition); T1 secondary
  success: static AND dynamic decodability both above control with CIs excluding 0 at nrmse parity (+2%)
  failure: no decodability change, or static up / dynamic down (the observer signature persists) -> distillation
    does not protect dynamics; the frozen-trunk + adapter route (LAB-ENC-1 outcome A) is the only shaping route
  controls: [constant-only, raw-pixel floor, deliberate-regression arm with the distillation target shuffled across
    clips, copy_detector CLEAN]
  cost: one v7-tiny 30k arm on the 4060 at bs 1-4 plus a DINOv3 feature pass over the corpus (banked? UNVERIFIED)
  tier: T0 / T1
  ties_to: [E-DEC-60, E-DEC-63, GS-2, v7f prereg]
```

---

### Not proposed, with reasons

- **A full-unfreeze vs freeze re-run of the postrain recipe**: E-DEC-64 already measured it (one variable) and the field's sign depends on trunk strength — a repeat adds nothing until the trunk changes.
- **LoRA / partial unfreeze of the tiny trunk**: the worst row in the only driving ablation that has it (`2603.24581` Table 5); not before LAB-ENC-2 says our trunk is in the erasing regime.
- **Swapping DINOv3 for V-JEPA 2 as refav1's trunk**: E-DEC-68 (9/9 for DINOv3 on static decodability) and What-Drives-Success (DINO > V-JEPA frozen) agree; LAB-ENC-1 tests the one thing V-JEPA 2 might carry that DINOv3 does not.
