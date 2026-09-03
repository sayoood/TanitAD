<title>PROPOSED_HYPOTHESES - SOTA pass: representation and decodability (theme 4 of 4)</title>

# Proposed register rows — theme 4 (representation and decodability), 2026-09-03

Register-ready rows for `GOALS_AND_CLAIMS.md`; **the Master Mind applies them, the Research Lab does not edit the register.** One variable each, both outcomes committed **before** the experiment, controls that must read known values, cost stated. Priors are PUBLISHED-PRIMARY from PDFs banked for `2026-09-03-sota-decodability/RESULT.md`. Ids `LAB-REP-1 … LAB-REP-4` are fixed by the `KNOWLEDGE_BASE.md` entries already in history (commit `4cd3cdb`).

Recommended order: **LAB-REP-1 (0 GPU, instrument — blocks P5) → LAB-REP-4 (0 GPU, criteria + the refav1 audit) → LAB-REP-3 (0 trunk compute) → LAB-REP-2 (one v7-tiny arm; MERGE with LAB-DRIFT-2 first)**.

⛔ **Cross-package dependency, stated up front:** `LAB-REP-2` and theme 1's `LAB-DRIFT-2` propose the same lever (a training-only cross-prediction head to ego kinematics on the `postrain30k` recipe) from two different primaries. **They must be merged into one arm before either is registered.**

---

## LAB-REP-1 — the capture-gated, clip-unit probe panel: the first admissible L3 (and the P3/P5 dual read)

```yaml
hypothesis:
  id: LAB-REP-1
  status: PROPOSED (Research Lab 2026-09-03; Master Mind applies)
  claim: >
    Under a capture-gated protocol with the CLIP as the unit of both the split and the interval,
    at least one banked arm's predicted latent zhat_{t+k} beats a PERSISTENCE baseline (z_t carried
    forward) on at least one physical target, at k>=1 - and the arms on which it does NOT are
    arms that fail Gate 1 (capture), not arms whose predictor was measured and found wanting.
  prior: >
    PUBLISHED-PRIMARY lib:2608.29998 - the gate order and the uncertainty-unit rule:
    Gate 1 capture (target readable from real encoded roots) -> Gate 2 real-effect resolvability
    (frozen readout resolves the effect at REAL endpoints) -> Gate 3 propagation, interpreted only
    if 1 and 2 pass, against persistence, zero-effect and an environment-endpoint oracle.
    "The complete environment family is the uncertainty unit; checkpoints, roots, actions,
    coordinates, and imagined samples are never treated as independent replications."
    n=96 evaluation families, 4,096 whole-family bootstrap resamples, readouts and thresholds frozen
    on 48 calibration families. Outcomes there: Cheetah/LeWM capture 0.579-0.609, real 0.127-0.198,
    predicted -6.33..-3.36 (3/3 severe, 3/3 oracle-deficient); PreJEPA not severe (-0.005..0.064)
    but 5/5 oracle-deficient.
    PUBLISHED-PRIMARY lib:2607.27017 - the certificate gate (a recurrent probe on RAW observations,
    run BEFORE any model claim) and the passthrough bound (an UNTRAINED copy of the architecture
    reads force 0.94 / position 0.97); the passthrough-immune metric is a FORECAST against
    persistence: VFX 0.51 held-out vs persistence 0.31, paired episode-bootstrap 95% CI of
    model-persistence [+0.14,+0.28] at 4,258 episodes, [+0.01,+0.17] at 800; camera-only VX
    strictly negative [-0.55,-0.15].
    PUBLISHED-PRIMARY lib:2606.31232 Table 5 - the transition rung: dx from dz, linear/MLP,
    20,000 trajectory-split consecutive pairs, 3 seeds; LeWM linear r 0.765 against a STATE rung of
    0.950 on the same models (Table 4).
    PUBLISHED-PRIMARY lib:2605.06388 SS-C.3 - train the readout on REAL latents, FREEZE it, apply it
    unchanged to PREDICTED latents; encoder read is the ceiling (V-JEPA 2.1 0.829/0.865 ->
    WM 0.781/0.840).
  one_variable: the ESTIMATOR AND THE GATE ORDER. No model is trained; the same banked checkpoints,
    the same windows, the same targets as the withdrawn L3 read.
  arms: postrain30k, postrain30k_freeze, rdw8p30k, k8clip05p30k, k60clip05p30k (banked; md5s in
    MODEL_REGISTRY.md 13.9/13.10). Lead-matched held-out clips (24 as in the leak audit; report n).
  protocol:
    - Gate 1 CAPTURE - fit the readout on real encoded roots, score on held-out CLIPS. Report per
      target. An arm whose Gate 1 fails is reported as UNCERTIFIED for propagation and is NOT scored.
      (rdw8p30k currently fails this: z_t pooled cross-clip R2 is -0.05..-0.09, MEASURED.)
    - Gate 2 REAL-EFFECT RESOLVABILITY - the frozen readout must resolve the change in the target
      between t and t+k at the REAL encoded endpoints. Fails => propagation not interpreted.
    - Gate 3 PROPAGATION - only for cells passing 1 and 2. Compare zhat_{t+k} against:
      PERSISTENCE (z_t carried forward), ZERO-EFFECT, the ENCODED FUTURE z_{t+k} (the oracle/ceiling),
      raw pixels, and the constant.
    - TRANSITION RUNG, both columns, never one: dz = z_{t+1}-z_t AND the raw endpoint pair
      [z_t, z_{t+1}], each with a RAW-PAIR floor (the same pair built from downsampled pixels).
      Targets: d(n_agents), d(lead_range_m), d speed, d yaw over K in {1,3,6}.
    - Readout family: cross-fitted ridge AND an early-stopped MLP, reported side by side, per
      lib:2607.27017 and lib:2603.21546 (selectivity gap delta = R2_MLP - R2_linear reported).
  estimator: >
    THE CLIP IS THE UNIT OF BOTH THE SPLIT AND THE INTERVAL. Leave-one-clip-out for the fit;
    episode-cluster bootstrap (paired where two columns are compared on identical windows) for the
    interval. No sqrt(n) over overlapping pooled scores under any circumstances. Print n_clips,
    n_rows and d on every row.
  controls_that_must_read_known_values:
    - constant-only == 0.0000 exactly
    - horizon-zero identity (k=0) == the capture score, exactly
    - persistence at k=0 == the model, exactly
    - shuffled-target (within clip) ~ 0
    - a POSITIVE control that must FIRE: score the ENCODED FUTURE z_{t+k} as if it were the
      predictor. If the ceiling column does not separate from z_t, the instrument is dead and no
      null below it is admissible (this is the check the banked panel never ran).
  cost: 0 GPU-h training. ~2-4 h dev-box CPU (the leak audit's re-scores ran at this scale);
    RTX 4060 optional. No pod, no Thor.
  success: >
    OUTCOME A - at least one arm passes Gates 1 and 2 and its predicted column beats PERSISTENCE on
    at least one target with a paired clip-bootstrap CI excluding zero. => L3 is PASSED for the first
    time on an admissible read; P5's bar is set at that effect size; the v7f prereg quotes it.
    OUTCOME B - every cell that passes Gates 1 and 2 has a predicted column whose CI includes or lies
    below persistence. => L3 is FAILED for the first time on an admissible read; the failure is
    attributed to the objective class (single-step, vision-only, jointly-learned targets = the
    lazy equilibrium of lib:2607.27017), NOT to the encoder; LAB-REP-2 becomes the top v7f arm and
    P5 is scored on the NEW arm, not on the banked line.
    OUTCOME C (must be reported, not folded into B) - no arm passes Gate 1 on any target. => the
    programme has no admissible decodability read at all and the readout, not the predictor, is the
    open question; LAB-REP-3 goes first.
  falsifier: any cell where the ceiling (encoded future) column fails to separate from z_t invalidates
    that cell's null, whichever outcome it would otherwise support.
  scope_limits: >
    T0-DIAGNOSTIC throughout. Nothing here is a driving claim. h>=2 columns are reported but carry
    the MM-E14 caveat (the h>=2 heads were never trained on this recipe) and may not be quoted as
    predictor quality.
```

---

## LAB-REP-2 — direct multi-horizon heads + a training-only cross-modal forecast target (one v7-tiny arm) ⚠️ MERGE WITH LAB-DRIFT-2

```yaml
hypothesis:
  id: LAB-REP-2
  status: PROPOSED (Research Lab 2026-09-03) - ⛔ BLOCKED on (a) merging with theme 1's LAB-DRIFT-2,
    (b) Master Mind confirmation of the labels-may-use-ego rule, (c) LAB-REP-1's instrument existing
  claim: >
    Replacing the single-horizon predictor with DIRECT heads at multiple horizons (k in {1,4,16}) and
    adding a training-only forecast target on ego kinematics raises the decodability of scene state
    from the predictor's own output above the persistence baseline - where the incumbent
    single-step, vision-only recipe does not.
  prior: >
    PUBLISHED-PRIMARY lib:2607.27017 SS-4.2 - the LAZY EQUILIBRIUM: single-step vision-only latents read
    directly visible, perfectly predictable object position at R2 0.04 against a certificate of 1.00.
    Two escapes COMPOSE: cross-modal prediction TARGETS -> 0.58; multi-horizon action-conditioned
    heads (delta in {1,4,16}) -> 0.89 with NO additional modality; both -> 0.98. Direct long-horizon
    heads beat autoregressive composition at matched horizon: 0.10 vs 0.19 arena units (static 0.21).
    Design rule 3 verbatim: "Ask for the future at multiple horizons, directly."
    The mechanism control that predicts our O14 null: VF RECEIVES force as input but never forecasts
    it and reads 0.15 against an untrained passthrough bound of 0.91; an equally sized proprioception
    target moves stiffness by -0.01 ("what matters is not extra work, but that the extra work's answer
    depends on the parameter").
    PUBLISHED-PRIMARY lib:2608.06799 - the transition-grounding form: supervise latent PAIRS with the
    net state change dq_{t,k} = q_{t+k} - q_t at MULTIPLE horizons k=1..T-1, training-only, heads
    discarded at inference (no inference cost). It matches an explicit action-IDM on action
    recoverability (0.80/0.86 vs 0.80/0.84) WITHOUT using action prediction, and beats it on
    gripper velocity (0.69/0.76 vs 0.67/0.73). Their argument against IDM as the transition target:
    "even from the same initial state, multiple action sequences can produce the same endpoint state
    change" - the net change is uniquely determined, the action is not.
  one_variable: ⚠️ THIS ROW AS WRITTEN IS TWO VARIABLES (multi-horizon heads AND a cross-modal target).
    It MUST be split into A (heads only) and B (target only) plus the conjunction, or the attribution
    cannot be made. lib:2607.27017 reports all three cells (0.89 / 0.58 / 0.98), so the three-cell
    design is the published one and is what should be registered.
  arms:
    - A: postrain30k recipe + direct heads at k in {1,4,16} (targets are the encoded futures at those
      horizons; no autoregressive composition)
    - B: postrain30k recipe + a training-only forecast head on ego kinematics (a, kappa; v per the
      Ledger's decode-target rule), heads discarded at inference
    - AB: both
    - CONTROL (must read a known value): an equally sized auxiliary head whose target does NOT depend
      on the scene - the published null (lib:2607.27017's proprioception-substitute cell, -0.01, and
      theme 1's JEPA-x REGRESS cell, drift 0.373 vs 0.361). If this control MOVES the read, the
      experiment measures capacity, not content, and is void.
  reads: LAB-REP-1's gated panel, unchanged, on all four arms; plus nrmse and the anchored actdiv
    read at the TRAINED speed scale (v/10, per the leak audit's finding 3).
  cost: 4 arms x v7-tiny (19M) on the dev-box RTX 4060 at bs 1-4. ⭐ Feasibility is PUBLISHED:
    lib:2607.27017's entire study, including its real-robot RH20T experiments, ran on ONE RTX 4060
    with ~5M-parameter models training in under an hour each.
  success: >
    OUTCOME A - arm AB (or A alone) clears persistence on >=1 target where the incumbent does not,
    paired clip-bootstrap CI excluding zero, AND the equal-size null control does not move.
    => the lazy equilibrium is OUR failure mode and the escape transfers; direct multi-horizon heads
    enter the v7f recipe and MM-E14 is closed by construction rather than deferred.
    OUTCOME B - no arm clears persistence, or the null control moves as much as the treatment.
    => the published escape does NOT transfer to driving video at our scale; the difference is
    recorded (their objects are large and action-driven at 16 steps; our scene change is dominated by
    ego motion), P5's bar is re-opened, and the next lever is the readout (LAB-REP-3), not the
    objective.
  falsifier: if arm A's k=1 head degrades against the incumbent's k=1 by more than the published
    accuracy-robustness trade-off (theme 1, lib:2512.24497 Remark 1), the horizon change is being paid
    for out of one-step quality and must be reported as such rather than as a free gain.
```

---

## LAB-REP-3 — readout vs latent, on the same banked tokens (0 trunk compute)

```yaml
hypothesis:
  id: LAB-REP-3
  status: PROPOSED (Research Lab 2026-09-03)
  claim: >
    Our 4x8 grid pool + 128->64 projection, not the latent, is what caps lead_range_m decodability:
    an attention-pooled readout over the SAME frozen tokens recovers range where the grid pool does
    not, while leaving n_agents unchanged.
  prior: >
    PUBLISHED-PRIMARY lib:2603.20327 SS-4.2 - mean-pooling the 196 spatial tokens per frame cut codebook
    utilisation from 62.5% to 5% AND flipped the grasp-angle contrast from p < 1e-4 to p = 0.357;
    "mean pooling erases spatial local semantic differences".
    PUBLISHED-PRIMARY lib:2411.04983 Table 2 - patch tokens 0.90 vs CLS 0.44 (Push-T success).
    PUBLISHED-PRIMARY lib:2603.24581 - 16 learnable query tokens per view, attention on lane markings:
    the driving-WM readout family for fine geometry.
    PUBLISHED-PRIMARY lib:2512.24497 - on frozen trunks, metric quantities are "only implicitly encoded
    and subject to quantization by the patch architecture".
    ⚠️ NO primary ablates grid-pool vs attention-pool on a driving RANGE target: the prior is
    directional only and the magnitude is ours to measure.
  one_variable: the READOUT (4x8 grid pool + 128->64 projection vs attention pooling over the same
    tokens with matched parameter count). Same frozen tokens, same targets, same clips, same fit rule.
  arms: banked token fields from rdw8p30k and postrain30k; and, if the readout shape is verified
    (UNVERIFIED today), refav1's banked fp8 DINOv3 patch tokens.
  reads: lead_range_m and n_agents, LAB-REP-1's gate structure, clip-unit estimator; parameter count
    MATCHED between readouts and printed.
  controls_that_must_read_known_values: constant-only 0.0000; raw-pixel floor; a MATCHED-PARAMETER
    grid-pool control (so the comparison is pooling geometry, not capacity).
  cost: 0 trunk compute; a small readout trained on banked features. Dev-box RTX 4060, hours.
  success: >
    OUTCOME A - attention pooling recovers lead_range_m by more than the matched-parameter control,
    paired clip-bootstrap CI excluding zero. => E-DEC-25's mechanism is CONFIRMED as readout geometry;
    GS-4's expensive branch (readout re-architecture) goes FIRST and the geometry loss is deferred.
    OUTCOME B - no recovery, or the matched-parameter control recovers as much. => the cap is in the
    latent, not the readout; GS-4's cheap branch (explicit geometry loss) runs first as already
    proposed, and E-DEC-25's attribution to the projection is narrowed to "capacity, not geometry".
```

---

## LAB-REP-4 — six controls into the criteria registry, and the refav1 oracle-input audit (0 GPU)

```yaml
hypothesis:
  id: LAB-REP-4
  status: PROPOSED (Research Lab 2026-09-03) - instrument/criteria work plus one audit
  claim: >
    refav1's tactical head fails the published leakage battery, and the failure is TRANSCRIPTION of
    an oracle channel rather than weak perception: withholding nav collapses the ranking to chance,
    a COUNTERFACTUAL nav makes the head follow the false command rather than the scene, and degrading
    the vision path does not reduce either effect.
  prior: >
    PUBLISHED-PRIMARY lib:2607.06925 - a goal-conditioned readout reads 0.90 and collapses to 0.27
    (chance 0.25) when the goal is withheld (3 seeds: 0.900 +/- 0.009 -> 0.270 +/- 0.034); under a
    COUNTERFACTUAL goal the prediction follows the FALSE instruction 94.5% [0.914,0.969] of the time
    and the true scene 2.3% [0.008,0.047] (N=256); a VALIDATED POSITIVE CONTROL (an engineered-leaky
    model) fires the probe at 0.97 against 0.03 for a no-goal baseline; the DOSE-RESPONSE refutes the
    "the other inputs are weak" explanation (alpha 1.0 -> 0.0 gives 0.975 -> 0.986, i.e. leakage never
    increases as the other input is degraded); leakage-controlled readouts must be run on the TRAINING
    render distribution (a fresh-scene variant under-counted 0.35 against an in-distribution 0.84);
    and the exposed/immune split is STRUCTURAL - goal in the planner COST is immune, goal conditioned
    into the PREDICTOR is exposed.
    PUBLISHED-PRIMARY lib:2607.27017 - the PASSTHROUGH BOUND: an UNTRAINED copy of the same
    architecture, probed identically, reads force at 0.94 and position at 0.97; the honest metric is
    the passthrough-immune one.
    PUBLISHED-PRIMARY lib:2312.03031 Tables 1-2 - the same experiment at system level in OUR domain:
    blank images move planning L2 from 0.37 to 0.46 while detection NDS goes 45.5 -> 0.0 and map mAP
    47.0 -> 0.0, whereas v x 0.0 moves L2 from 0.37 to 6.16; and Ego-MLP (ego status + driving command,
    NO perception) reaches L2 avg 0.35 against VAD-Base's 0.37.
    MEASURED (ours): AUC 0.873 [0.686,1.000] -> 0.520 [0.000,1.000] under nav_zero on the incumbent
    (ep2 0.922 -> 0.858); a nav-only predictor scores 0.684 accuracy against the model's 0.650 and the
    constant control's 0.650.
  one_variable: per read - the nav channel's CONTENT (true / zeroed / counterfactual / shuffled),
    with everything else fixed and the frames held on the TRAINING distribution.
  reads:
    - R1 nav withheld (already MEASURED; re-run inside the battery for comparability)
    - R2 COUNTERFACTUAL nav: feed a nav command naming a different manoeuvre; report the fraction of
      windows whose argmax follows the FALSE command vs the true scene label. NEVER RUN BEFORE.
    - R3 VALIDATED POSITIVE CONTROL: an engineered-leaky head (nav copied straight to the logits) must
      fire R2 near 1.0. Without it a "no leak" reading is a dead instrument, not a measurement.
    - R4 PASSTHROUGH BOUND: the same probe on an UNTRAINED copy of the head/adapter.
    - R5 DOSE-RESPONSE: scale the vision path (alpha 1.0 / 0.5 / 0.0) and re-run R2. The published
      prediction is that leakage does NOT increase as vision is degraded.
    - R6 ENDPOINT-SHUFFLED BOUND for any difference target (ours; TanitAD-novel - no primary reports it)
  criteria_registry_additions (the deliverable that outlives the audit):
    counterfactual-input control; validated positive control; passthrough/untrained-architecture
    bound; endpoint-shuffled bound for difference targets; NORMALISATION-STATE VERIFICATION before any
    rank/participation read (lib:2608.10145 SS-5.4: the same effective rank moved 16.5 -> 67.8 and
    11.9 -> 18.6 on checkpoints whose weights never changed); and the training-distribution rule for
    ablation frames.
  cost: 0 GPU training; the decoder panel's own budget was 560 forward passes / ~3 min on the RTX 4060.
    R2-R5 roughly quadruple that. No pod, no Thor.
  success: >
    OUTCOME A - R3 fires (>= 0.9) AND R2 shows the head following the counterfactual nav on a majority
    of windows AND R5 shows leakage flat or rising as vision is degraded. => the head is TRANSCRIBING
    an oracle channel; the fix is structural (nav out of the tactical dynamics, into the cost) and
    NOT a loss weight; D-REFAV1-TAC-DECODER-PANEL is upgraded from "nav-borne ranking" to
    "instruction leakage, published class, measured battery".
    OUTCOME B - R3 fires but R2 shows the head following the TRUE scene under a counterfactual nav.
    => the nav_zero collapse is a distribution-shift artefact of zeroing an input the head has never
    seen zeroed, not transcription; the correct control is the counterfactual, the withheld read is
    retired, and the ep2 vision-borne component (0.858) becomes the headline.
    OUTCOME C - R3 does NOT fire. => the instrument is dead; nothing in R1/R2/R5 is interpretable and
    the panel is rebuilt before any decoder claim is made.
  falsifier: if R4 (untrained passthrough) reads near the trained head's AUC, the whole panel is
    measuring input passthrough and every number in D-REFAV1-TAC-DECODER-PANEL is bounded by it.
  publication_note: >
    lib:2607.06925 SS-5.4 and SS-7 state TWICE that the authors "could not find a public-weights world
    model that conditions its predictor on language" and that demonstrating the leak in a released
    model is "the key remaining step to convert our characterization into a field-wide audit."
    refav1 is such a model with an oracle channel. This audit is a publishable contribution, not only
    an internal fix. ⚠️ First VERIFY the conditioning form (FiLM vs concatenation) - UNVERIFIED today.
```
