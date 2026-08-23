# PREREG — E-ENC-3WAY: which encoder should v7 be built on?

`PRE-REGISTRATION, 2026-08-22.` PI (verbatim): *"After we solved provably the
collapse and decodability problem, I would like to compare the three variants."*
Both outcomes are committed below **before** any arm runs.

---

## 0. ⛔ THE GATE THAT MUST CLOSE FIRST

This comparison **does not start** until an arm demonstrably clears collapse.
Reason, MEASURED 2026-08-22: v6F@20k's mean-pooled encoder tokens carry
**99.7 % of their variance in ONE direction** (effective rank 1.03); its `z_op`
reads 5.86 against the O6 gate's `absolute_floor` of **64**. A nine-day run
produced a driving number that could not have existed.

⚠️ **And the collapse is a TRAINING DYNAMIC, not an initialisation one** —
effective rank RISES to step 16,000 (8.96) and FALLS to 5.55 by 20,000. ⇒ If the
objective is the collapser, **variant B collapses too, just from a better start**.
Running the three-way before the collapse fix would measure the wrong thing.

---

## 1. The three variants

| | arm | encoder | trained? |
|---|---|---|---|
| **A** | `own-vit` | our own ViT on driving data, modern recipe (RMSNorm, QK-Norm, LayerScale — `ModernCausalBlock`) | from scratch |
| **B** | `dino-ft` | initialised from DINOv3, fine-tuned on driving | fine-tuned |
| **C** | `dino-frozen` | DINOv3, frozen; 4-brain architecture on top | frozen |

⭐ **C RUNS FIRST.** It needs no encoder training, so it is the cheapest, AND it
is simultaneously the clean re-test of REF-A — whose frozen-DINO arm scored
**2.1322 ± 0.1821** while carrying **two residual-init defects** in
`refa_v1.py` (`StrategicSubspacePredictor`, `TokenFieldPredictor`, both fixed
2026-08-22). D-A5's *"the frozen encoder is REF-A's ceiling"* rests on that
confounded number.

**PI's hypothesis, recorded before the run:** *"the driving problems of REF-A
frozen encoder is the wrong design of the on-top policy head/planner and also
the missing things we built in REF-A v1."* C is the arm that tests it.

---

## 2. ⛔ Validity conditions — each earned by a specific past failure

1. **MATCHED TOTAL PARAMS.** `MODEL_REGISTRY.md`: *"E-ENC decides at MATCHED
   TOTAL PARAMS — compare arms on this number, not on per-layer widths."* A
   frozen DINOv3-L contributes ~300 M of FREE capacity; A must be sized so the
   comparison is not "more parameters win". State the counted total per arm.
2. **THE ENCODER IS THE ONLY VARIABLE.** Identical readout, predictor, heads,
   planner, data, parity key, seed. REF-A vs REF-A v1 was never this clean,
   which is exactly why its verdict is contested today.
3. **GATE ORDER: rank → decodability → driving.** No arm earns a T1 driving eval
   until it clears §3. This is the week's central lesson.
4. **FOUR METRIC FAMILIES** on any driving claim (LONGITUDINAL / LATERAL /
   TACTICAL / STRATEGIC), never pooled, T1 primary, paired episode-cluster
   bootstrap. ADE alone is an incomplete result.
5. **PARITY** — `physicalai-train-e438721ae894`, skip-hash `f09e44db`. Any arm
   that re-selects episodes is refused.

---

## 3. The gates, with their floors already measured

| gate | metric | floor | measured today |
|---|---|---|---|
| **G-RANK** | participation ratio / effective rank of `z_op`, ≥1024 rows | frozen DINOv3 **8.56 / 17.25** on the same frames; gate `absolute_floor` **64** | v6F **2.70 / 5.86**; enc tokens **1.01 / 1.03** |
| **G-EGO** | linear probe R² over the mean predictor: speed, yaw, yaw-rate, d_ego | must beat the **raw-pixel floor** and the **constant control** | v6F speed **+0.0025**; DINOv3 **+0.1473**; pixel **−0.0177** |
| **G-DECODE** | E-DETECT-1 grid AP, episode-disjoint, cluster bootstrap | above `prior` AND above `pixel`, paired | — |
| **G-DRIVE** | T1 four families | vs flagship v1 **0.452 m** and REF-A **2.1322** | — |

⚠️ **Every panel carries a CONSTANT-ONLY control and a RAW-PIXEL floor, and
prints its `n` and `d`.** MEASURED 2026-08-22: four separate estimator failures
in one afternoon (raw-energy normalisation; λ chosen on the point estimate; λ
chosen on the test set; n≪d) each produced a confident, wrong number, and each
was caught ONLY because a control read the same value as the thing measured.

⚠️ **A linear-probe NEGATIVE is not a learnability negative.** Beating the
baseline proves learnable; failing proves only *not learnable by a linear map*.
Where a negative would decide an arm, re-run it with the nonlinear probe and its
**time-shuffled control**.

---

## 4. Both outcomes, committed in advance

| result | what it means | what we do |
|---|---|---|
| **C clears G-RANK/G-EGO and drives ≪ 2.13** | the PI's hypothesis holds — REF-A's failure was the on-top planner + the missing REF-A v1 pieces, NOT the frozen encoder | D-A5 is RETRACTED; v7 is built on a frozen/lightly-adapted pretrained trunk, and the programme stops paying to train encoders |
| **C clears the probes but still drives ≈ 2.13** | the encoder is fine and the *planner* is the constraint | the hierarchy work becomes the priority; encoder choice is settled as "frozen is enough" |
| **B beats C materially** | fine-tuning adds real driving value over frozen | pay for the patch-embed surgery; v7 = DINOv3-init |
| **A beats both at matched params** | our own recipe on driving data wins | keep training encoders — but the collapse fix must be what made it possible, and say so |
| **all three collapse under the objective** | the objective is the collapser and no encoder choice rescues it | STOP the three-way; the target/objective is the work item |

---

## 5. Cost, and why this is now affordable

The v7-tiny rig runs a **v6-identical arm in 29 minutes** (measured: 1,704.9 s
for 2,000 steps, 19.34 M params, all six o-terms, real PhysicalAI, 256×640).
Each variant gets a tiny arm first; only arms that clear G-RANK and G-EGO are
promoted to a full-scale run. A three-way tiny screen is **~90 minutes**, not a
GPU-week.

⚠️ **Known obstacle for B, stated up front:** there is NO code path. `--init-from`
loads a *v6 checkpoint* for stage init, not arbitrary pretrained weights, and
v6's encoder is `Conv2d(in_channels=9, …)` (a 3-frame stack) against DINOv3's 3
channels. B requires patch-embed surgery plus dim matching — real work, and it
should not be started until C has reported.

---

## 6. What this pre-registration REFUSES to conclude

* ⛔ Nothing here is a driving claim until G-DRIVE runs at T1.
* ⛔ **DINOv3's +0.147 speed R² is a FLOOR, not its ceiling** — it is a linear
  probe on mean-pooled tokens. REF-A's record has frozen-DINO speed at R² **0.61**
  with a trained adapter. Quoting +0.147 as "DINOv3 is weak" is an estimator
  error, and this document does not make it.
* ⛔ The rank collapse explains why **v6's S-W stage learns nothing**. It does NOT
  by itself explain bad driving: flagship v1 drove **0.452 m** at participation
  **4.92**, because its policies read the encoder directly (a supervised bypass)
  while v6's S-W has none. Do not conflate the two claims.
