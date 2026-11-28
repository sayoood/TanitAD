# E-TRUNK-2 — is ENVIRONMENT information decodable from the v6 trunk?

`MEASURED (ours; dev-box, CPU dual ridge)` · **T0-DIAGNOSTIC — decodability is a
representation property, NEVER driving performance** · checkpoint
**`v6F-SW-30k@20000`** (fp16 snapshot, **67 % of the 30 k run**) · 5,617 frames ·
130 episodes · **episode-disjoint** 5-fold · episode-cluster bootstrap ·
**no load added to Thor**.

⭐ **COMPLETE. The DINOv3 reference ran, and it INVERTS the conclusion E-TRUNK-1
was heading toward: the 40× pool is CHEAP, and the ENCODER is the defect.**

---

## 1. The question, and why E-TRUNK-1 could not answer it

The PI asked how we prove the trunk carries something useful about the
environment. **E-TRUNK-1 does not answer that**, and treating it as though it
did is the error this document exists to end:

| | asks | dominated by |
|---|---|---|
| **E-TRUNK-1** | can a predictor beat persistence at predicting the FUTURE FIELD? | **dynamics predictability** — the representation's scale and variance structure |
| **E-TRUNK-2** | is the environment information THERE, decodably? | **information content** |

They come apart badly. A **constant** representation is perfectly predictable
and carries nothing; a rich one can be *hard* to predict precisely because it
carries a lot. Only the second question is the PI's.

## 2. Design — a number with no reference is not a result

Every arm is scored on the **same frames, same folds, same probe**:

| arm | dims | role |
|---|---|---|
| `C-MEAN` | 0 | absolute floor, R² = 0 / AUC = 0.5 **by construction** |
| `C-PIXEL` | 640 | decoded frame, greyscale, downsampled **to the token grid** — trivial appearance |
| ⭐ `C-EGO` | 5 | speed, accel, yaw-rate — **car-following statistics with NO perception** |
| `v6_cells` | 2,048 | **the deployed operative latent** (`= d_op`) |
| `v6_tokens_pooled` | 12,288 | v6 tokens through the readout's **parameter-free 40× pool** |
| `v6_tokens` | **491,520** | the v6 encoder field **before** the readout |
| ⭐ `dino_tokens` | **655,360** | DINOv3 ViT-L/16 at a **matched 16×40 grid** — encoder reference |
| ⭐ `dino_pooled` | 16,384 | DINOv3 through **v6's own 40× pool** — the discriminating cell |

⚠️ **`C-EGO` is not optional.** `lead_gap_m` comes from `obstacle.offline`
cuboids, so it is a genuine environment target — but ego speed predicts headway
through plain car-following. **An arm that merely matches C-EGO has demonstrated
nothing about perception.** Same family as the sitclf leak.

⚠️ DINOv3 on a 256×640 frame yields a **16×40 patch grid — the same grid v6
produces** — so the pool applies identically and `dino_pooled` is a true
like-for-like of `v6_tokens_pooled`. Row order was **verified equal** to the
probe keys as an ordered list; the builder **refuses** rather than silently
misaligning.

**Targets** from the 130-clip `obstacle.offline` join (25,790 frames, 52.96
agents/frame, zero empty frames). ⛔ `lead_present` is **excluded and named**:
95.8 % positive, so a constant predictor scores 0.958.

## 3. Result

⚠️ **These are the SCALE-NORMALISED numbers and they supersede the first pass.**
v6 cells and DINOv3 activations differ by orders of magnitude, so an **absolute**
λ grid meant a different amount of shrinkage per arm. Each Gram is now divided by
its mean diagonal, making λ dimensionless. **This was not cosmetic:** `C-EGO`
→ `ego_yawrate` is an **identity map** and read **−1.81** before normalisation,
**+0.9845** after. Headline numbers are robust either way (`C-EGO`/`lead_gap_m`
+0.3340 both).

R² (regression) / AUC (binary), **bold = CI excludes the null**.

| target | `C-PIXEL` | `C-EGO` | `v6_cells` | `v6_tok_pooled` | `v6_tokens` | ⭐ `dino_pooled` | ⭐ `dino_tokens` |
|---|---|---|---|---|---|---|---|
| **`lead_gap_m`** | +0.0020 | **+0.3340** | −0.0176 | **−0.0151** | +0.0130 | **+0.3792** | **+0.4531** |
| `nearest_any_m` | +0.0227 | **+0.1194** | +0.0371 | +0.0260 | +0.0399 | **+0.3465** | **+0.3464** |
| `n_agents_log` | +0.0661 | **+0.1601** | +0.1700 | +0.0750 | +0.0821 | **+0.5766** | **+0.5646** |
| `occluded_frac` | −0.0082 | **−0.0081** | +0.0031 | +0.0008 | +0.0035 | +0.0299 | +0.1091 |
| ⭐ `ego_speed` | **−0.0510** | **+1.0000** | −0.0049 | **−0.0677** | −0.0340 | **+0.6957** | **+0.6872** |
| `ego_yawrate` | **−0.0546** | **+0.9845** | −0.0125 | −0.0077 | −0.0285 | −0.0084 | +0.0436 |
| `ego_accel` | −0.0141 | **+1.0000** | **−0.0339** | **−0.0446** | **−0.0270** | −0.0031 | −0.0276 |
| `left_occupied` | .4672 | .5009 | .5321 | .4982 | .4971 | **.8312** | **.8488** |
| `right_occupied` | .4703 | .5307 | **.5890** | .5447 | .5150 | **.8236** | **.8480** |
| `vru_ahead` | **.5967** | .5612 | .5728 | .5699 | .5722 | **.7095** | **.6993** |

⚠️ **`C-EGO` is a CEILING on the three ego targets, not a control** — it is the
identity map there. Its 1.0000 / 0.9845 / 1.0000 is the harness verifying itself.

### 3.1 The 2×2, read directly (`lead_gap_m` R²)

|  | unpooled | pooled (4×4) | **pool cost** |
|---|---|---|---|
| **DINOv3** | +0.4531 | +0.3792 | **−0.074 (−16 % rel.)** |
| **v6** | +0.0130 | −0.0151 | ~0 (nothing to lose) |

⇒ **The pool is CHEAP; the ENCODER is the defect.** Lane occupancy costs DINOv3
only **.8488 → .8312**.

### 3.2 ⭐ The spectrum, at ADMISSIBLE n, with a reference

⛔ **A rank read off the trainer log is INADMISSIBLE** — it is computed at
**n = 48** against d = 2048, so it is bounded by **47**, and `o6_rank_verdict`
returns `INCONCLUSIVE` ("cannot resolve rank"). Quoting `effective_rank 22.93`
as "2.3 of 2048" is the category error `v6.py` names explicitly, and it is logged
as **C128**. Computed instead on the **5,617 banked frames** (ceiling 2048/5616,
far above the 1024 bar):

| arm | participation ratio | effective rank | top-8 share |
|---|---|---|---|
| `v6_cells` (deployed, d=2048) | **4.90** | 515.6 | **0.806** |
| `v6_tokens_pooled` (d=12288) | **3.28** | 78.2 | **0.949** |
| `dino_pooled` (d=16384) | **40.77** | 2022.3 | **0.348** |

⇒ The defensible claim is **ANISOTROPY AGAINST A REFERENCE — 8.3× in
participation ratio, 0.806 vs 0.348 in top-8 share** — never "N dimensions".
⚠️ `o6_rank_verdict` returns `INCONCLUSIVE` for **all three including DINOv3**:
its criterion is retention-over-training, not a cross-arm comparison. The raw
statistics are comparable; the verdict machinery is not the instrument here.

### 3.3 What it says

1. ⛔ **No v6 arm decodes lead-vehicle distance.** All three at zero while ego
   state alone reaches **+0.3340** and DINOv3 **+0.4531**. 88.7 % of the oracle
   gap is longitudinal, so this is the programme's largest known defect,
   measured rather than inferred.
2. ⛔ **v6 cannot decode its OWN SPEED either (−0.0049) while DINOv3 reads it
   from the same frames through the same pool at +0.6957.** ⚠️ *"v0 is supplied
   as an action channel so the encoder needn't encode it"* was offered as an
   innocent explanation and **is refuted by this row**. The honest statement is
   that the latent is **impoverished on every axis tested** — environment and
   ego-motion alike — not specialised away from one toward the other.
3. ⭐ **The protocol has ample power — "we cannot detect it" is REFUTED.**
   DINOv3 in 16 cells reaches AUC **.83** on lane occupancy.
4. ⛔ **v6 is at CHANCE on adjacent-lane occupancy** — the one target where **no
   ego leak is possible** (`C-EGO` .5009 / .5307) — while DINOv3 gets **.83**.
5. **The only target v6 clears chance on is `right_occupied` (.5890)**, and
   `C-PIXEL` beats every v6 arm on `vru_ahead`.

## 4. Estimator — two faults found and fixed mid-run

⛔ **The first pass averaged PER-EPISODE R² and produced −182 against a pooled
+0.334.** Not a discrepancy — an invalid estimator: a per-episode R² divides by
*that episode's* variance, and an episode with a near-constant lead gap explodes.
Replaced with the doctrine's **episode-CLUSTER bootstrap of the POOLED
statistic**. *(Same family as the `overlapping_holdout_se` ban: an estimator that
also moves the point estimate.)*

⚠️ **λ grid.** The first grid (1e−2…1e5) had **68 folds select the MAXIMUM**.
Widened to **1e−4…1e8**, 30 remain — **all at MAX, none at MIN**. That is the
interpretation-safe direction: λ→max is the constant predictor, i.e. **the null
itself**, and no fit is under-regularised. ⭐ **Both load-bearing numbers —
`C-EGO`/`lead_gap_m` and `v6_tokens`/`lead_gap_m` — have ZERO edge folds.**

Protocol: folds **episode-disjoint** (the REF-A I-JEPA lesson — ~80 % of val
inside train made that number unusable); λ chosen on an **inner episode-disjoint
split of the train fold only**; `d ≫ n` so the **dual/Gram** ridge is used, which
is *exact*, with kernel-centering.

## 5. ⛔ What this does NOT license

* **NOT a matched-parameter claim.** DINOv3 ViT-L is **303 M** parameters
  pretrained on **LVD-1689M**; v6's encoder is far smaller and saw 2,376
  episodes. It is a strong **reference**, not a fair fight. The finding is that
  the v6 representation lacks facts a general encoder holds *through the same
  bottleneck* — not that v6 "should" have matched it at equal cost.
* **NOT a claim about the finished model.** Step **20,000 of 30,000**.
* **NOT "not present."** A **linear** probe measures *linear* decodability.
* **NOT an objective indictment by itself.** v6's trunk optimises
  action-conditioned future prediction, not representation quality. That it does
  not linearly expose headway is a **measured fact about the representation**,
  and a reason to ask whether the objective induces what the hierarchy needs.
* **NOT a v5.8 comparison.** No v5.8 arm was run. ⚠️ The premise *"in v5.8 we
  could not extract environment information"* **could not be located as a
  MEASURED result** — S-W's gate is P1/P3/P6 and *"consumes no v5.8f
  measurement."* Locate it before building on it.
* **NOT driving performance.** T0-DIAGNOSTIC.

## 6. ⚠️ Consequences — including a recommendation this REFUTES

1. ⛔ **`E_TRUNK_1_POOLING.md` §3/§5's parallel-token-path direction does not
   survive.** That document reasoned from *"the readout averages the dynamics
   away"* toward adding a 640-token path at the S-T boundary — at **40× the
   sequence cost**. E-TRUNK-2 shows the pool costs DINOv3 only 16 % of its
   headway R² and 0.016 AUC of its occupancy. **Bypassing the readout would buy
   almost nothing, because the pool is not where the information dies.**
   ⚠️ E-TRUNK-1 is not *wrong* — it measured what it measured — but its
   architectural reading was the wrong question, and the token path must not be
   funded on it.
2. ⭐ **REF-D's frozen-prior bet looks stronger, not weaker.** REF-D freezes
   Cosmos3-Edge and spends its budget on the hierarchy. This is direct evidence
   that a strong frozen encoder carries what our trained trunk does not.
   *(§3.2 of `REFD_DESIGN.md` declared frozen-vs-co-trained an open tension; this
   is a data point for the frozen side on REPRESENTATION CONTENT, and says
   nothing yet about closed-loop control.)*
3. **REF-A v1's DINOv3 choice is supported** on the same evidence.
4. **The open question is now the v6 OBJECTIVE, not its readout geometry.**

## 7. Manifest

| artifact | where |
|---|---|
| `e_trunk2_probe.py` (arms, dual ridge, cluster bootstrap, dino builder) | `…/simwam-analysis/code/` |
| `e_trunk2_targets.py` (environment targets from `obstacle.offline`) | `…/simwam-analysis/code/` |
| `e_trunk2_probe.json` (every point, CI, λ, edge count) | `…/simwam-analysis/raw/` |
| feature memmaps + cached Grams | scratchpad `sp2/e_trunk2_feat/` |
| DINOv3 fields (130 clips, 5,617 frames, 6.9 GB) | scratchpad `dinov3_fields/` |
| this document | `…/simwam-analysis/E_TRUNK_2_ENV_DECODABILITY.md` |
