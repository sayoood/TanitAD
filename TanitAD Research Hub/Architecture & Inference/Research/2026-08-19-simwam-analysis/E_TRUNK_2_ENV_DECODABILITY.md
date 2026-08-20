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

R² (regression) / AUC (binary), point with the **episode-cluster bootstrap** 95 % CI.

| target | `C-PIXEL` | `C-EGO` | `v6_cells` | `v6_tok_pooled` | `v6_tokens` | ⭐ `dino_pooled` | ⭐ `dino_tokens` |
|---|---|---|---|---|---|---|---|
| **`lead_gap_m`** | +0.0015 | **+0.3340** | −0.0165 | −0.0151 | +0.0181 | **+0.3841** | **+0.4549** |
| `nearest_any_m` | +0.0134 | +0.1164 | +0.0373 | +0.0260 | +0.0275 | **+0.3529** | **+0.3567** |
| `n_agents_log` | +0.0793 | +0.1611 | +0.1707 | +0.0749 | +0.0855 | **+0.5767** | **+0.5594** |
| `occluded_frac` | −0.0052 | −0.0239 | +0.0038 | +0.0008 | +0.0175 | +0.0356 | +0.1231 |
| `left_occupied` | .4807 | .5055 | .5391 | .4982 | .5032 | **.8303** | **.8462** |
| `right_occupied` | .4813 | .5390 | **.5987** | .5446 | .5295 | **.8242** | **.8420** |
| `vru_ahead` | **.6191** | .5630 | .5717 | .5700 | **.5823** | **.7093** | **.7041** |

Key CIs: `dino_tokens`/`lead_gap_m` **[+0.385, +0.533]**;
`dino_pooled`/`lead_gap_m` **[+0.320, +0.457]**;
`C-EGO`/`lead_gap_m` **[+0.254, +0.418]**;
`v6_tokens`/`lead_gap_m` **[−0.056, +0.076]**;
`dino_tokens`/`left_occupied` **[.812, .877]**; `v6_tokens`/`left_occupied` **[.430, .576]**.

### 3.1 The 2×2, read directly

|  | unpooled (640 tokens) | pooled (4×4 cells) | **pool cost** |
|---|---|---|---|
| **DINOv3** | +0.4549 | +0.3841 | **−0.071 (−16 % rel.)** |
| **v6** | +0.0181 | −0.0151 | ~0 (nothing to lose) |
| **encoder gap** | **+0.4368** | **+0.3992** | |

⇒ **The pooling is CHEAP and the ENCODER is the defect.** DINOv3 squeezed
through the very same parameter-free 40× pool keeps **R² +0.384 on headway**
and **AUC .830 on lane occupancy**; v6 sits at chance on both sides of the pool.

### 3.2 What it says

1. ⛔ **No v6 arm decodes lead-vehicle distance.** All three sit at zero while
   **ego state alone reaches +0.3340** and **DINOv3 reaches +0.4549**. The
   representation the whole hierarchy consumes carries **less headway
   information than a speedometer**. 88.7 % of the oracle gap is longitudinal,
   so this is the programme's largest known defect, measured rather than inferred.
2. ⭐ **The protocol has ample power — "we cannot detect it" is REFUTED.**
   DINOv3 in 16 cells reaches AUC **.83** on lane occupancy. Any claim that this
   battery is too weak to see environment information is now falsified.
3. ⛔ **v6 is at CHANCE on adjacent-lane occupancy** — the one target where **no
   ego leak is possible** (C-EGO .5055 / .5390, both spanning .5) — while a
   generic vision encoder gets **.83**.
4. **The only target v6 clears chance on is `vru_ahead`, and raw pixels do it
   better** (.6191 vs .5823). Appearance, not learned scene understanding.
5. `v6_tokens_pooled` (−0.0151) ≈ `v6_cells` (−0.0165) — the pooling
   reproduction is faithful, which is what licenses the `dino_pooled` cell.

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
