# E-TRUNK-2 — is ENVIRONMENT information decodable from the v6 trunk?

`MEASURED (ours; dev-box, CPU dual ridge)` · **T0-DIAGNOSTIC — decodability is a
representation property, NEVER driving performance** · checkpoint
**`v6F-SW-30k@20000`** (fp16 snapshot, **67 % of the 30 k run**) · 5,617 frames ·
130 episodes · **episode-disjoint** 5-fold · episode-cluster bootstrap ·
**no load added to Thor**.

⛔ **INCOMPLETE WITHOUT THE DINOv3 ARM (§6). Do not quote §3 as a conclusion
about v6 until that control has run.**

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
| ⛔ `dino_tokens` / `dino_pooled` | — | **NOT YET RUN** — §6 |

⚠️ **`C-EGO` is not optional.** `lead_gap_m` comes from `obstacle.offline`
cuboids, so it is a genuine environment target — but ego speed predicts headway
through plain car-following. **An arm that merely matches C-EGO has demonstrated
nothing about perception.** Same family as the sitclf leak.

**Targets** from the 130-clip `obstacle.offline` join (25,790 frames, 52.96
agents/frame, zero empty frames). ⛔ `lead_present` is **excluded and named**:
95.8 % positive, so a constant predictor scores 0.958.

## 3. Result

R² (regression) / AUC (binary), point with the **episode-cluster bootstrap** 95 % CI.
**Bold** = CI excludes the null.

| target | `C-PIXEL` | `C-EGO` | `v6_cells` | `v6_tok_pooled` | `v6_tokens` |
|---|---|---|---|---|---|
| **`lead_gap_m`** | +0.0015 | **+0.3340 [+0.254,+0.418]** | −0.0165 | **−0.0151 [−0.039,−0.000]** | +0.0181 [−0.056,+0.076] |
| `nearest_any_m` | +0.0134 | **+0.1164 [+0.020,+0.190]** | +0.0373 | +0.0260 | +0.0275 |
| `n_agents_log` | +0.0793 | **+0.1611 [+0.068,+0.235]** | +0.1707 [−0.019,+0.326] | +0.0749 | +0.0855 |
| `occluded_frac` | −0.0052 | **−0.0239 [−0.060,−0.004]** | +0.0038 | +0.0008 | +0.0175 |
| `left_occupied` | .4807 | .5055 | .5391 [.464,.612] | .4982 | .5032 |
| `right_occupied` | .4813 | .5390 | **.5987 [.523,.674]** | .5446 | .5295 |
| `vru_ahead` | **.6191 [.538,.699]** | .5630 | .5717 | .5700 | **.5823 [.503,.663]** |

### 3.1 What it says

1. ⛔ **No v6 arm decodes lead-vehicle distance.** All three sit at zero
   (−0.0165 / −0.0151 / +0.0181) while **ego state alone reaches +0.3340**. The
   representation the whole hierarchy consumes carries **less headway
   information than a speedometer**. This is the programme's largest known
   defect (88.7 % of the oracle gap is longitudinal) measured directly.
2. ⭐ **Pooling is NOT the culprit — and this CONTRADICTS what E-TRUNK-1
   suggested.** The unpooled 640 × 768 field (+0.0181) is **no better** than the
   pooled latent (−0.0165). The information is not being averaged away by the
   readout; on this evidence it is **not there to average**.
3. **The only target v6 clears chance on is `vru_ahead` — and raw pixels do it
   better** (.6191 vs .5823). That is appearance, not learned scene
   understanding.
4. `v6_tokens_pooled` (−0.0151) ≈ `v6_cells` (−0.0165) — the pooling
   reproduction is faithful, which is what licenses the pooled-DINOv3 arm in §6.

## 4. Estimator — one error found and fixed mid-run

⛔ **The first pass averaged PER-EPISODE R² and produced −182 against a pooled
+0.334.** That is not a discrepancy, it is an invalid estimator: a per-episode R²
divides by *that episode's* variance, and an episode with a near-constant lead
gap explodes. The doctrine's estimator is the **episode-CLUSTER bootstrap of the
POOLED statistic** — resample episodes, recompute pooled — and that is what §3
reports. *(Same family as the `overlapping_holdout_se` ban: an estimator that
also moves the point estimate.)*

⚠️ **λ grid.** The first grid (1e−2…1e5) had **68 folds select the MAXIMUM**.
Widened to **1e−4…1e8**, 30 remain — **all at MAX, none at MIN**. That is the
interpretation-safe direction: λ→max means the inner CV wanted the constant
predictor, i.e. **the null itself**, and no fit is under-regularised. ⭐ **Both
load-bearing numbers — `C-EGO`/`lead_gap_m` and `v6_tokens`/`lead_gap_m` — have
ZERO edge folds.**

Protocol: folds **episode-disjoint** (the REF-A I-JEPA lesson — ~80 % of val
inside train made that number unusable); λ chosen on an **inner episode-disjoint
split of the train fold only**; `d ≫ n` (491,520 vs 5,617) so the **dual/Gram**
ridge is used, which is *exact*, with kernel-centering.

## 5. ⛔ What this does NOT license

* **NOT "the v6 encoder is empty."** See §6 — without the DINOv3 arm this
  cannot be separated from *"this protocol detects nothing in anything."*
* **NOT a claim about the finished model.** Step **20,000 of 30,000**.
* **NOT "not present."** A **linear** probe measures *linear* decodability.
  Non-linearly encoded information would read as absent here.
* **NOT a v5.8 comparison.** No v5.8 arm was run. ⚠️ And the premise *"in v5.8
  we could not extract environment information"* **could not be located as a
  MEASURED result** — S-W's gate is P1/P3/P6 and *"consumes no v5.8f
  measurement."* Locate it before building on it.
* **NOT driving performance.** T0-DIAGNOSTIC.

## 6. ⭐ The deciding experiment — the PI's own suggestion, now mandatory

Two very different worlds produce §3, and nothing above separates them:

| world | what DINOv3 would show |
|---|---|
| **v6 genuinely lacks environment information** | DINOv3 recovers `lead_gap_m` well above C-EGO's +0.334 |
| **the protocol lacks power** (5,617 frames, d/n = 88, linear) | DINOv3 also lands at ~0 |

⇒ **Run `dino_tokens` (640 × d, matched granularity) and `dino_pooled`
(through the same 40× pool) on the SAME frames and folds.** `dinov3_extract.py`
is banked and the analogous job took ~3 min on the 4060. This is the arm that
converts §3 from suggestive to conclusive, and it also answers the second
question directly:

* v6tok ≈ dinotok, **both** collapse pooled → the **readout** is the defect
* v6tok ≪ dinotok → the **v6 encoder** is weak; no readout change saves it
* both ≈ 0 → **the battery, not the model, is what needs fixing**

⚠️ The naive form (v6 cells 16×128 vs DINOv3 640×1024) would be the
**E-ACTSTREAM-2 confound again** — encoder and granularity moving together. The
2×2 above is what removes it.

## 7. Manifest

| artifact | where |
|---|---|
| `e_trunk2_probe.py` (arms, dual ridge, cluster bootstrap) | `…/simwam-analysis/code/` |
| `e_trunk2_targets.py` (environment targets from `obstacle.offline`) | `…/simwam-analysis/code/` |
| `e_trunk2_probe.json` (every point, CI, λ, edge count) | `…/simwam-analysis/raw/` |
| feature memmaps + cached Grams | scratchpad `sp2/e_trunk2_feat/` |
| this document | `…/simwam-analysis/E_TRUNK_2_ENV_DECODABILITY.md` |
