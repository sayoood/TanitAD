# E-V6SHAPE — a v6-shaped LeWM at the reference's own training length encodes NEITHER environment NOR ego

`MEASURED (ours; dev-box RTX 4060, 2026-08-20/21)` · **T0-DIAGNOSTIC** ·
scored on the **same 5,617 keys, same order, same episode-disjoint folds** as
E-TRUNK-2, so every row sits in the same table as `v6_cells` and `dino_pooled` ·
**Thor untouched throughout.**

⚠️ **This supersedes `E_LEWM_1_ABLATION.md`, which was retracted as evidence
about LeWM** (~14 deviations from `github.com/lucas-maes/le-wm`). This harness is
built to v6's structure with the reference's settings and is a much closer
replication — but see §5, the Push-T caveat still stands.

---

## 1. What was built, and why v6 is the right shape

Reading the reference implementation established that **v6 already matches LeWM
on every structural axis my previous harness got wrong**:

| axis | LeWM reference | v6 | old harness |
|---|---|---|---|
| per-frame encoder → `z_1:T` | ✅ | ✅ `encode_window` | ✅ |
| predictor over a **history window** | ✅ 3 | ✅ **6** | ⛔ 1 |
| transformer predictor + pos emb | ✅ | ✅ | ⛔ MLP |
| residual `z + Δ` | ✅ | ✅ | ✅ |
| ⭐ **detach target** | ✅ **the CODE detaches** | ✅ | (made it an axis) |
| SIGReg on the per-frame latent | ✅ | ✅ | ✅ |

⛔ **The paper claims "no stop-gradient"; the code contradicts it** —
`F.mse_loss(pred_emb[...,-1:,:], goal_emb[...,-1:,:].detach())`. ⇒ v6's
`z_true_steps` detach is **faithful**, and I had twice quoted it as a deviation.

This harness therefore restores: `WINDOW=6`, AR transformer predictor with causal
mask, **CLS token → MLP projector with BatchNorm**, lr 5e-5, wd 1e-3, grad-clip
1.0, **additive** loss (`pred + λ·sigreg`, λ = 0.09), detach, SIGReg on the
per-frame latents. **Params matched to 0.8 %** across the readout axis
(6.37 M vs 6.42 M).

## 2. Result — at the reference's own 100 epochs

| arm | epochs | SIGReg | b/w | `lead_gap_m` | `left_occ` | `right_occ` |
|---|---|---|---|---|---|---|
| `cls192` | 7.4 | 1.919 | 3.27 | −0.0105 | .4474 | .5374 |
| ⭐ **`cls192`** | **100** | **1.468** | **2.11** | **−0.0068** | .5124 | .4754 |
| `v6shape` (d=2048 pool) | 7.4 | 3.648 | 5.00 | −0.0087 | .5343 | .5305 |
| **`random` — UNTRAINED** | 0 | — | 6.89 | −0.0156 | .5082 | .5659 |
| **supervised CONTROL** | 7.4 | — | 1.58 | **+0.9934** | **.9941** | **.9958** |
| *ref `dino_pooled`* | — | — | *2.47* | *+0.3792* | *.8312* | *.8236* |
| *ref `v6_cells`* | — | *7.83 stalled* | *4.56* | *−0.0176* | *.5321* | *.5890* |

⭐ **Training WORKED on every statistic we can name.** 13.5× more steps moved
`b/w` **3.27 → 2.11** — the best non-supervised value in the whole study, better
than **DINOv3's 2.47** — and SIGReg **1.919 → 1.468**, converging toward the
0.42 isotropic floor. Prediction loss fell **0.33 → 0.108**.

⛔ **And decodability did not move.**

## 3. ⭐⭐ What the latent actually contains — the finding

`cls192` at 100 epochs, all nine targets:

| target | `cls192` | `random` | supervised |
|---|---|---|---|
| `lead_gap_m` | −0.0068 | −0.0156 | **+0.9934** |
| `nearest_any_m` | −0.0131 | +0.0361 | +0.3966 |
| `n_agents_log` | −0.0313 | +0.0785 | +0.3676 |
| `left_occupied` | .5124 | .5082 | **.9941** |
| `right_occupied` | .4754 | .5659 | **.9958** |
| `vru_ahead` | +0.6034 | +0.5547 | +0.6582 |
| ⛔ **`ego_speed`** | **−0.0298** | +0.0334 | **+0.9758** |
| `ego_yawrate` | −0.0719 | −0.0114 | — |
| `ego_accel` | −0.0206 | −0.0672 | — |

⇒ ⛔ **IT ENCODES NEITHER THE ENVIRONMENT NOR THE EGO.** Not even its own speed,
which the supervised control reads at **0.9758** and frozen DINOv3 at **+0.6957**.
The single positive, `vru_ahead` +0.6034, is **below raw pixels' +0.6191**.

**This rules out all three standing stories at once:**

| story | verdict |
|---|---|
| **ego echo** (v1f's failure — supervised-on-trajectory collapsing to ego dynamics) | ⛔ **no** — `ego_speed` −0.0298 |
| **scene fingerprint** (the granularity argument) | ⛔ **no** — b/w 2.11, better than DINOv3 |
| **collapse** | ⛔ **no** — SIGReg 1.468, converged |

⭐ **A well-conditioned latent that is informationally empty on every measurable
axis, while its predictor demonstrably learns** (loss 0.33 → 0.108). Whatever it
encodes is *predictable forward* and is *none of our nine targets*.

## 4. ⚠️ The hypothesis this leaves — UNTESTED, and I have been wrong eight times

In a **forward driving camera** the next-latent target is dominated by
**ego-motion-induced optical flow**. The cheapest sufficient statistic for
predicting that is **low-level photometric / texture structure**, not object
semantics — so an objective that only asks "predict the next latent" can be
satisfied without representing a single object.

⚠️ **Recorded as a hypothesis with a mechanism.** *(Refuted so far, in order:
collapse-onto-ego · "2.3 of 2048 dims" (C128) · "collapse" as framing · "freeze
the encoder" (C129) · "the self-target is the problem" · Diaconis–Freedman ·
"undertraining" (twice) · "no history + wrong predictor". Eight.)*

**The cheap test:** probe for low-level targets — frame gradient energy, mean
brightness, optical-flow magnitude. If those decode well while semantics do not,
the hypothesis holds. Not run.

## 5. ⛔ What is still NOT excluded

**That this is an unfaithful replication.** The pre-registered gate — *the
baseline must decode* — **still has not passed**, and I did **not** run LeWM on
**Push-T**, which is the only way to separate *"the objective does not transfer
to driving"* from *"my code is wrong"*.

Remaining deviations from the reference: **512 SIGReg slices vs 1024** (declared:
Epps-Pulley is O(n²)/slice and 1024 cost 11.7 h/arm), **5-frame tick vs 1**
(declared: at k=1 a driving latent moves 1.12 % of its magnitude, so the identity
map explains 98.9 %), **128×320 patch-16 vs 224² patch-14**, and my ViT vs
`stable_pretraining.vit_hf` tiny.

⭐ **What raises this above "probably a bug":** the harness **reproduces v6's
signature** — `v6shape` (the 2048-d pooled readout) lands at b/w **5.00** against
v6's **4.56**, with elevated slow-converging SIGReg where v6 stalls at 7.83,
while `cls192` (192-d CLS readout) reaches **2.11 / 1.468**. **The readout axis
behaves exactly as LeWM's measured ~184-dim saturation predicts**, and that is a
discriminating result inside the study even though the gate is open.

## 6. What this does and does not license

✅ **Licensed:** *at this scale, on this corpus, in this implementation, an
action-conditioned next-latent objective with SIGReg produces a well-conditioned
latent carrying no measurable environment or ego content — and 9 confounds are
excluded by measurement.*

⛔ **NOT licensed:** any statement about **LeWorldModel**; any statement about
**v6 at 336 M on 2,376 episodes** (this is 6.4 M on 130 clips); any **T1/driving**
claim whatsoever.

⚠️ **And the readout finding is NOT a v6 verdict either.** That `cls192` beats
`v6shape` on b/w and SIGReg is a real, discriminating measurement — but with
**both at the decodability floor**, it says the 192-d CLS readout is *better
conditioned*, not that it is *better*.
