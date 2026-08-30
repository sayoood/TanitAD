# PRE-REGISTRATION — MM-E6: split drift into SELF-REFERENCE and ENVIRONMENT

**Written 2026-08-30 ~02:45, before any compute** · Master Mind · **Hypothesis** MM-E6 ·
**Tier** T0-DIAGNOSTIC · **Status** design complete, ⛔ NOT LAUNCHED (PI decision #5).

## Why this exists — it is the successor problem MM-E4/E5 produced

MM-E4 killed the direct attack: drift falls to *any* dynamics-perturbing term including a
meaningless one, and every such fall costs prediction. MM-E5 explained why: **drift and
prediction quality are largely the same property** — the predictability of Δz from z_t.
⇒ The programme's actual question is not "how much drift" but **"predictable BY WHAT"**:

- **environment-driven** predictability — Δz is foreseeable because the *scene* implies it
  (a car ahead is braking; the road curves). This is what a world model is FOR.
- **self-referential** predictability — Δz is foreseeable from the latent's own coordinates
  regardless of scene content (momentum in representation space). This is the pathology.

Today's single number cannot tell them apart, which is why every attack on it failed.

## The design

Split the latent by what the scene explains, then measure drift **within each part**:

```
z_env := P(z_t)            # the projection of z_t onto the span of scene features
z_res := z_t − P(z_t)      # the residual: latent content the scene does NOT explain
```
`P` = ridge from a FROZEN scene representation (DINOv3 embedding of frame t — an
independent encoder, so it cannot inherit our latent's pathology) onto z_t, **fit on the
FIT split only**, applied to the SCORED split. Then, with the existing drift instrument:

| read | meaning |
|---|---|
| `r_env` = drift computed from **z_env** alone | environment-driven predictability |
| `r_res` = drift computed from **z_res** alone | ⛔ self-referential predictability |
| `r_full` (today's drift) | the mixture we have been quoting |
| `share_self` = r_res² / (r_env² + r_res²) | **the number the programme has been missing** |

```yaml
controls (committed, each must read its known value):
  constant        : r == 0 exactly
  scene_shuffled  : P fit with scene features from OTHER clips ⇒ z_env carries no scene
                    information ⇒ r_env must collapse toward r of a random projection of
                    the same rank. ⛔ If it does not, P is not measuring the scene and the
                    run is void.
  rank_matched    : a RANDOM projection of the same rank as P, to prove r_env is not merely
                    "drift survives any projection" — the MM-E4 lesson applied preemptively
  fit_split       : P and every ridge fit on FIT only; scored split never tuned on
arms (all EXISTING checkpoints, no training):
  postrain30k · o14fut30k · emao14_30k · splitp30k (frozen) · e4_l4_crop
reads: r_env, r_res, r_full, share_self, all with the episode-cluster bootstrap
cost: probe-only on the 4060, ~2-3 h total; ⛔ ZERO training, ZERO Thor
```

## Committed outcomes

| outcome | criterion | consequence |
|---|---|---|
| **SELF-DOMINATED** | `share_self` > 0.6 on the trainable arms AND materially lower on the frozen arm | the pathology is confirmed AND localised — the target becomes `r_res`, which no term tested so far was aimed at; the drift attack REOPENS with a read that a meaningless regulariser cannot move (it would have to lower r_res while holding r_env) |
| **ENVIRONMENT-DOMINATED** | `share_self` < 0.4 across arms | ⛔ **the drift attractor was never the blocker** — high drift is the world model working, and the L4/T1 failures must be explained by something else entirely. This would retire a claim the programme has carried since E-DEC-61 |
| 🔶 **MIXED** | 0.4 ≤ share_self ≤ 0.6 | report both components per arm; the split becomes a standing reported quantity, no single-number drift claim is made again |
| ⛔ **VOID** | the scene-shuffled control does not collapse r_env, or the rank-matched control matches P | the projection is not measuring what it claims; read NOTHING, fix the instrument |

⭐ **Why this is worth running even if the answer is uncomfortable:** every drift number in
the programme — including the ones in the paper — is currently `r_full`, a mixture whose
composition is unknown. If ENVIRONMENT-DOMINATED fires, we have been treating a healthy
signal as a disease for three weeks. If SELF-DOMINATED fires, we get the first read that
the MM-E4 failure mode structurally cannot fake.
