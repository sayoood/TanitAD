# LeJEPA vs v6 vs what I proposed — I was wrong to reach for a DINO target, and the paper says why

`PUBLISHED-PRIMARY` (**banked 2026-08-20**, `2511.08544`, sha256
`fe06b548…`, 50 pp) + `MEASURED (ours)` · **T0** · written at the PI's challenge:
*"using DINO features … was actually not part of the approach proposed by the
LeWM paper which introduced SigReg."* **The PI is right.** This document
revisits the primary and compares the three designs.

⛔ **The paper was NOT banked until today** — the source of SIGReg, a load-bearing
v6 component, cited only through a docstring. That is the research-banking rule
violated on our most important dependency.

---

## 1. What LeJEPA actually is

**Balestriero & LeCun, "LeJEPA: Provable and Scalable Self-Supervised Learning
Without the Heuristics"** (2025-11-11). Two claims:

1. the **isotropic Gaussian** is the embedding distribution that **minimises
   downstream prediction risk**;
2. **SIGReg** (sketched, Epps–Pulley, Cramér–Wold) reaches it **without
   stop-gradient, teacher–student, EMA, or schedulers**.

The objective, verbatim in structure:

> **L_LeJEPA = (λ/V) Σ_v SIGReg({z_{n,v}}_{n=1..B}) + ((1−λ)/B) Σ_n L_pred**,
> with **L_pred = (1/V) Σ_{v′} ‖μ_n − z_{n,v′}‖²**, **μ_n = (1/V_g) Σ_v z_{n,v}**

⭐ Two structural facts follow, and both matter:

* **SIGReg is applied to the ENCODER EMBEDDINGS `z`, per view, across the batch
  of SAMPLES `n`.**
* **The predictive term is VIEW-INVARIANCE** — every view of sample `n` is pulled
  to the centroid of that sample's views. And views are *"data-augmentations for
  images, **frames for videos**"*.

Validation: 10+ datasets, 60+ architectures; ImageNet-1k **linear probe on a
frozen backbone, 79 % (ViT-H/14)**; train loss ↔ linear-probe accuracy Spearman
**94.52 %**.

## 2. ⭐ The granularity argument — why the guarantee does not transfer to us

LeJEPA's optimality is about the **marginal distribution over the samples `n`
that SIGReg is computed across**. The downstream task is assumed to live at
**that same granularity**:

| setting | one sample `n` | views | downstream task | aligned? |
|---|---|---|---|---|
| LeJEPA, images | an image | augmentations | classify **the image** | ✅ |
| LeJEPA, video | a clip | **frames** | classify **the clip** | ✅ |
| **v6 S-W** | a **window** | — | **decode / predict DYNAMICS INSIDE the window** | ⛔ **NO** |

⇒ **We applied a correct theorem at the wrong granularity.** Isotropy *over
windows* says nothing about informativeness *within* a window. A representation
that encodes only "which scene am I in" is **perfectly isotropic across a batch
of different episodes** and **trivially predictable forward in time**
(`z_{t+k} ≈ z_t`) — it satisfies **both** of v6's terms while carrying no
dynamics.

**MEASURED, directly on the banked latents (5,617 frames, 130 episodes):**

| representation | between-episode | within-episode | ratio |
|---|---|---|---|
| **`v6_cells`** | 0.0240 | 0.0053 | ⛔ **4.56×** |
| `dino_pooled` | 178.51 | 72.16 | **2.47×** |

(The 4.56× reproduces the independently banked 4.5×.)

⚠️ **And for LeJEPA-on-video this is not a bug — it is the DESIGNED OUTPUT.** Its
predictive term makes the embedding *invariant across frames of a clip*. A
clip-level fingerprint is exactly what that objective is for. **v6 does not use
LeJEPA's `L_pred`** (it uses action-conditioned future prediction), but it
imported SIGReg from a framework whose companion term removes within-clip
variation — and our latent carries that signature.

## 3. Where v6 deviates from the recipe — two concrete, checkable gaps

| | LeJEPA / ALPS-4B recipe | v6 S-W as running | evidence |
|---|---|---|---|
| **where SIGReg applies** | *"encoder embeddings AND predictor outputs at **all hierarchy levels**"* (`sigreg.py`) | ⛔ **operative latent only** — `# ---- O6: SIGReg on the operative latent ----`, on `states` (the READOUT output) | source read |
| **λ** | 0.1 | ✅ `w_o6 = 0.1` | live config |
| **slices** | 512 | ✅ `sigreg_slices = 512` | live config |
| **free_dims** | §B.3 relaxation exempts an ego subspace | ⚠️ `sigreg_free_dims = 0` ⇒ *"reproduces plain SIGReg on the full latent"* | docstring |
| **model selection** | **linear probe** (their own Fig. 1, ρ=94.52 %) | ⛔ never run until 2026-08-20 | E-TRUNK-2 |

⭐ **The encoder's 640×768 field is NOT regularised at all** — and E-TRUNK-2
measures it decoding nothing (`lead_gap_m` +0.013, spans zero). The one place
LeJEPA puts SIGReg is the one place v6 does not.

⭐ **And the probe is not a foreign metric.** Linear evaluation on a frozen
backbone is **LeJEPA's own headline benchmark**. By the paper's own methodology
v6's latent is failing — that is not an imported standard.

## 4. ⛔ My DINOv3-target proposal, judged against the primary

I proposed: keep the encoder trainable, predict **frozen DINOv3 future features**.

| | |
|---|---|
| **Is it LeJEPA?** | ⛔ **No.** LeJEPA's entire thesis is *"without the heuristics"* — **no teacher**. A frozen DINOv3 target IS a teacher. It is distillation with extra steps. |
| **Does it abandon the programme's thesis?** | ⚠️ **Partly.** `D-003`'s from-scratch main track is what makes the data-efficiency claim disruptive. Training our encoder to reproduce DINOv3's features caps us at DINOv3 and makes "we learned it from 2,376 episodes" untrue. |
| **Is it a bad idea?** | Not *bad* — DeepSight measures it working (Bench2Drive DS 86.23). But it is a **different programme**, and it should be proposed as one, not smuggled in as a fix. |

⇒ **I withdraw it as the primary recommendation.** It stays as a legitimate
comparison arm — which is exactly the status `D-003` assigns to frozen-encoder
approaches.

## 5. ⭐ The faithful alternative — fix the granularity, keep the doctrine

Three changes, all inside LeJEPA, none requiring a teacher:

1. ⭐ **Apply SIGReg at the granularity the downstream task lives at.** Constrain
   the **within-episode residual** `z_t − μ_episode` to be isotropic, in addition
   to (not instead of) the marginal. Same Epps–Pulley machinery, same λ, no
   teacher, no EMA. This directly attacks the measured 4.56× and is the minimal
   change consistent with the paper's own logic — its theorem says *make the
   distribution the downstream task is defined over isotropic*, and ours is
   defined over within-episode dynamics.
2. **Put SIGReg where the recipe says** — encoder embeddings and every hierarchy
   level, not the operative latent alone.
3. **Adopt LeJEPA's own model-selection metric**: linear probe. Their Fig. 1
   gives ρ = 94.52 % against train loss; we have the harness and it is 0-GPU on
   banked features.

⚠️ **(1) is a HYPOTHESIS with a mechanism, not a result.** It predicts that
within-episode SIGReg raises `lead_gap_m` decodability above the C-EGO line. That
is testable with E-TRUNK-2 at zero GPU **after** a short training run, and must
be **pre-registered with both outcomes** before it decides a GPU-day.

## 6. What this does NOT claim

* **Not that LeJEPA is wrong.** Its theorem is about the marginal it is computed
  over; the misapplication is ours. On its own benchmarks it reaches 79 %
  ImageNet linear probe and beats DINOv2/v3 ViT-S/16 transfer in-domain.
* **Not that within-episode SIGReg will work.** Unmeasured. §5(1) is a proposal.
* **Not a T1 claim** anywhere. C129 exists because I crossed that line once.
* **Not a full read of the paper.** 50 pages; I have read the abstract, the
  objective, the view definition and Fig. 1. §B.3 (the relaxation
  `sigreg_free_dims` implements) is **unread**, and it bears directly on gap 4 in
  §3.
