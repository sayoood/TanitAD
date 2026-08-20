# LeWorldModel vs v6 — the architecture IS ours, it WORKS, and our SIGReg has been stalled since step 11k

`PUBLISHED-PRIMARY` (**banked 2026-08-20**, `2603.19312`, 28 pp) +
`MEASURED (ours)` · **T0** · written after the PI corrected me: by *"LeWM"* they
meant **LeWorldModel**, not LeJEPA. Both are now banked; this is the directly
relevant one.

---

## 1. LeWM IS v6's architecture — and it works

**Maes, Le Lidec, Scieur, LeCun, Balestriero**, *"LeWorldModel: Stable
End-to-End Joint-Embedding Predictive Architecture from Pixels"* (v3, Jun 2026).
Figure 1, verbatim:

> *"the encoder maps frames into low-dimensional latent representations z₁:T. The
> predictor models the environment dynamics by **autoregressively predicting the
> next latent state z_{t+1} from the current latent state z_t and action a_t**.
> The encoder and predictor are **jointly optimized** using a mean-squared error
> (MSE) prediction loss. LeWM does **not rely on any training heuristics, such as
> stop-gradient, exponential moving averages, or pre-trained representations**.
> To prevent trivial collapse, the SIGReg [regularizer] …"*

**Two loss terms only** — next-embedding MSE + SIGReg — reducing tunable loss
hyperparameters *"from six to one"*. **~15 M params, one GPU, a few hours.**
Plans **up to 48× faster** than foundation-model world models.

⭐ **And §5.1 probes physical quantities — the same instrument as E-TRUNK-2.**
Push-T, linear and MLP probes:

| property | DINO-WM (frozen DINOv2) | PLDM | **LeWM** |
|---|---|---|---|
| Agent Location | r 0.977 | r 0.955 | **r 0.974** |
| Block Location | r 0.997 | r 0.938 | **r 0.986** |
| Block Angle | r 0.979 | r 0.745 | **r 0.902** |

⇒ **An end-to-end JEPA with SIGReg, no teacher and no pretrained encoder reaches
probe r ≈ 0.90–0.99, competitive with frozen DINOv2.**

## 2. ⛔ This REFUTES my own diagnosis, and I withdraw it

I argued the defect was the **self-referential target** — that predicting one's
own future latent is gameable, and proposed an external DINOv3 target. **LeWM has
exactly that self-target and it works.** The mechanism I proposed is not the
mechanism. *(Retraction chain: collapse-onto-ego → "2.3 of 2048 dims" (C128) →
"collapse" as framing → "freeze the encoder" (C129) → **"the self-target is the
problem"**, retired here.)*

⚠️ Also refuted, by measurement rather than argument: I suspected SIGReg had **no
statistical power** at d=2048/n=48 (Diaconis–Freedman). **It has ample power** —
measured at v6's exact live geometry:

| configuration | SIGReg (512 slices, n=48, d=2048) |
|---|---|
| isotropic N(0, I) | **0.420** |
| rank-2 collapsed | 2.997 |
| rank-1 collapsed | 3.560 |
| constant (total collapse) | 20.337 |
| — at LeWM's d=192 | 0.415 / 2.813 / 3.376 — **near-identical** |

## 3. ⭐⭐ THE MEASURED FINDING — SIGReg is not blind, it is LOSING

| | |
|---|---|
| **v6's live latent** | **SIGReg = 7.836** (independently recomputed; the run logs **7.834**) |
| vs isotropic | **18.7× worse** |
| vs **rank-1 collapsed** | ⛔ **2.2× WORSE THAN RANK-1 COLLAPSE** |

**And it has stalled:**

| step | `o6_sigreg` |
|---|---|
| 50 | 9.318 |
| 11,150 | 7.841 |
| 22,300 | **7.834** |

⇒ −16 % in the first 11 k steps, then **+0.09 % over the next 11,150** — flat for
half the run.

**Loss composition at step 22,300:** `o1_ctrl` 0.335 · `o1_fact` 0.519 ·
`o1_scene` 0.0004 · `o6_sigreg` 7.834 × λ 0.1 = **0.783** · total **1.720**.
⇒ **SIGReg is 45 % of the total loss — the single largest term — and it is not
moving**, while the prediction terms continue to fall.

⇒ **The optimiser has found a point where the two objectives cannot both be
satisfied.** SIGReg wants isotropy; the prediction terms are served by
episode-distinguishable structure. The equilibrium is a latent that is **neither
isotropic nor informative about dynamics** — which is exactly what E-TRUNK-2
measured (`lead_gap_m` −0.018) and what the **4.56× between/within** ratio
describes: a marginal that is a mixture of 130 episode clusters, which is
precisely what a normality test flags.

## 4. ⚠️ OUR OWN SOURCE NAMED THIS MECHANISM — but its remedy does NOT apply here

`sigreg.position_relaxed`, verbatim:

> *"Metric ego-position lives in a low-dimensional, structured (non-isotropic)
> subspace; plain SIGReg drives the WHOLE embedding toward an isotropic Gaussian
> and so **actively whitens exactly that structure** — the two objectives
> **partially cancel** (**the diagnosed step-21k regression mechanism**). The
> remedy: EXEMPT a fixed `free_dims`-wide subspace … `free_dims <= 0` reproduces
> plain SIGReg on the full latent."*

**The live run has `sigreg_free_dims = 0`.** The relaxation built to fix this is
**disabled**, and we are at **step 21,650** — the step number the docstring names.

⚠️ **I flagged `free_dims = 0` earlier as "an open item, not a diagnosis" and was
right to hold it there.** The stall since 11 k, the value worse than rank-1
collapse, the immovable 45 % loss share and the matching step number make the
MECHANISM (two objectives partially cancelling) well-evidenced. They say nothing
about whether *this* remedy fits *this* architecture — and it does not:

⛔ **CHECKED IN THE SAME TURN — AND THE PRECONDITION FAILS.** `v6.py` has **no
reserved ego-motion block**: `d_op` = 2048 is the flattened 16 cells x 128 from
the readout, with no semantic column layout. `position_relaxed` exempts *"the
first `free_dims` state columns"*, which in v6 would be an arbitrary slice of
cell 0. The relaxation is documented as *"shared by the flagship and REF-A
trainers"* — architectures whose state **does** reserve ego channels.

⇒ ⭐ **`sigreg_free_dims = 0` is CORRECT for v6 as built. It is not a
misconfiguration, and turning it on would exempt a meaningless slice.** The
step-21k coincidence is real and the *mechanism* (two objectives partially
cancelling) is measured — but **the named remedy does not apply to this
architecture**, and I nearly recommended it on the strength of a matching step
number.

## 5. The other deviations from LeWM, ranked

| # | axis | LeWM | v6 | assessment |
|---|---|---|---|---|
| 1 | **SIGReg outcome** | converges; collapse prevented | ⛔ **stalled at 7.83, worse than rank-1** | ⭐ **the finding** |
| 2 | **`free_dims`** | n/a (plain SIGReg at d≈192) | 0 — **correct**, no reserved block exists (§4) | ⛔ **struck** — not a defect |
| 3 | **latent dim** | **≈192**, measured threshold **184**, *"diminishing returns"* above | **2048** | **major** — 10× past their saturation point, and a far harder isotropy target |
| 4 | **stop-gradient** | ⛔ **none** (stated 4×) | ⚠️ target `z_true_steps` *"detached by the caller"* | **real deviation**, effect unmeasured |
| 5 | **loss terms** | **2** (*"six → one"* is their headline) | **7** (o1×3, o2, o3, o5, o6) | ⚠️ the exact complexity LeWM argues against |
| 6 | **SIGReg placement** | on `z_t` **and** `ẑ_{t+1}` (Fig. 1) | operative latent only | **real deviation** |
| 7 | scale | 15 M params, hours | 336 M, days | context, not a defect |

## 6. ⇒ What I now recommend, and what it replaces

**Replaces:** "predict external DINOv3 features" (withdrawn, §2) and "freeze the
encoder" (withdrawn, C129). **The approach is right — LeWM proves the recipe
works. Our INSTANCE of it is misconfigured.**

1. ⛔ **NOT the §B.3 relaxation** — checked, and v6 has no reserved ego block to
   exempt (§4). Struck before it could be acted on.
2. ⭐ **THE LEADING CANDIDATE: bring the latent dimension toward LeWM's measured
   regime.** They report a threshold at **~184** with *"diminishing returns"* and
   *"performance saturation"* above it; **v6 runs `d_op` = 2048, more than 10x
   past their saturation point.** A 2048-dim isotropy target estimated from
   **n = 48** samples per step is also a far harder constraint to satisfy than a
   192-dim one — which is consistent with a SIGReg term that stalls at 7.83 and
   never converges.
3. **Move SIGReg to where both LeWM's Fig. 1 and our own `sigreg.py` recipe put
   it** — encoder embeddings and predictor outputs, not the operative latent alone.
4. **Reconsider the detach** — LeWM states four times that it uses no
   stop-gradient, and SIGReg is what makes that safe.
5. **Use probing as model selection**, per LeWM §5.1 and LeJEPA Fig. 1.

## 7. ⚠️ Limits

* **T0 throughout.** No driving claim. C129 is why that line is drawn here.
* **§3 shows a STALL and an equilibrium, not a proven cause.** That SIGReg's
  stall *causes* the decodability deficit is a hypothesis with a mechanism; both
  are consequences of the same objective conflict, and separating cause from
  co-symptom needs an ablation.
* **The `free_dims` recommendation is CONDITIONAL** on §4's unverified reserved
  block.
* **I have read 28 pp partially** — Fig. 1, the abstract, §5.1, the latent-dim
  ablation. §3.1's exact loss, the hyperparameter table (§15) and App. F.2 are
  unread.
* **Push-T is not driving.** LeWM's probe results are on 2D/3D control tasks, and
  their transfer to a 2,376-episode driving corpus is not established.
