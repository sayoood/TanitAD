# v7-tiny — what the 29-minute rig found

`RESULT, 2026-08-22.` Tier **T0-DIAGNOSTIC** throughout. Nothing here is a
driving claim. All measurements on **24 HELD-OUT clips** pulled from the 2,270
episodes of `physicalai-train-e438721ae894` that neither arm ever saw
(`v7tiny-heldout24-w120-256x640cyl`, NON-PARITY by construction, provenance in
`raw/v7tiny_heldout24_provenance.json`).

---

## 1. ⭐ SOLVED — the residual-init defect, confirmed by a deliberate regression

Two arms of v6's **real** trainer (`train_v6_staged.py --stage S-W`), all six
o-terms at v6's own weights, same data, same seed, 2,000 steps, **one variable**:
`TANITAD_RESIDUAL_INIT_SCALE`.

| horizon | `fixed` (1e-3) | `regress` (1.0 = the v6 default) |
|---|---:|---:|
| h=1 (0.1 s) | −908 [−1549, −517] | −5,759 [−8466, −4180] |
| h=2 (0.2 s) | **−0.040** [−0.059, −0.029] | **−45,711** [−67539, −33281] |
| h=4 (0.4 s) | **−0.019** [−0.030, −0.013] | **−24,028** [−39322, −15865] |

Paired episode-cluster bootstrap over the same resampled clips: `fixed` beats
`regress` at every horizon, **p < 0.0001**.

* At h=2 the fix moves the predictor from **45,712× worse than hold to 1.040×** —
  from catastrophically broken to sitting essentially *on* hold.
* ⭐ **The gate can fail the arm it was built to catch.** §4 of `V7_TINY_DESIGN.md`
  required a deliberate regression that the gate must reject. It does. A gate
  that only ever returns "inconclusive" proves nothing; this one discriminates.
* ⛔ **But neither arm PASSES G2.** `fixed`'s best is −1.9 %: at hold, not past it.

Fix shipped in 3 modules / 6 instantiation sites / 5 models; the init regime is
now recorded natively by the trainer (`config.json`, the done-marker, and a
startup banner) — previously the two arms' configs were **identical on the one
variable that differed**.

## 2. ⭐⭐ FOUND — the trunk carries essentially no ego state

Linear probe, PCA k=128 fit on training clips only, λ chosen on a validation
split of the fit clips, episode-disjoint, n=1800 rows. R² is **over the
training-mean predictor**, so a representation carrying nothing reads 0.0000 —
and a constant-only control is carried in every panel and does read exactly that.

| target | v6F @20k latent | CI95 |
|---|---:|---|
| speed | **+0.0025** | [−0.0180, +0.0102] |
| yaw | −0.0001 | [−0.0005, +0.0002] |
| yaw_rate | **−0.4504** | [−0.9320, −0.2782] |
| d_ego | +0.0001 | [−0.0015, +0.0009] |

⚠️ **The honest qualification:** v6 supplies speed to the predictor through the
**action channel** (`_lift3` carries `v0/SPEED_SCALE`), so the architecture does
not *require* the encoder to encode it. That weakens "this is a bug"; it does not
weaken "this is a problem". A world model whose latent has no ego motion in it is
not carrying a state, and `yaw_rate` at −0.45 is worse than predicting the mean.

## 3. What was ELIMINATED, and at what cost

| hypothesis | verdict | evidence |
|---|---|---|
| the residual init explains the stall | ⛔ **partly only** | fix confirmed, but `fixed` still only reaches hold |
| **O6/SIGReg isotropy destroys temporal structure** | ⛔ **REFUTED** | `no-o6` +0.0016 vs `fixed` +0.0018 — identical (30 min) |
| our encoder is uniquely bad at temporal prediction | ⛔ **REFUTED** | raw pixels score +0.0010, v6F +0.0203 — the latent is **20× better than pixels** |
| the dynamics hides in a low-dim subspace of Δz | ⛔ **REFUTED** | top-2…32 PCs of the ego-explainable part: EM ≈ 0 at every k |
| the latent's Δ is ego-motion-driven | ⛔ **REFUTED** | ego motion explains **−0.0007** of Δz |

⭐ **The single most useful number in this whole pass is the pixel floor.**
"Per-tick change is ~98 % unpredictable" turned out **not** to be a property of
our encoder — it is what per-tick change looks like in *any* high-dimensional
visual representation at 10 Hz. Without that floor the finding would have been
written up as an encoder indictment, which is what nearly happened.

## 4. ⛔ Four methodology failures in my own instrument — and what caught each

This belongs in the record because the pattern matters more than any one bug.

| # | failure | what it produced | caught by |
|---|---|---|---|
| 1 | EM normalised by the target's **raw energy** when the target had a large constant (ego moves ~0.47 m/tick) | latent, [z,dz] and **pixels** all "+0.54" | the pixel floor matching to 3 decimals |
| 2 | λ selected on the **point estimate** | reported +0.130 with a CI straddling zero; hid the real +0.020 whose CI excluded zero | reading the whole sweep |
| 3 | λ selected on the **test set** | λ=10⁶ ⇒ constant predictor ⇒ everything "+0.0000" | the constant control reading identically |
| 4 | **n ≪ d** (2,050 features on ~700 rows) | ridge correctly collapsed to a constant; read as "the latent has nothing" | the constant control again |

**Common root cause: tuning on the scored data, and asking a 2048-dim question
with 700 rows.** Fixed once, properly: PCA fit on train only, λ on a validation
split of the fit clips, full sweep stored so a degenerate pick is visible on
sight, constant control in every panel.

⚠️ **A fifth, of a different kind — an INFERENCE error, not a code error.** The
oracle is **linear**. "A linear map cannot predict Δz" licenses only *not
linearly predictable*; it does **not** license *unpredictable by any predictor*,
which is what my verdict string asserted. The implication runs one way only:

    linear oracle BEATS hold  =>  target IS learnable        (valid)
    linear oracle FAILS       =>  not learnable BY A LINEAR MAP  (all it says)

Corrected in code; `code/v7tiny_mlp_oracle.py` exists to answer the nonlinear
question with two controls that decide readability — `ego` must score high, and
**time-shuffled Δz must score ~0** (structure surviving a shuffle is leakage,
not dynamics).

## 5. ⚠️ What this does NOT establish

* **Not a driving result.** T0 only.
* **Nothing about REF-A, flagship v1, REF-C v3 or REF-D.** Their trunks were not
  probed; the residual-init fix was applied to them, its *effect* there is
  unmeasured.
* **The 580× on v6 is not retracted** — verified at source that `z_true_steps` is
  *"ENCODED true future latents"*, so O5 is a latent-space loss and the
  measurement was on the right quantity. What changed is its **interpretation**:
  hold is a much stronger baseline in this representation class than the number
  makes it sound.
* **v7-tiny is 2,000 steps at 19 M.** Undertrained remains a live alternative for
  its own numbers; v6F @20k / 336 M is the load-bearing evidence.
* **D-A5's "the frozen encoder is REF-A's ceiling"** is now questionable but NOT
  overturned — the apples-to-apples frozen-DINOv3 column on these clips under
  this probe is the measurement that would settle it.

## 6. The rig itself is the durable deliverable

**29 minutes per arm**, v6's real trainer, real PhysicalAI data, 256×640, all six
o-terms, with a regression arm proving the gate discriminates. A two-arm ablation
is under an hour; the three-arm o-term ladder is ~95 minutes. That replaces
nine-day runs as the way this programme asks a question.

⚠️ It is **I/O-bound, not compute-bound** — GPU sat at ~1 % decoding PNG frames
at `--v2-lru 6`. Raising the LRU or pre-decoding a frame bank would cut the 29
minutes substantially, and at this size compute is nearly free.

## 7. Open, in priority order

1. **Frozen DINOv3 on these clips under this probe** — the column that decides
   whether v6's training *destroys* ego information a generic encoder has.
2. **The nonlinear oracle** — closes the linear-probe inference gap.
3. `o5-only` and `no-o1` arms of the ladder (running).
4. If the encoder is confirmed as the constraint: v7's design question is what
   supervises the trunk, not how big it is.
