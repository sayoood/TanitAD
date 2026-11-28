<title>PREREG_V7_SEED_POS — the positional seed repair, one variable, both outcomes committed</title>

# PREREG_V7_SEED_POS — repairing the DINOv3→`ViTEncoder` seed

**Owner** Architecture & Inference FlyWheel · **written** 2026-09-04 · **status** PRE-REGISTERED, not run
**Depends on** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-04-v7-seed-and-external-target/RESULT.md` (E-SEED-2, MEASURED)
**Relation to `PREREG_V7F.md`** this is rung **R3-minus-one**: it repairs the trunk *before* v7f's `trunk_policy` ladder spends an arm comparing `{frozen | full | anchored | lastk}` on a trunk that is not separable from a random encoder. ⛔ **If this prereg is skipped, `PREREG_V7F` is void for its trunk element**, by its own §2 rule (*"if a rung is skipped, this prereg is void for the element that skipped it"*).

---

## 0. Why this exists in one paragraph

`dinov3_seed_checkpoint.py` maps **292/292** DINOv3 block tensors and declares two losses honestly: DINOv3 is **RoPE-only** so `ViTEncoder`'s learned absolute `pos` is **left at `trunc_normal_(std=0.02)`**, and CLS/register/mask tokens are dropped. Both are claims about **tensors**. **MEASURED 2026-09-04** (E-SEED-2, 24 clips, 2,372 rows, 256×640, clip-disjoint, paired clip-cluster bootstrap): the trunk as `V6Stack.encode_window` actually feeds it is **not separable from a randomly-initialised encoder** (`seed_asis − scratch` = +0.1671 [−0.1742, +0.6124]) while published DINOv3 **is** (+0.5021 [+0.0779, +1.0576]). Adding ImageNet normalisation **and** zeroing `pos` recovers a **separated** +0.2165 [+0.0243, +0.4567] and leaves the seed **not separable from published DINOv3** (+0.1185 [−0.0960, +0.2628]). This prereg turns that step-0 reading into a trained-arm test.

---

## 0b. ⛔⛔ SCOPE NARROWED BEFORE THIS WAS EVER RUN — read this first

**E-SEED-2c (`…/2026-09-04-v7-seed-and-external-target/RESULT.md` §2.4) ran the SAME cached features through the register's OWN environment instrument** (`rangeprobe_rff.rff_fold` + `within_clip_r` + `panel_kfold.kfold_clip_scores`, imported not re-implemented) **and the repair's benefit does not appear on the scene axis**: `seed_imnet_pos0 − seed_asis` **+0.0295 [−0.0690, +0.1246] NOT separated**, and the repaired seed does **not** beat the raw-pixel floor (**+0.0061 [−0.1236, +0.1318]**). Under that instrument **not one speed contrast is separated either**, `dino_hf − scratch` included — so §0's ego numbers are a property of the linear/pooled-R² read.

⇒ **This prereg is kept, with its success criterion rewritten to require the SCENE axis, because the change is free and correct hygiene and because a null on a pre-registered cheap arm is a result.** ⛔ **It is no longer the programme's leading recipe change.** That is now `PREREG_V7F` §10 **D1 option A** — wrap the real `DINOv3ViTModel` so RoPE comes with the weights — because the seed loses a **separated** fraction of DINOv3's scene content (`dino_hf − seed_asis` **+0.2050 [+0.0896, +0.3233]**; still **+0.1755 [+0.0525, +0.2959]** after both repairs) and **no cheap wiring change recovers it**.

⭐ **Why the honest thing is to keep it rather than delete it:** the failure branch below was written before the arm existed, the arm was run, and it fired. Deleting the prereg would hide the one measurement in this package that refuted its own author.

---

## 1. Hypothesis (register id proposed)

> **`H-V7SEED-1` — Initialising `ViTEncoder.pos` to ZERO (rather than `trunc_normal_(std=0.02)`) when seeding from a RoPE-only trunk, with the ImageNet affine folded into the patch-embed `Conv2d`, raises the seeded trunk's decodability at step 0 to a level not separable from published DINOv3, AND that advantage survives a short trained arm rather than being erased in the first few hundred steps.**

Mechanism (HYPOTHESIS, not measured): a random absolute-position table is a **fixed additive perturbation** applied to patch embeddings produced by blocks that were **never trained with an absolute table**; under RoPE the positional phase is carried inside `Wq`/`Wk` (`lib:2403.13298` §3 Eq. 16, verbatim: *"RoPE does not need additional parameters for phase shift"*), so the transplanted projections meet a signal they were not co-adapted to. Zero is the unique value that leaves the transplanted operator undisturbed while remaining trainable.

---

## 2. SPEC block

```yaml
hypothesis: H-V7SEED-1
one_variable: pos_init            # {trunc_normal_0.02 | zeros}
                                  # the ImageNet fold is applied to BOTH arms
                                  # (it is not the variable; E-SEED-2 measured
                                  #  it is not separated on its own)
held_constant: [seed_source_sha256, enc_dim, enc_depth, enc_heads, patch,
                frame_h, frame_w, projection, corpus, clip_split, seed, steps,
                batch, window, o5_k, o5_target, w_o5, w_o6, cond_param, lr,
                probe_lambda_grid, pca_k, k_outer, k_inner, n_boot]
success: "AT STEP 0 AND ON THE SCENE AXIS -- (repaired - as_wired) `n_agents`
          delta > 0 under the REGISTER'S OWN instrument (rff_fold +
          within_clip_r + kfold_clip_scores) with a paired clip-bootstrap CI
          EXCLUDING 0, AND (repaired - pixel) > 0 with its CI excluding 0.
          !! THIS IS THE BINDING HALF AND IT IS CURRENTLY FAILING: E-SEED-2c
          MEASURED +0.0295 [-0.0690, +0.1246] and +0.0061 [-0.1236, +0.1318].
          The EGO half -- (repaired - as_wired) speed R2 delta > 0 under a
          linear ridge, MEASURED +0.2165 [+0.0243, +0.4567] -- is reported but
          is NOT sufficient on its own (E-DEC-17: ego is free);
          AND at step 2,000 of a matched tiny arm the repaired arm's
          Observer-Effect monitor ratio rho(2000)/rho(0) is NOT BELOW the
          as-wired arm's, paired CI excluding 0 in the repaired arm's favour
          OR containing 0 (i.e. the repair is not paid back);
          AND the repaired arm's step-0 gap to published DINOv3 remains
          NOT SEPARATED"
failure: "(a) the step-0 delta's CI contains 0 on the v7 launch geometry
             => E-SEED-2 does not replicate at that geometry and the repair is
             REFUTED there; report the geometry, do not generalise;
          (b) the repaired arm's monitor ratio at 2,000 steps is SEPARATED
             BELOW the as-wired arm's => zeroing `pos` buys a better start and
             a worse trajectory; report as SEED-REPAIR-WORSE, never as 'inert';
          (c) the deliberate-regression arm (see 4) CLEARS the gate
             => the gate is VOID and no arm passes it"
controls: [constant_only, raw_input_floor, deliberate_regression,
           time_shuffled, instrument_validity]
splits: {fit: "the 4-fold clip-disjoint TRAIN clips; PCA basis and ridge lambda
               fitted inside each fold's train clips only",
         val: "the inner 4-fold GroupKFold over those same train clips",
         test: "the out-of-fold clips; never tuned on, no hyper-parameter
                selected on a row it scores"}
```

⛔ **The one-variable check, diffed rather than intended.** The two arms differ in **exactly one tensor's initialisation**. The ImageNet fold is applied to **both**, precisely because E-SEED-2 measured it is **not** separated on its own (+0.0992 [−0.1100, +0.2910]) — putting it on one side would confound the arm with a change we have already measured to be individually null.

---

## 3. The change, exactly

In `stack/scripts/dinov3_seed_checkpoint.py`:

1. **`pos`**: emit an explicit **zero** tensor for the target's `pos` key instead of leaving it out of the seed dict (today it is the sole `left_at_init` key, so `ViTEncoder.__init__`'s `trunc_normal_` survives). The provenance stamp's `left_at_init_keys` must become `[]` and a new `zero_initialised_keys: ["pos"]` must appear, so the change is visible in every downstream artifact.
2. **ImageNet affine**: fold `x ← (x − μ)/σ` into `patch.weight` / `patch.bias` (`W' = W/σ` per input channel, `b' = b − Σ W·μ/σ`), the same algebraic fold the converter already performs for LayerScale, and record it in the stamp as `imagenet_affine_folded: true`.

⭐ **Both must land in the CONVERTER, not in the trainer.** `build_trunk_anchor` deepcopies `stack.encoder` **after** `apply_encoder_seed`, so a repair applied anywhere later leaves the anchor's frozen teacher carrying the defect it is supposed to anchor away from.

⚠️ **Compatibility.** Every existing seed checkpoint keeps its behaviour; the new stamp fields make old and new seeds distinguishable at a glance, and a run that mixes them must refuse.

---

## 4. ⛔ THE DELIBERATE-REGRESSION ARM — the gate must FAIL it

**`scratch`** — a `ViTEncoder` at random init, ImageNet-normalised input, everything else identical.
⛔ **Committed in advance: if the gate does not FAIL `scratch`, a PASS on the repaired arm means nothing and the panel is stamped VOID.**
**Already MEASURED at step 0** (E-SEED-2): `scratch` reads spatial speed **+0.0050 [−0.613, +0.384]** — the interval contains the no-information value — while `dino_hf − scratch` is **+0.5021 [+0.0779, +1.0576] SEPARATED**. The gate fails it today.

**A second regression arm for the trained rung:** `pos_random_frozen` — `pos` at `trunc_normal_` **and excluded from the optimizer**. If training simply learns its way out of a bad `pos`, this arm must **not** recover, and if it does the repair is a convergence-speed result rather than a representation result. That distinction is committed now so it cannot be chosen later.

---

## 5. Controls carried on every panel

| control | must read | why |
|---|---|---|
| **constant-only** | **exactly 0.0000** | the no-information value is known, not estimated (every R² is against the scored-set SST) |
| **raw-input floor** (`pixel`) | anything the arm must beat | a learned representation that does not beat raw input added nothing. MEASURED: −0.3977 spatial speed |
| **instrument validity** (`dino_hf − scratch`) | **SEPARATED** | ⛔ if it is not, the panel is **VOID**, which is exactly what happened to E-SEED-2's environment column (+0.1387 [−0.3232, +0.5740]) and why no environment conclusion is drawn there |
| **time-shuffled** feature stream | the constant-only value | a monitor that reads high on shuffled features is measuring the probe |
| **printed `n` and `d`** | n = 2,372 · clips = 24 · d_ambient 1,024 / 16,384 · d_probe 256 | `n ≪ d` is underpowered by construction, not a negative |

⛔ **λ is selected by an inner clip-disjoint `GroupKFold` on the fit clips only, and a λ at a grid edge is FLAGGED** — the 2026-08-22 failure family (λ chosen on the point estimate, λ chosen on the scored split, λ at the grid maximum collapsing every arm to the constant predictor).

---

## 6. Rungs, in order

| rung | question | one variable | cost | gate |
|---|---|---|---|---|
| **S0** | does E-SEED-2 replicate at the **launch geometry** (ViT-**B**/16, `--newest-frame-only --in-channels 3`)? | `pos_init` | **0 GPU-training**, ~15 min of forward passes; needs `dinov3-vitb16` pulled (not in the local HF cache — two probes) | step-0 delta separated, `scratch` fails, `dino_hf − scratch` separated |
| **S1** | is the repair paid back by training? | `pos_init` | 2 matched tiny arms, 2,000 steps | the Observer-Effect monitor ratio at 2,000 steps, paired |
| **S2** | ⭐ **V7A-2 — is the external target strong enough to dominate?** | *(a probe, not an arm)* | **0 GPU-training**: a forward hook on the banked `o7w1p0` arm | see §7 |

---

## 7. ⭐ V7A-2 — the dominance/conflict measurement, both outcomes committed

On the **already-banked** `o7w1p0` arm, log at the encoder output `z`, over a few hundred steps, the **two** scalars `lib:2505.08170` (MoKD) §3 defines — not one:

* **Gradient Dominance** `D = ‖g_ext‖ / ‖g_self‖`, where `g_ext = ∂L_O7/∂z` and `g_self = ∂(L_O5 + L_O6)/∂z`;
* **Gradient Conflict** `C = ⟨g_ext, g_self⟩` (and its cosine).

⛔ **NOT GradNorm.** `lib:1711.02257` is banked and its recommendation is **withdrawn** for this objective shape: on the two banked tables with an external supervised term co-trained with self-supervised auxiliaries it **loses to the weight sweep it would replace** (`lib:2010.08244` Table 1: CIFAR-10 **14.07 vs 13.76**, SVHN **7.68 vs 6.07**; `lib:2110.14048` Table 4: **81.03 vs 81.35**). Its own supporting result is NYUv2 with **three supervised tasks** and no self-generated target. It also swaps K weights for one α whose good value differs **12×** across its own two settings.

**Both outcomes committed in advance:**

* **Outcome A — `D ≪ 1` (the external gradient is dominated)**: `E-DEC-9` is a **DOMINANCE** failure and the external term is being drowned. ⚠️ **This does NOT license a bigger `w`**: MoKD §3 states the norm ratio is **non-stationary** (*"the norm of gradients varies throughout optimization"*), so a fixed weight is correct at one step only. The successor is a **scheduled or ratio-targeting** balance, reported with `D` over training, not a new point on the sweep.
* **Outcome B — `D ≈ 1` or `D > 1` and the environment target still loses**: the failure is **INTERFERENCE, not balance.** Reweighting is refuted outright.
* **Outcome C — `C < 0` materially**: the terms actively conflict and a projection method (PCGrad `lib:2001.06782`) is on the table.
* ⚠️ ⛔ **AND THE READ THAT LOOKS LIKE A NULL AND IS NOT:** `lib:2609.03150` measured **+1128 perplexity of negative transfer at `cos = −0.002 ± 0.003`** with near-disjoint routing (`J̄ ≈ 0.056`) — **interference WITHOUT conflict**, invisible to any conflict test. ⇒ `C ≈ 0` **may not be reported as "the terms do not interfere"**; it may only be reported as "no *conflict* is detectable by this statistic".
* ⛔ **There is no "no difference" outcome**: both scalars are defined on the same tensor at the same step. A missing number is a bug, not a result.

⚠️ **Its own control:** `D` must be reported **beside** the two raw norms, never as a bare ratio — the MM-E19 defect (a ratio moving because its denominator moved, read as the numerator moving) applies here verbatim.

---

## 7b. ⛔ THE RECIPE CONSEQUENCE THAT DOES NOT WAIT FOR V7A-2

Whatever V7A-2 returns, the field has already measured the *shape* of E-DEC-9 and the *shape* of its fix, and this prereg records it so the next arm is not another point on the sweep:

`lib:2509.10156` (LayerLock) **Table 3** — a self-generated-target latent loss added to an external-target pixel loss **as a weighted sum** drove SSv2 **50.1 → 3.7**; caption: *"Adding latent losses to the MAE paradigm without freezing leads to representation collapse"*. ⛔ **BOTH weight schedules — constant (3.7) and cosine (5.6) — collapsed.** Only **progressive freezing** worked, and the paper's mechanism is exactly ours: *a frozen layer's output cannot move to meet the predictor*, i.e. freezing converts a self-generated target into a fixed external one.

⭐ Our register holds the same result twice, measured on our own stack: `E-DEC-12` (distil, then self-supervise) **+0.1327** vs co-trained **−0.2553**; `E-DEC-14b-R` (freeze the distilled trunk, parity scale) **+0.3881** vs the trainable trunk **−0.0180**.

⚠️ **And the order is not free**: `lib:2302.14138` Table 1 — right-order staging beats joint by **+8.9**, wrong-order sits **15.1 BELOW** joint. A staged v7 arm must therefore pre-register its order, and its deliberate-regression arm is **the reversed order**.

---

## 8. What this prereg would look like if it FAILED — committed now

If **S0 (a)** fires — the step-0 delta's CI contains 0 at ViT-B/16 — then E-SEED-2 is a **ViT-L/16-only** result, the `pos` story is not general, and the honest write-up is *"a repair that is real at L/16 and not measurable at B/16"*. **I would publish that.**

If **S1 (b)** fires — the repaired arm's monitor is separated **below** the as-wired arm's at 2,000 steps — then zeroing `pos` buys a better start and a worse trajectory. That is `SEED-REPAIR-WORSE`, it is reported under that name, and the recommendation in `RESULT.md` §4 rank 1 is **withdrawn**, not softened.

If **S2** returns Outcome B, then the programme's entire `w`-sweep line (w = 1.0 → 0.2 → 10) is refuted and the compute queued behind it should be re-pointed at staging or subspace separation.

⛔ **And the outcome that would embarrass this document most, stated so it cannot be quietly avoided:** if `scratch` ever clears the gate, every conclusion in `RESULT.md` §2 is void, including the ones this prereg is built on.
