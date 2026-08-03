# PRE-REGISTRATION — is the situation classifier limited by MISSING TEMPORAL CONTENT?

**Date** 2026-08-03 · **Stream** sitclf temporal · **Substrate** dev box (RTX 4060), **0 pod GPU-h**
**Written and staged BEFORE any held-out number from this run was read.**

## 0. Why a new pre-registration is required

`…/2026-07-26-situation-classifier/PRE_REGISTRATION.md` §7 is binding:

> *"no re-sweep of any §2 constant after a held-out number is seen; **no post-hoc arm added to
> §5.2**; … **Any follow-up is a new pre-registration.**"*

`WIN = 8` lives in that document's **§5.2**, not §2, so a window sweep is not a forbidden re-sweep of
a label constant — it is a **new arm**, and new arms need a new pre-registration. This is exactly the
instruction the optimisation stream left: *"**Pre-register it** and run the sweep at full
training-set size, or leave `WIN=8`"* (`…/2026-08-03-sitclf-optimisation/SITCLF_OPTIMISATION.md:121`).

⛔ **No §2 label constant is touched by this study.** The detectors, their thresholds, `lead_s = 3.0`
and the 1.0 s minimum-useful-lead bar are all untouched.

## 1. The hypothesis, decomposed so it is testable

The PI brief proposes: *the situation classifier is limited by the same missing temporal content that
makes `long_accel` unrecoverable — a single RGB frame carries no relative velocity, no closing rate,
no TTC, which would explain a capacity curve that peaks at 129 parameters.*

⚠️ **As stated it is not yet a test, for two reasons established from source before this run:**

1. **The deployed head does not read one frame.** `sitclf.causal_window` stacks `WIN = 8` frames of
   features at offsets −7..0, so **0.7 s of history is already in the design matrix**. Anything a
   *linear* head could do with explicit differences it can already do with the stack, because
   differences are a linear function of it.
2. ⭐ **Nor is each feature a single RGB frame.** MEASURED at two probes: a real episode cache tensor
   is `frames_u8 [T, 9, 256, 256]`, and `stack/tanitad/config.py:17` reads
   `9 = camera (3-frame stack, D-015)` with `config.py:360` "3 RGB frames at 100 ms spacing
   channel-stacked". **Every v1 latent already integrates 0.2 s of motion**, and the WIN=8 stack of
   them spans **0.9 s** of motion-bearing evidence.

So the premise "single-instant, therefore motion-blind" is **false as stated for this classifier**.
What remains genuinely open decomposes into three mechanisms that are actually distinguishable:

| id | mechanism | why it is not the others |
|---|---|---|
| **H-T1** | **WINDOW LENGTH.** 0.9 s is too short for a manoeuvre defined over ~3 s (the label's own `lead_s`). | changes which rows enter the design matrix at all |
| **H-T2** | **MOTION SUBSPACE.** The PCA basis is fit on the marginal FRAME distribution, so it keeps maximum-*appearance*-variance directions; frame-to-frame change can sit in near-zero-marginal-variance directions and be truncated at rank 16 (16 PCs were measured to carry **97.0 %** of the variance). | changes *which* 16 of 2048 dimensions are kept — **not** a reparameterisation |
| **H-T3** | **PARAMETERISATION.** Even where the stack spans the differences, the optimiser may not find them; explicit motion channels could help the NON-LINEAR head. | a change of basis only; must be measured against the invariance control |

## 2. Arms (all VISION-ONLY at inference, per the binding PI ruling)

`REFERENCE = ridge_app16_w8` — the banked floor recipe exactly: PCA-16, WIN 8, **129 params/head**.

| group | arms | what moves |
|---|---|---|
| **A — window at MATCHED capacity** | `ridge_app128_w1`, `app64_w2`, `app32_w4`, **`app16_w8` (REF)**, `app8_w16`, `app4_w32` | flat dim held at **128 ⇒ 129 params** for every rung; only the window changes (0.0 s → 3.1 s) |
| **B — window at FIXED rank 16** | `ridge_app16_w{1,4,16,32}` | capacity is *allowed* to grow with the window (17 → 513 params) |
| **C — motion subspace** | `ridge_mot16_w8`, `ridge_app8mot8_w8`, `ridge_app16mot16_w8`, `ridge_mot16_w32` | PCA basis fit on Δ₁ of the 2048-d readout |
| **D — deployed transformer budget** | `tf_app16_w8_d128`, `tf_app16_w32`, `tf_app8mot8_w8`, `tf_app16_w8_diffparam` | so the verdict is not a ridge artefact |

## 3. Controls — every one able to FAIL

| control | definition | what its failure means |
|---|---|---|
| **NEG_FEAT** | every arm refit with image features permuted **ACROSS clips**, labels untouched; the permutation **preserves within-clip temporal order** | the null still sees real motion, just another drive's — so a temporal arm cannot be credited for "motion exists", only for *this clip's* motion. Fitted and scored **BEFORE** any real arm is quoted |
| **INVARIANCE** | `ridge_app16_w8_diffparam` — an **exactly invertible** linear remap of the reference's own window into `[f_t, f_t−f_{t−1}, …]` (`sitclf.diff_reparam`, round-tripped by `sitclf.undiff_reparam` in tests) | spans the identical hypothesis space, so its delta is standardisation + L2 penalty geometry and **nothing else**. It bounds how much of any "motion channels help" reading is attributable to reparameterisation |
| **C-POS (ORACLE)** | `CPOS_ORACLE_egofuture30` — ego over **(t, t+3.0 s]**, the label's own evidence window. ⛔ future-reading, built locally, never promoted to `stack/` | **MUST separate above the reference.** If it does not, these rows cannot be separated by anything and **no verdict may be issued** |
| **C-POW** | positive **CLUSTERS** counted and written to disk **before any score is read**; bar = **40** | `< 40` ⇒ that situation is `UNDERPOWERED_C_POW` and gets **no verdict** |
| **NEG_LABEL** | labels permuted across whole clusters | protocol manufactures signal ⇒ table void |
| **H-T2 SUBSPACE DIAGNOSTIC** | fraction of Δ₁ variance surviving projection onto the *appearance* basis vs the *motion* basis; principal angles between them | ⭐ can refute H-T2 **mechanistically, before any AP** — if appearance-16 already retains almost all Δ variance, there is no discarded motion subspace to recover |

## 4. Protocol

2 outer folds over whole clip **CLUSTERS**; 20 % of each train half's clusters select epoch (TF) or
λ (ridge); both PCA bases fit on that fold's FIT rows only; estimator is the **paired
episode-cluster bootstrap** (`taniteval` draws, imported not re-implemented), `B = 2000`, `seed 0`,
`separated` ⇔ the 95 % interval excludes 0.

⭐ **Every arm is scored on IDENTICAL rows** — validity is intersected with a full in-clip history at
`WIN_MAX = 32`, *not* at each arm's own window. Scoring a WIN=32 arm on rows a WIN=8 arm cannot reach
would confound the window with the row set and invalidate the paired estimator. This costs rows
against B4 and the count is reported.

Every arm reports **precision alongside recall at a fixed 5 % rank budget with BOTH denominators**
(`n_alarm`, `n_pos`), and the four binding metric families are reported per family.

## 5. ⭐ OUTCOMES — both committed in advance

| verdict | predicate | what the programme should then do |
|---|---|---|
| **CONFIRMED** | on a POWERED situation, some longer-window or motion-basis arm beats `ridge_app16_w8` with a paired CI **excluding zero** | the features/window ARE the constraint ⇒ buy a temporal encoder (BACKLOG **B5**), do not buy head capacity |
| **NO_EFFECT_ABOVE_MDE** | no window in {1,2,4,8,16,32} and no motion basis separates upward, **and** the C-POS oracle DID separate | the temporal-content hypothesis is **bounded-refuted for this task** at the stated MDE ⇒ redirect to the label definition, base rate, or FOV — **not** to a temporal encoder |
| **INDETERMINATE_C_POS_FAILED** | nothing separates, oracle included | the instrument, not the hypothesis, is the finding |
| **UNDERPOWERED_C_POW** | `< 40` positive clusters | no verdict; more clips, not more model |

⚠️ **Null language is constrained.** The parent pre-registration §7 states *"a `separated: false` is
**UNPOWERED, not refuted** — point estimates move a median 75 % on re-powering"*. A null therefore may
**not** be published as a bare refutation: it is reported as **no effect above the study's MDE**, with
the MDE stated as the widest paired-vs-reference CI half-width actually achieved.

⛔ **`roundabout` may not decide anything** — 39 positive clusters at B4, below its own C-POW bar.

## 6. What would make me wrong in a way I could not detect

* If the frozen v1 trunk's 3-frame stack already saturates the motion available at 256 px / 51.4°,
  every arm here inherits that ceiling and the study measures the *trunk*, not the *window*. The
  subspace diagnostic partly addresses this; a video-pretrained trunk (B5) is the only clean answer.
* The substrate is the B4 rebuild (400 train + 100 val local clips), **not** the parity cache.
  Absolute APs are **not** comparable to the banked table; every claim here is a within-substrate
  paired contrast on identical rows.
