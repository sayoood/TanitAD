# D-LATENT — ⛔ the latent-bottleneck thesis is FALSIFIED for `long_accel`, and the reason is worse than a bottleneck

**Date** 2026-08-03 · **Substrate** dev box (RTX 4060), **0 pod GPU-h**, no training pod touched ·
**Pre-registration** `Project Steering/PREREG_TEMPORAL_LATENT.md` (both outcomes fixed in advance)
**Run directory**
`TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-03-latent-bottleneck/`

| artifact | what it is |
|---|---|
| `results_mechanism.json` | ⭐ **the mechanism** — the decisive numbers |
| `results_precision_ladder.json` | how precise a speed read must be, and the failed one-mechanism test |
| `results_temporal_falsifier.json` | the pre-registered D probe — **35 arms** + the null + the oracle (pass 2, corrected substrate) |
| `raw/summary_tables.txt` + `summarize.py` | ⭐ every table below, emitted FROM the JSON — no number here is hand-transcribed |
| `raw/results_pass1_STACKAVG_INADMISSIBLE.json` | pass 1, kept — its defect is documented, not hidden |
| `raw/temporal_kv_cost.json` | ⭐ **what approach A costs**, MEASURED on all three REF-C presets |
| `raw/run_log*.txt` | full logs |

---

## 0.0 ⛔ THE PRE-REGISTERED VERDICT — **OUTCOME V (VIDEO-LIMITED)** — written first, as registered

`PREREG_TEMPORAL_LATENT.md` §4 committed me to putting this above every ranked approach if it
fired. It fired, with **every** admissibility condition met:

| pre-registered condition for OUTCOME V | measured |
|---|---|
| no temporal arm's paired ΔR²(`long_accel`) separated positive with R² ≥ +0.05 | ✅ **0 of 35 arms.** Not one of `v1` (2,048–18,432 features), `pix*`, `stk*`, `mot*`, linear or rbf |
| the ORACLE-INPUT arm still reaches ≈ +0.93 | ✅ **+0.9262** (9 features, true speed window) |
| `speed` still separates positive on the hand-built arms (the admissibility gate) | ✅ **6 arms separated**: `pix32_centre_rbf` +0.6693\*, `mot8_window_rbf` +0.5633\*, `mot16_window_rbf` +0.5200\*, `pix8_abstdiff_rbf` +0.4202\*, `stk8_abstdiff_rbf` +0.3898\*, `stk8_tdiff_rbf` +0.3531\* |
| the single-instant controls must NOT separate (outcome M) | ✅ none separated: `pix32_centre` −0.0552, `stk32_centre` −0.0559, `pix32_centre_rbf` −0.2561 |

⇒ **`long_accel` is not recoverable from monocular 10 Hz 256 px video at this n, above the measured
sensitivity floor — by ANY of 35 arms across four substrates and two kernels, while the same
arms recover `speed` and the oracle recovers accel at 0.93.** As registered: this **retires the
"the latent is the bottleneck" framing for `long_accel` specifically**. It does **not** retire it
for speed, TTC, headway or the manoeuvre decision, which must be argued on their own evidence.

⭐⭐ **AND THE FINDING THAT REFRAMES EVERYTHING — on this corpus `speed` is read mostly from
STATIC APPEARANCE.** A single 32×32 grayscale **still frame** through an rbf ridge reads `speed` at
**R² +0.6642 [separated]** — against the 18,432-feature, 800 ms latent window's **+0.7145**.
**One static frame is 93 % of the full learned latent's speed accuracy**, and it is **~1.75×** the
best *motion-only* arm in the whole panel (`pix8_tdiff_rbf` +0.3778, `stk8_tdiff_rbf` +0.3479\*).

⚠️ **Stated precisely, because the strong version is wrong.** Motion features DO carry speed — but
only nonlinearly, and only about half as well:

| basis | linear | rbf |
|---|---:|---:|
| **pure temporal difference** (`*_tdiff` / `*_diff`, all 4 substrates, 8–16,384 features) | **−0.0052 = the null on all 10 arms** | +0.1449 … **+0.3778** |
| **single static frame** (`pix32_centre`) | −0.0588 | **+0.6642** |
| full latent window (`v1_window`, 800 ms) | **+0.7145** | — |

(The linear null on differences is expected physics — `∂I/∂t = −∇I·v` is linear in `v` only at fixed
image gradient — which is exactly why the rbf rung was added.)

⇒ **Appearance dominates motion by ~1.75× for reading speed, and the full 800 ms learned latent adds
only 7 % over one still frame.** Nothing in this pipeline was ever *forced* to learn motion, because
appearance was almost sufficient for the metric being optimised. The `long_accel` null, the 88.7 %
longitudinal gap, and the appearance-dominant speed read are plausibly **one fact** rather than
three — ⚠️ HYPOTHESIS at programme scale, MEASURED only on comma2k19 highway (RANK 1 tests it).

---

## 0. FIVE MORE FINDINGS, in the order that changes decisions

1. ⛔ **THE BRIEFING'S MECHANISM IS FACTUALLY WRONG, and it is wrong in a way that matters.**
   The brief says *"A single RGB frame cannot carry relative velocity, closing rate, or TTC"*.
   REF-C's input is **not a single RGB frame**. `refc.py:241` sets `in_channels: int = 9` —
   *"D-015 3-frame RGB stack (latest = `[-3:]`)"* — and the stack is **sliding**, verified
   numerically on the corpus: `frames_u8[t][6:9] == frames_u8[t+1][3:6]` at **max |d| = 0.0**
   (likewise `[3:6] == [t+1][0:3]`). ⇒ the feature map REF-C keeps already spans **300 ms** of
   video in the channel dimension, and a conv stem over 9 channels forms a temporal difference at
   layer 1. **What `fmap[:, -1]` discards is history beyond 300 ms — not motion as such.**
   ⭐ **Independently corroborated by the sitclf-temporal stream on a DIFFERENT corpus and a
   DIFFERENT file** (`…/incoming/2026-08-03-sitclf-temporal/TEMPORAL_HYPOTHESIS.md` §1a: the
   PhysicalAI episode cache is `[199, 9, 256, 256]`, `stack/tanitad/config.py:17` and `:360`). Two
   streams, four probes, one conclusion — this correction is not a single-location read.

2. ⭐⭐ **THE ACTUAL MECHANISM, MEASURED: the frozen v1 latent has almost no 100 ms temporal
   resolution.** On the held-out episodes (`results_mechanism.json`):

   | quantity | frozen v1 latent | ORACLE (true speed window) |
   |---|---:|---:|
   | correlation of the **within-window speed derivative** with the true one | **+0.0891** | **+0.99997** |
   | SNR of that derivative (true increment std ÷ error increment std) | **0.856** | — |
   | regressing the speed **increment** DIRECTLY (not deriving it) — correlation | **+0.0007** | **+0.99997** |
   | that arm's `long_accel` R² | **−0.0622** *(= the empirical null −0.0626)* | **+0.8982** |

   ⇒ It is **not a parameterisation problem**. "Derive it from the speed track" and "regress the
   increment directly" are **both** exactly at the null. The 100 ms speed change is not in the
   latent window in any form a ridge over 18,432 features can reach.

3. ⭐⭐ **AND THE REASON, in one number: the latent is not TEMPORALLY SMOOTH — its frame-to-frame
   jitter along its own speed direction is 51× the physical signal.**
   Fit ONE linear speed readout `w` and apply it per window position. It is a good *scene-level*
   speed reader (held-out R² **+0.6767**), but the std of `w·(z_{j+1} − z_j)` is **29.695 m/s²**
   against a true increment std of **0.582 m/s²** — a **51.0× jitter ratio** — and it correlates
   **+0.0061** with the true increment.
   Supporting: mean cosine between positions 100 ms apart **0.98825**, across the 800 ms window
   **0.91398**, against a random other row **0.32770** — the representation separates **SCENES**
   (0.33) far more than **INSTANTS** (0.988), and what little it does move between instants is
   off-axis noise.
   ⇒ **Keeping W frames of this latent stacks W near-copies whose differences are noise.** That is
   why `v1_window` — all nine latents, 18,432 features — sits at the null, MEASURED, and it is why
   **approach A as briefed cannot work on the current trunk.**

4. ⛔ **"THE LATENT READS SPEED TOO COARSELY" IS ALSO REFUTED — the one-mechanism test FAILED.**
   Take the TRUE speed track, corrupt it with AR(1) noise matched to the real latent-read error's
   **σ = 5.943 m/s AND its within-window lag-1 autocorrelation 0.9999**, put it through the
   **identical** best Savitzky-Golay derivative: **R² +0.6495**. The real latent track through the
   same estimator: **−0.3773**. Paired Δ **+1.0268 [SEPARATED]**.
   ⇒ A track that coarse but correctly *structured* recovers the channel. The latent's error is not
   noise around a moving signal — **the within-window variation carries no signal to begin with.**

5. ⛔ **"THE ENCODER DESTROYS MOTION" IS NOT SUPPORTED.** The frozen v1 latent reads `speed` at
   **+0.7145** against the best hand-built substrate's **+0.6642** — only **1.1×** better, and the
   hand-built winner is a **single static frame**. The encoder is not throwing motion away; **there
   was little usable motion signal in the substrate to throw away**, and the learned latent is
   reading the same appearance cue a still frame gives.
   ⚠️ **Two instrument defects had to be fixed before this was readable** (§2): pass 1 averaged the
   9-channel stack (3 timesteps) into every window position, and pooling *before* differencing
   cancels opposing-sign gradients. On pass 1 the pixel family's positive control failed
   (`speed` ≤ +0.1064) and the whole family would have been VOID. Fixing both, plus adding rbf
   kernels (a linear map from brightness differences to velocity is physically mis-specified),
   raised it to **+0.6642 separated** and made the verdict admissible.

**⇒ The one-line answer to the PI's question.** The representation is not "missing temporal
content that a better architecture could route to the decoder". Two things are true at once:
**(a)** `long_accel` is not in this video at the strength `speed` is carried at (§0.0, pre-registered
falsification), and **(b)** the `speed` the model *does* read is an **appearance shortcut**, and the
latent's frame-to-frame motion along its own speed axis is **51× the physical signal and
uncorrelated with it** (§3.2b). Architecture changes that widen the temporal span (approach A) or
re-route existing tensors (approach B) are **measured dead on this trunk**. The only live lever is
the **encoder's pretraining objective** — and the programme now has a **0-GPU screen with numeric
targets** to gate any such run before it is paid for (§6).

---

## 1. WHAT WAS READ FROM SOURCE — the mechanism, quoted

`stack/tanitad/refs/refc.py:1412-1422` (`RefCModel.forward`):

```python
b, w = frames.shape[:2]
if self.cfg.hierarchy:
    fmap_all, pooled_all = self.encoder(frames.reshape(b * w, *frames.shape[2:]))
    pooled_seq = pooled_all.reshape(b, w, -1)
    pooled = pooled_seq[:, -1]
    fmap = fmap_all.reshape(b, w, *fmap_all.shape[1:])[:, -1]
    ctx = self.strategic(pooled_seq)
```

Four facts follow, each MEASURED from source and material to the ranking:

| # | fact | file:line | consequence |
|---|---|---|---|
| S1 | `W = 8` window frames | `refc.py:386` (`window: int = 8`) | the discarded history is 8 × 100 ms |
| S2 | each "frame" is a **3-frame RGB stack**, sliding | `refc.py:241`; verified numerically (§0.1) | the kept map already spans 300 ms |
| S3 | ⭐ **the encoder ALREADY runs on all `b*w` frames** and `[:, -1]` throws 7 of 8 feature maps away | `refc.py:1414-1418`; `hierarchy` default `True` and **`"hierarchy": true`** in REF-C-XL's run config (`MODEL_REGISTRY.md` §4.1) | **approach A costs ZERO extra encoder FLOPs** |
| S4 | the only temporal path to the decoder is `StrategicCtx` — a **GRU over the pooled sequence** → `d_ctx` → a **zero-init** `ctx_to_cond` → added to `cond`, which FiLMs the **MLP branch only** (`CrossAttnLayer.forward`: `q + mlp(film(norm(q), cond))`) | `refc.py:734-745`, `831-836`, `783-787` | history can **modulate** every anchor but can never be **attended to** |

S4 is real and is worth stating precisely: the anchor queries cross-attend **only** `fmap[:, -1]`'s
64 tokens; all 800 ms of history reaches them as one `d_ctx`-dim vector through a zero-init linear.
That is a genuine architectural asymmetry. **§3 measures that fixing it would not help, because the
tensor it would let the decoder attend to carries near-duplicate content.**

---

## 2. THE D PROBE — design, and what the pre-registration bought

Substrate held **identical** to the panel the null came from
(`…/Benchmarks & Eval/…/2026-08-03-idm-accel-recoverability/`): 50 content-clean comma2k19 episodes
`ep_00040…ep_00089`, `k=4` (9-frame NON-CAUSAL) windows at stride 2, episode-disjoint `i % 3` split
→ **33 train / 17 held-out (4,554 / 2,346 windows)**, inner split of TRAIN only (22 fit / 11 select),
exact kernel ridge over the whole α path (10⁻⁴…10¹⁰, 29 values **+ an exact-mean sentinel**), skill
gate 0.01, one-SE shrinkage tie-break, paired episode-cluster bootstrap B = 2000.

**Replication is bit-exact**, which is what makes the new arms comparable:

| arm | this run | the prior panel | |
|---|---|---|---|
| `NULL_train_mean` `long_accel` | **−0.0626** | −0.0626 | ✅ |
| `ORACLE_true_speed_window` `long_accel` | **+0.9262** | +0.9262 | ✅ |
| `v1_centre` `speed` / Δ vs control | **+0.6857 / +0.6908\*** | +0.6857 / +0.6908\* | ✅ |
| `v1_window` `speed` / Δ vs control | **+0.7145 / +0.7197\*** | +0.7145 / +0.7197\* | ✅ |
| `v1_diff` `long_accel` / Δ | **−0.0599 / +0.0027** | −0.0599 / +0.0027 | ✅ |

**Two instrument defects were caught by the controls, not by inspection.** Both are documented
rather than tidied away, and pass 1 is kept:

| defect | how it showed | fix |
|---|---|---|
| pass 1 collapsed all **9 channels** with `.mean(1)`, box-filtering **3 timesteps** into every window position | every pixel arm at the null on `speed` too | `latest_frame=True` takes sub-frame `[6:9]` — one row = one timestep; plus a `stk*` substrate exposing the within-stack 100 ms structure |
| pooling **before** differencing cancels opposing-sign gradients (`avgpool(|d|) ≠ |avgpool(d)|`) | `pix32_tdiff` read `speed` at −0.0052 = the null | `mot*` substrate: `|I_{t+1} − I_t|` at **full 256×256**, pooled only afterwards |
| a **linear** probe on pixel differences is physically mis-specified (`dI/dt = −∇I·v` is linear in `v` only at fixed `∇I`) | a linear-only pixel null would have been a property of the probe | rbf arms added on the small motion bases, matching the treatment the latent family already got |

⚠️ **Even after all three fixes the pixel family's positive control still fails** (§4.4). The value
of the pre-registration is that this outcome was written down in advance as VOID rather than being
read as evidence for "the video does not carry it".

---

## 3. THE MECHANISM — `results_mechanism.json`

### 3.1 The latent's within-window speed track is uninformative about its own derivative

Ridge (latent 9-frame window, 18,432 features) → the **whole 9-position speed sequence**; α chosen
on the inner split by mean sequence R²; scored once on the 17 held-out episodes.

| quantity | latent | oracle (true speed window) |
|---|---:|---:|
| `speed` R² at the window centre | **+0.6912** | +1.0000 |
| speed error σ at the centre | **5.943 m/s** | — |
| **TRUE** within-window increment std | 0.582 m/s² | 0.582 m/s² |
| **PREDICTED** within-window increment std | 0.407 m/s² | 0.582 m/s² |
| gain (pred ÷ true) | 0.700 | **0.999** |
| ⭐ **corr(predicted increment, true increment)** | **+0.0891** | **+0.99997** |
| error-increment std | **0.680 m/s²** | — |
| ⭐ **derivative SNR** (true ÷ error increment std) | **0.856** | — |

Read that gain row carefully: the predicted track is **not flat** — it moves 70 % as much as it
should. It simply moves **in an unrelated direction** (corr 0.089), and the error's own increment
(0.680) is **larger than the signal** it is supposed to reveal (0.582).

### 3.2 It is not the parameterisation — the increment is not directly regressable either

| arm (target = the true increment sequence) | corr(pred, true) | `long_accel` R² |
|---|---:|---|
| **latent** window → increments | **+0.0007** | **−0.0622 [at the null −0.0626]** |
| **oracle** speed window → increments | +0.99997 | **+0.8982** |

⇒ The programme has now tested **both** routes — derive from a predicted speed track (prior panel,
`derive_long_accel`, refuted) and regress the derivative directly (here) — and both land on the
null while the identical protocol on the identical rows reaches +0.90 from a real speed track.

### 3.2b ⭐⭐ The sharpest form: ONE shared speed direction, applied per position

§3.1 fits the whole 18,432-d window to each position's speed, so a position's prediction can be
carried by *other* positions' latents and the ridge can smooth. The clean question is whether the
latent **moves along its own speed direction** at 100 ms. Fit **one** linear readout `w` on the
train rows pooled over all positions, then apply it per position: `v̂ⱼ = w·zⱼ`, so
`Δv̂ⱼ = w·(z_{j+1} − zⱼ)`.

| quantity | value |
|---|---:|
| held-out `speed` R² at the centre | **+0.6767** *(≈ the windowed read's +0.6912 — the direction is a good SCENE-level speed readout)* |
| **TRUE** within-window increment std | 0.582 m/s² |
| **PREDICTED** increment std from that same direction | **29.695 m/s²** |
| ⇒ frame-to-frame **JITTER RATIO** | **51.0×** |
| corr(predicted increment, true increment) | **+0.0061** |
| derived `long_accel` R² | **−869.9** |

⭐ **This is the defect in one number.** The latent's speed direction is a *good* scene-level speed
readout and its frame-to-frame motion along that direction is **51× larger than the physical signal
and uncorrelated with it**. The representation is not temporally smooth: consecutive frames of the
same scene land far apart along the very axis that encodes speed. No decoder, no routing, and no
number of retained frames can recover a 0.58 m/s² signal from a 29.7 m/s² jitter.

⇒ **The design target this hands to approach C is concrete: temporal SMOOTHNESS / equivariance of
the latent along its task-relevant directions**, not span, not scale, not pooling.

### 3.3 The representation separates SCENES, not INSTANTS

Mean cosine similarity over the 17 held-out episodes' latent windows:

| pair | cosine |
|---|---:|
| adjacent window positions (**100 ms**) | **0.98825** |
| the window's two ends (**800 ms**) | **0.91398** |
| a **random other row's** centre | **0.32770** |

The drift is roughly linear in time (1 − 0.914 ≈ 8 × (1 − 0.988)), so the latent is not saturated —
it simply moves ~1.2 % per 100 ms against a 67 % scene-to-scene separation. **A decoder given all
eight of these token sets receives eight vectors that are 0.988–0.914 alike.** That is the
quantitative form of the finding that `v1_window` (18,432 features) is at the null.

---

## 4. THE PRECISION LADDER — `results_precision_ladder.json`

### 4.1 How accurate must a speed read be for `long_accel` to fall out?

TRUE speed track + controlled noise → best Savitzky-Golay derivative (the least-squares optimal
linear derivative over the 9-point window; the prior panel used a 2-point centred difference, which
understates every rung).

| σ (m/s) | white, central diff | white, best SG | AR(1)-matched, central diff | AR(1)-matched, best SG |
|---:|---:|---:|---:|---:|
| 0.00 | +0.8974 | **+0.9277** | +0.8974 | +0.9277 |
| 0.05 | +0.5198 | +0.9162 | +0.8448 | +0.9175 |
| 0.10 | −0.5516 | +0.8773 | +0.6994 | +0.8829 |
| 0.20 | −4.769 | +0.7317 | +0.1033 | +0.7569 |
| 0.35 | −16.30 | +0.3508 | −1.584 | +0.3804 |
| 0.50 | −36.79 | −0.3207 | −3.925 | −0.1576 |
| 1.00 | −142.3 | −4.078 | −18.92 | −3.650 |
| **5.94** *(the latent's actual σ)* | — | — | — | — |

**The target, interpolated in log-σ** (`precision_target` in the artifact):

| threshold | white, SG | AR(1)-matched, SG | white, central diff | AR(1), central diff |
|---|---:|---:|---:|---:|
| derived `long_accel` **R² ≥ 0.5** | **σ ≤ 0.281 m/s** | **σ ≤ 0.293 m/s** | 0.051 | 0.126 |
| derived `long_accel` **R² ≥ 0.7** | **σ ≤ 0.210 m/s** | **σ ≤ 0.218 m/s** | 0.030 | 0.100 |

⭐ **This is the number the programme did not have.** With the optimal 9-point derivative the
requirement is **σ ≈ 0.28 m/s**, not the ~0.1 m/s the 2-point analysis implied — a **2.8× easier**
target. The current latent is at **5.943 m/s**, i.e. **≈ 21× too coarse**.
⚠️ It corrects the prior panel's *"the speed track would need σ ≲ 0.1 m/s … a ~47× improvement"*:
that figure is specific to a **2-point centred difference**. With the right estimator it is
**σ ≲ 0.28 m/s and ~21×**. Both are quoted with their estimator, per the standing rule.

### 4.2 …but precision alone is NOT the mechanism — the one-mechanism test FAILED

| arm | `long_accel` R² |
|---|---|
| TRUE speed + AR(1) noise at the **real σ (5.943)** and the **real ρ (0.9999)**, best SG | **+0.6495** |
| the **REAL** latent-read track, same best SG | **−0.3773** |
| **paired Δ (sim − real)** | **+1.0268 — SEPARATED** |

A synthetic track with the same coarseness and the same autocorrelation recovers the channel at
+0.65. The real one does not. ⇒ **The latent's error is not "noise around the right track".** Its
within-window variation is uncorrelated with the true within-window variation (§3.1, corr 0.089),
which no scalar σ can express. **Precision is necessary and not sufficient; RESOLUTION is what is
missing.**

### 4.3 Four metric families — per family, never pooled

Computed for every arm in `results_temporal_falsifier.json → arms.*.four_families`
(`tanitad.eval.idm_families`), on the same windows, with the same estimator, and emitted by
`summarize.py` (table 5) so nothing here is hand-transcribed.
⚠️ **The values below are for the same headline arm and the same 2,346 windows as the panel this
replicates, and they agree with it** — the replication in §2 is bit-exact on the shared rows, so the
family numbers are a re-derivation rather than an independent second measurement. Read
`raw/summary_tables.txt` for this run's own emission.

| family | what the headline latent arm does | note |
|---|---|---|
| **LONGITUDINAL** | best latent arm `speed` MAE **4.719 m/s** (R² +0.7145) vs the null's 9.567; along-track MAE 5.842 m vs 12.035 | **distance-keeping (headway / time-gap / TTC) UNAVAILABLE, n = 0** — comma2k19 ships **no object annotation at all**. ⛔ A WORK ITEM, not a pass. The instrument exists (`idm_families.longitudinal`); the corpus does not carry the label |
| **LATERAL** | heading MAE 0.1213 rad, curvature 0.00619 1/m, yaw-rate 0.01953 rad/s, cross-track 0.2997 m | ⚠️ the arm is **worse than the null** on heading and curvature — this corpus is near-straight highway and lateral has almost no variance to explain. Not a defect of the latent; a limit of the substrate |
| **TACTICAL** | from the `long_accel` scalar: balanced accuracy **0.3333 = exactly chance**, recall **[0.000, 1.000, 0.000]**, fires `cruise` on **2,346 / 2,346** windows, support [402, 1745, 199] | ⭐ **the null made operational** — the channel drives **no longitudinal decision at all**. The ORACLE-input readout fires all three classes (BA 0.4707 [0.3850, 0.5695]) ⇒ the READOUT can discriminate; the INPUT cannot |
| **STRATEGIC** | **UNAVAILABLE, n = 2,346** | no route/goal label on comma2k19 (`idm_families.strategic` states the reason in the artifact). ⛔ WORK ITEM |
| ADE@2 s *(one row, never the result)* | 5.882 m [4.855, 7.016] vs the null's 12.047 m | |

### 4.4 The hand-built family, and why the verdict is admissible

The pre-registration's VOID clause was *"if `speed` fails to separate on the pixel arms, the pixel
substrate is inadmissible and the run is VOID for that family."* **On pass 1 it would have voided.
On pass 2 — with both substrate defects fixed and rbf kernels added — it does not.**

| substrate | best `speed` R² | separated? | vs the v1 latent's +0.7145 |
|---|---:|:--:|---|
| `pix*` pass 1 (9-channel stack **averaged**) | +0.1064 | 1 arm | 6.7× worse ⇒ **would have been VOID** |
| **`pix32_centre_rbf` — ONE static 32×32 grayscale frame** | **+0.6642** | ✅ | **0.93× — a still frame is 93 % of the latent** |
| `mot8_window_rbf` (full-res motion energy, 576 f) | +0.5582 | ✅ | 0.78× |
| `mot16_window_rbf` (2,304 f) | +0.5148 | ✅ | 0.72× |
| `pix8_abstdiff_rbf` / `stk8_abstdiff_rbf` / `stk8_tdiff_rbf` | +0.4150 / +0.3846 / +0.3479 | ✅✅✅ | |
| **frozen v1 latent** (`v1_window`, 18,432 f, 800 ms) | **+0.7145** | ✅ | — |

Three consequences, all load-bearing:

1. ✅ **The verdict is admissible.** Six hand-built arms separate positive on `speed`; **zero** of
   the 35 arms separate positive on `long_accel`. Outcome **V** fires as registered (§0.0).
2. ⭐⭐ **The best hand-built speed reader is a SINGLE STATIC FRAME** (+0.6642), beating the best
   motion-only arm (+0.3778) by **1.75×**. All **ten** linear pure-difference arms across all four
   substrates sit at exactly the null (−0.0052); their rbf counterparts reach +0.1449…+0.3778.
   ⇒ **Appearance dominates motion for reading speed here** — and the full 800 ms learned latent
   adds only **7 %** over one still frame.
3. ⚠️ **The `abstdiff` / `mot` arms do carry something real** (+0.35…+0.56, separated) — motion
   energy *is* informative about speed. It is simply **weaker than appearance**, and its own
   temporal derivative (`*_tdiff` on the motion field, i.e. the acceleration proxy) is at the null.

⛔ **What this does NOT establish.** The sensitivity floor here is two amplitude points
(planted signal at 0.01 / 0.05 of substrate RMS → R² +0.9953 / +0.9966, so the probe is not blunt at
those amplitudes), and the prior panel's carrier analysis — INHERITED, not re-run on the pixel
substrate — showed detection is **SNR-limited, not amplitude-limited**: a signal sharing a
high-variance direction can hide at true R² 0.6. **"`long_accel` is absent from the video" is not
proven; "it is not carried at the strength `speed` is carried at, at n = 17 held-out episodes" is.**

---

## 5. THE RANKED APPROACHES

Ranked by **expected value per GPU-day, conditioned on the measurements above**. Every rank carries
the evidence class and the measurement that sets it.

### ⛔ RANK 0 — the two things NOT to do, because they are already refuted

| # | approach | why it is dead | evidence |
|---|---|---|---|
| **A (as briefed)** | make the cross-attended tokens temporal: keep all `W` feature maps instead of `[:, -1]` | the tokens it would add are **near-duplicates** (cosine **0.98825** at 100 ms, **0.91398** at 800 ms), and the empirical test already exists: `v1_window` — **all nine latents, 18,432 features** — is **at the null** on `long_accel` while separating on `speed` in the same draw | MEASURED, `results_mechanism.json` + `results_temporal_falsifier.json` |
| **B (frame-difference tokens / temporal token bank)** | supply `fmap[:, -1] − fmap[:, -2]` (or a bank) as extra KV without retraining the trunk | the **latent difference carries nothing**: `v1_diff` **−0.0599** (Δctrl +0.0027, not separated), `v1_tdiff` and `v1_abstdiff` **at the null**; and regressing the increment directly gives corr **+0.0007** | MEASURED, same artifacts |

⚠️ **This is the deliverable's main saving.** Approach A is the obvious fix, it is cheap (§5.1), and
it would have been a reasonable thing to spend a GPU-week on. It is refuted for `long_accel` **at the
representation level**, before any training.

**⚠️ …but A is NOT refuted for everything.** The measurement above is about the **100 ms speed
derivative**. Over 800 ms the latent does move (cosine 0.914), and nothing here tests whether
attendable history helps **lateral topology, occlusion memory, or the strategic level** — which is
what `StrategicCtx`'s FiLM-only path (S4) actually starves. **A stays live for those, on its own
evidence, and it is cheap:**

#### 5.1 What approach A costs — MEASURED, `raw/temporal_kv_cost.json`

Because the encoder already computes all `W` maps (S3), the only cost is the decoder's KV growing
64 → 512 tokens.
⚠️ **That is true only under `hierarchy=True`.** With `hierarchy=False` the model takes the other
branch, `fmap, pooled = self.encoder(frames[:, -1])` (`refc.py:1686`), and approach A would then
cost a genuine **×W on the ENCODER** — by far the largest term. Every shipped REF-C config sets
`hierarchy` true (default; `"hierarchy": true` in REF-C-XL's run config), so the cheap case is the
real one — but the flag must be asserted, not assumed, at the point of change.

| preset | params | decoder MACs (KV 64 → 512) | ratio | peak CUDA MiB | +params for the seam |
|---|---:|---|---:|---|---|
| REF-C-small (54.7 M) | 47.9 M encoder / 2.95 M decoder | 0.157 → 0.377 GMAC | **×2.40** | 226.7 → 234.2 (**×1.033**) | **2,049 (0.0038 %)** |
| REF-C-base (104.2 M) | 90.5 M enc / 8.63 M dec | 0.856 → 1.560 GMAC | **×1.82** | 440.0 → 447.3 (**×1.017**) | **3,073 (0.0030 %)** |
| **REF-C-XL (251.9 M)** | 199.5 M enc / 22.7 M dec / 21.0 M imagination | 12.99 → 19.33 GMAC | **×1.49** | 1011.9 → 1015.7 (**×1.004**) | **4,097 (0.0016 %)** |

Why the ratio is far below ×8: the cross-attention is only **2.3–4.0 %** of a decoder layer's MACs
today (the per-anchor FiLM'd MLP dominates and does **not** grow with KV length); at KV 512 it rises
to **12.5–13.3 %**. **Peak activation memory is essentially unchanged (×1.004 on XL).**

**Exactly what changes, and what breaks:**
- `refc.py:1683` → keep `fmap_all.reshape(b, w, *…)` and flatten `(w, g, g)` into the KV axis;
  `feat_proj` is already per-token and needs no change.
- **+`W × d` temporal embedding + 1 scalar gate.** The identity-start discipline this codebase
  already uses (`lan_gate`, `cons_gate`, `ctx_to_cond` zero-init) maps onto an **additive attention
  bias** on the non-latest frames initialised large-negative: softmax weight ≈ e⁻²⁰ at step 0, so the
  build is **byte-identical**, and `∂L/∂gate ≠ 0`, so it is gated, not dead.
- ⛔ **H15 `ImaginationField` breaks.** It consumes `fmap [B, F, g, g]` (`refc.py:1324-1326`) and is
  **21.0 M params on XL, ON in the shipped XL config**. Either run it per-frame (**W× its FLOPs** —
  the one place where the cost really is ×8) or keep it on the latest frame only. This must be
  decided explicitly, not inherited.
- `state_dict` gains keys ⇒ resume from an existing REF-C checkpoint needs the gated-flag discipline
  (`tests/test_refc.py` pins the keys); an all-off build stays byte-identical.
- `refc_select.consequence_scores` reuses `feat_proj` and is unaffected.

*(Wall-clock ratios in the artifact were measured while another job held the GPU and are NOT
quotable; the MAC, memory and parameter columns are contention-free.)*

### ⭐⭐ RANK 1 — AUDIT THE APPEARANCE SHORTCUT PROGRAMME-WIDE (0 GPU, days of value)

**This was not on the brief and it outranks everything on it.** §0.0 and §4.4 measure that
`speed` — the channel the whole longitudinal story rests on — is read at **93 %** of the learned
latent's accuracy from a **single static frame**, and at **exactly the null** from every pure
temporal-difference basis. If our speed channel is an appearance cue (road class, lane width,
camera pitch, scene type) rather than a motion estimate, then:

* it is **corpus-specific and will not transfer** — which is exactly the shape of the measured
  cross-rig collapse (frozen v1 speed R² **+0.930 → −2.465** across a camera rig,
  `…/2026-07-22-idm-proof/results.json`), a result the programme has been reading as a
  *camera-geometry* problem;
* the **88.7 % longitudinal share of the oracle gap** is explained without any appeal to
  architecture;
* every arm conditioned on a predicted speed inherits the shortcut.

**What to run (all 0-GPU, on banked dumps):** the §6 screen plus a **frame-order-shuffle** control
(shuffle the window's frames; if `speed` R² is unchanged, the read uses no temporal order at all)
and a **single-frame-vs-window** contrast, on **v5f, REF-C and PhysicalAI-AV** — none of which has
ever been screened. ⚠️ Everything in this document is comma2k19 highway; the shortcut's *magnitude*
elsewhere is **UNKNOWN**, and that is precisely why it must be measured before it is believed or
dismissed.

### ⭐ RANK 2 — the encoder's PRETRAINING objective (approach C), gated by a 0-GPU screen

⚠️ **Justify it on `speed` / TTC / manoeuvre, NOT on `long_accel`.** §0.0 falsifies the
vision route for `long_accel` at this n and floor; an encoder proposal that promises to recover it
must be refused.

This is the only axis the measurements leave open, and it is where the PI's instinct was right.

**What our own programme has already settled about it — none of this is inheritable optimism:**

| measured constraint | number | source |
|---|---|---|
| **scale is refuted** | 48 M vs 90 M encoder: *"the 2.4× encoder/param cut did NOT degrade proposal quality"* | `MODEL_REGISTRY.md` §(REF-C sizing) |
| **from-scratch multi-task dynamics pretraining is refuted** | `dynenc-branchB` (GAIA-2 all-block camera conditioning + masked-latent SSL + action-conditioned forward prediction + SIGReg + supervised IDM + odometry grounding, 40 k steps, 2,466 clips) is a **WEAKER** dynamics substrate than plain flagship-v1: in-domain rig-A speed R² **+0.039 / −0.603** vs flagship-v1's **+0.862 / +0.910** | `MODEL_REGISTRY.md` §10.1 |
| **multi-rig data diversity is refuted** | held-out rig-B light-FT speed R² **−1.61** multi-domain vs **−1.65** single | `…/2026-07-22-idm-proof/results_multirig.json` |
| **frozen IMAGE-pretrained encoders hit a ceiling** | REF-A frozen DINOv2 plateaus at **2.14 m**; the no-speed control is 2.918 m | `MODEL_REGISTRY.md` |
| **the untried axis** | **frozen VIDEO-pretrained encoder (V-JEPA 2 ViT-L)** — *"the one experiment nobody in the field has run; our frozen-encoder ceiling was measured on IMAGE-pretrained encoders only"* | `BACKLOG.md` **B5** |

⇒ **Recommendation: B5 is the right next encoder experiment, and it must be gated by the screen in
§6 before any long run is authorised — with the appearance-shortcut audit (RANK 1) as its
acceptance criterion, not `long_accel`.** `dynenc-branchB` is the cautionary precedent: 40 k steps
were spent producing a latent that a 5-minute probe would have rejected.

⚠️ **Note the tension and do not paper over it.** Flagship-v1's encoder *was* trained with an
action-conditioned forward predictor — a temporal objective — and still gives cosine 0.988 at
100 ms. So "add a temporal objective" is **not** automatically sufficient; the objective must
penalise **temporal collapse** specifically. That is a hypothesis, not a result, and the screen is
how it gets tested cheaply.

### ⭐ RANK 3 — stop asking vision for `long_accel`; take it from the speed track (0 GPU)

**After §0.0 this is not merely the cheapest route — it is the ONLY measured one.** Already
measured:
`long_accel` is **R² +0.9262** recoverable from the **true 9-position speed window** by a
**9-feature** ridge. Wherever a speed track exists — CAN, or a downstream state estimate — the accel
channel should be **derived from that track, not read out of the latent**, and §4.1 now states the
requirement it must meet: **σ ≲ 0.28 m/s** for R² 0.5.

⚠️ **LEGALITY, stated plainly.** Under the binding rule (*labels may use ego; inference is
vision-only*), this is **admissible as a LABELLER and inadmissible as a PERCEPTION CLAIM**. It also
means REF-C's existing `v0` input — a **scalar** ego speed at inference (`refc.py:1432-1445`, with
`ego_dropout 0.5`) — cannot be upgraded to an ego speed *sequence* and then called a perception
result. It would be a legitimate **vehicle-integration** decision and must be labelled as one.

### RANK 4 — fix what the TACTICAL family actually shows (0 GPU, high value)

The `long_accel` channel currently predicts **`cruise` on 2,346 / 2,346 windows** — balanced
accuracy exactly at chance. Any downstream use of the IDM as a longitudinal pseudo-labeller is
labelling `cruise` unconditionally, **invisibly to ADE and to the scalar R²**. This is the binding
four-family rule paying for itself, and it is a correctness bug in a labeller, not a research
question.

### RANK 5 — measure the same thing on the arms that matter (cheap, and currently unmeasured)

Everything here is **flagship-v1 / comma2k19 / 256 px square**. The screen in §6 costs minutes per
arm and has **never been run on v5f (429-token, 176×624), on REF-C's ResNet trunk, or on
PhysicalAI-AV**. Two of those are the arms the programme is actually deciding about.
⛔ **Nothing in this document may be quoted as if it transfers to them.**

---

## 6. ⭐ THE DELIVERABLE THAT OUTLIVES THIS RUN — a 0-GPU latent screen with numeric targets

Any candidate encoder can now be gated **before** a training run is paid for. Three numbers, all
computable from a banked latent cache in minutes, all with an oracle positive control:

| # | screen | frozen v1 (FAIL) | oracle | **gate to pass** |
|---|---|---:|---:|---|
| 1 | ⭐ **JITTER RATIO** — std of `w·(z_{j+1}−z_j)` ÷ the true increment std, for the shared speed direction `w` | **51.0×** | 1.0× | **≤ 2×** — the single most diagnostic number, and the cheapest |
| 2 | corr(within-window speed derivative, true derivative) | **+0.0891** *(per-position: +0.0061)* | +0.99997 | **> 0.50** |
| 3 | derived `long_accel` R² (best 9-point SG derivative) | **−0.3773** | +0.9277 | **> +0.50**, equivalently speed-read **σ ≤ 0.28 m/s** |
| 4 | mean cosine between latents **100 ms apart** *(supporting, not decisive)* | **0.98825** | — | context for 1–3; a high cosine with high jitter means the motion is off-axis noise |

Implementation: `run_mechanism.py` + `run_precision_ladder.py` in this directory; the reusable
pieces are already in the repo instrument (`stack/tanitad/eval/accel_probe.py`, extended this run
with the `tdiff` / `abstdiff` adjacent-frame bases + a contract test).

**⇒ ESCALATION (see §8): this screen should become a pre-flight step in `GATE_PROTOCOL.md`.**

---

## 7. LIMITS — what this does NOT establish

* **One encoder, one corpus, one geometry.** flagship-v1 step 29999, 256 px square, 50 comma2k19
  highway episodes, 17 held out. **Nothing transfers to v5f or PhysicalAI-AV without re-running.**
* **One channel.** `long_accel`. The lateral and strategic families are reported but this design is
  not powered for them, and comma2k19 has no object or route labels at all (§4.3).
* ⛔ **"The information is not in the video" is NOT shown.** Outcome V says *not carried at the
  strength `speed` is carried at, at n = 17 held-out episodes, above the measured floor* — and that
  floor is **SNR-limited, not amplitude-limited** (INHERITED from the prior panel's carrier sweep:
  a signal on a high-variance direction can hide at true R² 0.6; two planted-amplitude points were
  re-measured here, +0.9953 / +0.9966).
* ⚠️ **The appearance-shortcut finding (§0.0) is measured on comma2k19 HIGHWAY only.** That corpus
  is ~74 % straight and its speed distribution is strongly tied to road class, which is exactly the
  condition that makes an appearance shortcut available. **Its magnitude on PhysicalAI-AV or v5f is
  UNKNOWN and must be measured, not assumed** — that is RANK 1.
* **The cosine and correlation screens are diagnostics, not a theory.** They predict this null well;
  their thresholds in §6 are proposed from **one** encoder and should be re-calibrated as soon as a
  second latent is screened.
* **n = 17 held-out episodes** is the independent unit and sets every interval width.
* Wall-clock numbers in `raw/temporal_kv_cost.json` were taken under GPU contention and are not
  quotable; MACs, memory and parameter counts are.

---

## 8. ESCALATIONS — these need someone else to act

0. ⭐⭐ **THE APPEARANCE SHORTCUT NEEDS A PROGRAMME-WIDE AUDIT (RANK 1), and it is not mine to
   schedule.** A single static frame reads `speed` at 93 % of the full learned latent's accuracy,
   and every pure-difference basis reads it at exactly the null. If that holds on PhysicalAI-AV, a
   large part of the longitudinal story — including the cross-rig collapse currently attributed to
   camera geometry (frozen v1 speed R² **+0.930 → −2.465**,
   `…/2026-07-22-idm-proof/results.json`) — has a simpler explanation. **Owner needed.**
1. ⭐ **`Project Steering/GATE_PROTOCOL.md` should adopt the §6 latent screen as a pre-flight gate
   for any encoder-training authorisation.** `dynenc-branchB` spent 40 k steps producing a latent
   the screen would have rejected in minutes (§5 RANK 2). This is a protocol change and is not mine
   to make.
2. ⭐ **`BACKLOG.md` B5 (frozen V-JEPA 2 video-pretrained encoder) is the top-ranked encoder
   experiment by this analysis** — and should be re-scoped to **run the §6 screen FIRST**, at which
   point it becomes an hours-scale decision instead of a GPU-week bet.
3. ⛔ **The prior panel's `long_accel` precision requirement is superseded.**
   `stack/scripts/idm_head.py` and
   `…/2026-08-03-idm-accel-recoverability/ACCEL_RECOVERABILITY.md` §4.7 state *"the speed track
   would need σ ≲ 0.1 m/s … a ~47× improvement"*. That is correct **for a 2-point centred
   difference**; with the optimal 9-point SG derivative it is **σ ≲ 0.28 m/s, ~21×** (§4.1). The
   docstring should carry the estimator with the number.
4. ⛔ **`ACCEL_RECOVERABILITY.md` §5.1's first lever is now REFUTED, not sharpened.** It concludes
   the remaining levers are *"the representation … or an ego-speed channel"*. Outcome V (§0.0)
   removes the representation lever **for this channel**: no representation tried — learned or
   hand-built, 8 to 18,432 features, linear or rbf — separates on `long_accel` while all of them
   separate on `speed`. The ego-speed channel is the surviving lever, and §4.1 states its
   requirement (**σ ≲ 0.28 m/s**, not σ ≲ 0.1).
5. ⚠️ **`BACKLOG.md` A7 (Delta-JEPA — displacement instead of concatenated endpoints)** gets a
   second, independent refutation here: `v1_tdiff` / `v1_abstdiff` (the true adjacent-frame
   difference bases, which A7 describes and which the old `diff` basis could not form) are at the
   null, and the direct-increment regression gives corr **+0.0007**.
6. ⚠️ **The four-family instrument reports LONGITUDINAL distance-keeping and STRATEGIC as
   UNAVAILABLE on comma2k19** (no object annotation, no route label). Per the binding rule these are
   **work items**: they need a corpus that carries the labels (PhysicalAI-AV has `obstacle.offline`
   on 97.44 % of clips), not a note.
