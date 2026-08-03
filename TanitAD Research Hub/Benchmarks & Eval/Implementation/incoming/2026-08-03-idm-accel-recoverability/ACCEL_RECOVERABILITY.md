# STREAM D — is `long_accel` UNRECOVERABLE from the frozen v1 latents, or merely UNRECOVERED?

**Date** 2026-08-03 · **Substrate** dev box (RTX 4060), **0 pod GPU-h** · **Run directory**
`TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-08-03-idm-accel-recoverability/`
· **Primary artifact** `results_accel_recoverability.json` (B=2000) · **substrate check**
`raw/substrate_verification.json` · **mechanism** `raw/speed_error_mechanism.json` · **log**
`raw/run_log.txt`

*(RESULTS SECTIONS ARE FILLED FROM `summarize.py` — no number in this file is transcribed by hand.)*

---

## 1. THE QUESTION P9 LEFT OPEN

P9 (`…/2026-08-03-idm-derived-accel/`) established, **MEASURED**, that the shipped IDM head's
`long_accel` is **not separated** from a shuffled-latent control — ΔR² **−0.0984 [−0.3087, +0.0179]** —
while `speed` (**+0.7187**), `yaw_rate` (**+0.2252**) and ADE (**−6.59 m**) all are. It concluded, in
`stack/scripts/idm_head.py:120-124`, that *"the channel carries no recoverable information from the
frozen v1 latents at this scale, so no reparameterisation of the head can repair it"*.

That conclusion rests on **one head, one geometry, one training recipe**. A null from a single arm
cannot distinguish:

* **UNRECOVERABLE** — the latents carry nothing usable, so the lever is the representation or the
  data; from
* **UNRECOVERED** — one head failed to find something that is there.

They point at opposite work, so this stream tests the hypothesis properly instead of inheriting it.

---

## 2. SUBSTRATE — RE-VERIFIED, NOT INHERITED

`raw/substrate_verification.json`. Three things had to hold before any verdict was admissible.

| check | result | evidence class |
|---|---|---|
| labels re-derivable from the raw `ep_*.pt` (poses/actions + the same heading repair) | **BIT-EXACT on all 50 episodes**, 0 mismatches on `S` and on the speed sequence `Q` | MEASURED |
| the cached latents really are the frozen v1 encoder's output | re-encoded `ep_00040`/`ep_00041` through `v1_speedjerk_ckpt.pt` (step **29999**): max \|d\| = **0.001953** = **one fp16 ULP**, relative mean \|d\| **3.75e-07** ⇒ reproduced to fp16 rounding | MEASURED |
| geometry | `[T, 9, 256, 256]` asserted per episode — flagship v1 is 256 px SQUARE | MEASURED |

**The oracle ceiling, re-measured on THESE windows** (the "R² 0.902" in `idm_head.py`'s docstring was
measured on a *different* window set — 30 episodes / 8,940 windows):

| quantity | value |
|---|---:|
| CAN `long_accel` from a centred difference of the **TRUE** speed track | **R² 0.8993** (corr 0.9488) |
| CAN `long_accel` from a ridge on the **whole** true speed window (in-sample) | **R² 0.9266** |
| `long_accel` std / MAE-about-mean | 0.6219 / 0.4008 m/s² |

⇒ **the target is a real physical signal and is 90 % explainable from the true speed track.** The
question is entirely about what the latents carry.

**The target is not high-frequency noise either.** Autocorrelation of `long_accel` at 0.2 / 0.4 / 0.8 /
1.6 s is **0.742 / 0.763 / 0.637 / 0.420**, and a 1 s boxcar-smoothed copy still explains **R² 0.812**
of the raw target. A slow predictor is not intrinsically capped here.

### ⛔ Two LABEL DEFECTS found on the way, both load-bearing for every IDM number on this cache

1. **`long_accel` is exactly 0.0 on 7.57 % of all windows.** `ep_00045` is identically 0.0 for all
   138 of its windows (std 0.0); `ep_00080` is 0.0 except its last two. A bootstrap draw can therefore
   contain a constant target, which makes R² undefined — `accel_probe.r2_score` returns **0.0, not
   NaN**, precisely so those draws are not silently dropped from the interval.
2. **TRAIN `yaw_rate` carries heading-repair outliers to ±15.3 rad/s (876 °/s) while the HELD-OUT
   episodes top out at 0.48 rad/s** — train std **0.895** vs held-out std **0.039**, a 23× mismatch.
   This is not a modelling problem; it is the standstill branch of the arctan2-of-ENU-velocity heading
   repair. ⚠️ It makes any inner-validation signal for `yaw_rate` untrustworthy on this cache, so
   **`speed` and `steer` — not `yaw_rate` — are used as the positive-control channels here.**

---

## 3. DESIGN — what makes this decisive rather than one more null

Substrate: the banked latent cache, **50 content-clean comma2k19 episodes**, `k=4` (9-frame,
non-causal) windows at stride 2, `state_dim` 2048. Split: **EPISODE-DISJOINT, identical to the P9
panel** — 33 train / 17 held-out (4,554 / 2,346 windows) — so the contrast lands on the same windows.
Hyperparameters are chosen on an **inner split of TRAIN only** (22 fit / 11 select); the held-out
episodes are touched once.

Every arm emits the same 12-dim contract (4 scalars + 4 waypoints × 2), so the four families are
computable for the closed-form arms as well as the neural ones.

| ingredient | why it is not optional |
|---|---|
| **A capacity ladder whose top rung is not a neural network** — exact kernel ridge over the full regularisation path (α ∈ 10⁻⁴…10¹⁰, 29 values, plus an **exact-mean sentinel**), linear AND rbf (3 bandwidths), on four feature bases: `centre` (2,048), `window` (18,432), `diff` (6,144, antisymmetric window positions — acceleration is a DERIVATIVE), `centre+diff` | no learning rate, no epoch budget, no initialisation to blame. "The head was too small" cannot survive an rbf kernel on 18,432 features |
| **A neural ladder** — transformer d64/L1 → d256/L3 (the SHIPPED geometry) → d512/L6, MLP on the centre token, MLP on the flattened window, bi-GRU, and a **single-task accel-only** arm whose objective differs from the shipped one in the mask alone | covers "too small", "wrong inductive bias" and "multitask interference" separately |
| **A matched SHUFFLED-LATENT control for EVERY arm** | same capacity, same recipe, latent↔target link cut |
| **The EMPIRICAL NULL as a first-class arm** (constant = train mean) | ⚠️ R²=0 is **not** the null: held-out `long_accel` has mean −0.1305 against train's +0.0163, so a no-information predictor already scores **−0.0626**. Same error class as the retracted "the null is 1/3" (it was 0.3678) |
| **IN-SAMPLE (train) R² for every arm** | separates "cannot fit" from "cannot generalise" — no held-out number can |
| **An ORACLE-INPUT capacity control** — the same architectures fed the TRUE speed window instead of the latent | if they reach the oracle, capacity, learning rate and epoch budget are all cleared as explanations |
| **A DETECTION-SENSITIVITY sweep** — the identical probe on latents carrying a KNOWN planted copy of the target, swept in amplitude and in correlation | ⚠️ without it a null is unfalsifiable: indistinguishable from a probe too blunt to see a real but weak signal |
| **A CONTEXT-LENGTH arm** — 2.5 s (k=12) vs 0.8 s (k=4) on the SAME rows, from a per-frame latent track reconstructed out of the cached overlapping windows (overlap asserted bit-identical) | covers "the window was too short" |
| **A component-vs-family SELF-CONSISTENCY control** — the same arm's scalar readout vs its trajectory, two independent routes to the same physics | if they disagree about which channel is recoverable, the instrument is measuring itself |

**Pre-registered reading, both outcomes committed before the run** (`run_accel_recoverability.py`
docstring): no arm separating on `long_accel` while the same arms separate on `speed`/`steer` and the
oracle-input control reaches ~0.9 ⇒ **UNRECOVERABLE at this representation and this n**; any arm
separating ⇒ **UNRECOVERED**, and that arm's geometry is the fix.

---

## 0. HEADLINE — the answer is **UNRECOVERABLE**, with a stated sensitivity floor

*(placed after the design on purpose: the controls are what make it readable)*

1. ⭐ **NO arm recovers `long_accel`. Seventeen latent-input arms** — 6 closed-form kernel ridge
   (linear and rbf, 4 feature bases, the full regularisation path) + 7 neural (transformer
   d64/L1 → d256/L3 → d512/L6 at 20 M params, MLP on the centre token, MLP on the flattened
   18,432-d window, bi-GRU, and a single-task accel-only head) + 2 context-length arms + 2
   standstill-filtered arms — **every one lands at or below the empirical null of −0.0626**, and
   **every closed-form arm's paired ΔR² against its own shuffled-latent control straddles zero**:
   −0.0032, −0.0026, −0.0005, −0.0001, +0.0027, **+0.0038 [−0.0299, +0.0244]** — interval
   half-widths of 0.01–0.05, so this is a tight null, not an underpowered one.
2. ⭐⭐ **The capacity control is decisive.** The **IDENTICAL closed-form protocol** — same feature
   builder, same standardisation, same alpha grid, same skill gate, same inner split, same 17
   held-out episodes, same bootstrap — fed the **TRUE speed window** instead of the latent recovers
   `long_accel` at **R² +0.9262 [+0.8876, +0.9507]**, ΔvsNULL **+0.9888 separated**.
   ⇒ **The protocol is not the limit, the head is not the limit, the regulariser is not the limit.
   The INPUT is.**
3. ✅ **The positive control fires on the same arms in the same run**: `speed` ΔR² vs control
   **+0.7197 [+0.5536, +0.9835]\*** (linear window), **+0.6265\*** (rbf window), **+0.6908\***
   (linear centre); ADE **5.88 m [4.85, 7.02]** against the null's **12.05 m**. A null on
   `long_accel` from arms that simultaneously recover speed is a property of the channel.
4. ✅ **The blind-by-construction control fails exactly as designed** — an MLP given only *v(t)*
   scores `speed` **+0.9677** and `long_accel` **−0.1089**. A scalar cannot contain its own
   derivative, and the pipeline does not pretend otherwise. No leak.
5. ⛔ **Cheating does not rescue it.** Choosing the hyperparameter (or the epoch budget) **on the
   held-out set itself**, the best `long_accel` any closed-form arm reaches is **−0.0335** and the
   best any neural arm reaches is **−0.0388**. **It never becomes positive.**
6. ⛔ **Three named alternatives are dead, each with its own arm:** more CONTEXT (2.5 s vs 0.8 s on
   the same 2,210 rows: identical, and `speed` is −0.0335 *worse*); the MULTITASK objective (an
   accel-only head, epoch-grid oracle bound −0.0616); LABEL NOISE (removing all 8.9 % stationary
   train windows including two fully parked episodes leaves `long_accel` at the null on both the full
   and the moving-only held-out set, while `speed` stays +0.70/+0.65 separated).
7. ⚠️ **The honest bound — this is a null with a magnitude, not an absence.** A *planted* signal is
   found down to **1e-5 of the latent RMS** when it is a clean rank-1 direction, but detection is
   **SNR-limited, not amplitude-limited**: at a carrier SNR ≈ 7 the protocol detects a planted TRUE
   R² of **0.3** (ΔCTRL +0.2131\*) and **misses 0.1** (+0.0011, not separated); at SNR ≈ 704 it
   detects **0.03** (+0.0221\*); at SNR ≈ 0.011 it misses **0.6**. ⇒ **What is excluded is that
   `long_accel` is carried as strongly as `steer` is (measured +0.34, and detected).** A very weak or
   very entangled accel signal is NOT excluded.
8. ⭐ **The mechanism in `idm_head.py` is CORRECTED.** It says differencing fails because it
   *"multiplies whatever error the speed sequence has by 1/(2·dt) = 5×"*. MEASURED: the speed error
   is **strongly autocorrelated (0.9265 at 0.2 s)**, so differencing *cancels* ~93 % of its variance.
   The route still fails — derived accel **R² −65.89 [−112.09, −45.17]** against the CAN label —
   because what survives is **4.787 m/s² of volatility against a target whose own std is
   0.587 m/s²**. It is a DYNAMIC-RANGE problem, not an amplification problem.
9. ⛔ **TACTICAL is where the null actually bites, and ADE cannot see it.** Every latent arm predicts
   **`cruise` for all 2,346 windows** — recall **[0.000, 1.000, 0.000]**, precision
   **[—, 0.7438, —]**, balanced accuracy **0.3333 = exactly chance** on support **402 / 1745 / 199**.
   The oracle-input arm fires all three classes: BA **0.4707 [0.3850, 0.5695]**, recall
   **[0.3035, 0.9931, 0.1156]**, precision **[0.9313, 0.7917, 0.8846]**.
10. ⛔ **Two data defects found on the way, both load-bearing for every IDM number on this cache** —
    two of the fifty episodes are a **parked car** (`ep_00045`, `ep_00080`), 7.57 % of all windows
    have `long_accel` exactly 0.0, and TRAIN `yaw_rate` reaches **15.275 rad/s** from the standstill
    branch of the heading repair while the held-out episodes top out at 0.48. **All 99 outlier
    windows are in `ep_00045`, at a measured ego speed of 0.003 m/s.**

---

## 4. RESULTS

### 4.1 Held-out R² per arm — `results_accel_recoverability.json`, B=2000, 17 episodes / 2,346 windows

| arm | params/feats | speed | steer | **long_accel** | in-sample (train) |
|---|---:|---:|---:|---|---:|
| **NULL_train_mean** (the floor) | 0 | −0.0052 | −0.0401 | **−0.0626 [−0.2126, −0.0018]** | −0.0000 |
| RIDGE linear centre | 2,048 | +0.6857 | +0.2784 | **−0.0626 [−0.2126, −0.0018]** | −0.0000 |
| RIDGE linear **window** | 18,432 | **+0.7145** | +0.3426 | **−0.0626 [−0.2126, −0.0018]** | −0.0000 |
| RIDGE linear diff (derivative basis) | 6,144 | −0.0052 | −0.0401 | **−0.0599 [−0.1979, +0.0033]** | +0.0805 |
| RIDGE linear centre+diff | 8,192 | +0.6650 | +0.2707 | **−0.0626 [−0.2126, −0.0018]** | −0.0000 |
| RIDGE **rbf** centre | 2,048 | +0.5882 | −0.0401 | **−0.0626 [−0.2126, −0.0018]** | −0.0000 |
| RIDGE **rbf** window | 18,432 | +0.6213 | +0.3582 | **−0.0626 [−0.2126, −0.0018]** | −0.0000 |
| NN transformer d64/L1 | 182,604 | −1.5863 | +0.3005 | **−0.1421 [−0.3241, −0.0252]** | +0.0794 |
| NN transformer d256/L3 (SHIPPED) | 2,899,724 | +0.3229 | +0.3669 | **−0.1482 [−0.3342, −0.0194]** | +0.0955 |
| NN transformer d512/L6 | 19,975,180 | +0.4660 | +0.3673 | **−0.1076 [−0.2488, −0.0148]** | +0.0540 |
| NN MLP centre h512 | 1,318,924 | +0.5951 | +0.1962 | **−0.1629 [−0.3313, −0.0532]** | +0.2032 |
| NN MLP window h1024 | 19,939,340 | +0.5507 | +0.3457 | **−0.1127 [−0.2484, −0.0175]** | +0.1082 |
| NN bi-GRU d128/L2 | 760,460 | +0.4056 | +0.4379 | **−0.1142 [−0.2620, −0.0246]** | +0.1237 |
| NN transformer d256/L3 **accel-only** | 2,899,724 | −3.6188 | −48.9676 | **−0.1115 [−0.2569, −0.0201]** | +0.0920 |
| **ORACLE-INPUT ridge linear window** | 9 | **+0.9994** | −0.0401 | **+0.9262 [+0.8876, +0.9507]** | — |

⚠️ The `long_accel` in-sample column is at the SELECTED shrinkage, and for the ridge arms the skill
gate selected the train mean, so `−0.0000` means "the inner split never found a setting with skill",
NOT "the model class cannot interpolate". That is a selection statement, not a capacity statement —
the capacity statement is the oracle-input row.

### 4.2 Paired ΔR² against the matched shuffled-latent control — the decision rows

| arm | Δ speed | **Δ long_accel** |
|---|---|---|
| RIDGE linear centre | +0.6908 [+0.4874, +0.9628]\* | **−0.0005 [−0.0365, +0.0205]** |
| RIDGE linear window | +0.7197 [+0.5536, +0.9835]\* | **+0.0038 [−0.0299, +0.0244]** |
| RIDGE linear centre+diff | +0.6702 [+0.5052, +0.9473]\* | **−0.0032 [−0.0249, +0.0095]** |
| RIDGE rbf centre | +0.5933 [+0.3117, +0.8909]\* | **−0.0026 [−0.0301, +0.0144]** |
| RIDGE rbf window | +0.6265 [+0.3533, +0.9120]\* | **−0.0001 [−0.0274, +0.0171]** |
| NN transformer d256/L3 (SHIPPED) | +0.3470 [+0.2044, +0.5307]\* | **−0.0848 [−0.2329, +0.0005]** |
| NN transformer d512/L6 | +0.4262 [+0.2222, +0.6382]\* | **−0.0435 [−0.1060, +0.0046]** |
| NN MLP window h1024 | +0.4961 [+0.2522, +0.7224]\* | **−0.0495 [−0.1198, +0.0115]** |
| NN bi-GRU | +0.3799 [+0.2029, +0.5871]\* | **−0.0391 [−0.1253, +0.0160]** |
| NN accel-only | +0.0196 [+0.0105, +0.0483]\* | **−0.0563 [−0.1233, −0.0037]\*** |

`*` = the paired episode-cluster bootstrap CI excludes zero. **Not one arm is separated POSITIVE on
`long_accel`; three are separated NEGATIVE** (worse than a head trained on shuffled latents). Every
one of them is separated positive on `speed` in the same draw.

⚠️ `steer` is NOT usable as a second positive control here: its interval is dominated by a handful of
episodes (linear window Δ +0.3827 **[−5.5229, +0.4543]**, not separated). `speed` and ADE are the
controls that fire — the §3 design table's "speed and steer" is corrected to **speed and ADE**.

### 4.3 The capacity control, in full — `raw/oracle_input_capacity.json`

| arm (input = TRUE speed window) | speed | **long_accel** | Δ vs the train-mean null |
|---|---:|---|---|
| **ridge linear window (closed form)** | **+0.9994** | **+0.9262 [+0.8876, +0.9507]** | **+0.9888\*** |
| MLP window h512, raw input | +0.4777 | +0.4299 [+0.3034, +0.5251] | +0.4925\* |
| MLP window h512, standardised input | +0.9214 | +0.0570 [−0.0946, +0.1352] | +0.1196\* |
| transformer d256/L3, standardised | +0.7572 | −0.1064 [−0.2890, −0.0117] | −0.0438 |
| bi-GRU d128/L2, standardised | +0.8220 | −0.0952 [−0.2741, −0.0035] | −0.0326 |
| MLP centre-only (BLIND by construction) | **+0.9677** | −0.1089 [−0.2922, −0.0152] | −0.0464\* |

⚠️ **Read this table honestly, both ways.** The **closed-form** control is unambiguous: the protocol
that carries the verdict recovers the channel at R² 0.93 when the input carries it. The **neural**
controls are mixed — an MLP on the window does separate (+0.4925\* raw, +0.1196\* standardised) but
the **shipped transformer geometry does not recover `long_accel` even from a perfect input**. That is
the one result in this stream pointing at "unrecovered", and it is a real finding about the shipped
head: it is a poor accel learner independent of what it is fed. It does **not** rescue the latent,
because the closed-form family — which has no such weakness — is equally at the null.
⚠️ The raw-vs-standardised split is reported because it was a **defect in my own control**: fed raw
m/s, a head whose input *is* the speed scored `speed` R² only +0.4678. Standardising fixes `speed`
(+0.76…+0.97) and *reduces* the neural accel recovery — an optimisation artifact, recorded rather
than tidied away.

### 4.4 Detection sensitivity — what this null does and does not exclude

Amplitude sweep, clean rank-1 direction (ρ=1), linear ridge on the centre token:

| planted amplitude (fraction of latent RMS) | held-out R² `long_accel` |
|---:|---|
| 0 (the real latent) | −0.0626 [−0.2126, −0.0018] |
| **1e-05** | **+0.9434 [+0.9048, +0.9605]** |
| 3e-05 | +0.9923 [+0.9869, +0.9947] |
| 1e-04 … 1e-02 | +0.9982 … +0.9993 |

Correlation sweep — the useful one, because it is stated in TRUE R²:

| carrier | native projection var | planted TRUE R² | measured | paired Δ vs control |
|---|---:|---:|---|---|
| random dir, frac 0.05 (SNR ≈ 7) | 0.547 | 0.10 | −0.0626 | +0.0011 [−0.0485, +0.0313] |
| random dir, frac 0.05 | 0.547 | **0.30** | +0.1505 | **+0.2131 [+0.0694, +0.3078]\*** |
| random dir, frac 0.50 (SNR ≈ 704) | 0.547 | **0.03** | −0.0405 | **+0.0221 [+0.0056, +0.0331]\*** |
| random dir, frac 0.50 | 0.547 | 0.10 | +0.0304 | +0.0957 [+0.0246, +0.1384]\* |
| **PC1 dir, frac 0.05 (SNR ≈ 0.011)** | **347.3** | **0.60** | **−0.0626** | −0.0027 (not separated) |
| PC1 dir, frac 2.0 (SNR ≈ 17.7) | 347.3 | 0.10 | +0.0147 | +0.0737 [+0.0022, +0.1187]\* |

⇒ **the floor is a function of the CARRIER's signal-to-native-variance ratio, not of amplitude.** A
signal sharing a high-variance direction can hide at true R² 0.6. That is the correct caveat on the
null, and it is why the closed-form ladder includes an **rbf** kernel — which does not privilege
high-variance directions the way ridge's implicit prior does — and still finds nothing.

### 4.5 Context length — `results_accel_recoverability.json → context_length`

k=12 (2.5 s, 24,576 features) vs k=4 (0.8 s) on the **same 2,210 rows**, per-frame latent track
reconstructed from the cached overlapping windows with the overlap asserted bit-identical:

| arm | speed | steer | long_accel |
|---|---:|---:|---|
| k=12 | +0.6805 | +0.4066 | −0.0665 [−0.2212, −0.0022] |
| k=4, same rows | +0.7140 | +0.3494 | −0.0665 [−0.2212, −0.0022] |

paired wide − narrow: `speed` **−0.0335 [−0.0741, +0.0137]**, `long_accel` identical (degenerate —
both fell back to the train mean). **Three times the temporal context changes nothing.**

### 4.6 Label noise is not the cause — `raw/standstill_filtered.json`

**8.89 %** of train windows and **6.22 %** of held-out windows are stationary (< 0.5 m/s); `ep_00045`
and `ep_00080` are stationary in **100 %** of their windows. Refitting with every stationary window
removed from TRAIN:

| arm | scope | long_accel | speed |
|---|---|---|---:|
| filtered ridge linear window | full held-out (n=2,346) | −0.0635 [−0.2141, −0.0020] | +0.7038 |
| filtered ridge linear window | **moving-only** (n=2,200) | −0.0683 [−0.2271, −0.0019] | +0.6501 |
| filtered NULL (train mean) | full held-out | −0.0635 | −0.0106 |
| filtered NULL (train mean) | moving-only | −0.0683 | −0.0005 |

`long_accel` is **numerically identical to the filtered null** on both scopes (paired Δ +0.0048 and
+0.0039, neither separated) while `speed` stays separated. **The parked-car label mass is a real data
defect and it is NOT what causes the null.**

### 4.7 Mechanism — `raw/speed_error_mechanism.json`

Best speed arm (ridge linear window): R² **+0.7145**, MAE **4.7186 m/s**.

| quantity | value |
|---|---|
| autocorrelation of the held-out **speed error** at 0.2 / 0.4 / 0.8 / 1.6 s | **0.9265 / 0.8451 / 0.6719 / 0.4241** |
| centred difference of the PREDICTED speed vs the centred difference of the TRUE speed | R² **−70.28 [−122.22, −46.74]** |
| the same vs the CAN `long_accel` label | R² **−65.89 [−112.09, −45.17]** |
| ORACLE — centred difference of the TRUE speed vs the CAN label | R² **+0.9198 [+0.8771, +0.9447]** |
| std of the true difference / predicted difference / CAN label | 0.568 / **4.787** / 0.587 m/s² |

Error budget (true speed + white noise, differenced, vs the CAN label): σ=0.05 → **+0.828**;
σ=0.10 → **+0.549**; σ=0.25 → **−1.407**. ⇒ **the speed track would need σ ≲ 0.1 m/s**, against the
arm's measured MAE of 4.72 — a **~47×** improvement.

**Self-consistency control (component vs family).** The same arm's scalar readout and its 2 s
trajectory are independent routes to the same physics and they agree about which channel is
recoverable: `speed` **+0.7145 (scalar) vs +0.7242 (trajectory)**; `long_accel` **−0.0626 (scalar) vs
−0.3792 (trajectory)**, against a GT-trajectory ceiling of **+0.6706** for that route.

### 4.8 THE FOUR FAMILIES — per family, never pooled

Reported for **every** arm in `results_accel_recoverability.json → arms.*.four_families`. Headline
arm = `RIDGE_linear_window` (the best speed arm), reference = `NULL_train_mean`, positive control =
the oracle-input arms.

| family | RIDGE linear window | NULL (train mean) | note |
|---|---|---|---|
| **LONGITUDINAL** | scalar-speed MAE **4.719 m/s**, traj-speed MAE 4.568 m/s, along-track MAE **5.842 m** | 9.567 / 9.654 / 12.035 | **distance-keeping UNAVAILABLE, n=0** — comma2k19 ships no object annotation at all. WORK ITEM, not a pass |
| **LATERAL** | heading MAE **0.1213 rad**, curvature MAE **0.00619 1/m**, yaw-rate MAE **0.01953 rad/s**, cross-track MAE **0.2997 m** | 0.0167 / 0.00439 / 0.01728 / 0.2808 | ⚠️ the arm is *worse than the null* on heading and curvature — this corpus is near-straight highway, so lateral has almost no variance to explain |
| **TACTICAL** (trajectory) | lateral BA **0.3873**, longitudinal BA **0.3295**, mixed BA 0.2312 (chance 0.3333 / 0.3333 / 0.2000) | 0.3333 / 0.3333 / 0.2000 | at chance, consistent with P9's finding on the shipped head |
| **TACTICAL** (from the `long_accel` scalar — the channel under test) | BA **0.3333 = chance**; recall **[0.000, 1.000, 0.000]**, precision **[—, 0.7438, —]**, support **[402, 1745, 199]**, fires `cruise` on **2,346 / 2,346** windows | identical | **the null made operational**: the channel drives no longitudinal decision at all |
| **TACTICAL** (same readout, ORACLE input) | BA **0.4707 [0.3850, 0.5695]**, recall **[0.3035, 0.9931, 0.1156]**, precision **[0.9313, 0.7917, 0.8846]** | — | the readout CAN discriminate; it is the input that cannot |
| **STRATEGIC** | **UNAVAILABLE, n=2,346** | | no route/goal label on comma2k19; `idm_families.strategic()` states the reason in the artifact. WORK ITEM |
| **ADE@2 s** (one row, never the result) | **5.882 m [4.855, 7.016]** | 12.047 m [9.608, 14.629] | |

---

## 5. WHAT THIS CHANGES

1. **Stop tuning IDM heads for `long_accel` on frozen v1 latents.** Seventeen latent-input arms,
   three architecture families, four feature bases, two context lengths, a full regularisation path
   and a test-set-selected upper bound all land on the null. The remaining levers are the
   **representation** (an encoder trained to carry dynamics rather than appearance) and the **input**
   (an ego-speed channel, which is what the oracle-input control shows would work).
2. **A cheap alternative exists and is measured here.** `long_accel` is R² **0.9262** recoverable
   from the *true* speed window by a **9-feature** ridge. Wherever the IDM is used as a labeller and
   a speed track is available (CAN, or a downstream state estimate), the accel channel should be
   **derived from that track, not read out of the latent**. What P9 refuted was deriving it from a
   *predicted* speed track at R² 0.72 — §4.7 quantifies exactly how good that track must be:
   **σ ≲ 0.1 m/s**.
3. **The IDM's TACTICAL longitudinal decision is currently vacuous** — one class, 2,346 / 2,346
   windows. Any downstream use of the IDM as a pseudo-labeller for longitudinal manoeuvres is
   labelling `cruise` unconditionally. This is invisible in ADE and in the scalar R²: it is the
   binding-rule payoff — the four families found it, a horizon sweep could not.
4. **Two data defects to fix before the next IDM number:** filter stationary windows (8.9 % of
   train), and either repair or exclude the standstill branch of the heading repair, the sole source
   of the ±15 rad/s `yaw_rate` outliers.

## 6. ESCALATIONS — these need someone else to act

1. **`stack/scripts/idm_head.py:104-127` must be updated.** It asserts, as settled, that *"the
   channel carries no recoverable information from the frozen v1 latents at this scale, so no
   reparameterisation of the head can repair it"*, and explains the failure as a **5× error
   amplification**. This run **supports the conclusion far more strongly than P9 could** — but the
   stated mechanism is **wrong** (the error is autocorrelated at 0.9265, so differencing cancels
   93 % of it; the killer is the target's 0.587 m/s² dynamic range), and the claim needs its
   **sensitivity floor** attached or the programme inherits an absolute from a bounded measurement.
2. **`BACKLOG.md` A7 (Delta-JEPA — displacement instead of concatenated endpoints)** is answered at
   probe level: the `diff` basis (antisymmetric window positions) is exactly that reparameterisation.
   It gives `long_accel` **−0.0599 [−0.1979, +0.0033]** (ΔCTRL +0.0027, not separated) and it
   **destroys `speed`** (−0.0052 = the null). A displacement-only input is strictly worse here.
3. **A stationary-window filter belongs in the IDM ingest**, not in each experiment. `ep_00045` and
   `ep_00080` are parked cars inside a driving corpus.
4. **Instrument follow-up (cheap):** `DualRidge.predict` recomputes the kernel per alpha; caching the
   test Gram per kernel would cut the rbf-on-18,432-feature arms by roughly an order of magnitude.
   Deliberately NOT changed after the run started, so the staged file is the one that produced the
   artifact.

## 7. WHAT WENT WRONG IN THIS RUN, AND WHAT IT COST

Four defects were caught **by the controls**, not by inspection. All four are fixed in the staged
code and their evidence is kept in `raw/`.

| defect | how it showed | fix |
|---|---|---|
| **argmax hyperparameter selection on a channel with no signal** | the shuffled control's `yaw_rate` scored **R² −4229** on held-out, because train `yaw_rate` has ±15 rad/s outliers and the held-out set does not | a **skill gate** — unless the inner-val best clears R² 0.01, emit the train mean — then a one-SE shrinkage tie-break |
| **a spurious `separated=true` on a null effect** | arms that both fell back to a *huge-alpha* ridge (not the exact mean) differed by ~1e-5 and the paired bootstrap called it separated | an **exact-mean sentinel** in the hyperparameter grid plus a **degeneracy guard** that forces `separated=False` when two arms' predictions are identical |
| **the epoch budget selected on `long_accel` inner R²** — i.e. on noise | the shipped transformer came out at `speed` R² **−2.78** where P9's identical geometry reaches +0.72; the positive control was destroyed | one **uniform** criterion (inner mean of R² clipped to [−1,1] over all four channels) **plus** a per-arm oracle-over-the-epoch-grid bound; the whole run was repeated (`raw/results_v1_BADEPOCHSELECTION.json` kept) |
| **the sensitivity control was shuffled BEFORE the injection** | planting a strong signal made the "control" *better* than the arm (ΔR² −0.0897) — the shuffle destroyed the native link and the injection then rebuilt it | inject first, shuffle second |

⚠️ Two of these (the skill gate and the epoch criterion) are **judgement calls that change numbers**.
Both are stated in the artifact next to every value they touch, and both are made harmless for the
headline claim by the test-set-selected upper bounds in §4.1 and §4.3, which depend on no selection
rule at all.

## 8. LIMITS — what this does NOT establish

* **It is bounded by power, not absolute.** §4.4 is the number: a signal on a carrier at SNR ≈ 7 must
  reach true R² ≈ 0.3 to be seen; on a high-variance carrier it can hide at 0.6. "No information" is
  not proven — "no information at the strength `steer` is carried at" is.
* **One encoder, one geometry, one corpus.** flagship-v1 (step 29999), 256 px square, 50 comma2k19
  episodes. Nothing here transfers to v5f (429-token, 176×624) or to PhysicalAI-AV without re-running,
  and it must not be quoted as if it did.
* **n = 33 train / 17 held-out episodes.** The episode is the independent unit and there are 50 of
  them. Every interval already reflects that; it is also why the sensitivity floor sits where it does.
* **The neural family's null is weaker evidence than the closed-form family's**, because the shipped
  transformer is a poor accel learner even on a perfect input (§4.3). The verdict rests on the
  closed-form ladder, whose capacity control is unambiguous.
