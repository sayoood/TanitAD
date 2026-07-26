# H2 — the attention-based SENSOR-NEED CLASSIFIER: pre-registration, training, held-out result

**Date:** 2026-07-26 (local, Europe/Berlin). **Author:** research engineer (H2 classifier stream).
**Pre-registration:** `PRE_REGISTRATION.md`, this folder, **written before any feature was extracted
and any head weight existed** (17:53 local / 15:53 UTC; every artifact in `artifacts/` is later).
**Host:** pod2 (A40). pod1 / pod3 / eval-pod untouched.

**Evidence classes:** `MEASURED` (ours + path) · `PUBLISHED` (cited) · `INHERITED` (another
agent/doc, NOT re-verified) · `ESTIMATED` · `HYPOTHESIS`.

🔒 **PhysicalAI-AV is gated-confidential.** No clip UUID and no raw content appears anywhere in this
folder; clips carry an integer index only. The UUID map exists solely on the dev box, outside the
repo.

*(Results sections are filled from `artifacts/h2c_results.json` by `scripts/h2c_report.py` — no
number in this document is typed by hand.)*

---

# 0. VERDICT IN ONE BOX

> ## **The head exists and is trained. The pre-registered comparison is UNDERPOWERED — but the run is not empty: it settles the efficiency claim, closes the C12 trap, and returns one WELL-POWERED negative about the frozen representation.**
>
> **The pre-registered rule (`PRE_REGISTRATION.md §7`, evaluated in code) returns `UNDERPOWERED`.**
> At the pre-registered operating point the primary arm beats **neither** baseline with a CI that
> excludes 0: vs random-at-matched-rate **Δrecall +0.1013 [−0.0158, +0.2727]**, vs the ego-state
> heuristic **+0.0229 [−0.1106, +0.1455]**. Paired episode-cluster bootstrap, B = 2000, 322 clips.
> **35 trigger-positive held-out clusters** — and the LABEL'S OWN effect is not separated on this
> same subset (§4). **This is UNPOWERED, not a refutation, and it must not be reported as either
> Outcome A or Outcome B.**
>
> ### ⭐ What IS settled, because it does not depend on the underpowered comparison
>
> **1. The efficiency claim is arithmetically true and INFORMATION-FREE, and this is the finding
> that should change how H2 is discussed.** MEASURED: the gate costs **0.072 % of one camera pass**
> (MACs) / 0.58 % (A40 wall-clock). Against always-on-7, *never escalating* saves **85.7 %** and a
> *perfect oracle* saves **85.6 %** — **the entire span between useless and perfect is 0.1
> percentage points.** Our operating point sits at **84.8 % [84.5, 85.1]**. **Nothing about a
> gating policy's compute saving discriminates between a good gate and a useless one.** The only
> axis that carries information is **recall at a fixed budget**, and that is where the evidence is
> thin: at B\* the head catches **42 of 306** events (recall **0.1373 [0.0188, 0.3199]**) and
> **10 of 36** behavioural-slice events.
>
> **2. ⛔ NO ARM IS ABOVE CHANCE — and that test is the one I nearly got wrong.** Comparing an AP
> interval to the full-sample base rate is *not* an above-chance test, because the base rate is
> itself random under episode resampling. The correct statistic is a **paired ΔAP against a constant
> score** (whose AP equals the base rate *inside every draw*). Measured that way:
>
> | arm | ΔAP vs chance | CI95 | above chance? |
> |---|---|---|---|
> | `head_img_ego` (PRIMARY) | +0.00268 | [−0.00278, +0.02639] | **no** |
> | `head_img` | +0.00024 | [−0.00421, +0.00918] | **no** |
> | `head_ego` | +0.00749 | [−0.00113, +0.03170] | **no** |
> | `heur_decel` | +0.00241 | [−0.00314, +0.01535] | **no** |
> | `heur_speed` | −0.00328 | [−0.00835, −0.00015] | **separated BELOW chance** |
>
> **The only separated result in the entire primary comparison is that ego SPEED is worse than
> chance** — trigger frames are *slower* (AUROC 0.2965), the same confound E1 recorded. Everything
> else is a point estimate with a CI through zero. The ordering (`head_ego` > `head_img_ego` >
> `head_img`) is the same in the training CV (0.0198 / 0.0097 / 0.0095) and in the held-out point
> estimates (4.18× / 2.60× / 1.80× base), and `head_img_ego` − `heur_decel` is **+0.00027
> [−0.00633, +0.01462]** — but **none of it clears chance, so it is a hint, not a finding.**
> ⚠️ **And even if it held, the conclusion would NOT be "ship a kinematic gate".** The ego arms key
> on the **trailing** 0.5 s acceleration, i.e. on braking already under way — the label's own
> confound (`P(already braking | trigger⁺) = 35.6 %` vs `14.1 %`; adjusting for it collapses the
> label's lift to 1.35× [0.82, 2.05], INHERITED). A reactive proxy is not the gate H2 needs.
> **Nothing here has demonstrated anticipation.** §6.1 d-bis.
>
> **3. The C12 trap is closed by construction on this corpus, and that is a real gain.**
> `P(L2_trigger | T_off) = 0.9623` — the counterfactual clause `a_req_seen < τ*` removes only
> **12 of 318** frames. **The composite IS its hard conjunct here**, so a null cannot be blamed on
> the easy half. C12's warning is discharged with a measurement rather than an argument.
>
> ### ⭐⭐ 4. THE WELL-POWERED RESULT — and it is the one that should redirect H2
>
> The pre-registered `T_seen` diagnostic was **mis-posed** (96.7 %-positive target trained with a
> rare-positive recipe) and **is not read**. Re-posed correctly as **`NOT_T_seen`** — *"an agent the
> encoder CAN see requires braking ≥ τ\*"*, **1,642 held-out positives across 101 clips**, i.e.
> **5.4× the positives and 2.9× the clusters** of the composite — the answer is unambiguous:
>
> | arm | AP | AP / base | ΔAP vs chance | above chance? |
> |---|---|---|---|---|
> | **`head_ego`** (NO camera) | **0.1226** | **3.74×** | **+0.0766 [+0.0506, +0.1353]** | ✅ **YES** |
> | `head_img_ego` | 0.0521 | 1.59× | +0.0060 [−0.0004, +0.0395] | no |
> | `head_img` | 0.0491 | 1.50× | +0.0031 [−0.0029, +0.0428] | no |
>
> **This is a POSITIVE CONTROL that fires, and it changes what "nothing is above chance" means.**
> The pipeline, the head, the CV and the estimator are all capable of finding a separated signal —
> they found one here. So the §0-point-2 null is a statement about **what the frozen image features
> deliver**, not about a broken rig.
>
> **Two things follow, and both are actionable:**
> 1. ⛔ **The frozen v1 state does not expose the most basic visual precondition the H2 gate needs**
>    — *"is there something ahead I must brake for?"* — even with 836 training positives and 1,642
>    held-out ones. It does not clear chance. *(Bounded claim: this is one head and one recipe on a
>    frozen representation. It does not prove the information is absent — a linear probe or a
>    fine-tuned trunk may still find it. It does prove this head cannot.)*
> 2. ⚠️ **Adding the image features to a WORKING ego head destroys it** — 3.74× → 1.59×, from
>    separated to not-separated. That is a capacity/optimisation failure signature (2048 × 8 image
>    dims swamping 2 ego channels at n = 836 positives), not evidence about information content —
>    and it says the next experiment must add vision in a **low-dimensional, regularised** way, not
>    by concatenation.
>
> ### The one number that decides what to do next
>
> **The universe is 582 of the label's 2,320 clips**, because only clips already decoded into the
> pod2 episode cache can be used. The bottleneck is **not science, it is a ~52 GB gated camera
> re-download plus a re-decode** — and the label, the head, the estimator and the whole pipeline in
> this folder run unchanged on the larger set. 🔴 **This is the escalation: authorise the corpus
> expansion, or H2 stays unanswerable at this n.**

---

# 1. What was built, and why it did not exist before

The L2 sensor-need label was GO-gated on 2026-07-26 — *"GO — training can start"* — and **nobody
started it**. This stream closes that gap: it trains the head, evaluates it on held-out episodes,
measures the efficiency claim the idea actually rests on, and compares it against four honest
baselines.

| | |
|---|---|
| **Target** | `L2_trigger` at **τ\* = 0.5 m/s², primary (out-of-crop) scope, resolvable-only** — imported from `l2_label.py`, **never re-implemented and never re-swept** |
| **Input** | the front camera only (Sayed's explicit Step-1 scoping) + the ego's own speed, which v1 receives by design (`action_dim = 3`) |
| **Encoder** | `flagship4b-speedjerk-30k` @ 29999 — **the deployed v1** (INHERITED, `MODEL_REGISTRY §1.2`; *not* `flagship4b-phase0-30k`, which is the no-speed ablation control). **Frozen**, STRICT-loaded via `tanitad.eval.ckpt_compat.build_world_from_ckpt`. 87,022,848 encoder + 98,432 readout params, state_dim 2048 (MEASURED, `artifacts/align_summary.json`) |
| **Head** | attention over an **8-step (0.8 s) window** of frozen 2048-d states + ego, → **per-camera independent Bernoulli** (never a softmax over mixed axes — `H2_SUBSTRATE §C.1`) |
| **Cost** | ~10 CPU-min of joins on the dev box · **9.6 GPU-min** of feature extraction · head training on one idle A40. No download. No training job perturbed. |

---

# 2. The substrate, and the three limits that bound what any answer here can mean

### 2.1 The episode ↔ clip join is PROVEN, not assumed

`build_episode` stores `episode_id = int.from_bytes(clip_id[:4])`. Replaying the cache's own recipe
(`discover_r0_clips → sorted → split_clips(val_frac=0.2, seed=0)`) reproduces **2376/2376 train** and
**600/600 val** stored episode ids, and the 24 corrupt-clip skips land exactly on 1798…1941.
(MEASURED, `artifacts/join_proof.json`.) Without this the whole study would rest on an assumed
ordering.

### 2.2 ⚠️ The universe is 582 clips, not the label's 2,320 — and that is the binding limit

Only clips that are **both** L2-labelled **and** already decoded into the pod2 episode cache can be
used. The rest would need a ~52 GB gated camera re-download plus a full re-decode. This was stated
in the pre-registration **before** any result, not after it.

### 2.3 ⚠️ The frozen encoder saw most of these clips during its own training

MEASURED: of the 359 held-out clips, **283 were in the encoder's own train split and 76 were not**
(TRAIN side: 176 / 47). The head's split is chunk- and clip-disjoint, but the *features* on most
held-out clips come from an encoder that has seen those pixels. A sensitivity restricted to the
encoder-unseen clips is reported in §7 whatever it says.

### 2.4 ⚠️ `obstacle.offline` is `scene:obstacles:autolabels:v2` — machine labels

`prov: "autolabel"`. Systematic misses of small or distant agents attenuate everything measured here.

---

# 3. ⚠️ The alignment guard tripped — and the adjudication is MEASURED, not argued

The label grid (`arange(t0, t1, 100 ms)` on the egomotion/obstacle clock) and the episode grid
(`linspace` on the camera clock, minus the 2 frames the D-015 stack consumes) are different indices
with no stored offset. Alignment is recovered per clip by maximising the Pearson correlation of the
two ego-speed series over an integer lag. The pre-registration admits a clip at **corr ≥ 0.99** and
declares the run **BLOCKED if more than 10 % fail**.

**MEASURED (`artifacts/align_summary.json`): 62 of 582 clips failed — 10.65 %, i.e. the guard
tripped.** Lags concentrate at +1…+3 (493 of 520 admitted); median admitted correlation 0.999904.

**The guard tripped on its own degeneracy, and this is measured rather than asserted.** Pearson
correlation is **undefined for a constant series**, so a clip driven at steady speed can be perfectly
aligned and still score ≈ 0. A **second probe** (`scripts/h2c_align2.py`, `artifacts/align_second_probe.json`)
re-aligned all 62 dropped clips with a statistic that *is* defined there — minimum RMSE between the
two speed series over the same lag search:

| | n | reading |
|---|---|---|
| dropped clips with best-lag **RMSE ≤ 0.25 m/s** | **19** | correctly aligned; median speed σ **0.036 m/s** — the statistic degenerated, the alignment did not fail |
| dropped clips with RMSE > 0.25 m/s | 43 | genuine alignment failure |
| …of which **no ≥ 60-sample overlap at any lag** | 12 | the label window and the episode barely overlap; these must be dropped under any rule |

⇒ **the genuine alignment-failure rate is 43 / 582 = 7.39 %, inside the pre-registered 10 % bar.**
Corroborating: among clips with a real speed profile (σ ≥ 1.0 m/s) the drop rate is **8.1 %**, and
Spearman(corr, speed-σ) = **0.62** across all 582.

**Adjudication: the run is NOT blocked, and the admitted set is NOT widened.** The primary analysis
uses exactly the 520 clips the *strict, pre-registered* rule admitted — the guard's output is kept,
only its 10 % trip is re-read. The choice is immaterial either way: **the 62 dropped clips carry
2 of 477 trigger positives (0.42 %)** (MEASURED, `artifacts/align_admission.json`).

*(This is a C13-class observation applied to my own guard: before trusting a threshold, ask what
value would make it fire — and whether the estimator can even be computed on the population it is
applied to.)*

---

# 4. ⭐ The power ceiling, measured BEFORE the classifier was read

The GO rests on **1,415** CONFIRM clips. The classifier is evaluated on the **322** of them that are
in the pod2 cache and clear the alignment floor. So the first question is whether that subset still
carries the decision-relevance the GO was granted for — asked and answered before any classifier
score was looked at (MEASURED, `artifacts/subset_lift.json`; paired episode-cluster bootstrap, ratio
form, B = 2000, `taniteval.ci._draws`):

| population | clips | trigger⁺ frames | trigger⁺ clips | **decision-relevance lift** | 95 % CI |
|---|---|---|---|---|---|
| **the 322 held-out clips this classifier is scored on** | 322 | 306 | **35** | **2.171×** | **[0.645, 4.469]** ❌ includes 1.0 |
| the 198 train clips | 198 | 169 | 25 | 3.212× | [0.743, 6.726] ❌ |
| all 26 chunks, full label table (context) | 2,320 | 1,987 | 231 | 2.518× | [1.731, 3.460] ✅ |
| **published GO** (INHERITED, `l2_confirm.json`) | 1,415 | 1,192 | 138 | **2.41×** | **[1.3998, 3.7041]** ✅ |

> ⚠️ **Read this before reading anything else.** On the exact clips this classifier is evaluated on,
> the **label's own** effect is *not separated from 1.0* — 35 positive clusters against the 138 the
> GO rested on. The point estimate is consistent (2.17× vs 2.41×) and the full 2,320-clip table
> reproduces the published effect at 2.518× [1.731, 3.460], so nothing here challenges the label.
> But it fixes a **ceiling on the strength of any claim this run can make about *decision-relevant*
> escalation**, and it is exactly the "a small-n non-separation is UNPOWERED, not refuted" situation
> the program has already measured (CI half-widths shrink ×2.8–3.9 from 40 → 600 episodes).
>
> **Predicting the trigger is still a well-posed supervised problem at this n** — AP against a
> **0.00305** base rate (306 positives in 100,238 camera-frame pairs) has its own power, better than
> the lift's — and that is what §5–§7 measure.

---

# 5. Training setup — what was actually run

### 5.1 The split (pre-registered, episode-disjoint)

| | chunks | clips admitted | windows | trigger⁺ (camera-frame) | trigger⁺ clips |
|---|---|---|---|---|---|
| **TRAIN** = the label's own **DEV** chunks | 10 | **198** | 31,032 | **169** | 25 |
| **HELD-OUT** = the label's own **CONFIRM** chunks | 16 | **322** | 50,119 | **306** | 35 |

Chunk-disjoint ⇒ clip-disjoint ⇒ episode-disjoint. Reusing the *label's* DEV/CONFIRM boundary means
the held-out side is the side on which τ\* had never been looked at; any re-cut would have moved
threshold-selection chunks into the test set.

**Model selection: 5-fold grouped CV inside TRAIN, grouped by chunk** (2 chunks per fold). The
held-out side is not read by `h2c_train.py` at all — that script computes no held-out metric; it
only writes scores for the evaluator.

### 5.2 The feature and the head

```
frozen  :  frames_u8 [T,9,256,256] --(/255)--> ViTEncoder(d768, depth12, patch16 -> 16x16 tokens)
                                   --> SpatialGridReadout(grid 4x4, d_readout 128) --> state [2048]
           (verified against the loaded checkpoint: readout params 98,432 == 768*128 + 128)
head    :  8 x (state[2048] (+ ego[v/10, a_pre/5]))  -> Linear -> +pos
           -> 2 x TransformerEncoderLayer(d, 4 heads, pre-norm, GELU, dropout 0.2)
           -> attention pooling over the 8 steps -> MLP -> 2 logits (cam_left, cam_right)
```

Per-camera **independent Bernoulli** — never a softmax over mixed axes (`H2_SUBSTRATE §C.1`; the
5-way maneuver-softmax defect). The encoder receives no gradient; features arrive pre-computed.
**Ego inputs are strictly causal**: `v(t)` and the trailing-0.5 s mean acceleration `a_pre(t)`.
`alon_fut_min` and `ego_dv4` (both future-looking, and both used by the label's *response*) are
never given to any arm.

### 5.3 The arms

| arm | inputs | role |
|---|---|---|
| **`head_img_ego`** | frozen states + ego | ⭐ PRIMARY — the deployable configuration |
| `head_img` | frozen states only | is there signal in the **image** at all |
| `head_ego` | ego only, same head | how much is just ego state, **learned** |
| **`heur_ego_both`** | ego only, **not learned** | the (v, a_pre) threshold-rule family, fitted on TRAIN to the same objective — the baseline that decides the verdict |
| `heur_ego_one` | ego only, not learned | same rule, one camera at random — half the cost, half the expected recall |
| `random_at_rate` | none | matched firing rate, 200 seeds |
| `always` / `never` | none | the two trivial endpoints |

⚠️ **An ego-only rule has no direction.** It cannot say *which* side camera to wake, so at a matched
compute budget it must either wake both (`_both`, 2 activations per firing frame) or pick one at
random (`_one`, half the recall). Both variants are reported; scoring the ego rule as if it knew the
camera would be the flattering error.

---

# 6. Held-out results

**How to read this section.** The unit is the **(camera, frame) pair** — firing the left camera when
the right one was needed is a miss, and only this unit scores it that way. `AP` is the
step-interpolated average precision (`h2c_stats.average_precision`), reported **against the base
rate**, never as accuracy. Every interval is a **paired episode-cluster bootstrap** over the 322
held-out clips, B = 2000, `taniteval.ci._draws`. **`overlapping_holdout_se` appears nowhere.**

⚠️ Ties in the non-learned ego scores are broken by row order, which is mildly **optimistic for
those baselines** — the safe direction for the verdict.

✅ **The operating-point row was re-derived independently of the evaluator** (a separate script
sharing no code with `h2c_eval.py`, reading only the raw score `.npz`): `θ* = 0.175311`,
camera-frame rate `0.031475`, recall `42/306 = 0.137255`, precision `0.013312`, lift `4.3607`,
behavioural slice `10/36`. **Every digit matches** — so the verdict does not rest on a bug in my own
reducer.

<!-- TABLES:RESULTS -->

### Discrimination — held-out, (camera, frame) pairs

*Base rate (chance AP) = **0.00305** (306 positives in 100238 pairs, 322 clips, 35 of them positive).*

| arm | AP (episode-cluster bootstrap CI95) | AP / base rate | AUROC |
|---|---|---|---|
| `head_img_ego` | 0.00795 [0.00285, 0.03071] | **2.60x** | 0.6832 |
| `head_img` | 0.00551 [0.00249, 0.01285] | **1.80x** | 0.6611 |
| `head_ego` | 0.01276 [0.00397, 0.03615] | **4.18x** | 0.6782 |
| `heur_speed` | 0.00199 [0.00108, 0.00320] | **0.65x** | 0.2965 |
| `heur_decel` | 0.00767 [0.00286, 0.01914] | **2.51x** | 0.5796 |

### Is any arm above CHANCE? (paired ΔAP against a constant score)

*A constant score has AP equal to the base rate **inside every bootstrap draw**, so this — not "does the AP interval clear the full-sample base rate" — is the correct above-chance test.*

| arm | ΔAP vs chance | CI95 | above chance? |
|---|---|---|---|
| `head_img_ego` | +0.00268 | [-0.00278, +0.02639] | no |
| `head_img` | +0.00024 | [-0.00421, +0.00918] | no |
| `head_ego` | +0.00749 | [-0.00113, +0.03170] | no |
| `heur_speed` | -0.00328 | [-0.00835, -0.00015] | below chance |
| `heur_decel` | +0.00241 | [-0.00314, +0.01535] | no |

### Paired AP deltas vs the primary (`head_img_ego`)

| contrast | ΔAP | CI95 | separated? |
|---|---|---|---|
| head_img_ego - head_img | +0.00244 | [-0.00003, +0.01804] | no |
| head_img_ego - head_ego | -0.00481 | [-0.01864, +0.00560] | no |
| head_img_ego - heur_speed | +0.00596 | [+0.00144, +0.02833] | **YES** |
| head_img_ego - heur_decel | +0.00027 | [-0.00633, +0.01462] | no |

### The operating point (pre-registered `theta*` fixed on TRAIN out-of-fold)

Budget `B* = 0.05` extra camera activations/frame ⇒ target camera-frame rate 0.0250; **realised 0.0315** (ratio 1.26x) — the calibration-transfer test.

| arm | firing rate | extra cams/frame | recall | precision | precision lift | recall on behavioural slice | missed |
|---|---|---|---|---|---|---|---|
| `head_img_ego` | 0.0315 [0.0221, 0.0424] | 0.0630 | 0.1373 [0.0188, 0.3199] | 0.01331 [0.00161, 0.03530] | 4.36 [0.62, 10.59] | 0.2778 | 264/306 |
| `head_img` | 0.0154 [0.0102, 0.0210] | 0.0308 | 0.0654 [0.0000, 0.1732] | 0.01295 [0.00000, 0.03819] | 4.24 [0.00, 11.80] | 0.0000 | 286/306 |
| `head_ego` | 0.0178 [0.0145, 0.0214] | 0.0356 | 0.1699 [0.0274, 0.3648] | 0.02915 [0.00392, 0.06388] | 9.55 [1.60, 20.58] | 0.2778 | 254/306 |
| `heur_ego_both` | 0.0274 [0.0209, 0.0342] | 0.0547 | 0.1144 [0.0277, 0.2420] | 0.01276 [0.00266, 0.02692] | 4.18 [1.07, 8.86] | 0.5833 | 271/306 |
| `heur_ego_both_rate_matched` | 0.0274 [0.0209, 0.0342] | 0.0547 | 0.1144 [0.0277, 0.2420] | 0.01276 [0.00266, 0.02692] | 4.18 [1.07, 8.86] | 0.5833 | 271/306 |
| `random_at_rate` | 0.0315 [0.0304, 0.0325] | 0.0630 | 0.0359 [0.0198, 0.0564] | 0.00349 [0.00157, 0.00587] | 1.14 [0.63, 1.80] | 0.0278 | 295/306 |

| paired recall delta | Δ | CI95 | separated? |
|---|---|---|---|
| head_img_ego - heur_ego_both | +0.0229 | [-0.1106, +0.1455] | no |
| head_img_ego - heur_ego_both_rate_matched | +0.0229 | [-0.1106, +0.1455] | no |
| head_img_ego - random_at_rate | +0.1013 | [-0.0158, +0.2727] | no |
| head_img - heur_ego_both_rate_matched | -0.0490 | [-0.1591, +0.0248] | no |
| head_img - random_at_rate | +0.0294 | [-0.0417, +0.1329] | no |
| head_ego - heur_ego_both_rate_matched | +0.0556 | [-0.0230, +0.1441] | no |
| head_ego - random_at_rate | +0.1340 | [-0.0110, +0.3247] | no |

### The efficiency trade-off CURVE (not a point)

| B (extra cams/frame) | head realised rate | **head recall** | head precision | head recall (behavioural) | heuristic recall | random recall | saving vs always-on-3 | saving vs always-on-7 |
|---|---|---|---|---|---|---|---|---|
| 0.005 | 0.0083 | **0.0392** | 0.02885 | 0.0000 | 0.0000 | 0.0027 | 0.664 | 0.856 |
| 0.010 | 0.0165 | **0.0686** | 0.02536 | 0.0833 | 0.0033 | 0.0054 | 0.661 | 0.855 |
| 0.020 | 0.0313 | **0.0752** | 0.01466 | 0.1389 | 0.0196 | 0.0103 | 0.656 | 0.853 |
| 0.050 | 0.0630 | **0.1373** | 0.01331 | 0.2778 | 0.1144 | 0.0247 | 0.645 | 0.848 |
| 0.100 | 0.1337 | **0.2026** | 0.00925 | 0.3611 | 0.1144 | 0.0500 | 0.622 | 0.838 |
| 0.200 | 0.2448 | **0.2843** | 0.00709 | 0.3611 | 0.3856 | 0.1001 | 0.585 | 0.822 |

### The efficiency ledger — where the saving actually comes from

| policy | extra cams/frame | cams/frame | saving vs always-on-3 | saving vs always-on-7 | recall of `L2_trigger` |
|---|---|---|---|---|---|
| **never escalate** (free and useless) | 0 | 1.000 | 66.7 % | 85.7 % | **0.000** |
| **L2 ORACLE** — fires exactly on the label | 0.0061 | 1.0061 | 66.5 % | 85.6 % | **1.000** |
| **`head_img_ego` @ B\*** | 0.0630 [0.0442, 0.0847] | 1.0630 | 64.5 % [63.8, 65.2] | 84.8 % [84.5, 85.1] | **0.1373** [0.0188, 0.3199] |
| **always escalate** | 2.000 | 3.000 | 0.0 % | 57.1 % | **1.000** |

> The saving is set by the POLICY SHAPE, not by classifier quality: between the oracle and our operating point the saving vs always-on-7 moves by well under a percentage point. **What the classifier has to earn is RECALL at that budget** — which is why recall, not saving, is the axis the verdict is decided on.

### Measured compute (A40, pod2)

- one frozen encoder+readout pass, batch 32: **3.244 ms** / camera-frame · analytic **23.40 GMAC**
- one head forward, batch 32: **18.9 us** / frame · analytic **16.9 MMAC**
- **head / encoder = 0.072 % of one camera pass (MACs), 0.58 % (wall-clock)** — the gate is nearly free, so the saving is set by the firing rate, not by the gate.

<!-- /TABLES:RESULTS -->

### 6.1 Reading the result

**a. Nothing separates from anything, and nothing clears chance.** Every pairwise contrast between
the learned arms and the non-learned ego rules has a CI spanning 0, and — on the *correct*
above-chance test (paired ΔAP against a constant score, §0 point 2) — **no arm is separated from
chance.** The two separations measured anywhere in the primary comparison are (i) `head_img_ego`
over **raw ego speed** and (ii) raw ego speed **below chance** — which are the same fact: ego speed
is *anti*-predictive here (AUROC **0.2965**: trigger frames are **slower**, the same confound E1
recorded). Beating it is not evidence of vision.

**b. The vision head is indistinguishable from a one-line deceleration rule.** `head_img_ego` −
`heur_decel` = **+0.00027 [−0.00633, +0.01462]**. The heuristic is *"rank frames by how hard the ego
is already decelerating"*. Two million parameters and a frozen 87 M encoder do not beat it.

**c. The best arm is the one with no camera.** `head_ego` (415 k params, inputs = `v`, `a_pre`) has
the highest AP of any arm on the held-out set **and** on the training-side CV (CV-AP **0.0198** vs
0.0097 / 0.0095). Adding 2048-d frozen features to two ego numbers *reduces* CV-AP by ~2×. The
pairwise delta is not separated, so this is a **direction, not a result** — but it is the *same*
direction in both places, and it is the direction that matters for the MoE design.

**d. Calibration transfers imperfectly, and that is reported rather than tuned away.** The threshold
fixed on TRAIN out-of-fold over-fires by **1.26×** on the held-out side (target camera-frame rate
0.0250, realised 0.0315). The baselines are matched to the *realised* rate so the comparison is
compute-matched anyway (amendment A3).

**⚠️ d-bis. The ego arms are probably exploiting the label's OWN known confound, and that limits what
"the signal is in the ego state" is allowed to mean.** `a_pre` is the **trailing** 0.5 s mean
acceleration — a rule keyed on it fires when the ego is *already decelerating*. The label work
MEASURED exactly that imbalance (`P(already braking | trigger⁺) = 35.6 %` vs `14.1 %`, and adjusting
for braking state collapses the label's own lift to 1.35× [0.82, 2.05]; INHERITED,
`l2_robustness.json`). So the ego arms' edge is at least partly a **reactive** proxy for a response
already under way — which is precisely what a *sensor-need* gate must not be, since the point of
escalating is to see something **before** reacting to it. **Read together: the vision head fails to
beat a reactive proxy, and the reactive proxy is not the gate we want.** Neither arm has yet
demonstrated anticipation.

**e. The trade-off curve is the answer to "how much does it cost in missed events".** Sweeping the
budget from 0.005 to 0.20 extra cameras/frame moves head recall **0.039 → 0.284** while the saving
vs always-on-7 falls **85.6 % → 82.2 %**. **At the widest budget the ego heuristic overtakes the
vision head (0.386 vs 0.284).** There is no budget at which the vision head recovers a majority of
the events.

---

# 7. Sensitivities and the C12 decomposition

<!-- TABLES:SENS -->

### C12 — the LABEL's own structure, before any model

| quantity | value |
|---|---|
| `T_off` (`a_req_off ≥ τ*`) rate | **0.634 %** (318 frames) |
| `T_seen` (`a_req_seen < τ*`) rate | **96.724 %** (48477 frames) |
| composite `L2_trigger` rate (frame level) | 0.611 % (306 frames) |
| **P(trigger \| `T_off`)** | **0.9623** |
| P(trigger \| `T_seen`) | 0.00631 |

### C12 — the composite decomposed into its two conjuncts

| conjunct | base rate | AP (CI95) | AP / base | AUROC |
|---|---|---|---|---|
| `T_off` | 0.0063 | 0.00846 [0.00404, 0.02125] | **1.33x** | 0.5818 |
| `T_seen` ⛔ **NOT READ — mis-posed instrument, amendment A1** | 0.9672 | 0.97673 [0.96868, 0.98368] | **1.01x** | 0.5748 |
| `NOT_T_seen` read off the MIS-POSED head ⛔ **NOT READ** | 0.0328 | 0.03779 [0.02781, 0.04958] | **1.15x** | 0.4252 |

> ⛔ The `T_seen` row and its complement are printed for completeness and **are not read**: `T_seen` is a 96.7 %-positive target and the pre-registered BCE + `pos_weight` recipe up-weights its MAJORITY class, so that head carries no information about the rare side. The corrected diagnostic is the next table.

### C12 — the CORRECTED conjunct diagnostic (`NOT_T_seen`), amendment A1

*Target: NOT_T_seen = (a_req_seen >= tau*) — an agent INSIDE the encoder crop requires braking >= 0.5 m/s^2. **1642 positives** in 50119 frames (3.28 %), **101 of 322 clips positive** — 5.4x the composite's positives and far better powered.*

| arm | AP (CI95) | AP / base | ΔAP vs chance | CI95 | above chance? | AUROC |
|---|---|---|---|---|---|---|
| `head_img_ego` | 0.05205 [0.03679, 0.07550] | **1.59x** | +0.00601 | [-0.00040, +0.03947] | no | 0.6544 |
| `head_img` | 0.04914 [0.03452, 0.07926] | **1.50x** | +0.00310 | [-0.00291, +0.04284] | no | 0.6349 |
| `head_ego` | 0.12263 [0.08052, 0.17431] | **3.74x** | +0.07659 | [+0.05055, +0.13529] | **YES** | 0.7776 |

### Sensitivities

| stratum | n positives | base rate | AP (CI95) | AP / base |
|---|---|---|---|---|
| `encoder_unseen_clips` | 34 | 0.00162 | 0.00169 [0.00044, 0.00443] | **1.04x** |
| `junction_in` | 73 | 0.00864 | 0.01364 [0.00215, 0.03535] | **1.58x** |
| `junction_out` | 233 | 0.00254 | 0.00784 [0.00210, 0.03834] | **3.09x** |
| `residual_scope_target` | 102 | 0.00102 | 0.00183 [0.00068, 0.00366] | **1.80x** |

<!-- /TABLES:SENS -->

### 7.1 Reading the sensitivities

- **⭐ `NOT_T_seen` (amendment A1) is the best-powered probe in this study, and it is the only place
  anything separates.** **1,642** held-out positives across **101** positive clips, against the
  composite's 306 across 35, and **836** training positives against 169. It asks the strictly
  easier, strictly more visual question: *is there an agent the encoder CAN see that requires
  braking?* **`head_ego` clears chance by +0.0766 [+0.0506, +0.1353]; neither image arm clears it at
  all**, and adding image features to the ego head takes it from 3.74× base to 1.59× and from
  separated to not-separated. Training-side CV agrees (CV-AP `head_ego` **0.0967** vs `head_img`
  0.0327 vs `head_img_ego` 0.0296, TRAIN base 0.0269).
  ⚠️ **Read the ego arm's success carefully** — it is the *same reactive channel* as §6.1 d-bis:
  when an agent ahead requires braking, the ego is usually already braking. An anticipation claim
  cannot be built on it, but a **positive control** can, and that is what it serves as here.
- **Junctions:** AP 1.58× base inside vs 3.09× outside — the same direction as the label's own
  junction null (0.45× [0.00, 1.40], INHERITED). H2's headline situation remains the weakest one.
- **Encoder-unseen clips:** 1.04× base on 34 positives — no signal, but far too few positives to
  distinguish "leak-free and useless" from "underpowered". Reported, not interpreted.
- **Residual scope** (the genuine off-front 36.4 %, where a second camera is the *only* remedy):
  1.80× base, descriptive only. The label itself is not separated there (1.66× [0.78, 3.06],
  INHERITED), so nothing here may be headlined as cross-camera.

---

# 7b. Training-side cross-validation — reported because it is where the arms first separate

CV-AP is **out-of-fold on TRAIN**, grouped by chunk, and it is the only quantity model selection was
allowed to read. It is **not a result** (C1: only held-out eval output is quotable) — it is reported
because the *ordering* of the arms is already visible here, before the held-out side was touched,
which is the strongest evidence that §6's ordering is not a held-out fluke.

<!-- TABLES:CV -->

### Training-side cross-validation (out-of-fold on TRAIN, grouped by chunk)

*TRAIN: 31,032 windows · CV base rate for reference is the mean of the two per-camera base rates.*

| arm \| target | selected config | selected epoch | **CV-AP** | full CV-AP-vs-epoch curve (first 6) | params |
|---|---|---|---|---|---|
| `head_img_ego|trigger` | pos_weight 20, d 256 | 1 | **0.0097** | 0.0097, 0.0040, 0.0035, 0.0030, 0.0037, 0.0037 … | 2,173,699 |
| `head_img|trigger` | pos_weight 20, d 256 | 1 | **0.0095** | 0.0095, 0.0047, 0.0031, 0.0033, 0.0030, 0.0030 … | 2,173,187 |
| `head_ego|trigger` | pos_weight 20, d 128 | 14 | **0.0198** | 0.0056, 0.0058, 0.0061, 0.0097, 0.0056, 0.0056 … | 415,107 |
| `head_img_ego|T_off` | pos_weight 20, d 128 | 2 | **0.0078** | 0.0058, 0.0078, 0.0056, 0.0068, 0.0063, 0.0055 … | 677,122 |
| `head_img_ego|T_seen` | pos_weight 100, d 128 | 8 | **0.9807** | 0.9777, 0.9731, 0.9723, 0.9737, 0.9754, 0.9736 … | 677,122 |
| `head_img_ego|NOT_T_seen` | pos_weight 100, d 256 | 1 | **0.0296** | 0.0296, 0.0281, 0.0246, 0.0271, 0.0249, 0.0235 … | 2,173,442 |
| `head_img|NOT_T_seen` | pos_weight 100, d 128 | 2 | **0.0327** | 0.0280, 0.0327, 0.0308, 0.0251, 0.0244, 0.0224 … | 676,866 |
| `head_ego|NOT_T_seen` | pos_weight 100, d 256 | 18 | **0.0967** | 0.0548, 0.0523, 0.0604, 0.0629, 0.0623, 0.0570 … | 1,649,154 |

| fold | chunks |
|---|---|
| 0 | 0036, 0928 |
| 1 | 0170, 1852 |
| 2 | 0174, 1870 |
| 3 | 0834, 2433 |
| 4 | 0868, 2503 |

<!-- /TABLES:CV -->

---

# 8. Limitations, stated plainly

1. ⚠️ **Power is the binding limitation, and it was measured before the result (§4).** 306 held-out
   positives across **35** positive clips; the label's *own* effect is not separated on this exact
   subset (2.171× [0.645, 4.469]). Any non-separation here is **UNPOWERED, not refuted**. ⛔ And it
   is not merely that the arms fail to separate *from each other*: **no arm separates from CHANCE**
   on the correct paired test. At this n the study cannot distinguish "a weak classifier" from "no
   classifier".
2. ⚠️ **One pre-registered instrument was mis-posed and is not read** (amendment A1): `T_seen` is a
   96.7 %-positive target trained with a rare-positive recipe. Found by reading my own result. Its
   corrected form (`NOT_T_seen`) is in §7 and it is a *diagnostic*, not an arm.
3. ⚠️ **169 training positives against a 2.17 M-parameter head.** The CV selects extremely early
   epochs (1–2 for the image arms), which is what an overfitting regime looks like; it is reported,
   not hidden. The final model is retrained on all of TRAIN for exactly that many epochs, so the
   primary arm's held-out scores come from a **single epoch** of optimisation — fragile by
   construction, and a direct consequence of the CV rule rather than a choice made after the fact.
4. ⚠️ **The frozen encoder saw 283 of the 322 held-out clips during its own training** (§2.3). The
   encoder-unseen sensitivity (§7) is the check, and it is itself small (76 clips before the
   alignment floor, 34 positives after).
5. ⚠️ **7.39 % of clips are genuine alignment failures** (§3) — the two clocks cannot always be
   reconciled from a speed series alone. 12 clips have almost no overlap between the label window
   and the episode at any lag.
6. **The universe is 582 of the label's 2,320 clips** — a ~52 GB gated camera re-download plus a
   full re-decode is what stands between this run and a 4× larger one. That is the single highest-
   value follow-up and it needs no new science.
7. **`obstacle.offline` is machine-labelled** (`prov: "autolabel"`); systematic misses of small or
   distant agents attenuate everything here.
8. **The compute model is a per-camera encoder-pass model.** It does not include image signal
   processing, the demosaic/ISP path, memory bandwidth for a second sensor stream, or the wake-up
   latency of a camera that is not already streaming — all of which make real-world always-on more
   expensive than modelled, i.e. the saving reported here is **conservative** on that axis and
   **optimistic** on the axis of ignoring activation latency. The head's own cost is reported two
   ways (0.072 % of an encoder pass by MACs, 0.58 % by A40 wall-clock); the gap is launch overhead
   on a tiny model, and the **MAC** ratio is the one used in the saving arithmetic.
9. **Wall-clock is A40 wall-clock.** It is not an Orin/Thor number and must not be quoted as one.
10. **AP ties are broken by row order**, which is mildly optimistic for the heavily-tied non-learned
    ego baselines — i.e. conservative for the primary arm, the safe direction.
11. **This is the front-periphery scope, not the cross-camera residual.** The label itself is not
   separated on the genuine off-front residual (1.66× [0.78, 3.06], INHERITED). Nothing here may be
   headlined as *"we learned when to switch on the side cameras."*

---

# 9. Recommendations — and two escalations raised here, not buried in a README

**1. 🔴 ESCALATION — authorise the corpus expansion. It is the only thing standing between H2 and a
decidable answer.** The label covers **2,320** clips; this run could use **582**, because only clips
already decoded into the pod2 episode cache have front-camera frames. The gap is a **~52 GB gated
camera re-download (26 chunk zips) + a re-decode**, not a scientific problem. Everything in this
folder — label, join, alignment, features, head, estimators, report — runs unchanged on the larger
set. **ESTIMATED** cost: ~30 min of HF pull at pod HF throughput, ~1.5 h of decode, ~30 GPU-min of
re-encode, and the head re-trains in **43 min** (MEASURED here). Expected effect: **~4× the
positives** (≈1,200 held-out trigger positives, ≈140 positive clusters — above the label work's own
n ≥ 40 bar and comparable to the 138 clusters the GO rested on). The program has MEASURED CI
half-widths shrinking **×2.8–3.9** going 40 → 600 episodes; that is the difference between
`UNDERPOWERED` and a verdict.

**2. 🔴 ESCALATION — the efficiency framing in `H2_PHASE1_PLAN.md` / `C-EFF` needs correcting in
place.** *"Selective activation saves 85.6 % vs always-on-7"* is true, MEASURED, and **carries no
information about the gate**, because *never activating* saves 85.7 % and a perfect oracle saves
85.6 %. Any future H2 artifact that headlines a compute saving without the **recall at that budget**
beside it is quoting a number that cannot fail. This is the same class as C13 (a guard that cannot
fire), applied to a benefit claim instead of a safety claim.

**3. Do NOT ship the kinematic rule as "the H2 gate" — but do use it as the bar the vision head must
clear.** The direction is consistent across the training CV and the held-out set (ego-only is the
best arm; the vision head is indistinguishable from a one-line deceleration rule), **but the ego
rule keys on the trailing 0.5 s acceleration, i.e. on braking already under way** (§6.1 d-bis), and
the label's own robustness table shows that adjusting for braking state collapses the effect to
1.35× [0.82, 2.05]. A gate that fires *after* the ego starts braking is reactive and defeats the
purpose of escalating a sensor. **The correct reading is not "ship the kinematic gate" — it is
"the vision head has not yet cleared a reactive proxy, and neither arm has demonstrated
anticipation."** Any future arm should be scored against `heur_decel` *and* in a braking-state-
stratified form.

**4. Do NOT re-sweep τ\*, do NOT add a post-hoc arm, and do NOT re-read the operating point.** The
pre-registration forbids all three, and `L1`'s death was exactly this error one level up. The
`NOT_T_seen` diagnostic in §7 is **not** an exception: it repairs a mis-posed *diagnostic*, it is
labelled as such, and it takes no part in the primary comparison.

**5. ⭐ The next pre-registration should test the REPRESENTATION, not the head — and `NOT_T_seen` is
the target, because it is the only place in this study where anything separated.** MEASURED here:
an ego-only head clears chance on it (+0.0766 [+0.0506, +0.1353]) while neither image arm does, and
adding image features to the working ego head **destroys** it (3.74× → 1.59×). Those two facts
together point at capacity/optimisation, not necessarily at missing information. The honest ladder,
in cost order, all on `NOT_T_seen` and all cheap:
   1. **a linear (ridge/logistic) probe on the frozen 2048-d state** — the lowest-variance reader;
      if *this* fails, no head rescues the representation;
   2. **ego + a low-rank image projection** (e.g. PCA-64) rather than raw concatenation — tests the
      swamping hypothesis directly;
   3. **a fine-tuned trunk** — only if (1) or (2) shows the information is there.
   Each is ~30 GPU-min on the features already on pod2. **Do (1) before spending anything else on
   H2.**

**6. Re-run this exact study after the expansion in (1) and change nothing else.** Every script here
is parameterised by the feature directory; the pre-registration, the split rule, τ\*, the operating-
point rule and the estimator are all fixed. A re-run is a **re-measurement, not a new design** —
which is what makes the expansion worth authorising.

---

# 9b. AMENDMENTS — recorded here, not by editing the pre-registration

| # | what changed | why, and what it can and cannot bias |
|---|---|---|
| **A1** | The `T_seen` conjunct head is **not read**, and a corrected `NOT_T_seen` diagnostic was run in its place (§7) | `T_seen` is a **96.7 %-positive** target and the pre-registered recipe (BCE + `pos_weight`) up-weights the *majority* class on it. Reading its complement would have been an over-read of a mis-posed instrument. `NOT_T_seen` poses the identical question as a rare-positive target at ~5× the power. **It is a diagnostic, not an arm**, and it takes no part in the primary comparison or the verdict — which was computed and frozen before it existed. |
| **A2** | The alignment guard's **10 % BLOCK branch was not taken**, on measured grounds (§3) | The guard's own statistic is undefined on constant-speed clips. A second probe puts the *genuine* failure rate at **7.39 %**, inside the bar, and the dropped clips carry **0.42 %** of positives. The admitted set was **not widened** — the strict rule's output is what the primary uses. |
| **A3** | The baselines are matched to the head's **realised** held-out firing rate as well as to the pre-registered budget | Matching a firing rate reads the held-out **score distribution**, never the held-out **targets**, so no test metric enters a selection path. Both forms are reported and they agree. |

---

# 10. Deliverable manifest

**Everything is in the repo working tree and STAGED (`git add`). Nothing was committed or pushed.**
Path: `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-26-h2-classifier/`

| artifact | what it is | where it lives |
|---|---|---|
| `PRE_REGISTRATION.md` | written before any feature existed; split, operating-point rule, success bar, both outcomes | **repo** |
| `H2_CLASSIFIER.md` | this document | **repo** |
| `artifacts/h2c_results.json` | **the held-out result** — discrimination, operating point, trade-off curve, C12 2×2, sensitivities, efficiency, every interval with its estimator | **repo** |
| `artifacts/universe.json` | per-chunk clips / frames / positives, per side, encoder-seen breakdown | **repo** |
| `artifacts/join_proof.json` | the 2376/2376 + 600/600 episode↔clip join proof | **repo** |
| `artifacts/align_summary.json` | per-clip lag, correlation, admission | **repo** |
| `artifacts/align_admission.json` | the guard-trip diagnostic (degeneracy vs genuine failure) | **repo** |
| `artifacts/align_second_probe.json` | the RMSE second probe on all 62 dropped clips | **repo** |
| `artifacts/subset_lift.json` | the label's own lift on the exact evaluation subset — the power ceiling | **repo** |
| `artifacts/train_summary.json` | CV grid, selected config/epoch per arm, fold composition, per-epoch train loss | **repo** |
| `artifacts/train_summary_c12fix.json` | the same for the corrected `NOT_T_seen` diagnostic | **repo** |
| `artifacts/c12_fix.json` | ⭐ the corrected C12 diagnostic — the only separated result in the study | **repo** |
| `artifacts/_tables.md` | the rendered tables, exactly as spliced into §6–§7b | **repo** |
| `artifacts/cost_model.json` | MEASURED A40 wall-clock + analytic MACs for encoder and head | **repo** |
| `checkpoints/head_*.pt` | **the 8 trained heads** — `{head_img_ego, head_img, head_ego} × trigger`, `head_img_ego × {T_off, T_seen}`, `{head_img_ego, head_img, head_ego} × NOT_T_seen` — each with its config, feature normalisation and target names | **repo** |
| `scripts/h2c_prep.py` | join + de-identified label bundle (dev box) | **repo** |
| `scripts/h2c_features.py` | alignment + frozen-encoder feature extraction (pod2) | **repo** |
| `scripts/h2c_align2.py` | the alignment second probe (pod2) | **repo** |
| `scripts/h2c_aligncheck.py` | the guard-trip adjudication (dev box) | **repo** |
| `scripts/h2c_train.py` | the head, the CV, all arms — `--plan primary` / `--plan c12fix` (pod2) | **repo** |
| `scripts/h2c_cost.py` | the compute measurement (pod2) | **repo** |
| `scripts/h2c_stats.py` | the estimators — paired episode-cluster bootstrap over an arbitrary reducer | **repo** |
| `scripts/h2c_eval.py` | the held-out evaluation (dev box, uses the repo's own `taniteval/ci.py`) | **repo** |
| `scripts/h2c_c12fix.py` | the corrected C12 diagnostic's evaluation (dev box) | **repo** |
| `scripts/h2c_subset_lift.py` | the power-ceiling check | **repo** |
| `scripts/h2c_report.py` | renders every table in §6–§7b from the JSON — no number is hand-typed | **repo** |
| `scripts/h2c_splice.py` | places the rendered tables into this document at its markers | **repo** |

**Not in the repo, and deliberately so:**

| | where | why |
|---|---|---|
| per-clip frozen features (`pod2:/workspace/h2clf/feats`, ~520 MB) | **pod2 only** | derived from a gated corpus, and rebuilt in **9.6 GPU-min** by `h2c_features.py` |
| the de-identified label bundle (`pod2:/workspace/h2clf/bundle`, 4 MB) | **pod2 + dev-box scratch** | rebuilt by `h2c_prep.py` in ~1 CPU-min |
| the clip-UUID map (`_LOCAL_ONLY_k2clip.json`) | **dev box only, outside the repo** | 🔒 gated corpus — UUIDs may never enter a derived artifact |
| the L2 label table (26 chunk parquets, ~24 MB) | dev-box scratch | rebuilt by the label stream's `l2_build.py` in ~5.5 CPU-min |

**Reproduction, end to end**

```
# dev box
python scripts/h2c_prep.py <l2tab> <ep_ids.json> <r0_selection.parquet> <bundle>
# pod2  (PYTHONPATH=/workspace/TanitAD/stack)
python3 scripts/h2c_features.py --bundle <bundle> --out <feats> \
        --ckpt /workspace/experiments/flagship4b-speedjerk-30k/ckpt.pt
python3 scripts/h2c_train.py --feats <feats> --bundle <bundle> --out <run> --epochs 30
python3 scripts/h2c_train.py --feats <feats> --bundle <bundle> --out <run_c12fix> \
        --epochs 30 --plan c12fix
python3 scripts/h2c_cost.py  --ckpt <same ckpt> --out <run>
# dev box
python scripts/h2c_subset_lift.py <l2tab> <bundle> artifacts/align_summary.json \
       artifacts/subset_lift.json
python scripts/h2c_aligncheck.py artifacts/align_summary.json <l2tab> <bundle> \
       artifacts/align_admission.json
python scripts/h2c_eval.py    --run <run>        --out artifacts --cost artifacts/cost_model.json
python scripts/h2c_c12fix.py  --run <run_c12fix> --out artifacts
python scripts/h2c_report.py  artifacts artifacts/_tables.md
python scripts/h2c_splice.py  artifacts H2_CLASSIFIER.md
```

**Timings, MEASURED on this run:** feature extraction **578.7 s** (520 clips, 1.11 s/clip, A40) ·
primary training **2,609 s** (5 models × 4-config × 5-fold CV + final) · corrected diagnostic
**1,623 s** · compute benchmark ~60 s · held-out evaluation ~10 CPU-min (B = 2000 over 322 clusters).
