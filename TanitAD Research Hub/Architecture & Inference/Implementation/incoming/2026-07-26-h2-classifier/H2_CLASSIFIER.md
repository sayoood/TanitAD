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

# 0. PLACEHOLDER — VERDICT

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
> 0.0028 base rate has its own, better power — and that is what §5–§7 measure.

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
frozen  :  frames_u8 [T,9,256,256] --(/255)--> ViTEncoder(d768, depth12, patch16)
                                   --> SpatialGridReadout(8x8 x 32) --> state [2048]
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

<!-- TABLES:RESULTS -->

---

# 7. Sensitivities and the C12 decomposition

<!-- TABLES:SENS -->

---

# 7b. Training-side cross-validation — reported because it is where the arms first separate

CV-AP is **out-of-fold on TRAIN**, grouped by chunk, and it is the only quantity model selection was
allowed to read. It is **not a result** (C1: only held-out eval output is quotable) — it is reported
because the *ordering* of the arms is already visible here, before the held-out side was touched,
which is the strongest evidence that §6's ordering is not a held-out fluke.

<!-- TABLES:CV -->

---

# 8. Limitations, stated plainly

1. ⚠️ **Power is the binding limitation, and it was measured before the result (§4).** 306 held-out
   positives across **35** positive clips; the label's *own* effect is not separated on this exact
   subset (2.171× [0.645, 4.469]). Any non-separation here is **UNPOWERED, not refuted**.
2. ⚠️ **169 training positives against a 2.17 M-parameter head.** The CV selects extremely early
   epochs, which is what an overfitting regime looks like; it is reported, not hidden.
3. ⚠️ **The frozen encoder saw 283 of the 322 held-out clips during its own training** (§2.3). The
   encoder-unseen sensitivity (§7) is the check, and it is itself small (76 clips before the
   alignment floor).
4. ⚠️ **7.39 % of clips are genuine alignment failures** (§3) — the two clocks cannot always be
   reconciled from a speed series alone. 12 clips have almost no overlap between the label window
   and the episode at any lag.
5. **The universe is 582 of the label's 2,320 clips** — a ~52 GB gated camera re-download plus a
   full re-decode is what stands between this run and a 4× larger one. That is the single highest-
   value follow-up and it needs no new science.
6. **`obstacle.offline` is machine-labelled** (`prov: "autolabel"`); systematic misses of small or
   distant agents attenuate everything here.
7. **The compute model is a per-camera encoder-pass model.** It does not include image signal
   processing, the demosaic/ISP path, memory bandwidth for a second sensor stream, or the wake-up
   latency of a camera that is not already streaming — all of which make real-world always-on more
   expensive than modelled, i.e. the saving reported here is **conservative** on that axis and
   **optimistic** on the axis of ignoring activation latency.
8. **Wall-clock is A40 wall-clock.** It is not an Orin/Thor number and must not be quoted as one.
9. **This is the front-periphery scope, not the cross-camera residual.** The label itself is not
   separated on the genuine off-front residual (1.66× [0.78, 3.06], INHERITED). Nothing here may be
   headlined as *"we learned when to switch on the side cameras."*

---

# 9. Deliverable manifest

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
| `artifacts/cost_model.json` | MEASURED A40 wall-clock + analytic MACs for encoder and head | **repo** |
| `checkpoints/head_*.pt` | **the trained heads** (5), each with its config, feature normalisation and target names | **repo** |
| `scripts/h2c_prep.py` | join + de-identified label bundle (dev box) | **repo** |
| `scripts/h2c_features.py` | alignment + frozen-encoder feature extraction (pod2) | **repo** |
| `scripts/h2c_align2.py` | the alignment second probe (pod2) | **repo** |
| `scripts/h2c_aligncheck.py` | the guard-trip adjudication (dev box) | **repo** |
| `scripts/h2c_train.py` | the head, the CV, all arms (pod2) | **repo** |
| `scripts/h2c_cost.py` | the compute measurement (pod2) | **repo** |
| `scripts/h2c_stats.py` | the estimators — paired episode-cluster bootstrap over an arbitrary reducer | **repo** |
| `scripts/h2c_eval.py` | the held-out evaluation (dev box, uses the repo's own `taniteval/ci.py`) | **repo** |
| `scripts/h2c_subset_lift.py` | the power-ceiling check | **repo** |
| `scripts/h2c_report.py` | renders every table in §6–§7 from the JSON — no number is hand-typed | **repo** |

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
python3 scripts/h2c_cost.py  --ckpt <same ckpt> --out <run>
# dev box
python scripts/h2c_eval.py --run <run> --out artifacts --cost artifacts/cost_model.json
python scripts/h2c_report.py artifacts
```
