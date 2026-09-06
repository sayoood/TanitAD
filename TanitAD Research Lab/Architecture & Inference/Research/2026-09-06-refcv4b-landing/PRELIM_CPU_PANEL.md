# PRELIMINARY — the zero-GPU ablation panel on `ckpt_30000.pt`

⛔⛔ **THIS IS NOT THE LANDING RESULT AND MUST NOT BE QUOTED AS ONE.** It is
`ckpt_30000.pt` (not the final 40,284), on **20 clips at `--window-stride 20` = 171 windows /
20 episodes**, run **CPU-only on the training pod while the run was still training**. The landing
read is **141 clips at stride 5 = 4,823 windows**, the grid refcv3 @40,284 was scored on. ⇒ levels
here are **not comparable** to any refcv3 number; only the SHAPES are informative, and each is
labelled with what would change it.

**Why it exists at all:** the PI has queued refcv5 for this A40. If the GPU is taken before the
landing panel finishes, this is what survives — and it cost no GPU.

**Artifacts:** `pod:/workspace/eval/ab20/{BASE,h19_off,sel_refined,ego_zero,frames_blind}.json`
and `d_*/` dumps; `raw/paired_ablate.py`; `raw/paired_h19.json`.

---

## 1. The arms at n = 171 windows / 20 episodes (episode-cluster bootstrap, `ade_dense_m`)

| arm | ADE (m) | CI95 |
|---|---|---|
| `os` — the model | **0.3055** | [0.2441, 0.3883] |
| `ha` — hold-action | **0.2860** | [0.2417, 0.3476] |
| **`ha0_ext` — the echo control** | **0.2769** | [0.2366, 0.3334] |
| `ha0` — constant velocity | 0.6542 | [0.4854, 0.8544] |

⛔ **Both trivial controls still beat the model at step 30,000 on this subset.** That is the same
defect shape refcv3 carries (`os − ha` = **+0.1423** [+0.1187, +0.1658] at 40,284, separated the
wrong way).

⭐ **But the margin is much smaller here: `os − ha` = +0.0195 m against refcv3's +0.1423 m — 7.3×
smaller.** ⚠️ **DIFFERENT SURFACE, DIFFERENT CHECKPOINT — this is a shape, not a comparison.**
Whether it survives on the 4,823-window grid at step 40,284 is **the single question the landing
eval answers**, and it is the one I will report first.

⚠️ `ha0` is **2.36× worse** than `ha0_ext` here, which is why an `ha0`-only bar certifies arms the
trivial control already beats. And `ha` (0.2860) and `ha0_ext` (0.2769) are **close but distinct**
on this surface — so the two halves of the acceptance bar are not degenerate at scale, even though
they read bit-identically on a 3-window probe.

---

## 2. `h19_off` — properly estimated, and the answer is INCONCLUSIVE, which is the honest third branch

Both runs share the **same window grid** — asserted, not assumed: `ws` and episode ids are compared
element-wise and the script REFUSES if they differ.

| | |
|---|---|
| BASE `os` | 0.3055 m |
| `h19_off` `os` | **0.3021 m** |
| paired delta | **0.0034 m, CI [−0.0040, +0.0111], `separated: false`**, `p_delta_gt0` 0.798 |
| estimator | `paired_episode_cluster_bootstrap` (`taniteval/ci.py`), n_boot 2000, seed 0 |
| windows whose selection flips | **10/171 = 5.85 %** |
| windows whose ADE changes at all | 10/171 = 5.85 % |
| ⛔ CONTROL, A paired against ITSELF | delta **0.0000000000**, lo 0.0000000000, hi 0.0000000000, `separated: false` — the estimator exercised end to end on a known null |
| CONTROL, base-vs-base selection flips | **0** |

**Verdict at this n: INCONCLUSIVE** — the pre-registered third branch, reported as such and not as
a null. The direction is very slightly toward *"removing the prior helps"* (0.3055 → 0.3021) and
the interval straddles zero.

⭐ **AND THE EFFECT SIZE IS THE FINDING, WHICHEVER WAY THE SIGN LANDS.** H19's anchor prior changes
**5.85 %** of picks and moves ADE by **0.0034 m** on an arm whose deficit to `ha` is 0.0195 m and
whose ADE is 0.3055 m. ⇒ **"repair the longitudinal kin3 head so H19's prior works better" is a
LOW-value refcv5 lever regardless of which way the landing read resolves it** — it is a
sub-1 %-of-ADE edge. Ranking levers by measured effect size, not by how interesting the mechanism
is, sends refcv5 elsewhere.

⚠️ It is NOT inert, and that was checked BEFORE this was written, because *a lever multiplied by a
zero coefficient is not a null about the lever*: `core.decoder.lat_to_anchor.weight` absmean
**2.802e-01**, `core.decoder.lon_to_anchor.weight` **7.047e-02**, neither all-zero, against
non-zero controls (`conf_head.weight` 1.730e-02, `goal_gate` 6.424e-02).

⭐ **An independent corroboration of the curve finding fell out of those weights:** the longitudinal
prior's matrix is **4.0× smaller in absolute mean** than the lateral one. The network itself
down-weighted the head whose cross-entropy cannot beat its own class marginal (eval CE 1.0090 nats
vs a prior-predictor floor of 0.8964).

---

## 3. What still has to come from the landing read

* the same panel at **4,823 windows / 141 episodes** on `ckpt_40284_FINAL.pt` — 28× the windows and
  7× the episodes, i.e. the power this read does not have;
* `sel_refined`, `ego_zero`, `frames_blind` at that n (running here on CPU, ~18 min/arm);
* the four families per arm, never pooled, each with its own paired interval;
* **turn recall per class beside ADE** — a ranking change that improves ADE by picking straighter
  paths is a regression wearing a win;
* the eval-set kin3 marginal, which converts §2's floor from a TRAIN EMA to a measured one.

---

## 4. ⭐⭐ `sel_refined` — SEPARATED, and it is WORSE. The pre-registered SUPPORTED branch is REFUTED.

This was the lever I expected most from: **zero parameters**, and the DiffusionDrive audit had
already measured that refcv3's ranking reads the **t = 0** confidence, *unchanged on 201/201
windows for every `steps` value* — naming the selection surface as "the lever the implementation
audit named". Turning it on:

| | |
|---|---|
| BASE `os` | 0.3055 m |
| `sel_refined` `os` | **0.3314 m** |
| paired delta | **−0.0259 m, CI [−0.0505, −0.0033], `separated: TRUE`**, `p_delta_gt0` 0.017 |
| estimator | `paired_episode_cluster_bootstrap`, n_boot 2000, seed 0, same asserted grid |
| selection flips | **51/171 = 29.82 %** — ⛔ **not a no-op**; it bites on nearly a third of windows |
| distinct anchors used | 18 → **16** (the ranking COLLAPSES the fan, it does not spread it) |
| ⛔ CONTROL A-vs-A | 0.0000000000 / 0 / 0 / `separated: false` |

### 4.1 And the per-class read says the same thing, which is what makes it unambiguous

My own pre-registration required turn recall beside ADE, *"because a ranking change that improves
ADE by picking straighter paths is a regression wearing a win."* Here ADE got **worse**, so the
inverse question had to be asked — and the answer is that it got worse **while also getting more
timid**:

| | BASE | `sel_refined` |
|---|---|---|
| LATERAL decision acc / κ | 0.9591 / 0.7039 | **0.9591 / 0.7039 — IDENTICAL** |
| `turn_left` recall (n_true 4) | 1.0000 | 1.0000 |
| `turn_right` recall (n_true 8) | 0.6250 | 0.6250 |
| LONGITUDINAL decision acc / κ | 0.8363 / 0.5782 | **0.8538** / 0.5899 |
| `steady` recall (n_true 130) | 0.8923 | **0.9385** |
| `brake_stop` recall (n_true 14) | **0.6429** | **0.5714** |
| `accelerate` recall (n_true 27) | **0.6667** | **0.5926** |
| `steady` n_pred | 130 | **139** |

⇒ **51 selection flips changed NO lateral manoeuvre decision at all** — they are within-manoeuvre
reshuffles — and they pushed the longitudinal decision toward the **majority class**.

⛔⛔ **AND THIS IS A TEXTBOOK CASE FOR THE PER-CLASS RULE: longitudinal ACCURACY went UP
(0.8363 → 0.8538) while BOTH minority recalls went DOWN.** `steady` is 130 of 171 windows (76 %),
so predicting it more often buys accuracy and loses the two decisions that matter. An accuracy-only
report would have called this an improvement. It is a regression on the family that owns
**88.7 %** of the oracle gap, and ADE agrees.

### 4.2 The mechanism, and the refcv5 consequence — which is the opposite of "flip the flag"

The ablation registry states it: `sel_refined` is `seen_in_training: **no**`. The refined-confidence
head was **never trained to rank**; `refc.py:1630` gates `score_emitted` on `steps > 0` and the
training objective never scores the refined estimate. So switching the ranking to it at eval time
is an **out-of-distribution use of an untrained head** — and that is exactly what a separated
regression looks like.

⇒ ⭐ **You cannot buy DiffusionDrive's selection mechanism by flipping a flag.** The audit is right
that refcv3/v4 ship the skeleton without the mechanism, but the missing part is **training the
selector**, not wiring it. **This is the measurement that motivates refcv5 WP-7 (`E-DDA-2b`
selector training) and de-motivates the zero-cost shortcut** — and it is worth having *before* the
GPU is spent, because the shortcut is the thing a hurried reading of the audit would try first.

⚠️ n = 171 / 20, `ckpt_30000.pt`, a 20-clip subset. The landing panel re-runs it at 4,823 / 141.
This is an **eval-time** intervention on **one** checkpoint, so `H-ESTIM-SEED-1`'s training-run
variance does not enter; the admissible claim form is *"this switch moves this metric on this
checkpoint"*, never *"this lever is worth X in refcv5"*.

## 5. `h19_off`, the per-class read

Lateral κ **0.7039 → 0.6738** and lateral accuracy 0.9591 → 0.9532 — i.e. removing the anchor prior
costs a little lateral decision quality while ADE is unchanged (§2). Longitudinal is essentially
flat (κ 0.5782 → 0.5671). ⇒ another reason the lever is low value: it is not free to remove, and it
is not worth repairing.

---

## 6. ⭐⭐ refcv4b @30,000 vs refcv3 @40,284 ON ONE SURFACE — and the win is LONGITUDINAL

Both models rolled through the **same adapter, same 20 clips, same stride, same grid, same labels,
same action units**, and the pairing is asserted element-wise (`ws` + episode ids) before anything
is computed. refcv3's checkpoint is `ckpt_step40284_frozen.pt`, md5 **`b1ed7075ff730d0993d2eaa3c86f6b56`**
— verified against `MODEL_REGISTRY.md` §4.5 rather than trusted by filename.

⭐ **THE INTERNAL CONTROL THAT MAKES THE COMPARISON VALID:** the model-free arms read
**bit-identically** across both runs — `ha` 0.2860, `ha0` 0.6542, `ha0_ext` 0.2769 — so the two
models are being scored on **one surface**, not two.

| arm | refcv3 @40,284 | refcv4b @30,000 |
|---|---|---|
| `os` ADE (m) | **0.4707** | **0.3055** |
| paired `refcv4b − refcv3` | **−0.1652 m, CI [−0.2318, −0.1090], `separated: TRUE`**, p 1.0, 171/171 windows differ | |
| relative | **−35.1 %** | |
| ⛔ CONTROL A-vs-A | 0.0000000000 / 0 / 0 / `separated false` | |

⇒ **refcv4b beats refcv3 by 0.1652 m, separated — while 10,284 steps LESS trained.** ⚠️ n = 171 / 20
and both are single-seed arms of **different training runs**, so `H-ESTIM-SEED-1` applies in full:
this interval answers *"would another draw of EPISODES say this?"*, never *"would another TRAINING
RUN say this?"*. And five levers moved at once — this is an **ARM delta, never a lever attribution**.

### 6.1 The gain is concentrated in the family that owns 88.7 % of the oracle gap

| tactical decision | refcv3 @40,284 | refcv4b @30,000 |
|---|---|---|
| LATERAL acc / **κ** | 0.9649 / **0.7554** | 0.9591 / **0.7039** |
| `turn_left` recall (n_true 4) | 1.0000 | 1.0000 |
| `turn_right` recall (n_true 8) | **0.7500** | 0.6250 |
| LONGITUDINAL acc / **κ** | 0.6842 / **0.2037** | **0.8363 / 0.5782** |
| `brake_stop` recall (n_true 14) | 0.3571 | **0.6429** |
| `accelerate` recall (n_true 27) | 0.3704 | **0.6667** |

⇒ refcv4b is **2.84× better on longitudinal decision κ** (0.2037 → 0.5782), nearly doubles both
minority recalls, and is **slightly WORSE laterally** (κ 0.7554 → 0.7039, `turn_right` 0.75 → 0.625).
⚠️ Those turn recalls rest on **n_true 4 and 8** — far too few to claim a lateral regression; the
landing read has 141 episodes.

### 6.2 ⛔ And the mechanism is confirmed by REMOVING it — `ego_zero`

| | |
|---|---|
| refcv4b `os` | 0.3055 m |
| `--ablate ego_zero` (keep = 0 **and** `v0 = None` at the core) | **1.1137 m** |
| paired | **−0.8082 m, CI [−0.9985, −0.6239], `separated: TRUE`**, p 0.0 |
| LONGITUDINAL κ | 0.5782 → **0.0653** — collapsed to near chance |
| `brake_stop` recall | 0.6429 → **0.1429** |
| LATERAL κ | 0.7039 → 0.6738 — **largely intact** |

⇒ **Two independent measurements agree on one mechanism.** Adding the measured ego block moves the
LONGITUDINAL family (refcv3 → refcv4b, κ 0.2037 → 0.5782); withholding it collapses the LONGITUDINAL
family and leaves the lateral one standing (κ 0.5782 → 0.0653 vs 0.7039 → 0.6738). The lever and its
ablation point at the same axis, which is what makes this an attribution rather than a coincidence —
even though the arm as a whole moved five levers at once.

### 6.3 ⛔ THE HONEST BOUND ON THE WIN

1. **refcv4b still LOSES to the trivial controls:** `os` 0.3055 against `ha` 0.2860 and `ha0_ext`
   **0.2769**. It has closed most of refcv3's deficit (`os − ha`: refcv3 **+0.1847** here,
   refcv4b **+0.0195** — 9.5× smaller on this surface) but has **not crossed the bar**, at step
   30,000 on 20 clips.
2. **The win is not vision-only.** `ego_zero` is the conservative reading of the PI's binding rule
   (*"for inference only vision"*, tempered by the 2026-09-02 ruling that measured `v0` at t0 is a
   legal initial state). At **1.1137 m** the ego-withheld arm is **3.65× worse** than the kept arm
   and worse than `ha0` (0.6542). ⇒ **whatever refcv4b has learned longitudinally, it has learned
   to lean on the measured ego block to express it**, and `--ego-dropout 0.5` has not bought a
   vision-only arm by step 30,000.
3. `frames_blind` — the panel's VOID gate — was still running when this was written; **the panel is
   not admissible until it is read**.
