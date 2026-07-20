# REF-C v1.2 — a learned re-scorer over the refined fan

**Date:** 2026-07-20 · **Arm key:** `refc-v12` · **Host:** `tanitad-pod3` (train) / `tanitad-eval` (score)
**Status:** IN PROGRESS — this file is updated as results land. Numbers below marked ✅ are read from
raw eval JSON on `tanitad-eval`; anything else is 🟥 UNVERIFIED.

---

## 1. The question

REF-C-XL denoises all 256 anchors but SELECTS on the t=0 classifier score computed over the
**un-refined** anchors; the denoise passes' own confidences are discarded. The cost is a pure
**ranking** deficit — the good plan is already in the fan.

REF-C v1.0 (training-free cost re-rank, sibling agent) recovered **0.0 %** of that gap: the best point
on the entire confidence↔cost blend is λ=0, i.e. the unmodified baseline, and pure cost is −171 %.

**v1.2 asks the successor question: can a LEARNED ranker do what a hand-written one provably cannot?**

Everything except selection is frozen — the same 252 M encoder+decoder, the same anchors, the same
2-step deterministic denoise. The only trainable object is a **1.65 M-param re-scorer**.

---

## 2. Baseline and ceilings — reproduced independently ✅

Read from `tanitad-eval:/root/taniteval/results/refc-v12-identity.json`, produced by
`stack/scripts/refc_v12_eval.py` with a **zero-init** head, 881 windows, the harness's own
8-split episode-disjoint jackknife:

| Quantity | Value | Meaning |
|---|---|---|
| `full_set` ADE@2s | **0.47144** | the frozen REF-C selection — the number to beat (G1/G2) |
| `heldout` ADE@2s | **0.4577 ± 0.0572** | same run, jackknife statistic |
| `base_2x` | **0.45403** | 45 % of windows pick >2× worse than the fan's best |
| `oracle_ade` (all 256) | **0.16395** | GT-informed, UNREACHABLE, and partly a lottery |
| `oracle_k_ade` (top-8) | **0.20263** | the honest ceiling for a top-8 re-scorer |
| `vocab_ade` (raw anchors) | **0.59917** | the quantisation floor — the offsets do real work |
| ranking gap | **0.30749 m** | `base − oracle` |
| `rank_acc` | 0.31101 | how often the frozen score already picks the fan-oracle |
| along / cross of the selected | 0.41676 / 0.13103 | the residual is dominantly **longitudinal** |

### 2.1 The chance floor — what makes the 0.3075 m interpretable

An oracle is a **minimum over many candidates scored against ONE realised future**, so part of any
oracle gap is the statistics of taking a min, not headroom a ranker could reach. The no-skill floors:

| Selector | ADE@2s | |
|---|---|---|
| chance over all 256 | **13.956** | the fan's typical member |
| chance inside the top-8 | **1.034** | a no-skill ranker restricted to the reachable set |
| frozen confidence | 0.4714 | |
| oracle inside the top-8 | 0.2026 | |

**The frozen confidence already captures 67.7 % of the chance→oracle span inside the top-8**
(`base_span_k` = 0.67653). That is the incumbent a learned re-scorer has to beat, and it is strong. The
remaining room is 0.2688 m of a 0.831 m span — 32 %.

**Identity check (the strongest guarantee in this arm):** the zero-init head reproduces
`refc-xl-30k` *to the digit* — `full_set` 0.47143933 vs the registry's 0.4714393,
`heldout` 0.4577 ± 0.0572, `fde@2s` 0.9724, `miss@2m` 0.1459, `tms` 0.2032, all identical.
The re-scorer is a **residual on the frozen ranking with a zero-initialised head**, so *any* movement
from 0.47144 is attributable to training and to nothing else. There is no re-implementation drift to
argue about.

### 2.2 A free control that fails: "just use the refined logits"

`refined_conf_ade` = **1.36593** — selecting on the DISCARDED refined-pass confidence is **2.9× worse
than the baseline**. REF-C never supervised the conf head at the denoise timesteps
(`refc_train.compute_losses` applies the anchor CE only to the t=0 `anchor_logits`), so that signal is
untrained noise. This matters: the obvious reading of the v1.5 fix — "keep the refined confidence and
select on it" — is *not* available to REF-C for free. It has to be **learned**.

---

## 3. Design

### 3.1 Frozen / trainable split

Frozen: the entire `RefCModel` (`refc-xl-30k`, step 29999, 251.9 M).
Trainable: `tanitad/models/refc_rescorer.py::RefCRescorer`, 1.65 M params:

| Module | Params |
|---|---|
| `q_proj` (frozen decoder query embedding 512→256) | 131,328 |
| `geom_proj` (20 explicit kinematic features) | 5,376 |
| `ctx_proj` (pooled 992 + condition 512 + v0) | 451,328 |
| 2 × candidate self-attention blocks (d=256, 4 heads) | 1,054,208 |
| input LayerNorms (`q`, context) | 4,034 |
| score / value heads + `base_gain` | 515 |
| **total** | **1,647,813** |

Per candidate the head sees a frozen decoder **query embedding** — the signal a hand-written cost
structurally cannot access — plus the refined trajectory's kinematics, the frozen logit itself, and
window context. The self-attention runs **across candidates**, so a proposal is scored relative to its
competitors; the frozen `conf_head` is a per-anchor `Linear(d,1)` and cannot express that.

**Which embedding is itself a swept variable** (`q_source`), because the two available ones are not
interchangeable and the difference is measurable:

| `q_source` | what it is | frozen linear readout of it |
|---|---|---|
| `t0` | the classifier pass over the un-refined anchors | **IS** the selection score, Spearman 0.907 vs ADE |
| `final` | the last denoise pass, over the refined trajectories | selects at **1.366 m** — 2.9× worse than baseline |
| `both` | concatenated | — |

The first design cached only `final`; that was a real gap, since `final` is precisely the
representation whose frozen readout is untrained noise. Both are now cached and compared, alongside
`--no-q` (no embedding at all) and `--no-context`.

    score = base_gain · standardise(base_logit) + score_head(tokens)     # score_head zero-init

Row-standardising the frozen logits is an increasing affine map (argmax provably unchanged, asserted by
tests) that puts the residual on a scale where confident re-orderings are reachable. Without it the
sharp incumbent logits (top-1 prob 0.654, top1–top2 gap 1.56 nats) make any additive correction a no-op
— the same arithmetic that drove v1.0's blend to λ=0.

### 3.2 Top-K restriction — a first-class axis, not a detail

The head scores only the **top-K candidates by frozen confidence** and re-orders within them.
Justification is measured, not aesthetic: the full-fan oracle (0.16395) is a min over 256 draws whose
typical member is ~14 m off — a lottery — while the **top-8 oracle is 0.20263, i.e. 87 % of the ranking
gap inside 3 % of the candidates**. Restricting removes the garbage tail that would otherwise dominate
every listwise/pairwise sum. K is swept (4 / 8 / 16 / 32 / all-256) and travels in the head config, so
the trainer and the eval adapter cannot drift apart on it.

### 3.3 The target — why soft, and why the sweep IS the experiment

Flagship v1.5 already applied a **hard argmin CE** to `sel_score` and it degraded as the fan sharpened
(`frac_sel_2x_worse` 0.099 → 0.40). The mechanism: once candidates converge to near-identical quality
the argmin index is a coin flip, and the CE keeps paying full loss for preferring one 0.30 m plan over
another 0.30 m plan.

v1.2 runs the classical learning-to-rank triad plus that control, off one architecture
(`rescorer_loss`):

| Arm | Objective | Note |
|---|---|---|
| `soft` | CE against `softmax(−ADE_i/τ)` | ListNet-top-1 / distillation. **τ→0 IS the v1.5 objective** — the temperature axis walks from the known failure to the safe regime |
| `pair` | `relu(m·|ΔADE| − Δscore)` over ordered pairs | distance-weighted margin: near-ties demand ~nothing, gross mis-rankings dominate |
| `regress` | smooth-L1 on predicted ADE, select `argmin` | pointwise value learning — **no temperature at all**, the control for "is the pathology inherent to ranking losses?" |
| `hard` | argmin CE | the v1.5 objective, reproduced deliberately |

A unit test pins `soft(τ→0) == hard` numerically, and another shows the mechanism directly: at the soft
target's own optimum on a 0.300 vs 0.301 m pair the soft gradient is **0** while the hard CE still
pushes with gradient ≈0.5 toward unbounded separation.

**The target is the JOINT along-track + cross-track error** (plain ADE). It is deliberately *not* a
speed/VTARGET objective — measured: a GT-perfect speed-matcher scores 1.1236 (worse than baseline), a
GT-perfect along-track-only ranker caps at 34 % of the gap, and VTARGET sits +1.42 m/s above v0, making
braking windows +0.51 m worse. Speed appears only as an input feature.

### 3.4 Training substrate

The frozen forward is constant work, so it is recorded once: `refc_v12_cache.py` runs the decoder in
the **exact TanitEval decode condition** (`model.eval()`, no denoise noise, no ego-dropout,
`nav_cmd=None`→`follow`, `v0` fed, window 8, **stride 8**, fp32) and stores per window the fan, the
per-anchor embeddings, the frozen logits, the context and the GT waypoints (~265 KB/window). The head
then trains for many epochs at ~0 GPU cost with the whole cache parked in VRAM.

`refc_forward_fan` re-derives the decoder orchestration in order to expose the discarded embeddings; a
test asserts it matches `RefCModel.forward` **bit-exactly** on `anchor_logits` / `anchor_traj` / `traj`
/ `sel_idx` (verified on both the dev box and pod3), so the cache cannot silently record a different
decode than the one the harness scores.

Split discipline: only `physicalai-train-e438721ae894` is read; the first 200 episodes form an
**episode-disjoint DEV split** used for all model selection. The 881-window TanitEval val set is
touched once, at the end, through the harness.

---

## 4. Results

### 4.1 Pilot sweep — the v1.5 temperature pathology REPRODUCES, as a temperature effect ✅

First real-data sweep: K=4, 6,362 train windows, 3 dev-episode-disjoint, 1,500 steps, no input
LayerNorm, `q_source=final`. Dev baseline (879→4,398 windows, 200 episodes) **0.4564**.

| arm | best dev ADE | gap recovered | degeneration (final − best) |
|---|---|---|---|
| `hard` (the v1.5 objective) | 0.4564 | +0.03 % | +0.0132 |
| `soft` τ=0.05 | 0.4564 | +0.03 % | **+0.0182** |
| `soft` τ=0.10 | 0.4557 | +0.23 % | +0.0201 |
| `soft` τ=0.20 | 0.4564 | +0.03 % | +0.0196 |
| `soft` τ=0.40 | 0.4564 | +0.03 % | +0.0156 |
| `soft` τ=0.80 | 0.4561 | +0.13 % | +0.0084 |
| `soft` τ=1.60 | **0.4543** | **+0.69 %** | **+0.0072** |

**The degeneration is monotone in temperature** — 0.0182 m at τ=0.05 down to 0.0072 m at τ=1.60 — and
the best arm is the warmest one tested. This is the flagship-v1.5 failure mode reproduced as a
*controlled* effect rather than an accident, and it is the first direct confirmation that the hard
target is the mechanism rather than a coincidence of that arm.

**But the absolute gains are ~0.** Every arm's best is within 0.002 m of doing nothing, and the trend
"warmer = less harm, asymptoting to the baseline" is the exact signature of a head with **no useful
signal to add**. The pilot therefore does not yet distinguish

  (a) the re-scorer is under-trained / mis-fed / over-parameterised, from
  (b) the residual ranking error is not predictable from the observation at all.

### 4.2 What is being run to separate (a) from (b)

Three changes and one new diagnostic, all live in the staged code:

1. **`q_source`** — §3.1. Feeding the head the embedding whose frozen linear readout is *already* a
   0.907-Spearman quality estimate is a materially different experiment from feeding it the one whose
   readout is noise.
2. **Input LayerNorms** on `q` and on the context vector — these are raw frozen activations with
   arbitrary scale.
3. **Capacity/regularisation arms** — 1.64 M params over ~15 k windows memorised outright in the pilot
   (dev degraded monotonically after step ~150).
4. **The fit-vs-generalise probe** — every arm is now scored on a fixed slice of its own TRAIN split
   with the identical code path. This is what actually separates (a) from (b):

   * recovers a lot on train, nothing on dev → the signal is **memorisable but not predictive**: the
     residual ranking error is future uncertainty, and the "gap" is not a recoverable target. That
     would also explain v1.0's 0.0 % and the fact that a GT-perfect speed-matcher is *worse* than the
     baseline.
   * recovers nothing on either → the head or the objective is too weak, and the arm is not yet
     evidence about the problem.

*(Stage B running — table lands here.)*

---

## 5. Verdict against the gates

*(pending)*
