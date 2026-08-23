# REF-C v1.2 — a learned re-scorer over the refined fan

**Date:** 2026-07-20 · **Arm key:** `refc-v12` · **Host:** `tanitad-pod3` (train) / `tanitad-eval` (score)
**Status:** COMPLETE. Every headline number is read from raw eval JSON on `tanitad-eval`
(mirrored into `Benchmarks & Eval/Implementation/incoming/2026-07-20-refc-v12/`).

> **One-line result.** A learned re-scorer over the frozen fan moves ADE@2s **0.47144 → 0.46251**
> (+2.9 % of the ranking gap) — *directionally* better on every metric, but the episode-clustered
> paired CI95 **[−0.0062, +0.0250] includes zero**. It clearly beats REF-C v1.0's hand-written cost
> (0.0 %), and the hard-argmin target is the worst arm in every configuration, replicating v1.5.
> **The decisive number is elsewhere: across 47 arms the head recovers at most 8.4 % of the gap *on
> its own training data*, so ~92 % of it is 2-second future uncertainty, not a ranking signal.
> Selection is no longer the productive lever on REF-C.**

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
every listwise/pairwise sum. K is swept (2 / 4 / 8 / 16 / 32 / all-256) and travels in the head config,
so the trainer and the eval adapter cannot drift apart on it.

> ⚠️ **The results qualified this reasoning** (§4.3.3): the garbage tail hurts the *pointwise* target
> badly (`regress` at K=all: −13 %) but the *listwise* soft target tolerates the full fan and in fact
> scored best there. K=8–32 is a flat plateau for both. The right K is target-dependent, not a
> universal property of the fan.

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

### 4.3 The main sweep — 47 arms, 19,802 cached windows

Cache: 900 episodes of `physicalai-train-e438721ae894` → **200 dev episodes (4,398 windows)** +
**700 train episodes (15,404 windows)**, stride 8, both embeddings. Dev frozen baseline **0.45646**,
dev oracle 0.15135, dev gap 0.30511, dev `2x` 0.4995.

Best 12 of 47 (`dev` = episode-disjoint dev split; `rec` = fraction of the ranking gap recovered):

| group | arm | dev ADE | dev rec | **train rec** | `2x` | degen |
|---|---|---|---|---|---|---|
| topk | `kall-soft-τ0.4` | **0.44432** | **+3.98 %** | +8.37 % | 0.4836 | −0.000 |
| topk | `k16-regress` | 0.44479 | +3.82 % | +5.92 % | 0.4879 | +0.000 |
| topk | `k32-soft-τ0.4` | 0.44486 | +3.80 % | +6.76 % | 0.4832 | +0.000 |
| qs-t0 | `k8-regress` | 0.44545 | +3.61 % | +5.97 % | 0.4825 | +0.000 |
| qs-final | `k8-regress` | 0.44572 | +3.52 % | +6.49 % | 0.4868 | +0.000 |
| geom-only | `k8-soft-τ1.6` | 0.44727 | +3.01 % | +4.59 % | 0.4952 | +0.000 |
| no-q | `k8-regress` | 0.44751 | +2.93 % | +5.14 % | 0.4930 | +0.001 |
| qs-both | `k8-regress` | 0.44752 | +2.93 % | +6.95 % | 0.4911 | +0.000 |
| qs-t0 | `k8-soft-τ0.4` | 0.44880 | +2.51 % | +5.02 % | 0.4893 | +0.006 |
| … | | | | | | |
| qs-t0 | `k8-hard` | 0.45509 | +0.45 % | +3.61 % | 0.4925 | +0.001 |
| no-q | `k8-hard` | 0.45570 | +0.25 % | −0.36 % | 0.4984 | +0.001 |
| topk | `kall-regress` | 0.49625 | **−13.0 %** | −6.28 % | 0.5252 | +0.000 |

Four things fall out, and only one of them is about temperature.

1. **The hard argmin target is the worst arm in EVERY group** — 0.45509 / 0.45570 / 0.45404 / 0.45268 /
   0.45086 across the five feature configurations. The v1.5 lesson replicates cleanly, and with the
   input LayerNorms + 19.8 k windows the *degeneration* is now near-zero everywhere, so this is a
   ceiling effect on the objective, not an instability. **Soft > hard, always.**
2. **Pointwise value learning (`regress`) is the strongest single target** in almost every group, with
   warm-listwise (`soft` τ=0.4) statistically tied at the top. The classical LTR ordering holds:
   pointwise ≈ warm-listwise > cold-listwise > hard-argmin.
3. **Top-K matters, but not the way the ceiling analysis suggested.** K=2 is too tight to re-order
   (+0.1 %); K=8–32 is a flat plateau; `soft` tolerates the full 256-wide fan (best arm) while
   `regress` **collapses** on it (−13 %) — a pointwise value head asked to regress the ADE of 248
   garbage candidates spends all its capacity there. K is a *target-dependent* choice.
4. **The frozen embedding is nearly worthless to the re-scorer.** `geom-only` — no `q`, no context,
   just the frozen logit plus the refined trajectory's own kinematics — reaches +3.01 %, against
   +3.61 % for the best embedding-fed arm. The 512-d decoder representation buys ~0.6 points of gap.
   And `q_source` barely separates: t0 0.44545 / final 0.44572 / both 0.44752. **The hypothesis that
   feeding the head the t=0 embedding (the one with the 0.907-Spearman linear readout) would unlock
   the problem is a clean NEGATIVE.**

### 4.4 The decisive diagnostic: the gap is ~92 % irreducible

**Across all 47 arms the best `train_gap_recovered` is 8.4 %.** The head cannot explain more than a
twelfth of the ranking gap *on data it is actively fitting*, with 1.65 M parameters, and shrinking the
head makes it worse (capacity is not the binding constraint). Dev tracks train closely (3–4 % vs
5–8 %), so this is not an overfitting story either.

That is the answer to the question the gap poses. The oracle is a **minimum over 256 candidates scored
against one realised future**; most of its distance below the incumbent is the statistics of that
minimum over aleatoric outcomes, not a ranking signal any observation-conditioned function could
extract. It is consistent with everything else measured: v1.0's hand-written cost recovering 0.0 %, a
GT-perfect speed-matcher scoring *worse* than the baseline, and a GT-perfect along-track-only ranker
capping at 34 %.

### 4.5 Final harness numbers — 881 windows, same protocol ✅

`refc-v12` is the **dev-selected** arm (`kall-soft-τ0.4`). `refc-v12-k16reg` is reported second and was
NOT pre-selected — it happened to do better on val, which is itself evidence the effect sits at the
noise scale. Selecting it as the headline would be selecting on the test set, so it is not.

| | baseline `refc-xl-30k` | **`refc-v12`** (dev-selected) | `refc-v12-k16reg` (post-hoc) |
|---|---|---|---|
| `full_set` ADE@2s | 0.47144 | **0.46251** | 0.45761 |
| `heldout` ADE@2s | 0.4577 ± 0.0572 | 0.4671 ± 0.0613 | 0.4546 ± 0.0563 |
| FDE@2s | 1.0061 | 0.9819 | 0.9750 |
| miss@2m | 0.1419 | 0.1373 | **0.1294** |
| TMS | 0.2135 | 0.2027 | 0.2077 |
| `frac_sel_2x_worse` | **0.45403** | **0.44722** | 0.45289 |
| oracle-in-fan gap | **0.30749** | **0.29856** | 0.29366 |
| gap recovered | 0 % | **+2.90 %** | +4.50 % |
| **paired Δ (m)** | — | **+0.00893** | +0.01383 |
| **paired CI95 (episode-clustered)** | — | **[−0.00616, +0.02500]** | [−0.00452, +0.03251] |
| **significant?** | — | **NO** | NO |
| windows improved / worsened / unchanged | — | 24.3 / 21.2 / 54.5 % | 20.8 / 19.1 / 60.2 % |

The paired test is the right one here: both arms run the same frozen decoder over the same 881 windows
and differ only in selection, so the per-window difference has far more power than the harness's
unpaired ±0.057 CI. **It still does not clear zero.** The point estimate is positive on ADE, FDE,
miss@2m, TMS and both mechanism metrics, and 24 % of windows improve against 21 % worsening — a real
but small directional effect that 40 val episodes cannot resolve.

---

## 5. Verdict against the gates

**G1 — beat the baseline 0.4714.** ⚠️ **Directionally yes, statistically no.** 0.46251 (dev-selected),
−0.0089 m, with an episode-clustered paired CI95 of [−0.0062, +0.0250] that includes zero. Every
secondary metric moves the same way (FDE −0.024, miss@2m −0.005, TMS −0.011). Called honestly: a
learned re-scorer produces a small positive effect that this validation set cannot certify.

**G2 — beat REF-C v1.0 (training-free cost re-rank).** ✅ **Yes, and this is the one clean win.** v1.0
recovers **0.0 %** of the ranking gap — its optimal blend weight is zero, i.e. the best hand-written
cost is *no cost at all*, and pure cost is −171 %. v1.2 recovers **+2.9 %** (dev-selected; +4.5 % for
the post-hoc arm) with the *identical* frozen decoder and fan. A learned ranker does do something a
hand-written one provably cannot. The honest gloss is that it is a qualitative win of small
quantitative size.

**G3 — the mechanism, before and after.**

| | before | after (`refc-v12`) |
|---|---|---|
| `frac_sel_2x_worse` | 0.45403 | 0.44722 |
| oracle-in-fan gap | 0.30749 m | 0.29856 m |

**The ceiling, stated honestly.** The 0.16395 full-fan oracle is GT-informed and unreachable, and it is
substantially a lottery: it is a min over 256 candidates whose typical member is 13.96 m off. Inside the
top-8 the frozen confidence already captures **67.7 %** of the chance→oracle span. The 0.3075 m "gap" is
therefore not a budget waiting to be claimed.

### 5.1 The finding that matters most

**~92 % of REF-C's ranking gap is not recoverable by any observation-conditioned re-scorer** — the best
fit *on the training data itself*, across 47 arms and five feature configurations, is 8.4 %. Combined
with v1.0's 0.0 %, the GT-perfect speed-matcher scoring worse than baseline, and the GT-perfect
along-track ranker capping at 34 %, the conclusion is that **selection is no longer the productive
lever on REF-C.** The remaining 0.30 m is dominated by genuine 2-second future uncertainty that no
re-ranking of a fixed fan can remove.

Three consequences worth acting on:

1. **Stop paying for selection research on REF-C.** Two independent arms (hand-written and learned)
   agree the headroom is ~0–5 %, and the train-fit ceiling explains why.
2. **The `refined_conf` result is a REF-C bug worth one line of code, not a research programme.**
   Selecting on the discarded refined-pass confidence scores **1.36593** — 2.9× worse than the
   baseline — because `refc_train.compute_losses` never supervises the conf head at the denoise
   timesteps. Any future anchored-diffusion trainer should either supervise every pass's confidence
   or stop computing it.
3. **`frac_sel_2x_worse ≈ 0.45` is not the alarm it looks like.** It is largely a statement about the
   oracle being a minimum over a wide fan, not about the selector being broken. Reporting it without
   the chance floor (13.96 m over the fan, 1.03 m inside the top-8) overstates the defect.

---

## 6. Reproduction

Everything here rebuilds from **repo code + the frozen checkpoint + the parity episode cache**. Note
that `/root` on a RunPod container is the overlay and does NOT survive a restart — the feature cache
lives there deliberately (it is derived, and rebuilds in ~32 min), while the head and every sweep
summary are persisted to `/workspace`.

```bash
# 1. record the frozen decoder's fan + embeddings (pod3, ~32 min for 900 eps)
PYTHONPATH=/workspace/TanitAD/stack:/workspace/TanitAD/stack/scripts \
python3 scripts/refc_v12_cache.py \
    --data-root /workspace/pai_epcache \
    --ckpt /workspace/experiments/refc-diffusion-xl-30k/ckpt.pt \
    --config xl --anchors /workspace/experiments/refc_anchors_full.pt \
    --out /root/refc_v12_cache --episodes 900 --dev-episodes 200 \
    --stride 8 --batch 22

# 2. train the head — one arm, or a sweep over targets / temperatures / K
python3 scripts/refc_v12_train.py --cache /root/refc_v12_cache \
    --out /root/v12_run --topk 8 --q-source t0 \
    --arms "soft:0.4,soft:1.6,soft:6.4,regress,hard"

# 3. score it through the harness (eval pod, 881 windows, ~35 s)
PYTHONPATH=/root/taniteval:/root/TanitAD/stack:/root/TanitAD/stack/scripts \
python3 scripts/refc_v12_eval.py --head /root/models/refc-v12/head.pt \
    --episodes 40 --tag refc-v12
```

**Guards that fail loud rather than drift** (`stack/tests/test_refc_rescorer.py`, 25 tests):

* `refc_forward_fan` matches `RefCModel.forward` **bit-exactly** (verified on the dev box AND on pod3)
  — the cache cannot record a different decode than the harness scores.
* `conf_head(q0) + maneuver_prior == anchor_logits` — proves `q0` really is the embedding behind the
  frozen selection score.
* identity-at-init holds at every K and with/without base standardisation.
* `soft(τ→0) == hard` numerically; and at the soft optimum on a 0.300/0.301 m pair the soft gradient is
  0 while the hard CE still pushes at ≈0.5.
* the loss gathers the head's own top-K slice of the full-fan ADE (a silent mismatch would train on
  the wrong rows).
* the paired bootstrap resamples **episodes**: a single episode carrying the whole effect does not
  read as significant.

---

## 7. Deliverable manifest

| Artifact | Location | Copies |
|---|---|---|
| Re-scorer model + losses + instrumented frozen forward | repo `stack/tanitad/models/refc_rescorer.py` | in `3d41bd0` * |
| Frozen-decoder feature cache builder | repo `stack/scripts/refc_v12_cache.py` | in `3d41bd0` * |
| Head-only trainer + sweep driver | repo `stack/scripts/refc_v12_train.py` | in `3d41bd0` * |
| TanitEval-compatible eval adapter (+ paired bootstrap) | repo `stack/scripts/refc_v12_eval.py` | in `3d41bd0` * |
| Contract tests (25, suite green at 611) | repo `stack/tests/test_refc_rescorer.py` | in `3d41bd0` * |
| This note | repo `TanitAD Research Hub/Benchmarks & Eval/Research/2026-07-20-refc-v12-learned-rescorer.md` | staged |
| All 8 sweep summaries + `BEST.json` + 3 raw eval JSONs | repo `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-20-refc-v12/` | staged |
| **Head `kall-soft-τ0.4`** (the `refc-v12` arm) | `tanitad-pod3:/workspace/experiments/refc-v12/head.pt` · `tanitad-eval:/root/models/refc-v12/head.pt` · local scratchpad | **3** |
| **Head `k16-regress`** (post-hoc arm) | `tanitad-pod3:/workspace/experiments/refc-v12/head_k16-regress.pt` · `tanitad-eval:/root/models/refc-v12-k16reg/head.pt` · local scratchpad | **3** |
| Raw results in the harness | `tanitad-eval:/root/taniteval/results/refc-v12{,-k16reg,-identity}.json` + `windows_*.pt` | 1 pod + repo mirror of the JSON |
| Feature cache (19,802 windows, 10.4 GB) | `tanitad-pod3:/root/refc_v12_cache` | **1, on the container overlay — NOT persistent** |

\* These five were staged by this agent and **swept into commit `3d41bd0` by the orchestrator while the
arm was still running** — not committed by me. Verified: every committed blob is byte-identical to the
final working copy (`git diff --quiet HEAD` clean on all five, and the late additions — `q_norm`,
`select_q`, `chance_ade`, `parse_arms`, `train_probe` — are present in the HEAD blobs).

**Single-copy / at-risk items**

* The **feature cache** lives on pod3's container overlay and will not survive a pod restart. It is
  *derived* and rebuilds in ~32 min from `/workspace/experiments/refc-diffusion-xl-30k/ckpt.pt` +
  `/workspace/pai_epcache` with the staged builder, so nothing is lost — but do not treat it as an
  asset.
* `windows_refc-v12*.pt` (per-window predictions, the A/B substrate) exist only on `tanitad-eval`.
* **`taniteval` remains uncommitted** (registry risk R2). This arm was deliberately built so that its
  eval code lives in the REPO (`refc_v12_eval.py` imports `taniteval.bench`/`taniteval.data` but is
  not part of them), so v1.2 is the first arm whose evaluator is version-controlled.

### Integration ask

1. **`taniteval/registry.py` has no `refc-v12` entry.** The arm is scored by a standalone repo script
   rather than `taniteval.runner run --model refc-v12`, deliberately, to avoid editing shared harness
   files while a sibling agent worked the same pod. Adding an entry (arch `refc+rescorer`, ckpt =
   frozen refc-xl-30k, extra field `rescorer_head`) is a small change and would make the row appear in
   `runner report` / `ab` like every other arm.
2. **`Project Steering/MODEL_REGISTRY.md` §4 needs a v1.2 row.** (The §4.1 single-clip-vs-corpus
   correction I was going to ask for landed independently in `bb3f6a7` while this arm ran; the
   corpus figures there — 0.4714 / 0.1640 / 0.3075 — match what `refc_v12_eval.py` reproduces
   from scratch, which is a useful cross-check of both.)
3. **The `refined_conf` bug** (§5.1 item 2) belongs in the REF-C row as a known defect, and in any
   future anchored-diffusion trainer's checklist.
