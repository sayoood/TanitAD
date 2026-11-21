# FAN CLIP — the longitudinal-band clip, run LOCALLY on the dev box

**Stream:** v5 follow-on, the experiment `V5_IMAGINATION_SELECTION.md §7.2` pre-registered and
explicitly **did not run**. **Date:** 2026-07-27.
**Host:** the **DEV BOX** (`FREEDOM2035`), NVIDIA RTX 4060, 8.00 GiB, SM 8.9, 24 SMs,
torch 2.11.0+cu128, Python 3.13.5. Venv `C:/Users/Admin/venvs/tanitad`.
⛔ **No pod was contacted.** pod1 / pod2 / pod3 / `tanitad-eval` were never touched.
🔒 No clip UUID and no PhysicalAI-AV raw content appears in any artifact here; episodes are
referenced by val index only.

> ⛔ **§0 (pre-registration) was written before any number in §3–§6 existed.** §2 (the blocker) was
> written after the artifact probe and before any clip was computed — it is what forced the
> two-half design, and it is stated as a falsification of the brief's premise, not absorbed.

---

## 0. PRE-REGISTRATION

### 0.1 The hypothesis

`V5_IMAGINATION_SELECTION.md §2.2` MEASURED that v4's 256-candidate fan spans **108.7 m of 2 s
along-track displacement per window** (−15.47 m to +100.57 m; a 2 s candidate that travels 100 m is a
**181 km/h** plan), and that **the world model does not veto an implausible plan — it obediently
simulates it**, so absurd candidates are *maximally self-consistent* and consistency-scoring ranks
them **first** (bias **+19.66 m**).

⇒ **H-CLIP: the fan's implausible longitudinal tail is the blocker. Remove it and selection
recovers.**

### 0.2 The bars — committed in advance, and NOT moved afterwards

| verdict | condition |
|---|---|
| **CONFIRM** | either selection route, on the clipped fan, beats **0.4907** (Bar A's in-sample ceiling of *feature-only* re-scoring) ⇒ the implausible tail was the blocker |
| **STRONG** | beats **0.4271** (deployed v1's world-model line, full-set) ⇒ report immediately |
| **REFUTE** | no material gain ⇒ the tail was not the blocker, stated plainly |

Also adjudicated, because it is the *prior* registration and must not be quietly dropped —
**`V5 §7.2`'s own bars**: CONFIRM if flat selection on the clipped fan beats **0.8563** by more than
A4's **−0.0857** (i.e. < **0.7706**); STRONG if it beats 0.4907; REFUTE if `oracle_in_fan` degrades
faster than the selection improves.

### 0.3 The two routes, frozen

| route | rule on the surviving candidates |
|---|---|
| **as-trained** | the head's own selector score (`argmax anchor_logits` for REF-C; `base_rank` position 0 for v4) |
| **imagination-consistency** | `argmin_c A1_cost` — the arm that failed at **0.7706** (v4 WM) / **0.5645** (v1 WM) |

### 0.4 The band — physics and configuration ONLY. This is the anti-privilege guarantee.

    s        := a candidate's 2 s along-track displacement
    band     := |s − v0·T| ≤ ½·a_max·T²     with T = 2 s     AND     s ≥ 0

Three inputs, and **nothing else**:

1. the **constant-acceleration kinematic reachable set** — textbook physics;
2. **`v0`, the observed initial speed** — a model *input*, observable at deploy time
   (it is literally the third action channel of the v1 speed contract);
3. a **fixed `a_max` grid**, swept: `0.5 · 1 · 1.5 · 2 · 2.5 · 3 · 4 · 5 · 6 · 8 · 10 · 15 · 20 ·
   30 · 50 · ∞` m/s². The **named anchor is `a_max = 2.5`**, which is
   `FlagshipV15Head.cfg.sel_accel_max` (`stack/tanitad/models/flagship_v15.py:139`, *"the 2 s
   reachable-speed clamp"*) — a **config constant of the model under test**, read from the source
   file, not fitted here.

> ⚠️ **No ground-truth future pose, no held-out error and no val-set outcome statistic enters the
> band.** A band tuned on `fan_err4` would be a privileged intervention; this one cannot be, by
> construction. And because the *whole grid* is reported, no verdict rests on any single choice.

Robustness variants also frozen in advance: **asymmetric** bands (comfortable acceleration +
emergency braking: `a_acc/a_dec` = 2.5/8.0, 1.5/8.0, 2.0/4.0) and a **reverse-permitting** variant.

### 0.5 Controls — every one reported at every band

| control | what it rules out |
|---|---|
| **random-on-survivors** | how much of any gain is the **CLIP** rather than the **ROUTE** |
| **anti-clip** | keep ONLY the candidates the band rejects — must be far worse |
| **oracle clip** (positive) | keep only the best-in-fan candidate — every route must land exactly on `oracle_in_fan`, or the masking plumbing is broken |
| **oracle survival** | does the best-in-fan candidate survive? *If clipping removes the good ones, that is the finding.* |

### 0.6 What makes this rule return a FAILING value — and the proof that it can

**A CONFIRM requires a measured route value below 0.4907.** The routes are evaluated by the *same*
masked-argmin used by the positive control, and that control **does** return 0.2505 — i.e. the
machinery demonstrably produces values far below the bar. Separately the harness returns
**15.8738** on a uniform-random pick, **45.5489** on an anti-oracle, **40.73 / 20.73** when the
cost matrix is shuffled against its windows, and **18.5–42.6** on the anti-clip. A harness that has
only seen good input has not been tested; this one has been driven wrong on purpose five ways and
failed loudly every time (§3).

### 0.7 Estimator

**Paired episode-cluster bootstrap**, `taniteval/ci.py`, **B = 2000**, resampling unit = **episode
cluster**, **40 clusters / 881 windows**, on identical windows. Single-arm intervals are the
unpaired `episode_cluster_bootstrap` on the same unit. **`overlapping_holdout_se` is never called**,
and no `legacy_*` block is quoted.

### 0.8 Parity

**Nothing here reads a corpus.** Every input is an artifact already committed to this repo. The dev
box's own episode cache is keyed **`14231cd29c74`**, which is **NOT** the canonical parity key
**`e438721ae894`** — it is never opened. §7 lists exactly which experiments that rules out.

---
--- MEASUREMENTS BEGIN ---
---

## 1. GATE — committed numbers reproduced before any new one was quoted

**MEASURED (ours, dev box) · CONFIRMED.** Artifact: `raw/fc_gate.json`. Verdict **GATE PASS**,
0 failures, wall **0.44 s**, pure CPU.

| # | test | result | |
|---|---|---|---|
| **S0.1** | all **9** v5 arms under **v4's** scorer WM | `0.8563 · 15.8738 · 0.2505 · 11.5298 · 10.3863 · 13.1805 · 1.7836 · 1.0653 · 0.7706` — all to **4 dp** | ✅ |
| **S0.1** | all **8** v5 arms under **v1's** scorer WM | `0.8563 · 1.2472 · 5.0746 · 6.2200 · 1.7836 · 0.5645 · 0.5645 · 0.2505` — all to **4 dp** | ✅ |
| **S0.2** | E-V5-3 depth axis at n = 256 | k1 **2.906** · k2 **1.423** · k8 **2.818** · k20 **11.530** | ✅ |
| **S0.3** | window alignment across three artifacts | REF-C GT vs Bar A's `tgt`@wp4 **max abs 0.0000 m**; `v0` **max abs 0.0000 m/s**; 881 windows / 40 episodes, same order | ✅ |
| **S0.4** | `MODEL_REGISTRY.md` full-set headline, recomputed from the staged fans | REF-C-**XL 0.4714** · **base 0.4728** · **small 0.5261** — all exact; `sel == argmax(logits)` on **100 %** of windows | ✅ |
| **S0.5** | failing input | random **15.8738** (committed 15.8738) · anti-oracle **45.5489** (committed 45.5488) | ✅ |
| **S0.6** | **shuffled-cost** negative control | as-trained route **40.7304**, A1 route **20.7319** vs as-trained **0.8563** — both destroyed | ✅ |

⭐ **S0.3 is what makes this stream commensurable.** The REF-C fans, Bar A's window dump and the v5
reduced dumps sit on the **byte-identical** 881 windows / 40 episodes, in the same order.

⚠️ **One bug this harness caught in itself, recorded because it would have produced a silent wrong
answer.** `base_rank[w, r]` is the **candidate index at rank r** (an argsort), *not* candidate `c`'s
rank. Using it directly as a cost gave a constant **40.1611** for the as-trained route across every
band — a plausible-looking, entirely fictitious null. It is now inverted explicitly and the code
**asserts** that the unclipped route reproduces the committed `A0` pick before any band is scored
(`fc_clip.py::rank_of_candidate`). The gate did not catch it because the gate reads the committed
`picks` tensor; only the new selection code was wrong.

---

## 2. ⛔ THE BLOCKER — the brief's premise is FALSE for this specific quantity

The brief states the clip is *"recomputable from the staged dump with NO new rollout"*, citing
`raw/v5_v4_windows_reduced.pt` and V5 §8's *"every bar in this document is recomputable from the
reduced dumps with NO GPU"*.

**That claim is TRUE for every published bar — §1 just discharged it — and FALSE for a longitudinal
band.** The reduced dump carries `fan_err4`, `costs`, `picks`, `cost_A1_by_k`, `base_rank`, `v0`,
`ep`, `canary_err`. It does **not** carry `fan`, `tgt`, `imag` or `ctrv`. A band on *2 s along-track
displacement* needs `fan[:, :, -1, 0]` — which is exactly what the reduce step dropped.

**Four independent probes** (per the operating standard: absence at one location is not absence):

1. exhaustive key walk of **both** reduced dumps (`v4` and `v1`) — no trajectory tensor;
2. repo-wide sweep of every `.pt` / `.npz` / `.npy` over 1 MB, and inspection of every plausible
   per-window dump (`rung1_perwindow_compact.pt`, `bi_perwindow_compact.pt`,
   `corridor_v4_30k_K185_perwindow_K20.pt`, `windows_flagship-v4-fromscratch-30k-produced.pt`,
   `bar_a_produced_windows.pt`, `fan_refc-{xl,base,small}-30k.pt`) — **v4's fan is in none of them**;
3. the dump's own `_note`: *"Full fan/imag/ctrv tensors stay pod-side (~150 MB) at `/workspace/_v5/`."*
4. V5 §8's manifest: **"Deliberately not staged (228 MB) … `tanitad-eval:/workspace/_v5/` ONLY"**.

**Regenerating it locally is also ruled out**: it needs the v4 checkpoint (pod-side) *and* the
canonical parity corpus (`e438721ae894`), and the dev box's cache is `14231cd29c74`. A sibling agent
correctly refused to use that cache today; so does this stream.

⇒ **Two halves, and the split is forced, not chosen:**

| half | fan | band | routes available |
|---|---|---|---|
| **A** | **REF-C-XL / base / small** — the only fans in the repo with per-candidate **trajectories** on the canonical 881-window deployment | the **EXACT** band on `s` | as-trained |
| **B** | **v4's fan** | a **SURROGATE** band (§4) | as-trained **and** imagination-consistency, under **both** scorer WMs |

**The fix is 902 kB and it is escalated in §8.**

---

## 3. HALF A — the EXACT band, on three real anchored fans. **The tail is not the blocker.**

`raw/fc_clip.json → HALF_A_exact_band_refc_fans`. Route = the as-trained selector restricted to
survivors. Paired episode-cluster bootstrap vs the *unclipped* as-trained pick, B = 2000, 40 clusters.

### 3.1 The over-dispersion replicates — it is a property of the anchored-fan family

⭐ **NEW, MEASURED here.** V5 §2.2 measured v4's envelope; nobody had checked whether it was a v4
accident. It is not:

| fan | anchors | 2 s along-track min → max | **per-window span (mean)** | implied speed at max |
|---|---:|---|---:|---:|
| v4 (committed, V5 §2.2) | 256 | −15.47 → +100.57 m | **108.74 m** | 181 km/h |
| **REF-C-XL** | 256 | −23.39 → **+95.27 m** | **104.61 m** | **171.5 km/h** |
| **REF-C-base** | 128 | −28.21 → +96.08 m | **108.23 m** | 172.9 km/h |
| **REF-C-small** | 64 | −33.07 → +92.95 m | **107.68 m** | 167.3 km/h |
| *ground truth* | — | — | — | *mean displacement 25.396 m* |

On REF-C-XL, **6.10 %** of candidates imply a mean speed above the **fastest ego speed anywhere in
the deployment** (36.55 m/s), **3.31 %** exceed 40 m/s (144 km/h), **5.44 %** travel *backwards*,
and **85.35 %** sit outside the ±2.5 m/s² reachable band. The pathology is real and large.

### 3.2 The sweep — REF-C-XL (256 anchors), unclipped as-trained **0.4714**, oracle **0.1640**

| `a_max` | % removed | survivors | **oracle survival** | oracle-in-survivors | **as-trained** | *random-on-survivors* | paired Δ vs unclipped |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.5 | 96.7 % | 8.5 | 0.680 | 0.3515 | 0.4954 | *1.14* | +0.0240 [−0.0097, +0.0585] |
| 1.0 | 93.7 % | 16.2 | 0.855 | 0.2389 | 0.4740 | *1.20* | +0.0025 [−0.0140, +0.0199] |
| 1.5 | 90.8 % | 23.6 | 0.925 | 0.1973 | **0.4708** | *1.38* | −0.0007 [−0.0065, +0.0065] |
| 2.0 | 88.1 % | 30.5 | 0.968 | 0.1754 | 0.4718 | *1.57* | +0.0004 [−0.0044, +0.0068] |
| **2.5** ⭐ | **85.4 %** | **37.5** | **0.986** | **0.1683** | **0.4734** | ***1.77*** | **+0.0020 [−0.0016, +0.0076]** |
| 5.0 | 73.0 % | 69.2 | 0.994 | 0.1666 | 0.4734 | *2.75* | +0.0020 [−0.0016, +0.0076] |
| 20.0 | 25.0 % | 191.9 | 0.994 | 0.1666 | 0.4734 | *9.13* | +0.0020 [−0.0016, +0.0076] |
| ∞ (no clip) | 0 % | 256 | 1.000 | 0.1640 | 0.4714 | *13.97* | — |

*Empty-survivor fallback (deterministic, GT-free: revert to the unrestricted argmin) fires on
**7 / 881** windows at `a_max = 0.5` and on **0** windows at every band from 1.0 upward on XL. It is
reported at every row in `raw/fc_clip.json` so a fallback can never silently carry a result.*

**Read the last two data columns together — they are the whole finding.**

> ⭐ **THE CLIP WORKS PERFECTLY AND CHANGES NOTHING.** At the model's own `sel_accel_max` band it
> deletes **85.4 %** of the fan, the best-in-fan candidate **survives on 98.6 % of windows**, the
> oracle bound barely moves (0.1640 → **0.1683**), and a **random** pick improves **7.9×**
> (13.97 → **1.77**). And the **as-trained selector does not move at all: +0.0020 m
> [−0.0016, +0.0076], not separated.**
>
> ⇒ **The implausible tail was never the trained selector's problem. It had already learned to
> avoid it.** The tail only ever hurt rules with no learned prior — a random pick, and
> imagination-consistency (§4.3).

**Replication.** REF-C-base: best clipped **0.4696** at `a_max=1.0` vs unclipped 0.4728 (−0.0032,
not separated). REF-C-small: best **0.5134** at `a_max=1.0` vs 0.5261. Across **45** band × arm
comparisons, **2** separated — both on REF-C-small (−0.0127 [−0.0266, −0.0017] and
−0.0081 [−0.0154, −0.0017]), **neither replicating on XL or base**. At 45 comparisons that is
what multiplicity produces; **I decline to quote either as an effect.**

**Band shape does not rescue it.** Asymmetric (2.5 / 8.0 emergency-braking) on XL: **+0.0020
[−0.0016, +0.0076]**. 1.5 / 8.0: **+0.0001 [−0.0055, +0.0071]**. Reverse-permitting 2.5 / 2.5:
**+0.0000**. 2.0 / 4.0: **+0.0011 [−0.0031, +0.0076]**. The null is flat in every shape tested.

**Controls fire.** Anti-clip (keep only the rejected candidates) on XL: **2.87 / 9.07 / 15.54** at
`a_max` = 2.5 / 8 / 20 — i.e. the rejected set *is* where the catastrophe lives. Positive control
(oracle clip): every route lands on **0.1640**, exactly `oracle_in_fan`.

### 3.3 The headroom is not in the tail — it survives the clip almost untouched

⚠️ **Read this subsection under the retraction published today in `2753d01`:** *"our fan is 41 %
better than our best shipped model"* is **WITHDRAWN**, because `oracle_in_fan` is a
**min-over-N-against-the-one-realized-future** and a single prediction is not. **I therefore do not
compare `oracle_in_fan` to any model's ADE here.** What is admissible is the **before-clip vs
after-clip contrast of the same statistic on the same fan** — which is exactly what a clip
experiment needs, and is the only way the number is used below.

| fan | band | `oracle_in_fan` **unclipped** | **in survivors** | headroom lost to the clip |
|---|---|---:|---:|---:|
| REF-C-XL | `a_max` 2.5 | 0.1640 | **0.1683** | **+0.0043 m (2.6 %)** |
| REF-C-base | `a_max` 2.5 | 0.1914 | 0.1992 | +0.0078 m (4.1 %) |
| REF-C-small | `a_max` 2.5 | 0.2213 | 0.2333 | +0.0120 m (5.4 %) |
| v4 (surrogate band, v1 WM) | `a_max` 2.5 | 0.2505 | 0.2514 | **+0.0009 m (0.4 %)** |

> ⇒ **Deleting 85 % of the fan for physical implausibility costs 0.4–5.4 % of its min-over-N
> headroom, and moves the trained selector by +0.0020 m.** Whatever is unrecovered was already
> living among candidates the band keeps. **The proposal-vs-selection question is not a tail
> phenomenon**, and constraining longitudinal admissibility — V5 §7.1's recommended redirect for
> S-2 — does not act on it.

### 3.4 Reconciling with the published "a big fan is adversarial" result

`2753d01` cites LLM-Assist Table 1: **PDM-Closed scores 92.51 at 15 proposals and 77.78 at 8,505** —
a big fan is an adversarial search against a scorer's approximation error — and notes ours is
**256, single-stage, unfiltered**, against published fan sizes of 15–20.

**Both facts are measured here, and they are not in tension.** Same fan, same windows, `a_max` 2.5
band vs no band:

| fan (survivors at the band) | scorer | clipped | unclipped 256 | **cost of the extra candidates** |
|---|---|---:|---:|---:|
| REF-C-XL (37.5) | random pick — *no scorer* | 1.765 | 13.968 | **7.9×** |
| REF-C-XL (37.5) | **as-trained selector** | 0.4734 | 0.4714 | **1.00×** |
| v4, surrogate band (29.6) | random pick — *no scorer* | 1.441 | 14.875 | **10.3×** |
| v4, surrogate band (29.6) | imagination-consistency (v4 WM) | 0.9589 | 11.5298 | **12.0×** |
| v4, surrogate band (29.6) | **as-trained selector** | 0.9080 | 0.8563 | **0.94×** *(mildly better with more)* |

⇒ **The fan is adversarial exactly in proportion to how weak the scorer is.** The published effect
reproduces on our fan for *rule-like* and *self-consistency* scorers — strongly — and **does not
reproduce for the trained selector** on either fan (1.00× on REF-C-XL; **0.94×**, i.e. mildly
*better* with all 256, on v4's). That is a sharper statement than either "the fan is too big" or
"fan size does not matter", and it is the same conclusion §3.2 reaches from the other direction.

⚠️ **This does not license "keep 256 anchors."** It says fan *size* is not what costs the trained
selector on **open-loop ADE against one realized future**. The published degradations are measured
on **closed-loop PDMS with collision / TTC / drivable-area / comfort terms**, which this metric
cannot see at all — and `2753d01`'s central finding is precisely that the missing ingredient is that
per-candidate **verdict**. A fan-size decision needs that metric, not this one.

---

## 4. HALF B — v4's own fan, both routes, both scorer world models

`raw/fc_clip.json → HALF_B_surrogate_band_v4_fan`.

### 4.1 The surrogate band, and exactly how it maps to the same physics

v4's per-candidate trajectories are not staged (§2), so the band is applied to
**`C2_wm_ref_proximity`** = `mean_j ‖cand_j − wm_ref_j‖`, the distance to the world model's roll-out
under the **observed** action. It is **ground-truth-free** (it never sees a future pose) and it is
already in the dump.

**The mapping is exact, not fitted.** A candidate that deviates from the reference by a constant
relative acceleration `a` has offset `½·a·(j·dt)²` at step j, so its mean over j = 1…20 at dt = 0.1 is
`½·a·dt²·mean(j²)` = **0.7175·a** metres. The identical `a_max` grid therefore maps onto a threshold
θ = **0.7175 · a_max** m.

**Two properties that keep this honest, both measured:**
- the fan is effectively **one-dimensional and longitudinal** — every rule's mean |terminal
  cross-track| is **0.75–1.00 m** (V5 §2.2 committed table) against a **108.7 m** along-track span,
  so a 2-D proximity band is a longitudinal band to second order;
- **the whole θ range is swept over two decades** (0.36 m → 35.9 m), so no conclusion depends on
  the constant 0.7175.

⚠️ **Named degeneracy:** as θ → 0 the survivor set collapses toward `argmin C2`, so *any* route
degenerates to arm **C2**. `mean_survivors` and `random-on-survivors` are printed at every band and
must be read with every row. The unclipped C2 rule (**1.0653** / **0.5645**) is carried in the table
as the reference a clipped route has to beat to have contributed anything.

### 4.2 Scorer = **v1's** world model (the good simulator, canary 0.4271)

Unclipped: as-trained **0.8563**, imagination-consistency **1.2472**, `oracle_in_fan` **0.2505**.

| `a_max` | θ (m) | survivors | oracle survival | oracle-in-surv | **as-trained** | **imag-consistency** | *C2 (no clip)* | paired as-trained Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **0.5** | 0.359 | **4.7** | **0.454** | 0.4059 | **0.6097** | 0.6529 | *0.5645* | **−0.2465 [−0.3537, −0.1388]** ✅sep |
| 1.0 | 0.718 | 12.2 | 0.842 | 0.3059 | 0.6477 | 0.6886 | *0.5645* | −0.2086 [−0.3153, −0.1080] ✅sep |
| 1.5 | 1.076 | 19.0 | 0.946 | 0.2678 | 0.6982 | 0.8107 | *0.5645* | −0.1581 [−0.2413, −0.0885] ✅sep |
| 2.0 | 1.435 | 25.0 | 0.981 | 0.2555 | 0.7493 | 0.9106 | *0.5645* | −0.1070 [−0.1667, −0.0568] ✅sep |
| **2.5** ⭐ | 1.794 | 30.6 | **0.996** | 0.2514 | 0.7940 | 0.9811 | *0.5645* | −0.0623 [−0.1181, −0.0214] ✅sep |
| 5.0 | 3.588 | 54.9 | 1.000 | 0.2505 | 0.8435 | 1.1605 | *0.5645* | −0.0128 [−0.0384, +0.0000] |
| ∞ | — | 256 | 1.000 | 0.2505 | 0.8563 | 1.2472 | *0.5645* | — |

**Best value reached on v4's fan by either pre-registered route, at any band, under either scorer:
`0.6097` [0.5190, 0.7068]** — this row.

### 4.3 Scorer = **v4's own** world model (canary 1.1381)

| `a_max` | survivors | oracle survival | **as-trained** | **imag-consistency** | paired imag Δ vs unclipped as-trained |
|---:|---:|---:|---:|---:|---|
| 0.5 | 3.3 | 0.175 | 0.8439 | 5.8898 | +5.0335 [+3.0279, +7.1241] ✅sep worse |
| 2.5 | 29.6 | 0.924 | 0.9080 | 0.9589 | +0.1026 [−0.0513, +0.3323] |
| **4.0** | 44.9 | 0.996 | 0.8438 | **0.8908** | +0.0345 [−0.0443, +0.1085] |
| **5.0** | 54.1 | 1.000 | **0.8385** | 0.8981 | +0.0418 [−0.0350, +0.1129] |
| ∞ | 256 | 1.000 | 0.8563 | **11.5298** | +10.6735 [+8.5244, +12.7886] ✅sep worse |

> ⭐ **The clip DOES fix the V5 §2.2 mechanism, and the size of the fix is the confirmation that the
> mechanism was correctly diagnosed: imagination-consistency goes 11.5298 → 0.8908, a 12.9×
> improvement, from removing candidates by physics alone.** The world model no longer gets to
> obediently simulate a 181 km/h plan, because that plan is no longer in the set.
>
> ⛔ **And it lands exactly on the as-trained selector, not past it** (+0.0345, not separated) — and
> **1.82× short of 0.4907.**

### 4.4 Attribution — the gain is not the clip's, and it is not the route's

Under v1's WM at `a_max = 0.5`: as-trained-on-survivors **0.6097**, imagination **0.6529**,
random-on-survivors **2.800** — so the routes *do* contribute over the clip alone. But:

> ⛔ **The band's centre is v1's world-model roll-out, and that reference ALONE scores 0.5645
> unclipped.** The best clipped result (**0.6097**) is **worse than the reference it is built from**.
> ⇒ **The clip adds nothing over the trajectory it borrows.** Every metre of the −0.2465 m gain is
> attributable to injecting a better world model's trajectory as the band centre — which is arm C2,
> already committed, already training-free, and already better.

**Secondary clip variable — a genuine negative result, reported because it is the case the brief
named.** Clipping instead on `A3_imag_kinematic` (the *imagined* acceleration magnitude, also
GT-free) **removes the good candidates**: oracle survival collapses to **0.200** at keep-q = 0.02 and
the oracle bound degrades **0.2505 → 7.6735**. Both routes get worse. ⇒ *"If clipping removes the
good ones, that is itself the finding"* — for **A3** it does; for the displacement band it does not
(98.6–99.6 % survival).

**Breadth clip on the model's own prior** (top-n by `base_rank`, extending V5 §5.2 to both routes):
the as-trained route is invariant by construction (0.8563 at every n), and imagination-consistency
under v1's WM is best at **n = 256** (1.2472) — restricting by the learned prior does not help it
either.

---

## 5. HALF C — the φ-free bound: is *any* depth of either ranking competitive?

Because v4's trajectories are missing, this asks the question without them. Any clip that only
*removes* candidates leaves a rule picking its **r-th choice** for some per-window r. So: what is
`E[ade_0_2s | the rule's r-th ranked candidate]`?

| scorer | rule | r1 | r2 | r3 | r8 | r32 | r128 | **ranks with mean < 0.4907** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v4 | A1 imagination-consistency | 11.5298 | 12.5999 | 13.6959 | 16.5291 | 18.6182 | 13.5952 | **0 / 256** |
| v1 | A1 imagination-consistency | 1.2472 | 1.2810 | 1.2826 | 1.6858 | 6.7225 | 16.3862 | **0 / 256** |
| — | as-trained (`base_rank`) | 0.8563 | 14.1775 | 40.6802 | 13.2817 | 10.7516 | 13.1670 | **0 / 256** |

**Not a single rank, for either rule, under either world model, has a mean below 0.4907 — or below
0.4271.** Both rules are already at their own best at r = 1.

⚠️ **Stated as support, not proof.** A clip induces a *per-window varying* r, and a per-window
choice can beat the best fixed r (in the limit it reaches the oracle). This profile therefore rules
out the *fixed-depth* family of tail removals, not literally every clip. It is consistent with §3
and §4 and adds no independent verdict.

---

## 6. VERDICT

> # **REFUTE.** The implausible tail was NOT the blocker.
>
> **The bars were registered for re-scoring v4's fan, so the adjudication is on v4's fan.**
> Best value reached there by either pre-registered route, at any band, under either scorer world
> model: **`0.6097` [0.5190, 0.7068]** (as-trained route, surrogate band at `a_max = 0.5`, v1's
> world model as reference).
>
> - vs **CONFIRM 0.4907** — misses by **1.24×**, and the arm's **own 95 % episode-cluster interval
>   excludes the bar** (lower bound 0.5190).
> - vs **STRONG 0.4271** — misses by **1.43×**.
> - And it is **worse than the unclipped rule it borrows its band centre from** (C2 = 0.5645,
>   already committed, already training-free) — so it is not even a local best.
>
> **On the EXACT band, on three real anchored fans, the null is flat:** with **85 % of candidates
> deleted** and the best-in-fan candidate surviving **98.6 %** of windows, the as-trained selector
> moves by **+0.0020 m [−0.0016, +0.0076]** on REF-C-XL — not separated, and the wrong sign. Best
> across all **45** band × arm comparisons: **0.4696** on REF-C-base (unclipped 0.4728, Δ −0.0032,
> not separated). ⚠️ *REF-C's selector already sits below 0.4907 **unclipped** (0.4714 / 0.4728);
> that is a property of a different and better selector, not of the clip, and it is not evidence on
> the registered bars.*

**Against `V5 §7.2`'s own bars, for completeness:** its CONFIRM (*beat 0.8563 by more than 0.0857*)
**does fire** under v1's world model (0.8563 → 0.6097, Δ −0.2465, separated) and **does not fire**
under v4's own (best Δ −0.0178). Its STRONG (< 0.4907) does not fire. Its REFUTE condition
(*`oracle_in_fan` degrades faster than the selection improves*) does **not** fire — at `a_max = 2.5`
the oracle degrades 0.0009 m while selection improves 0.0623 m.
⇒ The §7.2 CONFIRM is technically met and **carries no capability**: it needs a *foreign* world
model as the band centre, and that model's own trajectory (**0.5645**, C2, already committed and
training-free) is better than the clipped result.

**Tier: CONFIRMED, and DECISION-GRADE for the negative decision** — *"do not spend engineering on
constraining the fan's longitudinal admissibility as a route to selection quality."* It is separated
on 40 episode clusters, replicated across **four** independent fans (v4, REF-C-XL/base/small) and
**two** world models, flat in every band shape tested, and the harness reproduced **25 committed
values** (from `V5_IMAGINATION_SELECTION.md` and `MODEL_REGISTRY.md`) and detected **five**
deliberately-bad inputs before adjudicating.
**NOT decision-grade** for the positive fragments (§4.3's 12.9× repair of A1; REF-C-small's two
marginal separations) — single-run, and the latter does not survive multiplicity.

### 6.1 What is settled

1. **The tail is real** — 85 % of every anchored fan sits outside a ±2.5 m/s² reachable set; 6.1 %
   of REF-C-XL candidates imply a speed above the fastest ego speed in the deployment.
2. **Clipping it is safe** — the best-in-fan candidate survives on **98.6 %** of windows and the
   oracle bound moves 0.1640 → 0.1683. **It does not remove the good ones.**
3. **Clipping it does nothing for a trained selector** — +0.0020 [−0.0016, +0.0076] m.
4. **It repairs an untrained one, dramatically** — imagination-consistency 11.5298 → 0.8908 (12.9×);
   a random pick 13.97 → 1.77 (7.9×). **The tail was only ever a hazard for rules with no prior.**
5. **The clip costs 0.4–5.4 % of the fan's min-over-N headroom** while moving the selector +0.0020 m
   — so whatever is unrecovered was never in the tail (§3.3).
6. **The over-dispersion is a family property, not a v4 accident** — 104.6 / 108.2 / 107.7 m spans
   at 256 / 128 / 64 anchors.
6b. **The published "big fan is adversarial" effect reproduces for weak scorers (7.9×–12.0×) and
   NOT for the trained one (1.00×)** — §3.4.
7. **V5 §7.1's redirect needs amending.** It recommended re-scoping S-2 from *"convert the proposal
   advantage via selection"* to *"constrain the proposal distribution."* **Constraining the
   longitudinal proposal distribution is now measured, and it is not the lever.**

### 6.2 What I refuse to conclude

- **NOT** "the fan is fine", and **NOT** "the fan is 41 % better than v1" — that comparison is
  **withdrawn** (`2753d01`, min-over-N vs a single prediction). What is measured is narrower and
  still says something: the clip removes almost none of the fan's min-over-N headroom (§3.3), so if
  a surplus exists it is not in the tail.
- **NOT** "no clip can help." A band on *lateral* admissibility, on multi-modal coverage, or a
  *learned* admissibility model was not tested. Only the longitudinal/kinematic band was.
- **NOT** a v4-specific result for HALF A — REF-C's selector is a different, and better, selector
  (0.4714 vs 0.8563). HALF B is the v4-specific half, and it agrees.
- ⚠️ **NOT** "a longitudinal clip is worthless." **This REFUTE is scoped to `ade_0_2s`**, and that
  metric can only penalise a 181 km/h plan through its distance to the one realized future — it
  cannot see collision, TTC, drivable area or comfort. A clip that is worth nothing on ADE may still
  be worth something on a **closed-loop safety** metric, and `2753d01` argues exactly that such a
  per-candidate verdict is the program's missing ingredient. **The correct next test of an
  admissibility band is against a safety verdict, not against ADE — and this stream does not settle
  it.**
- **NOT** "the surrogate band equals the exact band." It does not; §4.1 states the mapping and its
  limits, and §2 states why the exact band could not be run on v4's fan.

### 6.3 Threats to validity I could not remove

| threat | status |
|---|---|
| v4's fan was clipped on a **surrogate** variable | **Real, and the reason for §8's escalation.** Mitigated by: the exact band on three other fans (§3), the exact 0.7175 physics mapping, a two-decade θ sweep, and HALF C. |
| tight bands collapse the survivor set (degeneracy) | **Named, and instrumented** — `mean_survivors` and `random-on-survivors` at every row; the best result (0.6097) sits at 4.7 survivors and is flagged. |
| REF-C ≠ v4 | **Real.** Different selector, different anchors (`fan_err4` correlation between the two fans is **0.259**). Both are reported separately and they agree. |
| n = 40 episodes | The **null** at the config band is tight (±0.0046 half-width on XL), so this null is **not** the unpowered kind. The two REF-C-small separations *are* small-n + multiplicity and are declined. |
| the band could still be privileged | **Excluded by construction** (§0.4) — physics + `v0` + a config constant, whole grid reported. |

---

## 7. LOCAL CAPABILITY — what the dev box can now run

Full note: **`LOCAL_GPU.md`** (this directory). Raw: `raw/fc_localgpu.json`. Headline:

- **`taniteval` suite: 449 passed in 59.06 s** on the dev box, CPU — matching the last pod run
  exactly. The harness is fully local.
- **This entire experiment ran on CPU in 27 s** — 3 REF-C fans × 16 bands + v4's fan × 2 scorers ×
  16 bands, each with a B = 2000 paired episode-cluster bootstrap. It queued behind nothing.
- **v4's planner head is 10.57 M parameters** and generates the fan at **batch 1024** in 8.00 GiB
  (5.62 GiB peak) — the whole 881-window deployment is **one pass**.
- **Ruled out here by parity, not by hardware:** anything that opens an episode. The dev-box cache is
  `14231cd29c74`, not `e438721ae894`. No corpus encode, no eval on real frames, no training arm, no
  600-episode re-adjudication.

---

## 8. ESCALATIONS — these must not sit in a file

1. ⭐ **Stage `fan[:, :, -1, 0]` from `tanitad-eval:/workspace/_v5/v5_v4_windows.pt`. It is
   [881, 256] fp32 = 902 kB.** That one tensor converts this stream's HALF B from a surrogate into
   the exact pre-registered experiment, with **zero** GPU and zero new rollout. The full 4-waypoint
   fan (7.2 MB) would additionally make every future re-scoring rule local. **Owner needed** — the
   dump exists on one disk and V5 §8 already flags it as single-copy.
2. ⭐ **Stage the per-window `states` [881, 8, 2048] (fp16 = 28.9 MB) plus the 10.57 M v4 head
   weights.** MEASURED here: with those two, fan regeneration *and* any head-side experiment become
   dev-box work at batch 1024. This is the highest-leverage 30 MB in the program.
3. ⛔ **`V5 §7.1`'s S-2 redirect should be amended before anyone acts on it.** It recommends
   re-scoping the selector work to *"constrain the proposal distribution."* Longitudinal constraint
   is now measured and is **not** the lever (§3.2, §6.1). The unrecovered surplus is **inside** the
   plausible band.
4. **`raw/fc_gate.json` independently re-derives three `MODEL_REGISTRY.md` full-set headlines**
   (REF-C-XL 0.4714 / base 0.4728 / small 0.5261) from the committed fan dumps, on the dev box, in
   0.44 s. That check is cheap enough to be a standing gate and currently is not one.

### 8.1 For `RETRACTION_LOG.md` — root-cause classes

- **C-II (unverified premise in a brief), again.** The brief carried *"recomputable from the staged
  dump with NO new rollout"* as a premise for the **clip**. It is true for the published bars and
  false for the clip's own input variable. Caught by probing the artifact before computing — the
  same resolution as V5 §0.13. **Class: a recomputability claim is scoped to the quantities that
  were reduced, and does not transfer to a new quantity.**
- **C-new: an index that is an ORDERING silently reads as a RANK.** `base_rank[w, r]` = candidate
  index at rank r. Used as a cost it produced a stable, plausible, entirely fictitious constant
  (40.1611) across a 16-point sweep. **A selection harness must assert that its unclipped route
  reproduces the committed pick before it scores any variant** — that assertion is now in
  `fc_clip.py` and it is the cheapest possible guard.
- **C-new: "the mechanism is confirmed" does not imply "the mechanism is the blocker."** V5 §2.2's
  diagnosis was *correct* — removing the tail repairs imagination-consistency by 12.9×. It was still
  not the thing standing between the program and better selection.
- **C-new: a REFUTE inherits the blind spots of the metric it was measured on.** `ade_0_2s` sees an
  implausible plan only as distance-to-one-future. **A plausibility intervention judged on ADE has
  been judged by an instrument that cannot represent plausibility**, and the verdict must be scoped
  accordingly (§6.2). Recorded so the next admissibility experiment is designed against a safety
  verdict from the start.
- **C-II again, and honoured rather than re-committed.** `2753d01` withdrew *"our fan is 41 % better
  than our best shipped model"* (min-over-N vs a single prediction) the same day. This document was
  drafted using `oracle_in_fan` as a ratio; it was rewritten to use it **only** as a before/after
  contrast of the same statistic (§3.3). ⚠️ **A retraction published mid-stream must be checked
  against a draft before it is filed, not after** — the log is a read-before-asserting instrument,
  and on a repo that advances hourly that means re-reading it at write time.

---

## 9. DELIVERABLE MANIFEST

**STAGED, NEVER COMMITTED, NEVER PUSHED.** All paths relative to
`TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-27-fan-clip-local/`.

| artifact | repo | elsewhere | note |
|---|---|---|---|
| `FAN_CLIP_LOCAL.md` (this file) | staged | — | §0 written before any measurement |
| `LOCAL_GPU.md` | staged | — | the capability note (Task 2) |
| `code/fc_common.py` | staged | — | loaders + `taniteval.ci` estimators |
| `code/fc_gate.py` | staged | — | the both-directions gate |
| `code/fc_clip.py` | staged | — | the clip sweep, all three halves |
| `code/fc_localgpu.py` | staged | — | the GPU capacity probe |
| `raw/fc_gate.json` | staged | — | every reproduced committed number |
| `raw/fc_clip.json` | staged | — | every band, every route, every CI |
| `raw/fc_localgpu.json` | staged | — | measured VRAM capacities |
| **inputs — none copied** | — | already in repo | `…/2026-07-26-v5-imagination-selection/raw/v5_{v4,v1}_windows_reduced.pt` · `…/2026-07-26-bar-a-selector/raw/bar_a_produced_windows.pt` · `taniteval/results/fan_refc-{xl,base}-30k.pt` · `…/2026-07-22-refc-small-30k/fan_refc-small-30k.pt` |
| ⛔ **`v5_{v4,v1}_windows.pt`** (full fan) | **no** | **`tanitad-eval:/workspace/_v5/` ONLY** | **single copy, and the reason HALF B is a surrogate.** §8.1 |

**Nothing exists in only one place except the pod-side tensor named above, which this stream did not
create and could not reach.** No pod was contacted. Nothing was launched, restarted, committed or
pushed. No steering file was edited. `pytest -q` in `taniteval/`: **449 passed**.
