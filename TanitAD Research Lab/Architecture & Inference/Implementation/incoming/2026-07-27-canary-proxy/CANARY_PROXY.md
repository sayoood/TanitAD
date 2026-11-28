# E-CP-1 — is there a DEPLOY-TIME proxy for world-model trustworthiness?

**Builds the named missing instrument from
`…/incoming/2026-07-26-v5-imagination-selection/V5_IMAGINATION_SELECTION.md` §2.3, recorded there
and in `Project Steering/V5_PLAN.md` §8 as a HYPOTHESIS with an unowned instrument.**

Date **2026-07-27** · **dev box only, no GPU, no pod contact, nothing launched** · every number
recomputed from `…/2026-07-26-v5-imagination-selection/raw/v5_{v4,v1}_windows_reduced.pt`.

> # HEADLINE
> **The signal exists — and the gate is not the product.**
>
> 1. **CONFIRM (the brief's literal question).** World-model trustworthiness **is** predictable from
>    deploy-time observables: `wm_canary_ade_2s` is predicted **out-of-fold, episode-disjoint** at
>    **R² 0.5526 / Pearson 0.7518** (R² **0.4203** with a single world model), against **R² 0.070**
>    for the only scalar previously tried (`v0`, r = 0.2645 — reproduced exactly).
> 2. **CONFIRM (conversion), with a prerequisite.** A learned gate recovers **164 %** of the oracle's
>    deployed value — **−0.1397 [−0.2289, −0.0634]**, separated, out-of-fold, stable across 5 fold
>    seeds — **but only using 2-world-model ensemble features.** Single-world-model gates are
>    **−0.0473 … −0.0602, UNPOWERED NOT REFUTED** at n = 40 (would separate at **n ≈ 61–129**).
> 3. ⛔ **And it is DOMINATED by the trivial baseline.** The 2-WM gate needs a second world model at
>    deploy time. With that same second world model you can instead score **ungated** and get
>    **−0.2918**, i.e. **2.1× more**. **The oracle gate the brief asked me to replace was worth
>    −0.0852 deployed; unconditional C2 under v1's world model is worth −0.2918 with no gate at all.**
> 4. ⭐ **The deployable answer: C2 is deployable UNGATED.** Best measured policy **0.5196–0.5221**
>    vs as-trained **0.8563** — paired **−0.3366 [−0.4507, −0.2310]**, separated, out-of-fold —
>    a **39 % cut in selector error, no oracle anywhere.**
>
> **Tier: CONFIRMED. NOT decision-grade** — single-run on 40 episodes, and the feature bank and model
> family were chosen after seeing this val set (§6.4). Reasons stated, not hedged.

---

## 0. PRE-REGISTRATION — written and staged before any proxy was measured

### 0.1 The gap, restated exactly (INHERITED from §2.3, then RE-MEASURED here as S0c)

On the **22.7 %** of windows where v4's world model is good (`wm_canary_ade_2s ≤ 0.55`), rule **C2**
— score fan candidates by distance to a **single** world-model roll-out used as a reference
trajectory — beats the as-trained selector **0.7085 → 0.3330**, paired
**−0.3754 [−0.5123, −0.2656]**, separated, 200 windows / 29 clusters. A **53 % cut in selector
error with zero training.**

⛔ `wm_canary_ade_2s` is computed **against ground-truth future poses**. The stratifier is an
**ORACLE**. The only deploy-time scalar previously tried supplies no proxy:
`corr(v0, canary_err) = 0.2645`.

### 0.2 ⭐ The estimand is a **POLICY**, not a stratum — the load-bearing design choice

A stratum mean **cannot be compared across gates**: a gate firing on 5 % of windows and one firing
on 95 % report their wins on different populations with different denominators, and the narrower one
always looks better. So **every row in this document is a deployable policy scored on all 881
windows / 40 episode clusters**:

```
π_g(w) = C2's pick   if the gate g fires on window w
         A0's pick   otherwise                       (A0 = the as-trained selector)
```

compared to `π_A0` with the **paired episode-cluster bootstrap** (`taniteval/ci.py`, B = 2000,
unit = episode). Under this framing:

- **the ungated baseline is the special case `g ≡ 1`**, so the trivial baseline and every proxy sit
  on **one axis with one denominator**;
- the oracle gate's whole-set value is `(n_sel/n) × stratum_Δ = 0.227 × (−0.3754) = −0.0852 m`;
- **`selected_frac` is on every row.** A gate firing on ~100 % or ~0 % is degenerate and the fraction
  makes it visible on sight. *(It caught three degenerate rows — §3.1, §4.3.)*
- where the ungated policy already wins, a second column **`Δ vs UNGATED`** is reported, because
  "separated vs A0" is not evidence a gate does anything when it fires on 99 % of windows.

### 0.3 Admissibility — enforced in code, not asserted

| tag | meaning | admissible as a gate input? |
|---|---|---|
| `DEPLOY-1WM` | computable from inputs + **one** world model's own forward pass | ✅ |
| `DEPLOY-2WM` | needs a **second** world model at deploy time | ✅ **at 2× imagination cost — flagged on every row** |
| `ORACLE` | touches ground-truth future poses (`canary_err`, `fan_err4`) | ⛔ **never as a gate input** |

`features()` in `code/canary_proxy.py` never reads `canary_err` or `fan_err4`. S0f asserts it.

⚠️ **A learned gate is *trained* on ground truth and *deployed* without it.** `fan_err4` supplies the
offline training label `u(w) = A0_err − C2_err`; the fitted gate consumes **only observables** and is
scored on **episodes it never saw**. That is supervised learning, not an oracle — but it is the
reason the tier is CONFIRMED and not decision-grade (§6.4).

⚠️ **`base_rank` in the reduced dumps is NOT a score ranking** (`v5_cost_curve.py`, E-H1 §9.2:
column 0 is the deployed pick, columns 1.. are anchor **index** order). The obvious
"head-vs-world-model rank agreement" feature is therefore **not constructible** from the staged
artifacts and is not used. The head's *pick* is available, so `cost__at_A0pick` carries that family.

### 0.4 Outcomes, committed in advance

- **CONFIRM** — a `DEPLOY-*` proxy whose gated policy is **separated** better than `π_A0` **and**
  survives out-of-fold, episode-disjoint. Report the recovered fraction of the oracle.
- **PARTIAL** — correlates with the canary but the **gated policy** is not separated. Report the
  correlation **and say plainly it does not convert.**
- **REFUTE** — nothing observable predicts trustworthiness. **Say so, do not re-scope.**

### 0.5 What makes this rule return a FAILING value — and the proof that it can

| S-test | direction | requirement | measured |
|---|---|---|---|
| **S0a** | fidelity | both dumps share `ep`, `v0`, `fan_err4`, the A0 pick, and the WM-free C1 cost | ✅ PASS |
| **S0b** | fidelity | reproduce the committed arm means from `fan_err4` alone, < 5e-4 | ✅ 7/7 exact |
| **S0c** | fidelity | reproduce §2.3's oracle stratum **exactly** | ✅ 200/29, 0.7085 / 0.3330, Δ = −0.3754 |
| **S0d** | ⚠️ **must FAIL** | a gate driven by **pure Gaussian noise** at 22.7 % must not win | ✅ **+0.0343 [−0.0051, +0.0744], not separated** |
| **S0e** | ⚠️ **must FAIL** | a 0 %-firing gate returns **exactly** 0 | ✅ **0.0000**, `separated=false` |
| **S0f** | admissibility | oracle arrays never reach `features()` | ✅ |

**Six further deliberately-failing controls run downstream** — a noise-fed learned gate (§4.3), a
noise-fed canary regressor (§3.1), and a noise-fed fold-seed sweep — **all fail as required**, three
of them by coming out *separated-worse*, which is the strongest possible form of the check.

---

## 1. ⭐ THE UNGATED C2 BASELINE — measured FIRST, and it settles the product question

`raw/canary_proxy.json § stage1` · **MEASURED (ours)** · tier **CONFIRMED** · `π_A0` = **0.8563**.
Negative Δ = better than the as-trained selector.

### 1.1 With **v4's own world model** (the world model that produced the fan)

| policy | `ade_0_2s` | paired Δ vs as-trained | sep | **selected frac** |
|---|---:|---|:--:|---:|
| **UNGATED — C2 everywhere** | 1.0653 | **+0.2090 [+0.0550, +0.3642]** | ✅ | **1.000** |
| ORACLE gate `canary ≤ 0.55` *(the §2.3 claim, as a policy)* | 0.7710 | **−0.0852 [−0.1190, −0.0548]** | ✅ | 0.227 |
| ORACLE gate, best canary threshold, **in-sample** | 0.7491 | −0.1072 [−0.1599, −0.0584] | ✅ | 0.401 |
| **CEILING — perfect per-window gate** *(unattainable)* | 0.6210 | **−0.2353 [−0.3287, −0.1583]** | ✅ | 0.336 |

⇒ Under v4's own world model **unconditional C2 is separated-WORSE.** A gate is genuinely required
here, and the oracle gate is worth **−0.0852 m** whole-set.

### 1.2 With **v1's world model** scoring v4's fan *(§3.1: feasible — the arms meet at the metric interface)*

| policy | `ade_0_2s` | paired Δ vs as-trained | sep | **selected frac** |
|---|---:|---|:--:|---:|
| ⭐ **UNGATED — C2 everywhere** | **0.5645** | **−0.2918 [−0.4233, −0.1598]** | ✅ | **1.000** |
| ORACLE gate `canary ≤ 0.55` | 0.5538 | −0.3025 [−0.4190, −0.2021] | ✅ | 0.745 |
| ORACLE gate, best canary threshold, **in-sample** | 0.5416 | −0.3147 [−0.4358, −0.2022] | ✅ | 0.880 |
| **CEILING — perfect per-window gate** *(unattainable)* | 0.4497 | −0.4065 [−0.5157, −0.3097] | ✅ | 0.549 |

### 1.3 ⛔ THE FINDING THE BRIEF ASKED ME TO LOOK FOR FIRST — the trivial baseline wins

> - The **oracle** gate on v4's world model is worth **−0.0852 m** deployed.
> - **Ungated** C2 under v1's world model is worth **−0.2918 m** deployed — **3.4× more, no gate.**
> - Even the **perfect per-window gate** on v4's world model (unattainable; it needs the answer)
>   reaches only **−0.2353 m** — **still worse than ungated C2 under v1's world model.**

⇒ **The entire proxy search on v4's world model is bounded above by a number that unconditional
application already beats.** Building the gate was never the highest-value move; **swapping the
scoring world model was**, and that is available today with zero training.

**Where the "53 %" goes.** The §2.3 headline **−0.3754** is a *stratum* number on 22.7 % of windows;
its deployed value is `0.227 × 0.3754 = 0.0852 m`. The 53 % error cut is real **and it applies to a
fifth of driving**. Unconditional C2 under a better simulator cuts less per window (34 %) **on all
of it**, and 34 % of everything beats 53 % of a fifth.

### 1.4 The second finding — under v1's world model the canary is a **weak gate even as an oracle**

Headroom above ungated, under v1's world model: the perfect gate adds **−0.1148 m**. The **canary
oracle** adds only **−0.0107 [−0.0680, +0.0349], not separated** — it recovers **9.3 %** of the gate
headroom that exists. ⇒ **`wm_canary_ade_2s` is not a good gate even when you are allowed to cheat**,
and any proxy for it inherits that ceiling.

⭐ **This reframes the brief's question, and the reframing is what pays** (§4.2): the useful
instrument is **not** "a proxy for the canary" but **"a predictor of where C2 beats A0"**. They are
measurably different targets — the best canary-correlated feature has ρ = +0.573 with the canary but
only −0.429 with the gate utility.

---

## 2. THE FEATURE BANK — what was tested, and what was NOT

73 per-window scalars, all `DEPLOY-*`. Families, mapped to the brief's candidate list:

| # | family | n | brief's category | admissibility |
|---|---|---:|---|---|
| I | `v0` | 1 | input-side scene statistics | `DEPLOY-1WM` |
| II | cost-distribution shape per rule (min / mean / std / p90 / range / top-1–top-2 gap / normalised margin / **value at the head's own pick** / its z-score), for A1, A2, A3, C1, C2 | 45 | predictor-internal + decision margin | `DEPLOY-1WM` |
| III | **world-model vs analytic-model divergence** — mean and abs-mean of `A1 − C1`, their per-window rank correlation across the 256 candidates, argmin agreement | 7 | **self-consistency without ground truth** | `DEPLOY-1WM` |
| IV | **roll-out drift over the horizon** — per-k mean consistency at k = 1…20, six growth ratios, log-log drift slope | 15 | latent/step drift over the roll-out | `DEPLOY-1WM` |
| V | **2-world-model ensemble disagreement** — abs-diff, rank corr, argmin agreement between v1's and v4's reference rolls | 5 | ensemble variance | ⚠️ `DEPLOY-2WM` |

### 2.1 ⚠️ NOT TESTED, stated plainly rather than quietly dropped

- **Latent-space signals** — attention entropy, latent-norm drift, eval-mode/dropout variance. The
  reduced dumps carry **metric trajectories only, no latents**. Getting them needs a GPU re-run of
  the harness on the eval pod (**≈11 GPU-min per arm**, per the V5 manifest). **Not run** — §1.3 says
  the marginal value on v4 is bounded by a number ungated already beats, and the eval pod is shared
  with a live IDM v3 stream.
- **Re-encode-the-frame divergence**, and the admissibility condition the brief flagged: at window
  `t` the vehicle has only seen frames `≤ t`, so a 2 s roll-out from `t` can be compared to a
  re-encode only at `t + 2 s`. **It is inherently a LAGGED signal** — usable to gate the *next*
  decision, never the current one. Anyone building it must state that; a same-window formulation
  would be an oracle in disguise.
- **Perturbed-input roll-out disagreement** (dropout / jittered inputs) — needs the same GPU re-run.
  Family V is the *cheap* stand-in: two independently trained world models instead of two
  perturbations of one.

---

## 3. TARGET A — is the canary itself observable? **YES, and this is the instrument.**

`raw/canary_proxy.json § stage3 …learned_canary_regressor…` · 5-fold **episode-disjoint** OOF.

| arm | feature set | model | **OOF R²** | OOF Pearson | OOF Spearman |
|---|---|---|---:|---:|---:|
| **v4** | ALL (incl. 2-WM) | ridge | **0.5526** | **0.7518** | 0.7032 |
| v4 | ALL | GBM | 0.5034 | 0.7104 | 0.6771 |
| **v4** | **1WM only** | GBM | **0.4203** | **0.6492** | 0.5537 |
| v4 | 1WM only | ridge | 0.3576 | 0.6240 | 0.5569 |
| v4 | *`v0` alone — the previously-tried scalar* | — | *0.070* | *0.2645* | — |
| ⚠️ **v4 NEGATIVE CONTROL — 24 pure-noise features** | — | ridge / GBM | **−0.0336 / −0.0981** | −0.2043 / −0.0027 | — |
| v1 | ALL | ridge | 0.2138 | 0.4743 | 0.4726 |
| v1 | 1WM only | ridge | 0.2089 | 0.4694 | 0.4718 |
| ⚠️ **v1 NEGATIVE CONTROL** | — | ridge / GBM | **−0.0316 / −0.1201** | — | — |

⇒ **CONFIRM on the brief's literal question.** v4's world-model 2 s fidelity is predictable
out-of-fold at **Pearson 0.75**, explaining **7.9×** the variance `v0` explains, and **R² 0.42 with a
single world model.** The negative controls return **negative R²** as required.

**The strongest individual observables** (Spearman vs the canary; `corr(v0, canary) = 0.2645`
reproduced exactly as a sanity anchor):

| feature | family | ρ vs canary | ρ vs **gate utility** | admissibility |
|---|---|---:|---:|---|
| `ens__C2_absdiff_mean` — the two world models' reference rolls disagree | V | **+0.573** | −0.429 | `DEPLOY-2WM` |
| `C2__min` — the fan's closest approach to the WM's own trajectory | II | **+0.532** | −0.328 | `DEPLOY-1WM` |
| `C2__at_A0pick` — how far the head's own pick sits from the WM's roll | II | **+0.520** | −0.202 | `DEPLOY-1WM` |
| `ens__C2_corr` | V | −0.438 | +0.293 | `DEPLOY-2WM` |
| `A3__range` — spread of imagined accelerations across the fan | II | −0.408 | +0.274 | `DEPLOY-1WM` |
| `v0` *(the previously-tried scalar)* | I | +0.231 **(Pearson 0.2645 — reproduced exactly)** | −0.169 | `DEPLOY-1WM` |

⚠️ **Read the last two columns together — they are the whole §4 story.** The features that best track
the canary are **not** the features that best track where C2 pays. ρ collapses from **0.573 → 0.429**
and **0.532 → 0.328** when the target changes.

---

## 4. TARGET B — does it CONVERT? The gated-policy results

### 4.1 Single-feature gates, with the feature chosen INSIDE the folds

Reporting "the best of 73 features × 2 signs" after inspecting all of them is a 146-hypothesis search
dressed as one test. `oof_feature_and_threshold_gate` therefore chooses **feature, sign and
threshold on the training episodes of each fold**.

| arm | selection | Δ vs as-trained | sep | Δ vs **ungated** | sep | **frac** |
|---|---|---|:--:|---|:--:|---:|
| **v4** | **nested, honest** (ALL) | **+0.0193 [−0.0443, +0.0876]** | ❌ | −0.1898 | ✅ | 0.413 |
| v4 | nested, honest (1WM) | +0.0257 [−0.0206, +0.0811] | ❌ | −0.1834 | ✅ | 0.280 |
| v4 | *post-hoc best-of-146* — `ens__C2_absdiff_mean` | *−0.0490 [−0.1117, +0.0116]* | ❌ | −0.2580 | ✅ | 0.394 |
| **v1** | **nested, honest** (ALL) | −0.3291 | ✅ | **−0.0374 [−0.0865, +0.0003]** | ❌ | 0.885 |
| v1 | nested, honest (1WM) | −0.3222 | ✅ | −0.0304 [−0.0805, +0.0094] | ❌ | 0.838 |
| v1 | *post-hoc best-of-146* — `ens__C2_at_other_argmin_z` | *−0.3344* | ✅ | *−0.0426 [−0.0924, −0.0033]* | ✅ | 0.875 |

⇒ **No single observable converts.** On v4 the honest nested gate is on the **wrong side of zero**.
On v1 the honest nested increment misses separation. The post-hoc rows separate — and are exactly the
selection artefact the nested protocol exists to expose. **Reported as a diagnostic, not a result.**

### 4.2 ⭐ The learned gate — and the target that pays is the UTILITY, not the canary

`f_θ(observables) → û = Â0_err − Ĉ2_err`, gate fires iff `û > 0` (**τ = 0 — no threshold tuning at
all**). 5-fold episode-disjoint OOF.

**v4's world model** — the arm where the oracle claim lives. Bar to beat: the oracle's **−0.0852**.

| gate | Δ vs as-trained | sep | **frac** | **recovered fraction of the oracle** |
|---|---|:--:|---:|---:|
| ORACLE `canary ≤ 0.55` *(the thing being replaced)* | −0.0852 [−0.1190, −0.0548] | ✅ | 0.227 | 1.00 |
| **⭐ learned utility gate, ALL, ridge, tuned τ** | **−0.1397 [−0.2289, −0.0634]** | ✅ | 0.302 | **1.64** |
| learned utility gate, ALL, ridge, **τ = 0** | −0.1261 [−0.2161, −0.0440] | ✅ | 0.401 | 1.48 |
| learned utility gate, ALL, GBM, τ = 0 | −0.1212 [−0.2130, −0.0438] | ✅ | 0.350 | 1.42 |
| learned utility gate, **2WM features only** | −0.0881 [−0.1430, −0.0380] | ✅ | 0.355 | 1.03 |
| ⚠️ learned utility gate, **1WM only**, GBM | **−0.0602 [−0.1400, +0.0081]** | ❌ | 0.355 | 0.71 |
| ⚠️ learned utility gate, **1WM only**, ridge | −0.0473 [−0.1389, +0.0307] | ❌ | 0.392 | 0.56 |
| **PREDICTED-CANARY gate** `ĉ ≤ 0.55`, ALL, ridge | **−0.0383 [−0.0626, −0.0159]** | ✅ | 0.160 | 0.45 |
| PREDICTED-CANARY gate, tuned quantile, ALL, ridge | −0.0598 [−0.1199, −0.0029] | ✅ | 0.481 | 0.70 |
| PREDICTED-CANARY gate `ĉ ≤ 0.55`, **1WM**, ridge | +0.0044 [−0.0096, +0.0186] | ❌ | 0.104 | — |
| ⚠️ **NEGCTRL noise gate, ridge** | **+0.0000**, fires on **0.000** | ❌ | 0.000 | — |
| ⚠️ **NEGCTRL noise gate, GBM** | **+0.0573 [+0.0173, +0.1028] separated-WORSE** | ✅⛔ | 0.232 | — |
| ⚠️ **NEGCTRL noise predicted-canary** | +0.0000 / −0.0008, fires on 0.000 / 0.002 | ❌ | ~0 | — |

**Two things to read here.**

1. **The literal instrument works: replacing the oracle `canary ≤ 0.55` with a *predicted*
   `ĉ ≤ 0.55` recovers 45 % of the oracle's value, separated.** Its agreement with the oracle gate it
   replaces is **accuracy 0.817, precision 0.638, recall 0.450** — it fires on a **subset** of the
   oracle's windows and is right about two thirds of the time.
2. ⭐ **Predicting the canary is the WRONG objective.** Gating on the **direct C2-vs-A0 utility**
   scores **−0.1397** against the predicted-canary gate's **−0.0383** — **3.6× better, and 1.64× the
   ORACLE canary gate itself.** A perfect canary proxy would still only be worth −0.0852. §1.4 said
   the canary is a weak gate even as an oracle; this measures the cost of aiming at it.

**v1's world model** — here ungated already wins, so **the incremental column is the only honest one.**

| gate | `ade_0_2s` | Δ vs as-trained | sep | **Δ vs UNGATED** | sep | **frac** |
|---|---:|---|:--:|---|:--:|---:|
| UNGATED reference | 0.5645 | −0.2918 | ✅ | 0.0000 | — | 1.000 |
| ORACLE `canary ≤ 0.55` | 0.5538 | −0.3025 | ✅ | −0.0107 [−0.0680, +0.0349] | ❌ | 0.745 |
| **⭐ learned utility gate, ALL, ridge, τ = 0** | **0.5196** | **−0.3366 [−0.4507, −0.2310]** | ✅ | **−0.0449 [−0.0938, −0.0053]** | ✅ | 0.670 |
| **⭐ learned utility gate, 1WM only, ridge** | **0.5221** | **−0.3342 [−0.4469, −0.2281]** | ✅ | **−0.0424 [−0.0904, −0.0035]** | ✅ | 0.679 |
| learned utility gate, ALL, GBM, τ = 0 | 0.5231 | −0.3332 | ✅ | −0.0414 [−0.0900, −0.0029] | ✅ | 0.731 |
| PREDICTED-CANARY gate, ALL, ridge | 0.5656 | −0.2907 | ✅ | **+0.0010 [−0.0591, +0.0653]** | ❌ | 0.838 |
| **CEILING** — perfect per-window gate | 0.4497 | −0.4065 | ✅ | −0.1148 [−0.1631, −0.0767] | ✅ | 0.549 |
| ⚠️ **NEGCTRL noise gate, ridge** | 0.5645 | −0.2918 | ✅⚠️ | **+0.0000**, fires on **1.000** | ❌ | 1.000 |
| ⚠️ **NEGCTRL noise gate, GBM** | 0.5801 | −0.2761 | ✅⚠️ | **+0.0156 [+0.0028, +0.0279] WORSE** | ✅⛔ | 0.923 |

⚠️ **Look at the two NEGCTRL rows' `Δ vs as-trained` column: −0.2918 and −0.2761, both "separated".**
A noise gate looks like a win on that axis because it inherits the ungated policy wholesale — its
`frac` is 1.000 and 0.923. **This is precisely the degeneracy the pre-registration required
`selected_frac` and the incremental column to catch, and it caught it.**

⇒ On v1's world model the learned gate adds **−0.0449** over unconditional application, separated,
and **recovers 39 % of the ceiling's available headroom — 4.2× what the canary ORACLE recovers.**
⚠️ **And it is fold-seed fragile** (§4.4).

### 4.3 Which family carries it — one learned gate per family, alone

| family | v4: Δ vs as-trained | sep | v1: Δ vs **ungated** | sep |
|---|---|:--:|---|:--:|
| **V — 2-WM ensemble disagreement** | **−0.0881 [−0.1430, −0.0380]** | ✅ | −0.0265 [−0.0674, +0.0047] | ❌ |
| II — C1 cost shape | −0.0433 [−0.1217, +0.0175] | ❌ | −0.0227 | ❌ |
| II — A3 cost shape (imagined accelerations) | −0.0259 [−0.0759, +0.0187] | ❌ | +0.0009 | ❌ |
| II — **C2 cost shape** | −0.0121 [−0.0769, +0.0558] | ❌ | **−0.0386 [−0.0877, +0.0012]** | ❌ |
| III — WM-vs-analytic self-consistency | −0.0035 | ❌ | −0.0276 | ❌ |
| II — A1 cost shape | −0.0032 | ❌ | −0.0196 | ❌ |
| II — A2 cost shape | +0.0077 | ❌ | −0.0275 | ❌ |
| **IV — roll-out drift over the horizon** | **+0.0108 [−0.0498, +0.0718]** | ❌ | −0.0103 | ❌ |
| ⛔ **I — `v0` alone** | **+0.0411 [+0.0133, +0.0752] separated-WORSE** | ✅⛔ | −0.0084 | ❌ |

Three readings:

1. **Only the 2-world-model ensemble family is separated on its own.** Every single-world-model
   family fails alone; the 1WM result of §4.2 comes from *combining* them.
2. ⛔ **`v0` alone is separated-WORSE than doing nothing.** The scalar §2.3 tried is not merely a weak
   proxy — used as a gate it is **actively harmful**. That is a stronger statement than
   `corr = 0.2645` and it closes that thread.
3. **Roll-out drift — the most theoretically attractive family — is the weakest.** `+0.0108` on v4,
   `−0.0103` on v1, neither separated. §2.2 of the V5 document predicts exactly this: a world model
   with no plausibility prior reproduces whatever it is asked to, so its self-consistency drift is
   **uninformative about its accuracy**. The measurement confirms the mechanism.

### 4.4 ⚠️ Fold-seed stability — 40 episodes, 5 folds, one assignment can flatter a gate

| gate | Δ range across 5 fold seeds | same sign | **all separated** |
|---|---|:--:|:--:|
| **v4 ALL ridge τ = 0** (vs as-trained) | **[−0.1329, −0.1239]** | ✅ | ✅ **ROBUST** |
| ⚠️ v4 **1WM** ridge τ = 0 (vs as-trained) | [−0.0445, −0.0222] | ✅ | ❌ **UNPOWERED** |
| ⚠️ v1 ALL ridge τ = 0 (**vs ungated**) | [−0.0481, −0.0217] | ✅ | ❌ **borderline** |
| ⚠️ v1 1WM ridge τ = 0 (vs ungated) | [−0.0501, −0.0189] | ✅ | ❌ **borderline** |
| ⚠️ **NEGCTRL noise, v4** | [+0.0000, +0.0000], fires on 0.000 | — | ❌ ✅*(fails as required)* |
| ⚠️ **NEGCTRL noise, v1** (vs ungated) | [+0.0000, +0.0000], fires on 1.000 | — | ❌ ✅*(fails as required)* |

⇒ **Only v4's 2-WM gate is robust.** The v1 increment and the v4 1WM gate keep their **sign** at every
fold assignment but lose separation at some. Per `MODEL_REGISTRY.md` §1.2a (half-width shrinks
**×2.8–3.9, mean ≈3.4**, going 40 → 600 episodes) these are **UNPOWERED, NOT REFUTED**:

| result | Δ | half-width | **n episodes to separate** *(ESTIMATED, √n from the measured half-width)* |
|---|---:|---:|---:|
| v4 1WM utility gate, GBM | −0.0602 | 0.0740 | **≈ 61** |
| v4 1WM utility gate, ridge | −0.0473 | 0.0848 | **≈ 129** |
| v1 incremental gate, worst fold seed | −0.0217 | 0.0442 | **≈ 167** |

All three are **far below the 600-episode harvest**, so all three become decision-grade there at no
extra cost.

---

## 5. VERDICT

### 5.1 Against the pre-registered outcomes

| question | verdict |
|---|---|
| **Is there an observable signal that predicts when this world model's 2 s imagination is trustworthy?** | ⭐ **CONFIRM.** OOF R² **0.5526** / Pearson **0.7518** episode-disjoint (R² **0.4203** single-world-model) vs **0.070** for `v0`. Noise controls return negative R². |
| **Does gating on it recover the win?** | **CONFIRM with 2 world models** — **−0.1397 [−0.2289, −0.0634]**, separated, **164 %** of the oracle, stable across 5 fold seeds. **UNPOWERED, NOT REFUTED with 1 world model** (−0.047 … −0.060; separates at n ≈ 61–129). |
| **Does a *canary* proxy specifically convert?** | **PARTIAL.** ρ = 0.75 with the canary; the predicted-canary gate recovers **45 %** of the oracle, separated — but a **utility** gate on the same features recovers **164 %**. **Aiming at the canary costs 3.6×.** |
| **Is C2 deployable — gated or ungated?** | ⭐ **UNGATED, with v1's world model. −0.2918 [−0.4233, −0.1598], separated, zero training, no gate.** Adding the learned gate reaches **0.5196**, **−0.3366 [−0.4507, −0.2310]** — increment −0.0449, separated at the headline fold seed but **fold-seed fragile**. |

### 5.2 ⛔ The dominance argument — why the instrument I built should probably not ship

The **only robustly separated** gate on v4's world model needs **2-world-model ensemble features**,
i.e. a second world model at inference. **With that second world model in hand you can instead score
with it directly, ungated, and get −0.2918 — 2.1× more than the gate's −0.1397.**

> **The 2-WM gate is dominated by its own prerequisite.** Stating it plainly because it is the kind of
> result that gets quietly omitted: **I built the named missing instrument, it works, and measuring it
> honestly on one axis shows it is the wrong thing to deploy.**

**What should ship instead**, in order:

1. **C2 with v1's world model, applied unconditionally.** −0.2918, separated, zero training. One
   extra world-model forward pass per window (one roll-out, not 256 — this is C2, not A1).
2. *Optionally* the learned 1WM utility gate on top: **0.5221, −0.3342 [−0.4469, −0.2281]**, increment
   −0.0424 — **re-adjudicate at 600 episodes before funding it.**
3. **Nothing here rescues the v4-world-model gate.** Its ceiling is below the ungated alternative.

⚠️ **And carry §3.4's caveat with it:** v1's world model *alone* scores **0.4271**; projected onto
v4's 256-anchor fan it scores **0.5645** — the fan's **0.1374 m quantisation tax**. C2 is the answer
to *"select from v4's fan"*, not to *"produce the best 2 s trajectory"*.

### 5.3 What is NOT settled, and what I refuse to conclude

- **NOT** "no single-world-model gate exists." −0.0473/−0.0602, consistently negative across five
  fold seeds, needs **n ≈ 61–129**. **UNPOWERED, NOT REFUTED.**
- **NOT** "latent-space trustworthiness signals don't work." **I did not test them** (§2.1) — no
  latents in the staged dumps.
- **NOT** "the canary is unpredictable." It is predictable at ρ = 0.75. It is **the wrong target**.
- **NOT** a claim that v1's world model is the best available scorer. It is the best of **two** tested.
  §3.5's finding that C2 tracks simulator quality 1:1 means **any Bar-B improvement converts directly**
  — that lever is now measured to be worth more than any gate.

### 5.4 Threats to validity I could not remove

| threat | status |
|---|---|
| **feature bank + model family chosen after seeing this val set** | **Real, and the reason the tier is CONFIRMED not decision-grade.** Mitigated by: τ = 0 needs no tuning; 5 fold seeds; nested feature selection (§4.1) reported alongside; six negative controls all fail. **Not removed.** The clean test is a fresh episode set. |
| bootstrap CI on an OOF-assembled policy | The point estimate is honest OOF; the interval does not propagate fold-selection variance. §4.4's seed sweep is the stand-in. |
| n = 40 episodes | Named on every unseparated row, with the n that would separate it. |
| single fan, single head | Every result is conditional on v4's 256-anchor fan. §7's fan-conditioning stream may move all of it. |
| 2-WM features assume the two world models are independent | v1 and v4 share a corpus and a parity key; their disagreement is a *lower* bound on true epistemic uncertainty. |
| ⚠️ **the stable-argsort-over-ties bug** (`taniteval/rank_metrics.py`, scored 1.726× chance) | **Excluded, measured.** No ranking-vs-chance comparator is used here at all. The four `argmin` features were checked directly: **0 / 881 windows have a tied argmin in any of the five cost matrices**, so no tie-break by row order occurs. |

---

## 6. WHAT THIS UNBLOCKS — and three escalations

### 6.1 Streams

| stream | what it gets |
|---|---|
| **`V5_PLAN.md` §8** | The last bullet — *"the good-WM gate is ORACLE-GATED and therefore not a capability… finding an observable canary proxy is cheap, concrete and **unowned**"* — is now **OWNED AND ANSWERED**. Proposed replacement text in §6.3. |
| **`Project Steering/Gates/flagship-v5-retrain.PREP.md`** | A **training-free −0.2918 m post-processor** that needs only a grounded step-readout, plus the measured fact that **the scoring world model matters 3.4× more than any gate**. A v5 retrain should be evaluated with C2 applied, or it will be compared against a weaker baseline than the one already available. |
| **Bar B (`wm_canary`)** | Strengthened. §3.5 measured C2 tracking simulator quality 1:1; §1.3 now measures that **swapping the simulator beats the best possible gate**. Bar B is the highest-leverage bar in this stream. |
| **The 600-episode harvest** | **Three ranked entries** with their required n: v4 1WM utility gate (−0.0602, n ≈ 61), v4 1WM ridge (−0.0473, n ≈ 129), v1 incremental gate (−0.0217 worst-seed, n ≈ 167). All well inside 600. |
| **v2corpus arm (pod1)** | Inherits the ungated C2 post-processor **and** the finding that it should be scored with the *best available* world model, not its own. |
| **E-V5-4 fan-conditioning** | §4.3 measures that `A3` (imagined-acceleration spread) and `C1` (analytic-consistency) cost shapes are the two most informative single-WM families — both are **longitudinal-plausibility** signals, the same lever E-V5-4 is pulling. |

### 6.2 The cheapest discriminating experiment this points to, pre-registered in outline

**Score v4's fan with the best world model in the program, not the second-best.** §1.3 shows the
simulator swap is worth 3.4× the oracle gate. v1 was chosen because it was already dumped — **it was
never established as the best available scorer.** Sweep every checkpointed arm with a grounded
step-readout as the C2 reference (REF-B v2, REF-C XL, v3enc, v4.1), pick by `wm_canary`, re-run C2.

- **CONFIRM** if any arm's ungated C2 beats **0.5645** by more than the learned gate's −0.0449 —
  i.e. the simulator axis dominates the gate axis a second time, independently.
- **REFUTE** if v1 is already the best available scorer, in which case **0.5645 is the ceiling of the
  ungated route** and the gate becomes worth funding after all.
- ⚠️ **Both outcomes committed here before anyone runs it. I did not run it** — it needs GPU and the
  eval pod is shared.

### 6.3 THREE ESCALATIONS — these must not sit in a file

1. ⭐ **DECISION AVAILABLE TODAY, and it needs an owner: apply C2 with v1's world model, ungated.**
   **−0.2918 [−0.4233, −0.1598]**, separated, zero training, one extra roll-out per window. It is
   **3.4× the deployed value of the oracle gate this brief was written to replace.** This is a
   decision, not a finding.
2. ⛔ **`V5_PLAN.md` §8's last bullet is now stale and should be rewritten**, from *"ORACLE-GATED
   therefore not a capability… an observable proxy is unowned"* to: **"MEASURED 2026-07-27: the
   canary IS observable (OOF R² 0.5526 / ρ 0.7518) and a learned gate recovers 164 % of the oracle —
   but the gate is DOMINATED: unconditional C2 under v1's world model is worth −0.2918 against the
   oracle gate's −0.0852. Ship the ungated rule; the gate is a 600-episode question."**
   ⚠️ **I did not edit `V5_PLAN.md`** — agents do not edit steering files. This needs the coordinator.
3. ⚠️ **The `−0.3754 / 53 %` figure must never travel without its selected fraction.** It is a stratum
   number on 22.7 % of windows; its deployed value is **−0.0852**. Quoted bare it overstates the
   deployable win by **4.4×**, and it is already in `V5_PLAN.md` §8 and `V5_IMAGINATION_SELECTION.md`
   §2.3 without the conversion. **This is the M12 relay failure class** — the surface travels, the
   denominator does not.

### 6.4 For `RETRACTION_LOG.md` — root-cause classes

- ⭐ **C-new: "a stratum win is not a deployable win — the selected fraction IS part of the number."**
  A conditional improvement's deployed value is `frac × Δ`. Reported without `frac`, −0.3754 reads
  4.4× larger than the −0.0852 it is worth. **Every conditional/gated result in this program should
  carry its firing rate.** Generalises to every "it works when X" claim.
- ⭐ **C-new: optimise the objective you are paid for, not its most legible correlate.** The canary was
  the legible target; the **utility** was the paid one. Aiming at the canary cost **3.6×**
  (−0.0383 vs −0.1397) *with identical features and identical folds*. Whenever a gate is proposed for
  an oracle stratifier, check whether the stratifier is even a good gate — here it was not (§1.4:
  the canary oracle recovers 9.3 % of available headroom).
- **C6 (confounded comparison), avoided by construction.** Without the `selected_frac` column and the
  incremental-vs-ungated column, **two pure-noise gates on v1 would have been written up as separated
  wins** (−0.2918 and −0.2761 vs as-trained). The pre-registration caught them.
- **C-II (unverified premise in a brief) — none found.** Every inherited number in the brief
  (−0.3754, −0.2918, 0.2645, 0.7085 → 0.3330, 22.7 %) **reproduced exactly** from the primary
  artifacts (S0b/S0c). Recorded because the previous two streams each caught one.

---

## 7. DELIVERABLE MANIFEST

**STAGED, NEVER COMMITTED, NEVER PUSHED.** All paths relative to
`TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-27-canary-proxy/`.

| artifact | where it lives | only one place? | note |
|---|---|---|---|
| `CANARY_PROXY.md` (this file) | `repo:` staged | no | pre-registration (§0) staged before any proxy was measured |
| `code/canary_proxy.py` | `repo:` staged | no | the whole instrument, 4 stages, CPU-only · md5 `93a97f4f…` |
| `raw/canary_proxy.json` | `repo:` staged | no | **every number in this document** · 203 kB · md5 `2b8b11fb…` |
| `raw/canary_proxy_s01.json` | `repo:` staged | no | S-tests + stage 1 alone, written before stages 2–3 existed (banking) · md5 `f2954cc7…` |
| `raw/canary_proxy_s012.json` | `repo:` staged | no | stage-2 sweep before the nested protocol was added · md5 `3e9ff46f…` |
| inputs: `v5_{v4,v1}_windows_reduced.pt` | `repo:` `…/2026-07-26-v5-imagination-selection/raw/` | no | **not copied** — read in place, unmodified |

**Nothing lives in only one place. No pod was contacted. No GPU was used. Nothing was launched,
restarted, committed or pushed. No steering file was edited.**

**Reproduce everything, no GPU, ~7 min:**
```
/c/Users/Admin/venvs/tanitad/Scripts/python.exe code/canary_proxy.py \
    --stage 0123 --out raw/canary_proxy.json
```
It refuses to adjudicate if any S-test fails (`raise SystemExit` before stage 1).

⚠️ **Dev-box parity note:** this analysis is **parity-independent** — it consumes the *persisted
per-window outputs* of the pod-side harness, which were produced on the canonical
`physicalai-train-e438721ae894` (2376 episodes, skip-hash `f09e44db`). The dev box's own local cache
(`14231cd29c74`) is never touched. No episode was re-selected.

**Suite status:** `taniteval` CI/rank tests **131 passed** (`pytest -q -k "ci or rank"`). This work
adds no code under `stack/` or `taniteval/`.
