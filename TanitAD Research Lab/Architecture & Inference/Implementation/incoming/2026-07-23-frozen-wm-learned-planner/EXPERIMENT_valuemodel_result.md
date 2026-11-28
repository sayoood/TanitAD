# Experiment — learned value-model + CEM search over the frozen v1 WM (the crux)

**Author:** frozenwm-valuemodel subagent · **Date:** 2026-07-24 (Europe/Berlin) · **Host:** pod1
`tanitad-pod` (A6000, FREE, under `gpu_lock frozenwm-valuemodel`) · **Evidence:** `MEASURED` — raw
`artifacts/valuemodel_results.json` + `artifacts/perwin_value.pt`, harness `artifacts/value.py`. Cache path
only (no `tanitad.lake`). Frozen v1 WM read-only; **no WM parameter updated. pod2/pod3/eval untouched.**

**The crux for the frozen-WM-as-flagship-contender call.** CEM search with the **GT future** as its cost hits
**0.132** (4.5× better than feed-forward W 0.599). A DEPLOYABLE contender needs that search at **test time
WITHOUT the GT future** — i.e. a **learned value/cost model** replacing the GT cost. This is the only route
that could make frozen-WM beat from-scratch (~0.45).

**Pre-registration (committed in advance):** learned-value-search **≤ 0.45** → frozen-WM + search-via-
learned-value **IS a viable flagship CONTENDER** (a real Sayed decision) · **≈ 0.599** → the value-model's
error re-introduces the aleatoric gap, frozen-WM stays the **fallback** · in-between → report the fraction
closed.

**Method.** Same frozen v1 WM, 12-ep val, same cache. (1) **Collect** on-distribution training data: run
cold-init CEM with the *GT* cost on 350 train windows, logging every explored candidate's (window,
rolled-trajectory, TRUE cost). (2) **Train** a value model `V(z_state, rolled-waypoints) → predicted cost`
(L1 regression on the true cost). (3) **Test**: run cold-init CEM on the val windows ranking candidates by
the **learned V** (no GT), and measure the selected plan's **TRUE** open-loop ADE. Paired vs W (0.599), the
GT-search ceiling (warm 0.132 / cold 0.247), and the oracle (0.4045).

**Diagnostic that explains the outcome — within-window rank-correlation** between V's predicted cost and the
true cost across candidates. A value model can only learn `E[cost | state]` (the *expected* future); it
cannot know *this* window's actual future. If the correlation among *good* candidates is low, V cannot do
the per-window future-matching that lets GT-search reach 0.132 — the aleatoric wall — and CEM will instead
**adversarially exploit V's errors** (pick candidates V wrongly rates as cheap).

---

## 1. Result

| planner over the frozen WM (12-ep val) | ADE@2s | CI95 | vs W (paired) |
|---|---:|---|---|
| GT-search, warm (uses actual future) — **ceiling** | 0.1322 | [0.087, 0.184] | — |
| GT-search, cold (uses actual future) | 0.2471 | [0.149, 0.375] | — |
| oracle-action (frozen WM, GT actions) | 0.4045 | [0.310, 0.514] | — |
| **W — feed-forward planner** | 0.5989 | [0.374, 0.854] | — |
| CV floor | 0.8463 | — | — |
| **V-search (learned value, NO GT) — deployable** | **1.0162** | [0.809, 1.273] | **+0.417 [+0.237, +0.605] SEP (WORSE)** |

**Diagnostics:** value-model held-out cost-prediction L1 = **0.115 m** · within-window rank-corr(V, true
cost) = **0.613** · matched cold-GT-search on the same val = **0.248** (reproduces the mpc 0.247 — harness
validated).

**Verdict: `worse-than-0.599` — the learned value does NOT close the gap; naive value-search is WORSE than
feed-forward W (1.02 vs 0.599, paired-separated) and nowhere near the ≤0.45 bar.** Frozen-WM + learned-value-
search is **NOT** a viable flagship contender by this route; **frozen-WM stays the fallback (feed-forward
W = 0.599 remains its best deployable planner).**

## 2. Paired episode-cluster bootstraps (same windows, B=2000)

| contrast | Δ | CI95 | separated | reading |
|---|---:|---|:--:|---|
| V-search − W | +0.4173 | [+0.237, +0.605] | ✅ | **worse than feed-forward** — the value/search adds nothing, it hurts |
| V-search − cold-GT-search | +0.7682 | [+0.587, +0.988] | ✅ | the entire gap to the search ceiling is V's imperfection |
| V-search − oracle | +0.6116 | [+0.432, +0.850] | ✅ | far above the action ceiling |
| V-search − CV | +0.1699 | [−0.162, +0.524] | ❌ | barely around the CV floor |

## 3. Interpretation — two compounding failures, and what the 0.132 number really is

**(a) The aleatoric wall.** A value model supervised by the true cost can only learn `E[cost | state]` — the
*expected* cost over the future distribution. It cannot know *which* future this window will bring, because
that information is simply not in the state (the driver's upcoming intent). So V ranks candidates by expected
cost, whose minimiser is the conditional-mean-future trajectory — **exactly what feed-forward W already
outputs.** The within-window rank-corr (0.613) looks moderate, but it is driven by the *easy* rejection of
wild bad candidates; among the *good* candidates — where the GT-search's 0.132 advantage lives — V cannot
tell which one matches the actual future.

**(b) Adversarial exploitation — why V-search is even WORSE than W.** CEM *optimises against V*. With ~180 k
candidates it finds the ones V **wrongly** rates cheap (V's blind spots / off-distribution corners) — the
same failure class as the amortised-MPC distill-trap, now on the value side. So the search actively selects
V's errors, landing at **1.02 — worse than not searching at all.** The whole gap between cold-GT-search
(0.248, where V *is* the perfect cost) and V-search (1.02) is this exploitation of the learned value.

**(c) What the 0.132 GT-search number actually is — and the honest update it forces.** GT-search minimises
distance to **the expert's actual future trajectory**, which the ego does **not** control in an open-loop
imitation eval — it is a *fixed demonstration*. So the W(0.599)→search(0.132) gap is **the gap between
prediction and hindsight**, not a controllable "planning headroom." The missing ingredient is *the actual
future*, and no deployable method (feed-forward, bigger planner, distilled action-prior, or learned-value-
search) can supply it. **This updates the amortised-MPC note** (`EXPERIMENT_amortised_mpc_result.md`, which
read "search finds better plans → product path worth pursuing"): the value-model prototype settles it — the
product path (test-time search + learned value) does **not** work for open-loop imitation, because the prize
was hindsight-privileged. *(In a genuine CLOSED-LOOP setting the ego controls the future and value-search
could pay off — but that is a different eval needing a simulator/reward, not this metric.)*

**The complete frozen-WM deployable picture (all MEASURED, this folder):** feed-forward W **0.599** ·
bigger planner **flat ~0.60** (capacity not the lever) · distilled search-action prior **1.40** (brittle) ·
learned-value-search **1.02** (exploited). **Every deployable route hits the ~0.60 aleatoric wall or worse;
only future-peeking GT-search reaches 0.132.**

## 4. Honest limits

- **12-ep val, 350-window collect** — a prototype scale; a full contender would scale the collect and the
  value model, and likely use an **ensemble/conservative value** to blunt the adversarial-exploitation
  failure (TD-MPC2 uses value ensembles + a policy prior for exactly this). This prototype tests the
  *straightforward* learned-value-search; a negative here does not prove *no* value method can help, but it
  measures whether the cheap/obvious one does.
- **The aleatoric wall is structural, not a training artifact:** a value model supervised by the actual
  cost can only learn its *expectation* over futures; the residual — which future actually occurs — is not
  in the input, so no amount of value-model capacity recovers the GT-search's per-window future-matching.
- **Cold-init CEM** (no GT seed, since W's weights were not saved) is the deployable setting; the GT-search
  ceilings are reported at both cold and warm init for a fair frame.

## 5. Recommendation for the contender call

**Frozen-WM is a solid cheap FALLBACK (~0.60 feed-forward, degradation-free), NOT a search-matching flagship
contender.** The value-model route — the only one that could have made it a contender — **fails** (1.02,
worse than feed-forward), from a structural aleatoric wall plus adversarial exploitation of the learned
value. The 0.132 GT-search number was **hindsight-privileged** (peeking at the expert's actual future in an
open-loop metric), not a controllable planning gain, so no deployable method reaches it. **Do not commit
frozen-WM as the flagship on the expectation of closing the gap** — across four deployable routes it does
not. If a value-based planner is still wanted, it belongs in a **closed-loop** setting (ego controls the
future, needs a simulator/reward) with **TD-MPC2-grade machinery** (value ensembles + conservative targets +
a policy prior to blunt exploitation) — a real project, not a cheap win, and out of scope for an open-loop
imitation contender. **Frozen-WM's value stands where the earlier notes put it: a safe, cheap ~0.60 fallback
if from-scratch stumbles.**
