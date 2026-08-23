# Experiment — bigger feed-forward planner on the frozen v1 WM (contender evidence)

**Author:** frozenwm-bigplanner subagent · **Date:** 2026-07-24 (Europe/Berlin) · **Host:** pod1
`tanitad-pod` (A6000, FREE, under `gpu_lock frozenwm-bigplanner`) · **Evidence:** `MEASURED` — raw
`artifacts/bigplanner_{v2,large,mlp}.json` + `artifacts/big.py`. Cache path only (no `tanitad.lake`).
Frozen v1 WM read-only; **no WM parameter updated. pod2/pod3/eval untouched.**

**The question for Sayed's frozen-WM-as-flagship-CONTENDER call.** The feed-forward analytic-gradient planner
**W = 0.599** vs CEM search over the same frozen WM = **0.132** (4.5×) → "the planner is the headroom." Is
that gap a **feed-forward CAPACITY limit** (a bigger planner closes it) or does it fundamentally need
**test-time search** (the distill-trap — not cheaply deployable)?

**Pre-registration (committed in advance):** bigger feed-forward planner **≤ 0.45** → viable DEPLOYABLE
flagship contender (no value model) · stays **~0.599** → architecturally limited, only test-time search
reaches the ceiling (→ value-model path / frozen-WM stays the cheap fallback).

---

## 1. Headline

**Feed-forward capacity is NOT the lever. Scaling the planner 11× in W's own head family is a FLAT line —
W 0.599 → mlpbig(30.8 M) 0.601 → mlpwide(42.6 M) 0.599 — the better recipe ties W (0.588), and bigger
*query-decoder* planners OVERFIT and do worse (0.82–0.86). No feed-forward variant approaches the ≤0.45
bar; they saturate at ~0.60.**

**Verdict: `~0.599` outcome → feed-forward is architecturally limited.** The W→search gap is **not** closed
by feed-forward capacity or recipe; matching the 0.132 search ceiling fundamentally needs **test-time
search** (a learned value/cost — the distill-trap already showed a fast action-prior fails). **Frozen-WM +
feed-forward is a solid cheap FALLBACK (~0.60, paired-beats every trivial floor), NOT a search-matching
flagship contender.**

## 2. The sweep (12-ep val, 265 windows, same frozen WM, same 8,803-window train, analytic-gradient)

| planner | head | params | ADE@2s | CI95 | train loss | paired vs W |
|---|---|---:|---:|---|---:|---|
| **W** (baseline, 3k steps) | MLP | 3.77 M | **0.5989** | [0.374, 0.854] | ~0.50 | — |
| **wplus** (W arch, better recipe) | MLP | 4.0 M | **0.5878** | [0.377, 0.823] | 0.365 | −0.011 [−0.039, +0.014] **ns** |
| **mlpbig** (8× — d768, 5 enc) | MLP | **30.8 M** | **0.6007** | [0.393, 0.831] | 0.374 | +0.002 [−0.032, +0.029] **ns** |
| **mlpwide** (11× — d1024, 4 enc) | MLP | **42.6 M** | **0.5994** | [0.391, 0.836] | 0.392 | +0.001 [−0.041, +0.035] **ns** |
| med (d512, 4 enc + 2 dec) | query | 18.9 M | 0.8204 | [0.479, 1.250] | 0.527 | +0.222 [+0.075, +0.428] **SEP (worse)** |
| large (d768, 5 enc + 3 dec) | query | 52.8 M | 0.8565 | [0.504, 1.316] | 0.510 | +0.258 [+0.105, +0.488] **SEP (worse)** |
| — CEM search ceiling (not feed-forward) | — | — | 0.1322 | [0.087, 0.184] | — | — |
| — oracle-action (frozen WM, GT actions) | — | — | 0.4045 | [0.310, 0.514] | — | — |

**The clean capacity curve is the MLP-head family** (W's own head, size varied without the query-decoder
confound), and it is **FLAT: W 0.599 → wplus 0.588 → mlpbig 0.601 → mlpwide 0.599.** Scaling **11×** (3.77 M
→ 42.6 M) and a longer warmup+cosine recipe move ADE by **< 0.015 m, none paired-separated.** Capacity is
decisively not the lever.

**The query-decoder bigger planners (med, large) are strictly WORSE, paired-separated** — a bigger, harder-
to-regularise head **overfits** on 8,803 windows (train loss ≈ W's while val degrades to 0.82–0.86). So
"just make the planner bigger" not only fails to help, the naive bigger architecture actively hurts.

## 3. Paired episode-cluster bootstraps — the best feed-forward planner (mlpbig, 30.8 M)

| contrast | Δ | CI95 | separated | reading |
|---|---:|---|:--:|---|
| mlpbig − W | +0.0018 | [−0.032, +0.029] | ❌ | **8× capacity = no change** |
| mlpbig − CV | −0.2456 | [−0.527, −0.017] | ✅ | still beats the CV floor |
| mlpbig − oracle | +0.1962 | [−0.023, +0.421] | ❌ | within noise of the ceiling — same as W |
| mlpbig − search | +0.4686 | [+0.291, +0.669] | ✅ | far above the search ceiling (unreachable feed-forward) |

## 4. Interpretation — why feed-forward saturates at ~0.60

A feed-forward planner predicts the **2 s future action sequence from the 8-frame past window**. Its error
above the oracle-action ceiling (0.4045) is dominated by the **aleatoric uncertainty of the future** — the
driver's upcoming intent is not determined by the past window, so a deterministic planner regresses toward
the mean future and pays an irreducible ADE. **W (3.77 M) already saturates this**; 8× more capacity cannot
reduce *unknowable* future uncertainty, so mlpbig ties W. The CEM search escapes it only because it
optimises **per window against the actual future as its cost** (a privileged, test-time signal) — which is
exactly why the search 0.132 is **not** a feed-forward-reachable number and why distilling it into a fast
action-prior failed (the distill-trap, `EXPERIMENT_amortised_mpc_result.md`: prior 1.40).

**This settles the contender question:** the W→search gap is a **test-time-search vs amortised-policy gap**
(the classic model-based-planning gap), **not** a feed-forward-capacity gap. Closing it deployably needs a
**learned value/cost model** so search can run at test time without the GT future — the model-based-RL
half of TD-MPC2, which needs a reward/value we do not yet have offline.

## 5. Recommendation for the contender call

- **Frozen-WM + feed-forward planner is NOT a search-matching flagship contender** — feed-forward saturates
  at **~0.60** regardless of capacity (8× → no change) or recipe. It is a **solid, cheap, degradation-free
  FALLBACK** (~0.60, paired-beats CV/hold-v0, WM canary untouched) — worth keeping as the safety net if
  from-scratch stumbles, but it will not reach the search ceiling feed-forward.
- **The only path to the 0.132 ceiling is test-time search with a learned value** (not a bigger policy).
  That is a real, promising direction (the search prize is 4.5×) but it is a **new arm with a value model**,
  not a cheap deployable win — and it is the honest thing to tell Sayed before committing frozen-WM as the
  flagship.

## 6. Honest limits

- **12-ep val, 8,803-window train** — same setup as the W prototype, so the *capacity comparison is clean*
  (only the planner varies). Absolute numbers inherit the prototype's scale caveats; a full-40-ep read is
  data-limited on the eval pod (separate note).
- **Data-scaling caveat (the one residual confound):** the *query-head* bigger planners overfit on 8,803
  windows, so their degradation is partly data-limited. But the **clean MLP-head result is not an
  overfitting artifact** — mlpbig's train loss (0.374) is *not* collapsed below W's and its val simply
  **ties** W; a bigger MLP planner neither fits train much better nor generalises worse, it **plateaus**.
  That plateau — not a degradation — is the evidence that capacity is not the lever. A definitive
  "architecturally impossible even with more data" claim would need a data-scaling sweep; the actionable
  contender conclusion ("capacity/recipe do not cheaply close it") holds regardless.
- **≤0.45 (near oracle) was the right bar**, not the 0.132 search number: a feed-forward planner's best case
  is predicting the *true* actions → the WM's own 0.4045; the 0.132 is a per-window search-privileged number.
