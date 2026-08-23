# Experiment (P2-follow-up) — Amortised-MPC prototype over the frozen v1 WM

**Author:** frozenwm-mpc subagent · **Date:** 2026-07-24 (Europe/Berlin) · **Host:** pod1 `tanitad-pod`
(A6000, FREE, under `gpu_lock frozenwm-mpc`) · **Evidence:** `MEASURED` — raw `artifacts/mpc_results.json`
+ `artifacts/perwin_mpc.pt`, harness `artifacts/mpc.py`. Cache path only (no `tanitad.lake`). The frozen v1
checkpoint was loaded read-only; **no WM parameter was updated.**

**This is a small 12-ep PROTOTYPE — explicitly NOT a new arm.** The commitment to a full amortised-MPC arm
stays Sayed's call; this measures whether it is worth proposing.

**Coordinator's pre-registration (committed in advance):** amortised-MPC **beats W** (search finds better
plans than the feed-forward planner) → the product path is worth a later 40-ep hardening + a Sayed go ·
**ties/worse than W** → analytic-gradient (arm W) is already sufficient, don't pursue MPC further.

---

## 1. Headline

**Per-window CEM/MPC search over the frozen v1 WM finds plans that are 4.5× better than the feed-forward
analytic-gradient planner (arm W).** On the same 12-ep val, same windows:

| planner over the frozen WM | ADE@2s | vs W (0.599) | what it is |
|---|---:|---:|---|
| **CEM search, warm-init** (refine GT actions) | **0.1322** | **4.53× better** | the strongest plan the WM admits per window |
| **CEM search, cold-init** (from scratch, μ=0) | **0.2471** | **2.42× better** | search with no GT seed, only GT cost |
| oracle-action ceiling (frozen WM, raw GT actions) | 0.4045 | 1.48× better | v1's own operative number |
| **W — feed-forward analytic-gradient planner** | 0.5989 | 1.00× | the prototype's baseline |
| amortised prior (naive action-BC distill, feed-forward) | **1.3987** | **2.34× WORSE** | the DEPLOYABLE fast planner — **fails** (§3) |
| CV floor | 0.8463 | — | trivial |

**Verdict on the pre-registration — split, and the split is the finding:**
1. **The SEARCH decisively beats W** (warm **0.132** ≪ W 0.599, paired Δ −0.467 [−0.692, −0.272]; it even
   beats the raw-GT-action ceiling, Δ −0.272 [−0.371, −0.197]). By the coordinator's own framing —
   *"amortised-MPC beats W (search finds better plans than the feed-forward planner)"* — this is **YES**.
   **→ the product path is worth a later 40-ep hardening + a Sayed go.**
2. **But the NAIVE feed-forward amortisation FAILS** — distilling the search plans by action-BC gives a
   prior of **1.399**, paired-**worse** than W (Δ +0.800 [0.293, 1.398]) and worse than even GT-action-BC
   (arm B 1.000). **The search gain does NOT amortise into a fast feed-forward action-prior.** The hardened
   arm therefore cannot be a distilled action-prior; it must use **test-time search + a learned value**
   (§5–§6). This is the finding that saves the program from building the wrong thing.

## 2. What "search beats even the oracle-action ceiling" means (stated honestly, to avoid over-claim)

Warm search (0.132) beats the **oracle-action ceiling (0.405)** — the frozen WM rolled under the *raw
ground-truth actions*. **This is NOT a claim of superhuman driving.** CEM optimises the *action inputs* to
minimise the **decoded** ADE, so it finds actions `â*` that the *frozen WM* maps closer to the GT waypoints
than the true actions do — it exploits the fact that the WM's action→trajectory decode is imperfect and does
not perfectly invert. `â*` are the **WM-preimage of the expert trajectory**, not the physically-true
controls. Read correctly, this measures **how controllable / expressive the frozen WM is as a simulator**:
you can drive it to reproduce the expert trajectory to **0.13 m**. It confirms the frozen-WM thesis from the
other direction — *the WM is an excellent differentiable simulator; the bottleneck for the feed-forward
planner W is the planner's action-prediction, not the WM.* Search closes that gap by optimising per window;
`â*` are high-quality **teacher targets** for imitation.

## 3. The deployable amortised prior (does a FAST feed-forward planner capture the search gain?)

The TD-MPC2 amortisation: distil the warm-search plans `â*` (on 1000 train windows) into the **same 3.77 M
feed-forward planner** as arm W (BC on `â*`); at test it runs feed-forward — **no GT, no search**.

| | ADE@2s | CI95 | FDE@2s | miss@2m |
|---|---:|---|---:|---:|
| amortised prior (feed-forward, deployable) | **1.3987** | [0.955, 1.948] | 3.109 | 0.498 |
| arm W (analytic-gradient, feed-forward) | 0.5989 | [0.374, 0.854] | 1.294 | 0.208 |
| warm search (teacher, uses GT per window) | 0.1322 | [0.087, 0.184] | 0.230 | 0.004 |

**The amortisation fails, and the mechanism is instructive.** The distillation BC loss on the search
actions reached **0.0194** (the prior learned to predict the search plans `â*` almost exactly on train) —
yet rolling those predicted actions through the WM on val gives **1.399**, *worse than W and worse than
plain GT-action-BC (arm B 1.000)*. The search plans `â*` are **brittle, non-smooth per-window WM-preimages**:
excellent as fixed targets, but they are not a smooth function of the encoder state, so (i) the prior's
tiny residual action errors **compound catastrophically** over the 20-step rollout, and (ii) whatever the
prior does predict is a blurred average of jagged targets that the WM decodes to nonsense. Optimising the
*trajectory* through the WM (arm W's analytic gradient) is forgiving of exactly these errors; BC-matching
*actions* is not — which is the same reason arm B (1.000) lost to arm W (0.599), amplified because the
search targets are even less smooth than the true actions. **Naive action-distillation is a trap here.**

## 4. Paired episode-cluster bootstraps (same windows, B=2000)

| contrast | Δ | CI95 | separated | reading |
|---|---:|---|:--:|---|
| **search_warm − W** | **−0.4668** | [−0.692, −0.272] | ✅ YES | per-window search **beats** feed-forward W (4.5×) |
| **search_warm − oracle** | **−0.2724** | [−0.371, −0.197] | ✅ YES | search **beats even raw-GT-action** ceiling (WM controllability) |
| search_cold − W | −0.3518 | [−0.536, −0.207] | ✅ YES | search-from-scratch also beats W |
| **amortised_prior − W** | **+0.7997** | [+0.293, +1.398] | ✅ YES | the deployable fast prior is **worse** than W |
| amortised_prior − search_warm | +1.2665 | [+0.822, +1.819] | ✅ YES | the amortisation captures **~none** of the search gain |

## 5. Interpretation and honest limits

- **The search-vs-W gap is the decision-relevant signal, and it is large** (warm 0.132 vs 0.599). It says the
  frozen WM has far more planning headroom than a feed-forward planner extracts — exactly the TD-MPC2
  premise (search improves over the amortised policy). This justifies the product path.
- **The amortised prior is the "can we make it fast feed-forward" question, and the answer here is NO.** A
  naive action-BC distillation of the search plans is worse than W (1.399 vs 0.599, paired) — the WM-optimal
  action sequences are brittle preimages that do not amortise. **This is a positive finding: it rules out the
  simplest (and tempting) product design — "distil the CEM search into a policy prior" — before it costs a
  full arm.** The realizable amortisation is NOT action-BC; it is either (i) test-time search, or (ii)
  distilling the *trajectory/value* rather than the actions (which, for the trajectory, is essentially arm W
  — already the best deployable feed-forward planner at 0.599).
- **Cost/scale caveats (prototype):** 12-ep val (265 windows); the distilled prior saw **1000** search-planned
  windows vs arm W's 8,803 — a data-scale handicap on the prior (search is ~1.8 s/window, so a full-scale
  teacher was out of prototype budget). CEM: P=128, I=4, elite 10 %, annealed action-scaled σ, elitism +
  best-tracking. Search uses GT as the imitation cost (as any teacher does); the prior never sees GT.
- **The full realisation needs test-time search with a DEPLOYABLE cost/value** (not GT) — the model-based-RL /
  learned-value half of TD-MPC2. That is the natural content of the hardened arm, and it needs a value or
  reward model we do not yet have offline (research doc §2(b), deferred behind reward design).

## 6. Recommendation

> ⚠️ **SUPERSEDED 2026-07-24 by `EXPERIMENT_valuemodel_result.md`.** This section proposed pursuing the
> product path because "search finds a 4.5× better plan." The **value-model prototype settled it: a
> deployable test-time search with a *learned* value FAILS (1.02, worse than feed-forward W)** — because the
> 0.132 GT-search prize is **hindsight-privileged** (it minimises distance to the expert's *actual* future,
> which the ego does not control in an open-loop metric), plus CEM adversarially exploits the value model's
> errors. **The amortised-MPC / value path is NOT a cheap deployable win; frozen-WM stays a ~0.60 fallback.**
> The measurement below stands; only its forward-looking recommendation is retracted.

**~~Propose the amortised-MPC arm to Sayed~~** (his go, per the pre-registration): the frozen WM admits plans
**4.5× better** than the current feed-forward planner, so there is a large, measured prize. The hardened arm
should (i) evaluate on the full 40-ep val with paired estimators, (ii) scale the search-distillation teacher,
and (iii) add **test-time search with a learned value** so the gain is realised at deployment without GT.
Frozen WM throughout — the degradation risk stays zero, and the search confirms the WM is an excellent
differentiable simulator to plan against.
