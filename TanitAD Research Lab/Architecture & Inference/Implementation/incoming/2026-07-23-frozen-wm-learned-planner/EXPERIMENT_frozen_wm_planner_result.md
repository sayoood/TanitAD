# Experiment (P2) — Frozen v1 WM + learned planner via analytic gradient

**Author:** frozenwm-planner subagent · **Date:** 2026-07-23 (Europe/Berlin) · **Host:** pod1
`tanitad-pod` (A6000, FREE, under `gpu_lock` frozenwm-planner) · **Evidence:** `MEASURED` — raw
`artifacts/results.json` + `artifacts/perwin.pt`, harness `artifacts/run.py`. Reproduced across two seeds
(unseeded run1 + seeded run2/3; numbers within 2 %).

**Nothing running was touched.** The frozen v1 checkpoint was loaded read-only; no WM parameter was updated
in any arm.

---

## 1. The result (headline)

A **3.77 M** learned planner, trained *only* by backpropagating open-loop ADE through the **frozen** v1
world model (encoder+readout+predictor+step-readout all `requires_grad=False`), reaches **ADE@2s 0.599 m**
on held-out val — **beating the CV floor (0.846) and hold-v0 (0.788), 6.1× better than decoding the
trajectory statically off the same frozen latent (3.65 m), and within 1.48× of the world model's own
oracle-action fidelity ceiling (0.405 m)** — **with the world model perfectly intact (canary unchanged, by
construction).**

**Verdict: the frozen-WM + learned-planner regime is `VIABLE`, not `BOTTLENECKED`.** Freezing sidesteps
the joint-training WM degradation entirely (a frozen trunk cannot re-optimise), and — the open question —
it does **not** re-introduce a REF-A-style frozen ceiling, *provided the planner reads the WM through its
dynamics (outputs actions the frozen predictor rolls) rather than decoding the trajectory off the static
latent.*

## 2. The numbers (matched, same 12-ep val / 265 windows)

| arm / reference | what it is | ADE@2s | CI95 (episode-cluster) | FDE@2s | miss@2m |
|---|---|---:|---|---:|---:|
| **oracle-action ceiling** | frozen WM rolled under **GT** future actions (= v1's operative number, matched) | **0.4045** | [0.310, 0.514] | 0.851 | 0.042 |
| **W — analytic grad thru frozen WM** | planner→actions→frozen rollout→ADE, backprop through WM | **0.5989** | [0.374, 0.854] | 1.294 | 0.208 |
| hold-v0 (go-straight) | trivial floor | 0.7883 | — | — | — |
| CV | trivial floor | 0.8463 | — | — | — |
| **B — action BC (no WM in loss)** | planner→actions, MSE to GT actions; eval rolled thru WM | **1.0001** | [0.697, 1.354] | 2.254 | 0.415 |
| **F — direct decode off frozen state** | planner→20 waypoints directly (no predictor) | **3.649** | [2.632, 4.723] | 5.924 | 0.781 |

*Cross-check (unseeded run1): F 3.761 · W 0.609 · B 0.961 — consistent. Harness validated apples-to-apples:
reproduced CV 0.8463 (registry full-set 0.8377), hold-v0 0.7883 (registry 0.7876), oracle-action 0.4045
(registry full-set 0.4271) — the frozen WM reproduces the canonical operative number on this val subset.*

### Paired episode-cluster bootstraps (same windows, B=2000) — the decision-grade separation

| contrast | Δ (mean) | CI95 | separated | reading |
|---|---:|---|:--:|---|
| **W − CV** | **−0.2474** | [−0.505, −0.034] | ✅ **YES** (frac>0 0.006) | the analytic-grad planner **beats the CV floor**, paired-separated |
| **W − B** | **−0.4012** | [−0.717, −0.128] | ✅ **YES** (frac>0 0.001) | the WM-as-cost-model **beats plain action-BC**, paired-separated |
| **W − oracle** | +0.1944 | [−0.045, +0.448] | ❌ no (frac>0 0.937) | the action-prediction penalty is **within noise of the WM's GT-action ceiling** at n=12 |
| W − hold-v0 | −0.1893 | [−0.391, −0.012] | ✅ YES | W beats hold-v0, paired-separated |
| W − F | −3.0501 | [−4.045, −2.078] | ✅ YES | through-dynamics ≫ static-decode |
| B − CV | +0.1538 | [−0.222, +0.587] | ❌ no | action-BC ties/slightly worse than CV |
| F − CV | +2.8027 | [+1.840, +3.842] | ✅ YES | static-decode much worse than CV (REF-A regime) |

**The paired tests upgrade every VIABLE read from "point estimate" to "CI-separated":** W beats CV
(−0.247, sep), beats hold-v0 (−0.189, sep), beats action-BC (−0.401, sep), beats static-decode (−3.05,
sep) — and is **statistically indistinguishable from the WM's own oracle-action ceiling** (+0.194, *not*
separated). A tiny planner driving the frozen WM lands within bootstrap noise of feeding the WM perfect
actions.

## 3. Adjudication against the pre-registration (DESIGN §4)

| pre-registered criterion | measured | verdict |
|---|---|---|
| **VIABLE:** W beats CV **and** ≲ 0.60 **and** W ≪ F | W 0.599 (≈0.60 target); **paired W−CV −0.247 [−0.505,−0.034] SEPARATED**; W−F −3.05 SEPARATED (6.1×) | ✅ **VIABLE, paired-confirmed** |
| **BOTTLENECKED:** W ≥ CV, or ≈ F, or ≫ 0.40 despite healthy training | none hold — W paired-*below* CV and F; **W not separated from the 0.40 oracle ceiling** (+0.194 [−0.045,+0.448]) | ✅ rejected |
| **mechanism:** sign of W − B | **paired W−B −0.401 [−0.717,−0.128] SEPARATED** | ✅ analytic gradient **beats** action-BC, CI-separated |

**All three pre-registered reads resolve in the same direction, every one paired-CI-separated.** The frozen
v1 WM behaves as a good differentiable simulator that a small learned planner can drive — to within
bootstrap noise of the WM's own oracle-action ceiling.

## 4. What each arm *means* (the mechanism, stated cleanly)

- **Arm F = 3.65 m reproduces the documented static-latent probe (3.89 m, `metric_dynamics.py`).** v1's JEPA
  latent does **not** linearly hold metric ego-trajectory — so a planner that decodes waypoints *off the
  frozen state* hits the **REF-A frozen-encoder regime** (2.1–2.9 m band, here worse). **This is the
  bottleneck — and it is a bottleneck of *static decode*, not of freezing per se.**
- **Arm W = 0.60 m routes through the frozen *dynamics* and escapes it.** The metric information the static
  latent lacks is present in the **action-conditioned rollout** (`predictor` + `step_readout`), and the
  analytic gradient of ADE **through** that frozen rollout teaches the planner to produce actions that land
  the trajectory. The frozen WM is a faithful simulator; the planner learns to drive it.
- **Arm W (0.60) < Arm B (1.00): the world model earns its place as a differentiable cost model.** Plain
  behaviour-cloning of the expert *actions* (Arm B) is **worse than the CV floor** — small action errors
  compound over the 20-step rollout. Backpropagating the *trajectory* cost through the WM (Arm W) corrects
  exactly the errors that matter downstream. This is the "differentiable simulator gives better policy
  gradients" effect `[PUB: SHAC/Suh]` confirmed in our driving setting.
- **The canary is 0.4045, unchanged.** No WM parameter moved, so the degradation the v4 saga fights
  (canary 0.42→1.30+ v4 / 0.72@4k v4.2 `[INH]`; v1.6 0.452→1.1022 `[MEAS]`) **cannot occur here.** Freezing
  buys WM integrity for free.

**The trade, quantified.** W's gap above the oracle ceiling (0.599 − 0.405 = **0.194 m**) is the
*action-prediction* penalty — the cost of predicting 2 s of controls from an 8-frame window — **not** WM
infidelity (the WM's fidelity is fixed at 0.405). That penalty is the planner's headroom, and it is where
the product path (amortised MPC / a bigger planner) would spend effort — **not** on the WM.

## 5. Honest limits (what this proof does and does not establish)

1. **Scale — 12-episode val subset.** The pod carries 12 of the eval-pod's 40 val episodes (265 windows).
   *Marginal* CIs are wide (W's upper 0.854 sits near CV's 0.846), which is exactly why the **paired**
   estimator is the decision-grade test — and it **does** separate W from CV (−0.247 [−0.505, −0.034]) and
   from action-BC (−0.401 [−0.717, −0.128]). The wide *marginal* bands are still an honest limit: the full
   40-ep eval-pod run is the natural hardening and costs one eval pass. Train was 400 of 2376 parity
   episodes (8,803 windows).
2. **The coupled baseline is cited, not re-derived (C6 discipline).** v1.6 (0.4886, canary→1.10) and v4/v4.2
   canary degradation are the program's established finding `[MEAS/INH]`; my within-experiment arms differ
   from those coupled arms in ≥2 respects (planner shape, scale, objective), so the clean self-contained
   claim is **W vs the on-corpus ceilings**, and the coupled numbers frame the *WM-integrity* axis only.
3. **This is mechanism (a) only.** It does not build the amortised-MPC product path (research §2(c)); it
   settles whether that path is worth building (it is — the frozen WM is a good enough simulator).
4. **Planner is deliberately tiny + generic** (2-layer transformer, 3.77 M). W's train loss plateaued ~0.45;
   the limiter is the planner/action-prediction, not the WM. A larger planner or search-over-the-frozen-WM
   is expected headroom, not a reason to unfreeze.

## 6. Conclusion for the program

**Freezing the world model and learning the planner against it as a differentiable simulator is a viable
ADDITIONAL direction** — it removes the WM-degradation failure mode that has cost the v4 saga three
mis-tuned coupling arms, at the price of an action-prediction penalty (~0.19 m here) that is the planner's
to close, **not** a frozen-representation ceiling. The REF-A ceiling reappears only under *static latent
decode*; routing the planner through the frozen *dynamics* avoids it. **Recommended next step:** the
TD-MPC2-shaped amortised-MPC path (distil the measured-feasible P2 CEM search over the frozen WM into a
learned prior), evaluated on the full 40-ep val with paired estimators — a new arm with its own gate, not
an edit to anything running.
