> Build commissioned & greenlit by Sayed, 2026-07-19 (V3_HIERARCHICAL_PLANNING_DESIGN §8 P2 de-risk).
> Status: **measured result** — CEM planner + cost run over the FROZEN v1 flagship world model on physicalai val. Nothing trained; nothing committed. Eval pod (`tanitad-eval`, A40).
> Decisive gates: **G1 (planning > heads) PASS** · **G4 (closed-loop drift < head) PASS**. Verdict: the v3 planning thesis is validated at zero training cost, with a crisp, honest lateral ceiling that scopes P3/P4.

# P2 — CEM planning over the frozen v1 world model (training-free v3 de-risk)

**The question (spec §0/§8):** should the driving decision come from *planning* — rolling alternative action sequences through the world model and scoring their predicted consequences against a goal — rather than from a *supervised trajectory head* (which degenerates: v1's tactical head is **3.38 m** ADE@2s, worse than constant-velocity's 0.82 m)? P2 answers this **before training anything**: a CEM planner + a hand-built cost drive the frozen v1 flagship's proven operative world model. A win here validates the entire v3 architecture pivot at zero cost.

**What was built:** a new module `taniteval/planner_p2.py`. It loads the frozen v1 flagship (`flagship-30k`, step 29999, action_dim 3 `[steer,accel,v0]`) and its grounded step-readout, mints a target-speed label offline, and runs Cross-Entropy-Method search over future action sequences — each candidate rolled through the **exact gate operative rollout** (`metric_dynamics.rollout_decode`: encode window → predictor K steps under the action seq → per-step metric Δpose → SE(2) accumulate; 0.452 m with true actions). Nothing in the world model or readout is trained. The planner is *evaluated*, not fit.

---

## 1. Method (tight)

| Piece | Instantiation |
|---|---|
| **Frozen WM** | `flagship-30k` operative predictor + `grounding.step['op']` step-readout, loaded via `loaders.load` (strict). The 0.427/0.452 m grounded rollout, reused verbatim. |
| **Decision variable** | future action sequence `fa` = [steer, accel] × 20 steps (2 s @ 0.1 s); v0 channel = observed current speed, held constant (leakage-safe, matches every trainer). Open-loop keeps the observed last action fixed (apples-to-apples with the true-action rollout); closed-loop lets the planner emit `a0` directly. |
| **VTARGET (offline label)** | per window, 85th-pct of the future speed over the next 10–20 s, dropping steps braking harder than 1.5 m/s² (free-flow only); falls back to current speed when the free-flow sample < 3 s. Provenance: **kinematic** (no VLM sign-read pass on the eval pod → v_source is an honest gap). 94.2 % of windows got a valid free-flow target. |
| **Proposal set** | v1 has no multi-mode anchored decoder (that is v2), so per the spec fallback: a coarse **steer×accel constant-action grid** (5 steer × 3 accel-around-reach + coast = 16 seeds) **+ the v1 tactical head's own 0.5 s control as one learned proposal-prior seed**. CEM inits from the best seed per window. |
| **CEM (spec M6)** | N=64 samples, 3 iterations, elite-8, per-window Gaussian over the 20×2 action tensor, clamped to the action envelope (\|steer\|≤0.03, \|accel\|≤2.5). Fully batched over windows×samples. |
| **Cost J** | `w_v·(v̂−v_target)² + w_c·(accel²+jerk²) + w_s·steer_rate² − w_p·progress`. v̂ = planned trajectory's step-speed. **gap/TTC barrier SKIPPED (v0)** — our front-cam+pose data carries no lead-agent boxes (per the spec's "skip gap term v0"). Weights `(1.0, 0.1, 50, 0.02)` **engineered from physical scales, NOT fit to GT ADE** (that would make G1 circular); a 3×3 sensitivity sweep confirms robustness (§2.5). |
| **Protocol** | physicalai val, 40 eps open-loop (880 windows) / 20 eps closed-loop; window 8, stride 8; CI = 8-split episode-disjoint jackknife (bench protocol). All baselines recomputed in the same pass → apples-to-apples. |

---

## 2. Results

### 2.1 Open-loop planner ADE@2s — the G1 test (880 windows, 40 eps)

| Path | ADE@2s (m) | vs planner |
|---|---|---|
| **CEM planner (this build)** | **0.893 ± 0.114** | — |
| Tactical head (v1, the head being challenged) | 3.150 (in-run) · 3.38 (canonical) | planner **−2.26 m** |
| Constant-velocity (trivial floor) | 0.825 | planner +0.07 m |
| Operative rollout, TRUE actions (WM ceiling) | 0.452 | planner +0.44 m |

**G1 — does the plan beat the tactical head?** head − planner = **+2.257 ± 0.329 m, CI-separated → PASS.** The training-free planner cuts the head's error by **72 %** (3.15 → 0.89 m). Against the canonical 3.38 m head the margin is larger still.

The planner does **not** beat constant-velocity overall (0.893 vs 0.825) and sits at ~2× the true-action operative ceiling (0.452). Both facts are explained — and turned into signal — by the decomposition below.

### 2.2 Where the planner wins and loses — straight vs curved

| Stratum | n | Planner | Operative (trueA) | CV | Head |
|---|---|---|---|---|---|
| Straight (< 5° net heading) | 634 (72 %) | **0.564** | 0.393 | 0.439 | 3.297 |
| Curved (top-10 % curvature) | 89 (10 %) | **2.114** | 0.484 | 2.426 | 3.344 |

On **straight** windows the planner (0.564 m) is within 0.17 m of the true-action WM ceiling and crushes the head. On **curved** windows it degrades to 2.11 m — yet still beats CV's 2.43 m (the steer-grid proposals capture *some* curvature that CV's zero-yaw path cannot), while the true-action rollout stays at 0.48 m.

### 2.3 The honest signature — the residual is *lateral*, and it is expected

- Planner error decomposes to **long-RMSE 1.41 m / lat-RMSE 1.97 m → only 34 % of the 2 s squared error is longitudinal** (66 % is lateral / cross-track).
- Speed-decoupled path-geometry cross-track RMSE = **0.445 m**; longitudinal speed-bias **+0.47 m/s** (mild over-prediction, consistent with the known high-speed longitudinal weakness and the 85th-pct target).
- **v_target tracking:** planned speed lands **1.03 m/s** from the minted target vs GT's own 1.54 m/s — the planner tracks the strategic target *better than the log does*, confirming the longitudinal cost works as designed.

This is the mechanism, not a surprise: **the P2 cost is longitudinal + comfort + progress only — it carries no lateral/route/goal term** (the strategic goal module is P3). So the planner nails longitudinal control and defaults laterally to the smoothest low-curvature option its proposals + WM allow. The lateral residual *is the measurement of what P3/P4 must add*.

### 2.4 Closed-loop drift — the G4 test (imagination-in-the-loop, 20 eps, 221 windows)

The imagination-in-the-loop harness (`closedloop.py`) is reused verbatim; only the PLAN step is swapped — the tactical head is replaced by **per-tick CEM** (a lighter N=48×2-iter budget for the 20-tick loop; smoke-checked that per-tick vs every-0.2 s replanning is numerically equal), and because the planner emits actions, `a0` is executed directly (no pure-pursuit inversion the head needs).

| Metric | CEM planner | v1 head baseline |
|---|---|---|
| closed-loop ADE@2s (m) | **1.038 ± 0.202** | 1.685 ± 0.098 |
| closed-loop FDE@2s (m) | **2.194 ± 0.455** | 3.530 |
| divergence rate (> 5 m @2s) | **8.7 % ± 4.6** | 22.2 % |
| reference: true-action rollout | 0.424 | 0.452 |

**G4 — closed-loop drift < the 1.685 m head baseline? PASS.** The planner drifts **38 % less** (1.038 vs 1.685 m, CI-separated), ends **38 % closer** at 2 s (FDE 2.19 vs 3.53 m), and **diverges 2.5× less often** (8.7 % vs 22.2 %) — a genuine stability win: smooth v_target-tracking actions compound far more gracefully than the head's erratic waypoints. The residual gap to the true-action rollout (0.42 m) is again the compounding cost of the lateral-blind cost + imitation-era WM, not instability.

### 2.5 Weight sensitivity (G1 robustness — weights not fit to GT)

Across a 3×3 sweep of the comfort/progress weights (w_c ∈ {0.05, 0.1, 0.2} × w_p ∈ {0.01, 0.02, 0.04} — a 4× range, 8-ep subset), the planner ADE@2s moves only **0.647 → 0.669 m (a 3.4 % swing) and beats the head in all 9 configs** (head 3.134 m, weight-invariant). The center weights (0.1, 0.02) sit mid-range, not cherry-picked. **G1 is not a tuning artifact** — the planner-beats-head verdict is invariant to the cost weights within a 4× band.

---

## 3. Verdict on the decisive gates

- **G1 (planning > heads): PASS, decisively and CI-separated.** A CEM planner over the *frozen* v1 world model, with zero training, beats the supervised tactical head by 2.26 m (72 % error reduction). The core v3 claim — heads degenerate, planning-over-a-world-model recovers the decision — holds on today's checkpoint.
- **G4 (closed-loop drift < head 1.685 m): PASS.** 1.038 ± 0.202 m — 38 % below the head, with 2.5× lower divergence (8.7 % vs 22.2 %). The planner is both more accurate *and* more stable in the loop (§2.4).

## 4. The v3 thesis — what P2 proves and what it scopes

**Proven at zero training cost:** the world model v1 *already* contains a good enough action→consequence map that a hand-built cost + CEM turns it into a driver that dominates the supervised head. The head is not the world model's limit — it is a lossy readout of it, and planning bypasses that loss. This is the single result the v3 pivot was betting on, and it lands.

**Honestly scoped (the frozen-v1 ceiling):** planning-over-v1 does **not** yet beat constant-velocity open-loop or the true-action operative rollout, and its residual is **66 % lateral**. Two causes, both structural and both already in the v3 build order:
1. **The cost is lateral-blind (P2 by design).** No route/goal/lateral term exists until the strategic module (P3) mints and conditions on a lateral goal. The 2.11 m curved-window error is the cost-of-no-lateral-goal, measured.
2. **The frozen v1 WM was trained for open-loop imitation, not for planning.** Its consequence predictions are only as good as its imitation-era dynamics; a WM trained *for* planning (P4, goal-conditioned tactical predictor + operative retained) is what closes the curved-window gap to the 0.48 m the true actions already reach.

**Net:** P2 validates *planning > heads* and localizes the remaining work with a number. It does **not** claim planning-over-a-frozen-imitation-WM is a finished driver — and that honest boundary is exactly the P3/P4 mandate, now evidence-backed rather than asserted.

## 5. Concrete implications for P3 / P4

1. **Add the lateral goal to the cost (P3, highest leverage).** 66 % of the residual is lateral and untouched by P2. The strategic ROUTE/LANEOBJ + a goal-consistency / path-progress-toward-goal term is the direct lever; expect the curved-window 2.11 m to be where it pays.
2. **The v_target label works — train the strategic head to predict it (P3).** Offline it already guides the planner to track speed better than the log. The P3 strategic VTARGET head replaces the oracle mint with an inferred token (goal-dropout ≥0.5, per the vocabulary freeze).
3. **A goal-conditioned tactical predictor / feature-space rollout (P4)** is what lifts the WM from imitation-era to planning-grade and closes the curved-window gap toward 0.48 m.
4. **CEM is sufficient and cheap** at this scale (N=64×3, fully batched, ~0.7 s/window incl. all baselines) — the Diffusion-ES upgrade (v3.1) is not on the critical path for the de-risk.

## 6. Limitations & provenance (stated, not hidden)

- **v_target is an offline oracle label** minted from the log's future speed (the spec-sanctioned "VTARGET minted offline for val"). It uses no GT *position/steering*; it is a legitimate strategic-target stand-in for the de-risk. P3 trains the strategic module to predict it at inference.
- **No gap/TTC/collision term** — our data has no lead-agent boxes or HD map. The barrier is stubbed off (spec v0). This is a *drift/longitudinal* de-risk, not a safety-closed-loop.
- **Closed-loop is self-referential** (the WM is both planner-state-estimator and simulator) — inherited from the imagination-in-the-loop harness; an external photoreal sim (AlpaSim/NuRec) remains the only cure and is unrunnable on this pod.
- **Weights engineered, not GT-fit;** the sensitivity sweep (§2.5) is the guard against a circular G1.
- Non-destructive: read-only over frozen checkpoints; the only artifact is `results/planner_p2_flagship-30k.json`. Nothing committed.

## 7. Repro

```
# eval pod tanitad-eval (A40), frozen flagship-30k
python3 -m taniteval.planner_p2 --arm flagship-30k --episodes 40                # open-loop (G1)
python3 -m taniteval.planner_p2 --arm flagship-30k --closed-loop --cl-episodes 20 --replan-every 1  # G4
```
Module: `/root/taniteval/taniteval/planner_p2.py`. Results: `/root/taniteval/results/planner_p2_flagship-30k.json`.
Baselines (same harness): `closedloop_flagship-30k.json` (operative 0.452, CV 0.825, closed-loop head 1.685), `plan_flagship-30k.json` (tactical head 3.38).
