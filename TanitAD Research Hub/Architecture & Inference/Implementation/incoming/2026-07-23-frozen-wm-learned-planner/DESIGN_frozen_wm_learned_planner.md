# Design — Frozen v1 WM as a differentiable simulator + a learned planner

**Author:** frozenwm-planner subagent · **Date:** 2026-07-23 · **Status:** design bank (P1) + the
pre-registered experiment run as P2. Additional direction — **modifies nothing that is running**
(v4.2b, Branch B, REF-C weights untouched; this uses a read-only copy of v1's checkpoint).

Evidence tags as in the research doc (`[PUB]/[MEAS]/[INH]/[EST]/[HYP]`).

---

## 1. The asset map — what we already have, exactly

| Need | Asset (path) | Evidence |
|---|---|---|
| A trained, differentiable WM | flagship v1 `flagship-30k` = trained ViT encoder + SpatialGridReadout + `OperativePredictor` d768×10 + `HierarchicalGrounding.step['op']` | `[MEAS: tanitad-pod:/root/models/flagship-30k/ckpt.pt; loaders.load]` |
| A differentiable action→trajectory rollout | `tanitad.models.metric_dynamics.rollout_decode(predictor, states, aw, fa, step_readout, k)` — rolls the predictor `k` steps under fed actions, decodes per-step Δpose, SE(2)-accumulates; **gradient flows to states, predictor, step_readout AND the fed actions** | `[MEAS: source read + smoke: grad→fed actions norm 0.85, 95 % nonzero]` |
| The WM's own fidelity ceiling | operative rollout under **GT** future actions = **0.4045** on the pod's 12-ep val (`[MEAS: this experiment refs]`), **0.4271** full-set 40-ep `[MEAS: REGISTRY §1.2]` | matched apples-to-apples |
| Canonical eval geometry | `taniteval` (`rollout.py`, `driving.py` ADE = `de.mean(1)` over wp [5,10,15,20]), val `physicalai-val-0c5f7dac3b11` | `[MEAS: reproduced CV 0.8463 vs REGISTRY 0.8377; hold-v0 0.7883 vs 0.7876]` |
| A latency-feasible search planner over the WM | P2 CEM imagine-and-select — **20.82 ms p50 @ K=8** | `[MEAS: REGISTRY §1.2]` |

**The key structural fact that shapes the design** `[MEAS: metric_dynamics.py docstring + our Arm F]`:
v1's metric ego-motion lives in the **action-conditioned dynamics** (`predictor` + `step_readout` on
rolled transitions), **not** linearly in the static latent (a held-out MLP probe off the latent gets
3.89 m). **Therefore the planner must read the frozen WM through the dynamics — it must output actions the
frozen predictor rolls — not decode a trajectory off the frozen state.**

---

## 2. Architecture — a small learned planner on the frozen WM

```
 8-frame window ──[ FROZEN v1 encoder+readout ]──► states z[8,2048]   (grad OFF)
                                                        │
                                          ┌─────────────▼──────────────┐
                                          │  LEARNED PLANNER  π (3.8 M) │   2-layer Transformer over the
                                          │  proj 2048→384 · +pos · 2×  │   8 window tokens, last-token
                                          │  TransformerEncoder · MLP   │   readout, MLP head
                                          └─────────────┬──────────────┘
                                                        │ â = [steer,accel]×20   (the plan)
                     append constant v0 (=poses.v/10)   │
                                                        ▼
 states z[8,2048] ──►[ FROZEN OperativePredictor d768×10 ]──roll 20 steps under â──►
                     [ FROZEN step_readout ]──Δpose──► SE(2) accumulate ──► trajectory ŵ[20,2]
                                                        │
                                        cost = ADE(ŵ, expert)  (+ smoothness, optional)
                                                        │
                        ◄──── analytic gradient BACKPROP through the FROZEN WM ────
                             (WM params get no update; only π updates)
```

- **Planner input:** the frozen window states (the encoder is run once and cached — `encode_window` is
  per-frame independent, so caching per frame is exact `[MEAS: source read]`).
- **Planner output:** a 2-D control sequence `[steer, accel] × 20`. The speed channel `v0 =
  poses.v/SPEED_SCALE(10)` is appended as the held-constant 3rd action channel, **matching
  `rollout.append_ego` byte-for-byte** so the fed action matches the checkpoint's `action_dim=3`.
- **Seam discipline (why *this* seam):** the planner reaches the WM through the **existing action channel**
  — the program's **only 1-for-1 seam family** (`A4`: ego-through-actions REF-A 3.73→0.83; speed channel
  2.918→0.452) `[MEAS: ARCHITECTURE_WIRING_COMPARISON.md §2.4]`. No new conditioning surface, no FiLM-into-
  a-vector (the 0-for-4 family). The planner *is* the action source; nothing about the WM changes.

**Two control arms sharing the trunk, to locate the ceiling:**
- **Arm F — static-decode control:** planner reads the frozen states and regresses the 20 waypoints
  **directly** (no predictor). Measures the frozen-*encoder* direct-decode ceiling — the REF-A regime on
  v1's grounded state. Predicts ~3.9 m by the latent-probe measurement.
- **Arm B — action-BC control:** planner outputs actions supervised by MSE to the **expert actions** (no WM
  in the loss); evaluated by rolling those actions through the frozen WM. Isolates what the
  analytic-gradient cost model buys over plain behaviour cloning of controls (research §2 mechanism (d)).

---

## 3. Training recipe

| Item | Value | Rationale |
|---|---|---|
| Frozen | encoder + readout + predictor + step_readout, all `requires_grad=False`, `eval()` | the WM is the fixed differentiable simulator |
| Learned | planner π only (**3.77 M params** `[MEAS]`) | "small learned planner" |
| Objective (Arm W) | ADE of the rolled trajectory vs expert, mean over 20 steps | analytic gradient through the frozen WM (research §2(a)) |
| Optimiser | AdamW lr 3e-4, wd 1e-4, cosine | program default |
| Steps / batch | 3000 / 24 (W) · 48 (F,B) | W's 20-step BPTT graph is deeper → smaller batch |
| Gradient guard | `clip_grad_norm_(π, 1.0)` | SHAC/BPTT caveat `[PUB: arXiv:2204.07137]` — analytic grads over 20 steps can spike |
| Data | 400 train episodes → **8,803 windows** (frozen-encoded once, cached); eval on the pod's **12 val episodes / 265 windows** | cheapest proof; train/val are the parity train vs val splits → episode-disjoint by construction |
| Eval metric | open-loop **ADE@2s** = mean L2 over wp [5,10,15,20] (`driving.py` def), + FDE@2s, miss@2m | canonical, apples-to-apples |
| Statistic | **episode-cluster bootstrap** over the val episodes, B=2000 | program estimator (`CLAUDE.md`) |
| **Canary** | **not needed as a controller — it is frozen by construction.** The WM's GT-action rollout is unchanged (0.4045) because no WM parameter moves. This is the whole point: the degradation the v4 saga fights cannot occur here. | `[MEAS: WM params frozen]` |

**Truncation fallback (pre-declared):** if Arm W's analytic gradient is pathological (loss NaN/oscillation,
grad-norm clipping saturating), truncate the BPTT to the last N steps with a stop-gradient on the earlier
window states (SHAC-style). *Not expected at 20 steps; declared so the response is not improvised.*

---

## 4. The cheapest discriminating experiment — PRE-REGISTERED (this is P2)

**Question:** does the frozen v1 WM behave like a **viable differentiable simulator** (a learned planner
reaches near the WM's own 0.40 fidelity ceiling, cleanly beating the CV floor) or like a **frozen-encoder
bottleneck** (stuck near the CV floor / REF-A range despite healthy training)?

**Design:** train Arms F, W, B on the frozen WM; evaluate open-loop ADE@2s on the held-out val episodes;
anchor against three MEASURED references computed on the **same** windows — the oracle-action ceiling
(0.4045), the CV floor (0.8463), hold-v0 (0.7883) — and against the cited coupled results.

**Pre-registered readout (both outcomes committed in advance):**

| Outcome | Criterion (on the held-out val) | Interpretation |
|---|---|---|
| **VIABLE** | Arm **W** beats CV (`< 0.85`, CI-separated) **and** lands within a modest factor of the 0.40 oracle ceiling (target `≲ 0.60`), i.e. the action-prediction penalty is small, with **W ≪ Arm F** | freezing is a viable **additional** direction: no WM degradation (frozen), **no crippling frozen ceiling** — the frozen WM is a good simulator and the planner learns effectively against it via analytic gradient |
| **BOTTLENECKED** | Arm **W** stuck `≥` CV floor, or `≈` Arm F, or `≫` 0.40 despite healthy training | frozen-WM ceiling like REF-A — the frozen dynamics do not admit a good learned planner → **report the bound honestly**; the design pivots to (c) amortised-MPC or accepts partial unfreezing |
| **mechanism** (secondary) | sign of **W − B** | if `W < B`, the analytic-gradient cost model beats plain action-BC (the WM adds value as a differentiable cost); if `W ≈ B`, BC is sufficient and (a) is not worth its complexity over (d) |

**What each arm's number *means* (stated before seeing it, to prevent post-hoc storytelling):**
- **Arm F ≈ 3.9 m** would merely reproduce the documented static-latent probe → confirms the metric info is
  in the *dynamics*, not the latent. (Not a knock on freezing — a knock on *static decode*.)
- **Arm W near 0.40–0.60** → VIABLE. **Arm W near CV/REF-A** → BOTTLENECKED.
- The gap **W − 0.40** is the *action-prediction* penalty (predicting 2 s of controls from an 8-frame
  window), **not** WM infidelity — because 0.40 already fixes the WM's action→trajectory fidelity.

---

## 5. The coupled baseline, honestly

v4 is not trained, so there is no clean "coupled v4 ADE" to diff against. The MEASURED coupled evidence is:
- **v1.6** (unfreeze 4 ViT blocks + predictor under a planner loss): ADE **0.4886** heldout, **canary
  0.452→1.1022 (+144 %)** `[MEAS: REGISTRY §1.4b]`.
- **v4 / v4.2 warm-start coupling:** canary **0.42→1.30+** / **0.72@4k** `[INH: planner-wm-gradient-
  coupling/DESIGN.md]`.

**The contrast the frozen arm draws:** a coupled planner lands around v1's ADE (~0.45–0.49) **but pays a
degraded WM** (canary +60–190 %); the frozen arm's WM canary is **0.40, unchanged, for free**. So the
decision is not "which ADE is lower" alone — it is **"does the frozen arm reach competitive ADE *while
keeping the WM intact*."** If Arm W lands in the VIABLE band, freezing dominates on the WM-integrity axis
at little-to-no ADE cost — a genuinely additive direction. **Caveat (C6):** my within-experiment arms and
the v4 coupled arms differ in ≥2 respects (planner shape, train scale, objective); the coupled numbers are
cited as the program's established finding, **not** re-derived here — the clean, self-contained claim is
Arm W vs the on-corpus ceilings.

---

## 6. Product path (beyond this proof)

If VIABLE, the destination is **research §2(c) amortised MPC (TD-MPC2 shape)**: run the measured-feasible P2
CEM search over the **frozen** WM and distil it into a fast learned prior. The frozen WM removes the
degradation risk that makes v4's joint training fragile; the learned prior removes the search latency;
`--strategic`/goal conditioning enters as KV tokens on the planner (1-for-1 seam family), never as a
gradient into the frozen trunk. This proof (mechanism (a)) is the gate that decides whether (c) is worth
building — i.e. whether the frozen WM is a good enough simulator to plan against.
