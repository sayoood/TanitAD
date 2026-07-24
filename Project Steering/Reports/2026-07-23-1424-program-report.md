# TanitAD Program Report — 2026-07-23 14:24 Berlin

*Resuming the 3×/day filed program-report cadence (lapsed since 2026-07-15). Evidence class on every
number: MEASURED (ours, artifact) · PUBLISHED (cited) · HYPOTHESIS. Decision-grade numbers cite the
registry or raw eval JSON, never prose.*

## Headline
Two big bets are mid-flight — **flagship v4.2b** (the corrected joint hierarchical planner) and
**Branch B** (our own rig-robust dynamics encoder) — and both are early-healthy. The closed-loop
**RL-vs-alternatives** question is decisively resolved: RL is the *last* lever; the junction failure needs
closed-loop-aware **training** (Gate-1), not inference tricks. Orin deployment is settled (FP16).

## 1. Flagship (v4 line) — architecture stable, schedule finally being tuned right
The architecture is unchanged across v4/v4.1/v4.2/v4.2b (Sayed's design: v1 world model + strategic layer
+ tactical & operative **anchored-diffusion** planners, **trained jointly, nothing frozen** — verified:
encoder 149/149 params require grad, `gnorm_encoder`>0 live, effective batch 64 = v1). What we've been
iterating is the **training schedule**:
- **v4** (lr_trunk 3e-4): degraded the WM (canary → 1.3+). ✗
- **v4.1** (halve-to-zero controller): starved the planner → **10k gate FAIL, held-out ade 0.852** (MEASURED). ✗
- **v4.2** (cap-and-hold floor 0.25): **floor-too-high CONFIRMED** — interim eval @step4000 **4wp ade 0.987,
  canary 0.722**, WORSE than v4.1@10k at <half the steps (MEASURED, harness-validated). The floor degraded
  the WM faster than the planner benefited (negative loop). ✗
- **v4.2b** (floor 0.15, fresh-from-v1): **LAUNCHING NOW** — the pre-registered fix (lower floor → less WM
  degradation while still un-starving the planner). v4.2's ckpt preserved.

*Resolved side-question (Sayed):* "v4 is faster than v1 → it must have frozen parts" — NO. The whole chain
trains; v4 is faster only because v4.1 ran effective-batch 16 vs v1's 64 + no grad-checkpoint. Per-sample
work is nearly identical. v4.2/v4.2b now match v1's effective batch 64.

## 2. Own-encoder (IDM / YouTube-scale pretraining) — Branch B live, mechanism working
- **Branch A** (cheap warm-start GAIA-2 conditioning): **FAILED** cross-rig (ablation +0.09 on a −2.2 collapse).
- **Branch B** (from-scratch, all-block camera-conditioned video-SSL, 2466-clip multi-rig): **LIVE, healthy**
  (~step 11k/40k, loss 10.2→~1.0, IDM 5.8→0.3–0.8). ⭐ **Camera-conditioning demonstrably working** — all 12
  blocks learned from zero-init, rig-A-vs-B token-delta **2.7–7.5/block** (vs Branch A's +0.1). The decisive
  test is the **held-out-rig transfer eval at step 40k (~13h)** — does it beat the −2.1 cross-rig ablation?

## 3. Closed-loop RL-vs-alternatives — RESOLVED (Sayed's research question)
Full cited verdict: `Architecture & Inference/Research/2026-07-23-closed-loop-wm-training-verdict.md`
(3 research angles, ~40 sources + 3 in-house experiments). **For our logs-only, no-fast-sim, safety-critical
regime, RL is the LAST lever, not the first.**
- **Free inference floor is RULED OUT for junction off-road** (MEASURED, Gate 0/0b + rung-3): guidance-selection,
  gradient-synthesis, AND WM-MPC all fail to fix it — because the plan is on-road but the **ego departs**, i.e.
  it's a closed-loop **execution** failure, not planning. The single-step re-plan already captures the
  imagination benefit (WM-MPC ties it).
- **→ Gate-1 (closed-loop-aware training)** — RoaD/CAT-K-style or analytic-gradient through the diff-WM — is the
  measured-justified fix. On-policy **rollout collection is RUNNING now** (idle capacity). Needs Sayed's go for
  the fine-tune.
- **Ship** the gradient-nudge floor as a free **safety override** (intersection collisions 0.71→0.43).

## 4. Benchmarks / closed-loop (all within-sim, ~3.2× OOD — relative only)
- REF-C base **beats** flagship-v1's tactical head closed-loop (paired n=12 + native-res confirm; the n=1
  "flagship wins" was a lucky scene — RETRACTION C7).
- **Scenario-stratified (balanced 38-scene):** flagship **TIES** REF-C on roundabout+highway, loses on
  straight/traffic-light; **both collapse at intersections** (the joint target). ΔScore −0.123.
- AlpaSim on the A40: ~0.8–1.0× real-time @854 / 0.29× native, **renderer-bound** (model tick ~90 ms).

## 5. Deployment (Jetson Orin/Thor) — DONE
FP16 is the deployment precision. **INT8 is not worth it** — no latency win (2.1% faster enc / 2.1% *slower*
pred = QDQ overhead) + readout-head activation collapse + compounding rollout error. Tick clears 10 Hz in FP16.

## 6. Fleet
| pod | stream |
|---|---|
| `tanitad-pod2` | flagship **v4.2b** launching (floor 0.15) |
| `tanitad-pod3` | **Branch B** own-encoder (~11k/40k, healthy) |
| `tanitad-eval` | **Gate-1** on-policy rollout collection (idle-capacity prep) |
| `tanitad-pod` (pod1) | spare / v4.2b-eval host |

## 7. Decisions for Sayed
1. **v4.2b** — proceeding per the looping rules (act on the pre-registered result); **veto available**.
2. **Gate-1** — green-light the closed-loop-aware training after v4.2b? It's the measured-justified junction
   fix; rollout-data prep is running so it's launch-ready.
3. **Branch B** — continues to its 40k transfer eval (auto).

## 8. Retractions & process this period (root-cause classes in `RETRACTION_LOG.md`)
- **C7** — n=1 "flagship beats REF-C" reversed by the n=12 paired suite.
- **C2** — my "ablation crashed" false alarm (it had completed; results in a subdir I didn't check).
- **Data-integrity invariant** — val split `physicalai-val-f1b378f295ae` leaks 78% into train; never eval on it.
- **Process (mine):** over-requested a classifier-blocked HF push (corrected — won't route around it); overstated
  the agent-monitor issue as universal (it's a delivery-gap for *some* agents). And the passive failure this
  reminder caught — idling a decisive result on a rubber-stamp instead of acting per the looping rules.
