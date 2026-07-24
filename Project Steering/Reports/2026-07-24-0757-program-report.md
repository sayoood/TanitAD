# TanitAD Program Report — 2026-07-24 07:57 Berlin

*3×/day filed program report (D-025). Evidence class on every number: MEASURED (ours + artifact) ·
PUBLISHED (cited) · HYPOTHESIS. Decision-grade numbers cite the registry / raw eval JSON, never prose.*

## Headline
**The flagship is on its cleanest footing yet.** After four warm-start v4 arms failed by degrading the
world model when the planner coupled, `flagship-v4-fromscratch-30k` (v1's proven co-evolve-from-random
recipe) is **holding the WM-integrity canary stable through 39 % planner coupling** — the exact failure the
warm-start arms could not survive. And the own-encoder line reversed from a decisive failure to a **GO,
decision-grade** YouTube-IDM scale-up. The one decision gating forward progress is Sayed's YouTube-IDM go.

## 1. Flagship (v4 line) — from-scratch, co-evolution looking real
The whole v4 line is a fixed architecture (**verified from the running `config.json` + `flagship_losses.py`,
not prose**): v1 WM + **anchored-diffusion planner** (`head_cfg.decoder`: diffusion_steps 2, noise_std 0.1,
256 anchors) + the **full operative/tactical/strategic hierarchy**, FiLM-wired (`flagship_losses.py:244`
`strategic ctx → tactical intent → operative`), trained jointly.

- **v4 / v4.1 / v4.2 / v4.2b = all FAILED** (MEASURED): coupling a planner degraded a *warm-started* WM.
  v4.1 10k gate `ade_0_2s` **0.8522** [0.75,0.98] (`flagship-v4.1-10k.json`); v4.2 @4k **0.9869** / canary
  0.722; v4.2b canary drifted 0.42→**0.70** (held-reads), floor-tuning exhausted (floor 0.25→0.72, 0.15→0.70).
- **Root cause = warm-start artifact.** The cosine pre-probe settled the fork for ~0 GPU: seam
  cos(g_wm,g_plan) = **+0.0043** (near-orthogonal) → gradient-surgery is a no-op, **REFUTED**. So the coupled
  path = from-scratch (v1's existence proof).
- **`flagship-v4-fromscratch-30k` (LIVE, pod2, PID 108011):** step **4350**, λ_plan **0.39** (ramping to 1).
  ⭐ **canary HOLDING through coupling: 15.67(0)→2.98(2000)→2.587(2500)→[4.63 noise](3000)→2.175(3500)→
  2.162(4000)** — stable ~2.16 at 39 % coupling, NOT rising. oracle_ade **0.342 (below v1's 0.4271)**, val
  ade@2s ~0.63–0.74, eff-batch 64, restarts 0. ⚠️ **not yet the full verdict** — needs the canary holding to
  λ_plan=1 + the **10k gate (~11h)**; judge the descent trajectory, NOT the warm-start ≤0.55 bar (from-scratch
  canary baseline is 15.67, not 0.42). The v4 gate now renders COMPLETE (3 missing emitters built).
- **Fallback ready:** frozen-WM investigation complete end-to-end → it is a solid **~0.60 deployable
  fallback**, NOT a search-matching contender (see §7 retractions). Keep as the safety net if from-scratch stumbles.

## 2. Own-encoder / YouTube-IDM — reversed from failure to GO
- **Branch B (`dynenc-branchB`, from-scratch GAIA-2 camera-conditioned, 40k) = FAIL** (registry §10.1,
  `results_branchb_transfer_e50_CONVERGED.json`): cross-rig speed R² **−0.667** vs **flagship-v1 frozen
  +0.657** (paired CI excludes 0 on 3/4 arms). From-scratch camera-conditioning is **REFUTED** for rig-robustness.
- ⭐ **The cheap substrate exists:** flagship-v1's encoder + a **multi-domain** readout head transfers
  (rig-B speed **+0.657** / yaw +0.504; cross-CLASS fisheye→comma **+0.585**). The −1.17 was a head-diversity
  artifact, not an encoder ceiling.
- ⭐⭐ **YouTube-IDM = GO, DECISION-GRADE.** Downstream WM-pretraining ablation: pseudo-labels capture
  **~96 %** of real-label pretraining value (8 seeds); **parity validation on the ACTUAL target: 109 % speed /
  107 % traj / 71 % yaw** of the real-label ceiling (`results_idm_parity_validation.json`, 4 seeds, all beat
  floor CI-sep). The label-R²~0.63 proxy understated it. Residual: v1's cross-class gap caps novel-rig
  *absolute* quality, not the pretraining *value*. **→ scale-up is Sayed's licensing-gated commitment.**

## 3. Closed-loop research / Gate-1 — all roads lead to the renderer
- **Gate-1 (closed-loop-aware FT):** mechanism WORKS (junction offroad **11→7**, collisions 5→1) but a
  promotable run is HELD — data-bound (~13–22 real junction eps → memorization, leave-3-out held-out Δ≈0) +
  the instrument gap.
- **D2 recovery-aug lever:** halves held-out lane-departures + **generalizes** (beats Gate-1's memorization)
  but decoder-only is **Pareto-bound** (departure↓ ⟺ ADE↑).
- **RefcCL (Sayed-greenlit, encoder-in-loop) = NOT promotable (branch c, MEASURED):** even a material encoder
  move (feat_cos 0.966) doesn't unblock the trade — it's **intrinsic to the single-step synthetic-recovery
  objective**, not a frozen-encoder artifact. ⭐ **2 findings: (a) REF-C's encoder is SAFELY fine-tunable —
  the v4 WM-degrade hazard is avoidable (de-risks camera-cond + co-train); (b) the real escape = a
  closed-loop-CONSISTENT objective (RoaD/CAT-K on-policy) → needs a low-OOD renderer.**
- **Convergence: the closed-loop program is now gated on ONE thing — a faithful low-OOD renderer** (AlpaSim =
  3.2× OOD; the real-footage instrument is map/agent-free). Not on more planner/encoder tricks.

## 4. Benchmarks & closed-loop
- ⭐ **REF-C base BEATS flagship-v1 closed-loop — now TRIPLE-confirmed** across independent instruments:
  n=1 (scene-dependent, retracted) → n=12 NuRec AlpaSim → **n=40 real-footage low-OOD** (ADE@2s 0.564 vs
  1.488; departure-rate 0.013 vs 0.032; both at 1.02–1.20× OOD). LEADERBOARD §5.5. The low-OOD instrument
  decomposes flagship-v1's deficit as **longitudinal, not lane-keeping**.
- The **low-OOD-vs-safety-metric gap is ~fundamental**: reactive-agent off-road/collision needs a sim (OOD);
  low-OOD needs real footage (no agents). Resolving both = a lower-OOD renderer (hard).

## 5. Deployment (Orin/Thor)
- **FP16 is the deployment precision; INT8 rejected** (MEASURED: no latency win + readout activation collapse).
  Tick clears 10 Hz. The v4 diffusion tick (denoise-count knob) profiles when from-scratch converges.

## 6. Fleet (verified ~07:5x Berlin, nvidia-smi ground truth)
| pod | stream | state |
|---|---|---|
| `tanitad-pod2` | **from-scratch flagship** | 🟢 step 4350, λ_plan 0.39, canary holding ~2.16, 10k gate ~11h |
| `tanitad-eval` | reserved | ⚪ free, held for the 10k formal gate |
| `tanitad-pod` (pod1) | — | ⚪ free (frozen-WM investigation COMPLETE) |
| `tanitad-pod3` | — | ⚪ free (YouTube-IDM de-risked; next step = Sayed's scale-up) |
**Idle honestly stated:** the cheap-experiment phase is exhausted — every research question this period is
resolved. The 3 idle pods have no high-value autonomous work; the next moves are the flagship gate, Sayed's
decisions, or bigger commitments (the low-OOD renderer, the hierarchy proof needing a converged flagship).
The YouTube-IDM go would immediately re-fill them.

## 7. Decisions for Sayed
1. 🔴 **YouTube-IDM SCALE-UP** — GO, decision-grade (§2). The build (v1+multi-domain-head → pseudo-label
   YouTube → pretrain → FT on parity) + the **licensing/GDPR sign-off** (ship URL-pointers + pseudo-labels,
   not bytes; face/plate check) are the commitment. **This is the one decision gating forward progress.**
2. **HF-storage cleanup** — `Sayood/` is over quota (403), blocking the Branch B ckpt backup + old pushes.
   Your action (upgrade or delete repos); I offered to enumerate the repos.
- Resolved off-plate this period: own-encoder pivot (cheap v1 path) · frozen-WM contender (fallback, no
  value-model arm) · RefcCL (greenlit, not-promotable) · IDM cheap-test (=GO).

## 8. Retractions this period (root-cause classes in `RETRACTION_LOG.md`)
- **C5** — *"from-scratch canary descending, co-evolution CONFIRMED"* asserted from a **single** eval point
  (step-500); step-1000 bounced. Corrected same-iteration (n=5 now confirms the descent). Cost 0, headline
  reached chat.
- **C6** — *"CEM search 0.132, 4.5× → the planner is the headroom on a frozen WM"* — the 0.132 is
  **hindsight-privileged** (peeks at the expert's actual future, which the ego doesn't control open-loop);
  the deployable learned-value search = **1.02, worse than feedforward 0.599**. The gap is prediction-vs-
  hindsight, NOT planning headroom. **A privileged-input arm is not a headroom estimate.**
