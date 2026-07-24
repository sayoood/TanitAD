# TanitAD Program Report — 2026-07-24 12:57 Berlin

*3×/day filed program report (D-025). Evidence class on every number: MEASURED (ours + artifact) ·
PUBLISHED (cited) · HYPOTHESIS. Decision-grade numbers cite the registry / raw eval JSON, never prose.*

## Headline
**The flagship is now three-quarters through its coupling ramp and holding — the single most important
read of the program is trending clearly positive.** After four warm-start v4 arms failed by degrading the
world model on planner coupling, from-scratch co-evolution is holding the WM-integrity canary stable at
76.7% coupling with its best driving numbers yet. Separately, the closed-loop-improvement direction —
which I'd twice over-declared "closed" — reopened: its apparent Pareto wall was largely a scoring artifact
(a real, honest reframe). The one decision gating forward progress remains Sayed's YouTube-IDM go.

## 1. Flagship (v4 line) — from-scratch, co-evolution now convincing
Architecture (verified from the running `config.json` + `flagship_losses.py`, MEASURED): v1 WM +
**anchored-diffusion planner** (decoder diffusion_steps 2, 256 anchors) + the **operative/tactical/strategic
hierarchy**, FiLM-wired (`flagship_losses.py:244`), trained jointly.

- **v4/v4.1/v4.2/v4.2b = all FAILED** (warm-start artifact: coupling degraded a warm-started WM; canary rose
  to 0.7–1.3). Cosine pre-probe (cos +0.0043, near-orthogonal) REFUTED gradient-surgery for ~0 GPU → the
  coupled path is from-scratch (v1's existence proof).
- **`flagship-v4-fromscratch-30k` (LIVE, pod2, PID 108011):** **step 6600, λ_plan 0.767 (76.7% coupled).**
  ⭐ **Canary HOLDING through 3/4 of the coupling ramp** (stable ~2.2–2.6 band: 2.36→2.57→2.63 across
  steps 6000–6500; the descent 15.67→~2.2 already done in Phase A) — NOT the v4.x runaway. **oracle_ade
  0.314 (below v1's 0.4271), plan_ade 0.99, WM loss 2.28 (lowest yet), val ade@2s ~0.54–0.59, miss@2m 0.21.**
  The WM co-evolves AND drives better as it couples. ⚠️ still not the full verdict — needs holding to
  λ_plan=1 + the **10k gate (~7h)**; judge trajectory not the 0.55 warm-start bar (from-scratch canary
  baseline was 15.67). eff-batch 64, restarts 0. v4 gate now renders COMPLETE (3 emitters built).
- **Fallback:** frozen-WM = a ~0.60 deployable fallback, NOT a contender (all 4 deployable routes hit the
  aleatoric wall; the 0.132 "search" was hindsight-privileged — retracted C6).

## 2. Own-encoder / YouTube-IDM — GO, decision-grade (unchanged, awaiting Sayed)
- **Branch B (from-scratch camera-conditioning) = FAIL** (registry §10.1): cross-rig speed R² −0.667 vs
  flagship-v1 frozen +0.657. REFUTED for rig-robustness.
- ⭐ **v1's encoder + a multi-domain head IS the cheap substrate** (rig-B +0.657, cross-class comma +0.585).
- ⭐⭐ **YouTube-IDM = GO, DECISION-GRADE:** downstream ablation ~96% of real-label pretraining value;
  **parity validation on the ACTUAL target 109% speed / 71% yaw** (`results_idm_parity_validation.json`,
  4 seeds). Residual: v1's cross-class gap caps novel-rig *absolute* quality, not the pretraining *value*.
  **→ scale-up is Sayed's licensing-gated commitment.**

## 3. Closed-loop research / Gate-1 — reopened by a metric fix (the period's twist)
- Gate-1 (offroad 11→7) held on data (memorization at ~13–22 junction eps). D2 recovery-aug halves
  departures + generalizes but looked Pareto-bound; RefcCL (encoder-in-loop) not-promotable; LOWOOD-CL
  (on-policy) BOUND (base rarely fails on the low-OOD source → RoaD/CAT-K starved).
- ⭐ **REFRAME (MEASURED, `a1f26c92`): the "Pareto trade" was LARGELY a knife-edge-L2-metric artifact.**
  Under a fair lane-tolerance band (`band_ade2d(1.0)`, base 0.1997) the raw-ADE cost **fully vanishes (CI∋0)
  for 3/4 configs, −74% for naive-D2** — exact-path L2 was scoring benign in-lane recovery as a cost. The
  real residual is a small, **n=12-underpowered departure signal**. **A more-powered departure eval (n≈40
  cross-fit) is RUNNING** to settle it: WIN → the lever is a net win, direction reopens; BOUND → n=12 noise.
- **Adopt `band_ade2d` as the fair closed-loop metric.** Bonus banked: REF-C's encoder is safely FT-able
  (RefcCL canary held at a material move — de-risks camera-cond / co-train). Renderer paths remain for the
  *separate* reactive-agent collision problem only.

## 4. Benchmarks & closed-loop (unchanged)
- REF-C base **beats** flagship-v1 closed-loop — TRIPLE-confirmed (n=1 retracted → n=12 NuRec → **n=40
  real-footage low-OOD**: ADE@2s 0.564 vs 1.488). LEADERBOARD §5.5.

## 5. Deployment — FP16 (INT8 rejected). The v4 diffusion tick profiles when from-scratch converges.

## 6. Fleet (verified ~12:5x Berlin, nvidia-smi)
| pod | stream | state |
|---|---|---|
| `tanitad-pod2` | **from-scratch flagship** | 🟢 step 6600, 76.7% coupled, holding; 10k gate ~7h |
| `tanitad-eval` | departure-power eval | 🟢 `a1f26c92` — the closed-loop reopen-or-closed verdict, landing |
| `tanitad-pod` (pod1) | — | ⚪ held (research exhausted) |
| `tanitad-pod3` | — | ⚪ held (YouTube-IDM de-risked → scale-up is Sayed's) |

## 7. Decisions for Sayed
1. 🔴 **YouTube-IDM SCALE-UP** — GO, decision-grade. Build + licensing/GDPR sign-off (URL-pointers +
   pseudo-labels, not bytes; face/plate check) are the commitment. **The one decision gating progress + the
   one that re-fills the idle pods.**
2. **HF-storage cleanup** — `Sayood/` over quota (403); your action (offered to enumerate repos).

## 8. Retractions this period (`RETRACTION_LOG.md`) — a NAMED meta-pattern
This period I **over-declared closure/certainty 4×**, each reopened by a cheap follow-up:
- **C5** "canary descent confirmed" (n=1) · **C6** "planner is the headroom" (0.132 was hindsight-privileged)
  · **C3** "closed-loop needs a renderer" (over-broad; but the cheap test that checked it came back BOUND,
  re-confirming the instrument-need) · **C3** "closed-loop is Pareto-bound/closed" (the trade was a
  knife-edge-metric artifact).
- **Lesson logged: before declaring a direction closed/bound/resolved, run the cheapest metric-or-power
  check FIRST — the closure claim is the single one most worth a $0 test.** All caught same-session at ~0
  cost, but the firm claims reached chat/reports; I'm biased toward premature certainty and correcting for it.
