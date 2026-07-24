# TanitAD Program Report — 2026-07-24 17:57 Berlin

*3×/day filed program report (D-025). Evidence class on every number: MEASURED (ours + artifact) ·
PUBLISHED (cited) · HYPOTHESIS. Decision-grade numbers cite the registry / raw eval JSON, never prose.
Flagship numbers are at pod wall-clock 17:09 UTC (step 9550).*

## Headline
**The flagship's crux question is answered: from-scratch co-evolution WORKS.** The v4 world model
reached **full planner coupling (λ_plan = 1.0)** and the WM-integrity canary *improved* through it
(2.6 → 1.4) — the exact failure mode that killed all four warm-start v4 arms. **10k gate ≈ 1 h**
(CONTINUE expected). The fleet is fully utilized (flagship + v2-corpus build + IDM pilot, no idle GPU).
One decision gates a large data-scale step: non-CC YouTube licensing.

## 1. Flagship (v4 line) — from-scratch co-evolution DECISIVELY working
Architecture (MEASURED from the running `config.json` + `flagship_losses.py`): v1 WM +
**anchored-diffusion planner** (diffusion_steps 2, 256 anchors) + the **operative/tactical/strategic
hierarchy**, FiLM-wired, trained jointly.
- **v4 / v4.1 / v4.2 / v4.2b = all FAILED** (warm-start artifact — coupling degraded a *warm-started* WM).
  v4.1 10k `ade_0_2s` **0.8522** [0.75, 0.98] (`flagship-v4.1-10k.json`). Cosine pre-probe seam
  cos(g_wm,g_plan) = **+0.0043** (near-orthogonal) → gradient-surgery REFUTED for ~0 GPU → from-scratch.
- **`flagship-v4-fromscratch-30k` (LIVE, pod2, PID 108011): step 9550, λ_plan = 1.0 — FULLY COUPLED.**
  ⭐ **canary_ade@2s DESCENDED through coupling: 15.674 (rand-init) → 2.59@7000 → 1.371@9000 → 1.566@9500**
  (controller "ok", vs_base −14.3) — the warm-start arms instead ROSE to 0.7+ here. **val ade@2s
  0.531@9000 (n=881), miss@2m 0.188, oracle_ade@2s 0.283.** restarts 0. The WM co-evolves AND drives
  better as it fully couples — v1's existence proof reproduced. **10k gate ≈ 1 h** (pace ~7 s/step).
- **Gate-prep DONE:** `eval_flagship_v4.py` + `gate_emitters.py` + `speed_benefit.py` all on the eval pod
  (`/root/v4eval/stack/`) → the 10k gate renders a COMPLETE verdict incl. `speed_benefit` (the
  from-scratch log lacks `g_op_fwd_ade_m`, so the fresh emit is required — now enabled).

## 2. Own-encoder / Branch B / YouTube-IDM
- **Branch B (from-scratch GAIA-2 camera-conditioning) = FAIL** (registry §10.1): cross-rig speed R²
  **−0.667** vs **flagship-v1 frozen +0.657**. Refuted for rig-robustness.
- **v1 encoder + a multi-domain IDM head IS the cheap substrate** (rig-B speed +0.657). **YouTube-IDM =
  GO, decision-grade:** downstream pseudo-label pretraining captures **96% (proxy) / 109% speed, 71% yaw
  (parity)** of real-label value; direct extracted-vs-GT speed R² **0.62–0.66**, longitudinal 0.60, yaw ~0
  (weak, caveated).
- **Pilot (pod3):** CC harvest DONE (**80 clips / 64 licensed videos**), P3 pseudo-label DONE (80 latents +
  `pseudo_labels.json`), **P4 downstream-lift RESUMED**. ⚠️ **CC-licensed dashcam is scarce (80 clips) →
  a large harvest needs Sayed's non-CC licensing decision.**

## 3. Closed-loop research / Gate-1 / frozen-WM
- **D2 REF-C closed-loop (recovery-aug) = CLOSED honestly.** The "ADE cost" was largely a knife-edge-
  L2-metric artifact; the departure *benefit* **reverses at full power** (n=12 +0.0089 → n=40 −0.0302,
  C5-retracted). Not promotable. Durable: the machinery, REF-C encoder safely FT-able.
- **Gate-1 (closed-loop-aware FT):** mechanism works (junction offroad 11→7, collisions 5→1) but a
  *promotable* run is HELD — data-bound (memorization at ~13–22 junction eps) + instrument-gapped.
- **D1 frozen-WM + learned planner = ~0.60 deployable FALLBACK** (W 0.599 ADE@2s, WM canary unchanged by
  construction). Value-model crux FAILED (learned-value 1.02 > W 0.599); the "0.132 search headroom" was
  hindsight-privileged (C6). Not a contender that beats the coupled path.

## 4. Benchmarks & closed-loop
- ⭐ **Traffic-light scenario (SC-14) + a TLC (Traffic-Light-Compliance) metric BUILT** + the beyond-ADE
  suite (LAL/TMS/OKRI/CNCE/LOPS). **First REAL beyond-ADE numbers** (dev-box 4060, comma val): decision-tick
  **latency p50 14.33 ms**, TMS 0.0435 (expert-log ref), CNCE 210k. Full closed-loop TLC/LAL/OKRI/LOPS
  **renderer-gated** (MetaDrive confirmed absent at 3 probes).
- **REF-C base > flagship-v1 closed-loop — triple-confirmed** (n=1 → n=12 AlpaSim → **n=40 real-footage
  low-OOD**: ADE@2s **0.564 vs 1.488**). LEADERBOARD §5.5.

## 5. Deployment (Orin/Thor)
- **FP16 is the deployment precision; INT8 rejected** (no latency win + readout activation collapse). Tick
  clears 10 Hz. The v4 diffusion-tick knob profiles when from-scratch converges.

## 6. Dataset enlargement (v2 corpus, NEW this period) — Sayed-directed
- **Enlarge to ~50 h within NVIDIA PhysicalAI-AV, balanced distribution** (Sayed 2026-07-24). New "v2"
  canonical corpus for the NEXT flagship gen (breaks parity by design; current run finishes on 13 h).
- **Phase 1 DONE:** 104.6 h available (2× headroom); balanced 9,000-clip selection hits target EXACTLY
  (turns 14.25→28%, lane_keep 60→45%, junction-clip presence 38→61%). Key `physicalai-v2bal-4b7eeeac222d`.
- **Phase 2 BUILDING (pod1, detached):** ⭐ JPEG-compressing the cropped 256px frames = **982 GB → ~25 GB**
  (bit-identical to parity, full 256px kept — the "fit the quota" path Sayed chose). **388/9000** at last
  check, resumable. Integration flag: v2 training needs a thin `load_compressed` Dataset wrapper.
- Current 13 h corpus profiled (MEASURED): 13.13 h, lane_keep 59.6% / turns 14.25% / accel+brake 26%,
  30k steps = 4.73 epochs.

## 7. Fleet (verified ~17:09 UTC, nvidia-smi + heartbeats)
| pod | stream | state |
|---|---|---|
| `tanitad-pod2` | **from-scratch flagship** | 🟢 step 9550, λ_plan 1.0, canary 1.4–1.6, **10k gate ~1 h** |
| `tanitad-pod3` | **YouTube-IDM pilot** | 🟢 P3 done (80 latents), P4 downstream-lift resumed |
| `tanitad-pod` (pod1) | **v2-corpus build** | 🟢 388/9000, JPEG 25 GB, building detached (resumable) |
| `tanitad-eval` | reserved + gate-prepped | ⚪ for the imminent 10k gate |
**No idle GPU.** Every pod carries high-value work.

## 8. Decisions for Sayed
1. **Non-CC YouTube licensing** — for a *large* IDM harvest (CC-only yields ~80 clips; the mechanism is
   de-risked, the scale is licensing-gated).
2. **HF-storage cleanup** — `Sayood/` at 403; blocks the REF-C low-OOD 2nd arm + ckpt backups.
3. *(offered)* persist the full work-package status as a living `PROGRAM_OVERVIEW.md`.

## 9. Retractions this period (`RETRACTION_LOG.md`)
- **C5** — *"D2 recovery-aug halves held-out departures + generalizes"* (from n=12) → **reverses at n=40**
  cross-fit (departs 3.3× more). Forward use must cite the n=40 BOUND.
- **C8** — *"TanitEval / lake ingest modules stranded on worktrees"* → FALSE, a **truncated mtime-sorted
  Glob artifact**; main has all modules and is the newest/most-complete copy. Lesson: verify presence with
  `git ls-files` / a narrow glob, never a capped Glob.
- Same-day self-corrections (caught by reading state, before deciding): the IDM harvest read as "dead/
  stalled" twice → it had actually completed (80 clips); the pilot read as "stalled at 16 latents" → 80.
