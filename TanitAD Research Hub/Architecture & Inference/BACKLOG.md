# Architecture & Inference — Experiment Backlog

Prioritized roadmap (D-020 §4). Each run: execute ≥1 item, report measured numbers, re-prioritize.
Format per item: goal / method / resource / expected number / falsifier.

## P0 — next run

A. **Well-powered D1 discriminator on the pod (info-lost vs less-linear).** Goal: give the loop's live
   D1 investigation (`0284a5c`) a decision-grade verdict. Method: this run's `probe_capacity_ladder.py`
   with **PCA-to-active_k + `--episodes ≥50`**, run on the actual **14k-vs-21k (or 30k) checkpoint pair**
   (pod, idle-only, or Colab). Resource: pod GPU (uncontended) or 4060 if ckpts pulled. Expected: at n≥50
   the MLP is no longer starved → a real linear-vs-MLP gap emerges or it does not. Falsifier: **positive
   gap on the later ckpt** ⇒ "less-linear" (info intact, readout mismatch) ⇒ readout/schedule remedy;
   **no gap** ⇒ D1 regression is NOT a decode-capacity artifact ⇒ escalate toward coordinate-frame /
   highway-normalisation (cf. `c0b22b7`). **This run established the method + the D≫N fix + a directional
   step-6500 read (no nonlinear advantage, gap −15 %/−39 %).** Gate: D1 (D-004 — no config change from a
   BLOCKED gate). Artifact: `Implementation/incoming/2026-07-11-d1-probe-capacity-ladder/`.

B. **Characterise Sub-JEPA subspace-SIGReg (arXiv 2605.09241) as the `iso_active` remedy.** Goal: raise
   `iso_ratio_active` (0.27 at 6500) toward the 2605.26379 optimal-planning precondition by regularising
   the active-k subspace toward Gaussian instead of all 2048 dims (global iso 2e-8, budget wasted on the
   dead tail). Method: design-note first (which dirs, loss form), then a 4060 smoke arm through the
   bake-off harness; primary read = `iso_ratio_active` + `rms_offdiag_corr` (orthogonality instrument),
   secondary = D2/imag_rel. Falsifier: iso_active does not rise / D2 regresses → drop. Gate moved =
   ISOTROPY admissibility (NOT D1). **D-018 Tactic → escalate before any trained-config change.**

1. **[✅ DONE 2026-07-08 — spectral-sizing on real trained latents]** step-6500 `ckpt_full.pt`: fit
   R²=0.99, rank ≈43, knee 31, k*=21 → OVER-PROVISIONED. **Remaining:** re-run at the FINAL Stage-0 ckpt.
   Artifact: `Research/2026-07-08-spectral_step6500.json`.
2. **[✅ DONE 2026-07-09 → DECISION-GRADED by D-027]** K-step rollout: my first arm (K=2 vs K=1, −64 %
   @1-step, "K must match horizon") fed the loop's operative K=4 arm (`859caa8`, imag_rel 8.13→1.03);
   **Sayed accepted D-027 (rollout_k=4 for all post-30k training, `ff6a409`).** Backlog items 2b/2c
   (operative K sweep + extend horizons) are **RETIRED — settled at K=4.** Artifact:
   `Research/2026-07-09-...md` + `Implementation/kstep_bakeoff_probe/`.

## P1

3c. **REF-A/REF-B reference builds (Sayed 2026-07-09)** — full build plans in
   `Project Steering/REFERENCE_ARCHITECTURES.md`: REF-A frozen-DINOv3 WM (feature-cache training,
   stability measures specified) behind the K-step arms; REF-B 4B vision-action E2E (no WM,
   budget-matched) post-30k on pod1. Pre-registered decision rules included.
3d. **Multi-cam encoder optimization — experiment #1 (batched multi-cam encode)** — extend
   `latency_cnce_baseline.py` with N in {1,4,7} batched encodes on the 4060; expected ~2-2.5x
   single-cam latency at N=7 (not 7x). Investigation doc:
   `Research/ENCODER_MULTICAM_OPTIMIZATION.md` (attack order + production-readiness table).


3. **RoPE + AdaLN conditioning** — one-lever bake-off vs FiLM/learned positional embedding
   (2512.24497 "AdaLN+RoPE best"). Smoke-scale first (d256 on 4060, 1k steps, probe fit — reuse the
   `kstep_bakeoff_probe` harness), promote to Colab arm only if smoke shows ≥ +2% probe fit. Falsifier:
   Δ within noise → close. **Prior lowered (arXiv 2605.08567):** AdaLN vs cross-attn is a wash for
   LOW-dim actions and our actions are 2-D → expect small Δ; keep AdaLN (not cross-attn), test cheap.
3b. **[✅ DONE 2026-07-10 — orthogonality/isotropy instrument]** Shipped `spectral_orthogonality.py`
   (intake `2026-07-10-orthogonality-instrument/`, 8 tests): step-6500 → `iso_ratio_active=0.250`,
   `cond_active=246`, `rms_offdiag_corr=0.428` → **NOT-YET-ADMISSIBLE** (SIGReg isotropy not converged;
   D-021 "optimal" reading waits for the final ckpt). **⚠ Branch `worktree-arch-inf-20260710` UNMERGED in
   main — recommend orchestrator merge.** Cross-checked by 2026-07-11 (iso_active 0.269, active_k 19).
   **Remedy candidate = Sub-JEPA subspace-SIGReg (new P0 #B).**
4. **H4 arm-B: frozen DINOv3 world model — IN MOTION BY THE LOOP (`cda93df`, 2026-07-11):**
   `stack/scripts/dino_precompute.py` shipped (v3-with-v2-fallback, latest-frame per-timestep, fp16 token
   grids) = step (a) of the design below. WATCH: pick up steps (b)/(c) (predictor-on-frozen training +
   gate-matched probe comparison) if the loop doesn't reach them. Original design fixed:**
   (a) precompute DINOv3-**B/16** features once over the comma epcaches (16×16 grid @256px matches
   our readout geometry; pod2 post-arms or Colab T4 — embarrassingly parallel); (b) train
   predictor+readout on frozen features (`--data cached` path, ~110M trainable, low VRAM, fast);
   (c) same D1/D2 probe protocol vs our 15k ckpt (preview) and 30k (decision-grade), same held-out
   routes, I7-checked. B/16 = fair-compute match to our 99.5M encoder; DINOv3-L later as the
   foundation-scale upper bound. Caveat stands (2512.24497: DINO > V-JEPA for planning readouts —
   the arm most likely to beat us; be honest). Owner: loop or Wed agent, whoever reaches it first;
   queued behind the K-step arms on pod2.
4b. **E2E behavior-cloning reference arm (Sayed ask 2026-07-09 — the D4 opponent):** same encoder
   trunk + direct action-regression head (camera→steer/accel), trained on the same comma caches at
   matched steps via `--data cached`. Purpose: gate D4's learned baseline — does imagine-and-select
   beat direct regression at matched data/compute? Closed-loop comparison in CARLA once the
   camera path exists; open-loop preview = action-prediction error + D2-style ranking. Cheap
   (~100M trainable, reuses everything).
4c. **Pixel-prediction WM reference (Phase 1, small-scale, budget line):** generative-decoder
   variant to make "latent beats pixels on sample efficiency" (H3) measured, not cited. Explicitly
   NOT Phase 0.
5. **Tactical horizon ablation** — measure D2 at horizons {8, 16} separately (gate runner already
   emits per-horizon rows); decide if the 16-horizon head earns its 26.5M params.

## P2 / theory watch

6. **σ-gated tactical MoE** (route on ImaginationField.logvar) — WP4/Phase-1; needs epistemic
   interface; design note first.
7. **SIGReg-vs-spectral-contrastive theory gap** (2606.27014 constants don't transfer to SIGReg) —
   watch Balestriero/Klindt/PKU-Yisen-Wang lineages for a bridging result; escalate if a paper
   directly bounds SIGReg-trained planning regret.

## Done / retired
- (2026-07-08) Gate runner D1–D3 + spectral module shipped via intake; integrated with D-017 rework.
