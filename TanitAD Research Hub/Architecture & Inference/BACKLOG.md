# Architecture & Inference — Experiment Backlog

Prioritized roadmap (D-020 §4). Each run: execute ≥1 item, report measured numbers, re-prioritize.
Format per item: goal / method / resource / expected number / falsifier.

## P0 — next run

1. **[✅ DONE 2026-07-08 — spectral-sizing on real trained latents]** Ran `run_spectral.py` on the
   step-6500 `ckpt_full.pt` (24 val eps, 7,176 pairs, 4060): **fit R²=0.99, rank ≈43, knee 31, k*=21 →
   OVER-PROVISIONED** (knee ≪ 512, well inside the predicted 20–60 band → efficiency-moat evidence for H3).
   Feeds D-021. **Remaining:** re-run at the FINAL Stage-0 ckpt (rank still climbing 35→43) before any resize.
   Artifact: `Research/2026-07-08-spectral_step6500.json` + note §5b.
2. **[✅ DONE 2026-07-09 — K-step rollout bake-off, first measured arm]** Matched-compute K=2 vs K=1
   (2×2000 steps, real comma2k19, 4060, 11.74 M reduced-but-real probe, OFAT-verified). Rollout is
   **nearly free (+0.5 % wall-clock, 0 params)**. **Metric lesson:** the planned falsifier (D2 dir-acc)
   **saturated at 1.0** → non-discriminative; the discriminative signal is **`imag_rel`**, on which K=2
   cuts 1-step latent-pred error vs persistence **2.914→1.049 (−64 %)** but does NOT help the 4-step
   horizon → **K must match the decode horizon**. D1 FAIL + D3 BLOCKED at this scale ⇒ no decision-grade
   claim (D-004). Artifact: `Research/2026-07-09-...md` + `Implementation/kstep_bakeoff_probe/`.
   **Superseded by P0 #2b + #2c below.**

2b. **Decision-grade K-step sweep K∈{1,2,4} at OPERATIVE scale** — the real bake-off arm.
   Method: matched-compute trained arms from the pod2 step-8k `ckpt_full.pt` (Phase C, idle-pod or
   Colab L4/A100-with-ledger-row); primary metric **`imag_rel` per horizon** (NOT dir-acc — it
   saturates) + D3 ratio once imagination horizons are extended. Expected: K=4 lowers `imag_rel` at
   horizons ≤4 without D2 regression; falsifier: no `imag_rel` improvement at matched steps → drop
   K-step. **D-018 Tactic → escalate to Sayed before it touches the trained config.**

2c. **Extend imagination horizons past 0.4 s** (predictor imagines k∈{1,2,4}=0.4 s; the plan's D3 is 2 s).
   Couples with 2b (K must cover the horizon). Method: add longer horizons to `predictor.horizons` in a
   prototype; measure `imag_rel`/D3 at 1 s/2 s. Falsifier gate: D3. Escalate before trained-config change.

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
3b. **[✅ DONE 2026-07-10 — orthogonality/isotropy instrument built + measured]** Intake pkg
   `incoming/2026-07-10-orthogonality-instrument/` (`spectral_orthogonality.py` + `run_orthogonality.py`
   + 8 tests, pure torch; target = extend `stack/tanitad/eval/spectral.py`). Measured on step-6500:
   `active_k=21` / `cov_effective_rank=24.93` **exactly reproduce** the spectral numbers (cross-check);
   `iso_ratio_active=0.250` / `rms_offdiag_corr=0.428` → **NOT-YET-ADMISSIBLE** (SIGReg isotropy not
   converged at step-6500) → the D-021 over-provisioning finding stays *descriptive*, the *optimal-planning*
   reading is NOT licensed (instrument blocks it, D-004). **Superseded by 3b-final below.**

3b-final. **Re-run orthogonality + spectral at the FINAL Stage-0 ckpt** (15k preview / 30k decision-grade)
   — the admissibility gate on any D-021 resize. Method: `run_orthogonality.py --ckpt <final> --cache-dir
   <val cache DIR>` (+ `run_spectral.py`). Expected: `iso_ratio_active` climbs toward 1 as SIGReg converges.
   **Falsifier:** isotropy stalls low → withhold any resize AND escalate "raise SigReg weight" (D-018 Tactic).
   No resize is admissible until this passes (D-004/G-AI1).

3b-live. **Live `iso_ratio_active` training row** (from the 2026-07-10 rec #3) — add active-subspace
   isotropy to the trainer's collapse-health log next to `erank`, so SIGReg-isotropy convergence is
   watchable in-flight and feeds the "raise SigReg if it stalls" decision with a direct signal. Cheap
   (reuses `spectral_orthogonality` on the readout batch). Ship as intake. Falsifier: adds >2 % step time → gate to eval-only.

3b-ham. **[Phase-1 lever, from HamJEPA 2605.20107]** symplectic/Hamiltonian cross-view predictor coupling
   as the non-isotropic alternative to isotropic SIGReg for structured driving geometry (beats SIGReg
   +3.5–10.6 probe pts on structured tasks). One-lever bake-off vs the FiLM/SIGReg baseline; **only if**
   the final-ckpt orthogonality gate shows isotropy is a poor fit for the driving cost geometry. Changes
   the trained objective → **D-018 Tactic, escalate.** Design note first.
4. **H4 arm-B: frozen DINOv3 world model — PROMOTED (Sayed ask 2026-07-09), design fixed:**
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
