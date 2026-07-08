# Architecture & Inference — Experiment Backlog

Prioritized roadmap (D-020 §4). Each run: execute ≥1 item, report measured numbers, re-prioritize.
Format per item: goal / method / resource / expected number / falsifier.

## P0 — next run

1. **Spectral-sizing on real trained latents** — validate D-008 readout dim 2048 against the
   knee of the action-conditioned transition spectrum (arXiv 2606.27014 method).
   Method: pull latest `ckpt.pt` from pod (`/workspace/runs/`), run
   `stack/scripts/run_spectral.py` on ≥2k held-out comma windows locally (4060).
   Expected: knee 20–60 (step-5000 preview: R²=0.997, erank≈35, knee≈22); verdict
   OVER → efficiency-moat evidence. Falsifier: knee > 512 ⇒ 2048 justified, H3 story weakens.
   Wall-clock ~30 min. Feeds proposed D-021.
2. **K-step rollout loss bake-off (K=2)** — test the "multistep-as-augmentation" finding
   (arXiv 2512.24497) on our predictor.
   Method: branch trainer config, feed predictions back for K=2 alongside single-window loss;
   2×3k-step arms at matched compute on Colab A100 or idle pod; compare D2 dir-acc + imag-rel
   on the same probe protocol. Expected: D2 ≥ +0.02 or drop it. Falsifier: no D2/D3 improvement
   at matched steps → record negative, close item.

## P1

3. **RoPE in FiLM/AdaLN conditioning** — one-lever bake-off vs learned positional embedding
   (2512.24497 "AdaLN+RoPE best"). Smoke-scale first (d256 on 4060, 1k steps, probe fit),
   promote to Colab arm only if smoke shows ≥ +2% probe fit. Falsifier: Δ within noise → close.
4. **H4 arm-B prep: frozen DINOv3 encoder path** — implement the frozen-encoder variant behind
   a config flag in a prototype (Implementation/ folder, NOT stack/); measure probe fit on 500
   comma windows vs our step-latest checkpoint. Caveat from 2512.24497: DINO > V-JEPA for
   planning readout — this is the arm most likely to beat us; be honest.
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
