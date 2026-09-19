<title>LEDGER B11 — physics-informed operators and scientific ML</title>

# LEDGER B11 — physics-informed neural operators + scientific ML

`APPEND-ONLY. Created 2026-09-13 (LAB-RUN-012) — no ledger existed; the last live primary (2604.01349) was WITHDRAWN (TRACKS 2026-09-09). Transfer question: dynamics priors the predictor now learns from scratch.`

## 2026-09-13-01 — First live B11 primary since the withdrawal: physics parameters enter as ACTION coordinates

`FULL TEXT` · arXiv **2609.10464v1** (9 Sep 2026) · SG-JEPA — full entry in `LEDGER_A2_jepa.md` 2026-09-13-01.
B11 reading: no PDE prior, no operator — the physical law is **conditioned**, not **imposed**: a scalar `g` concatenated to the action, and a rollout objective that makes the latent update compose. Zero-shot OOD gravity generalisation with 3-D error −31–48 % vs DINO-WM.
**Transfer:** the cheapest dynamics prior available to us is an *episode-constant conditioning coordinate* (limit, friction proxy, day/night) — `E-B11-SGJ-1`. ⚠️ One scalar, toy physics; gains attributed to the encoder.

## 2026-09-18-01 — Latent Generative Solver: the generative transition is the load-bearing term, and input noise buys a contraction bound

`HTML primary via summariser` · arXiv **2602.11229** · banked. Removing flow matching: 5/10-step L2RE **+145 % / +155 %** (~5× the noise ablation). Input noising `x̃ = (1−k)x + kz` with bound `E‖δ_{s+1}‖ ≤ L_T(1−k)E‖δ_s‖ + C·sup‖η‖` ⇒ contraction when `L_T(1−k) < 1`. Wins 0/16 at 1 step, 16/16 at 20; 20-step L2RE 56.1 % → 30.2 %. Limits: 2-D; isotropic noise may fail at multimodal bifurcations.
**Our position:** the third independent line (after WA-JEPA, FlowR2A) against deterministic latent regression ⇒ LR14-5; input noising is a zero-architecture drift lever (FS18-5). Also scanned: `2605.30542` (query-conditioned physically viable WMs, banked abstract-only).
