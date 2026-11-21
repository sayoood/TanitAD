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

## 2026-09-19-01 — ⛔ CORRECTION to 2026-09-18-01 (append-only; the entry above is not rewritten)

`FULL TEXT (local pypdf, 28 pp)` · arXiv **2602.11229**. The 09-18 entry was written from the fetch **summariser**. Re-read against the banked PDF:
* ⛔ **Does not reproduce:** *"LGS wins 0/16 systems at 1 step but 16/16 at 20"* (09-18 RESULT row 1). **The primary says:** LGS *"matches the strongest deterministic baseline at one step, wins on 15/16 systems at both 5- and 10-step rollout, cuts 20-step L2RE from 56.1% to 30.2%"*, at **13–77×** less dynamics-step compute.
* ✅ **Reproduces:** flow removed **+145/+155 %** (5/10-step); noising `k` removed **+26/+29 %**; context removed **+35/+34 %**; `k` and context each cost **≤ 16 %** at 1 step; bound `E‖δ_(s+1)‖ ≤ L_T(1−k)E‖δ_s‖`, steady-state bounded when `L_T(1−k) < 1`. The ablation chain is **superadditive** (+277 % vs +218 % summed).
**Consequence:** FS18-5's committed read (h=1 may degrade ≤ 16 %) **stands**. Quote the headline only as **"15/16 at 5 and 10 steps"**. Root-cause class: **summariser extraction quoted as a primary number**. Of the other summariser-read items, DriveZero verified clean; World Tokens, DriveVLA-M0 and OneWM-VLA are pre-committed for a local re-read.
