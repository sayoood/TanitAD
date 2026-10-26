# Red-team disposition (2026-09-29)

Review: `TanitAD Research Hub/Architecture & Inference/Research/2026-09-29-hicap/RT_redteam_paper.md` (3 blockers, 16 major, 18 minor).
Status after the author's pass: FIXED = text/protocol changed; MEASURED = new evidence added; OPEN = needs data or a PI decision.

| id | finding (short) | status |
|---|---|---|
| B1 | InfoNCE optimum/bias stated for the unmasked loss only; masked optimum is not PMI; toy β tuned on the test metric | FIXED (Prop. + Remark in §theory; default loss = level-wide softmax); MEASURED (`hicap_pmi_toy_v2.py`: unmasked critic reaches the PMI optimum, held-out β = 1 in 5/5 seeds; masked best β 0.5–0.75) |
| B2 | Outcome-InfoNCE cannot rank candidates (free c(s,a)); future-frame targets smuggle the action | FIXED (Prop. `future` rewritten; Tier 2 = cost heads on counterfactual costs; aux InfoNCE on exogenous descriptors only; action-shuffled control voids Tier 2) |
| B3 | H-HC6 "≤ 20 % of draws below nominal" unreachable by split conformal (≈ 50 % under exchangeability) | FIXED (spread/5th-percentile criteria; PAC form with episode as unit; §prelim reading corrected: SD 2.5–2.8× the exchangeable value) |
| M1 | Non-reactivity is causal, not observational CI | FIXED (Prop. `ident`) |
| M2 | Mask proposition non-sequitur; 0.5 s differences are lenient | FIXED (Prop. `mask`) |
| M3 | Distillation bound scope/vacuity at τ = 100 | FIXED (single-level scope, δ_s < margin/200 stated) |
| M4 | Coarse-then-compose exact only for separable scores | FIXED (Prop. `cost`; recall measured, H-HC3) |
| M5 | G6 tunes β on validation | FIXED (held-out training episodes) |
| M6 | Goal head + situation-like teacher goals reach the decision | FIXED (R4 geometry goals only; NC6 goal-shuffle; disjointness no longer claimed) |
| M7 | Ego ruling not cited; scope beyond velocity | FIXED (cited; PI decision added); ego-zeroed row reported for every family |
| M8 | Non-inferiority on ADE only | FIXED (per-family margins δ_f required in every family) |
| M9 | H-HC0 kNN includes v₀ | FIXED (v₀-only kNN arm) |
| M10 | Intro misattributions (92.2 % belongs to refcv3; registry §4.6–4.8 not in repo; tags) | FIXED |
| M11 | "Fixed scorer" numbers came from three models | FIXED (REF-C-XL restricted to 64/128/256, registry) |
| M12 | Contamination test on the wrong split | FIXED in text; screen on the 147 v7 eval IDs OPEN (BACKLOG E14) |
| M13 | Audio absence asserted with the mp4 unprobed | FIXED (wording); ffprobe OPEN (BACKLOG E8) |
| M14 | Novelty overstated / missing prior art | FIXED (hierarchical softmax, TDM, beam-optimal trees, conformal cascades cited; claims reduced to combination) |
| M15 | Strategic band truncated by clip length | FIXED (L1 effectively [8,12] s) |
| M16 | Speed ratio singular at standstill | FIXED (v(t) − v₀ with a standstill sub-vocabulary) |
| m1–m18 | minors | FIXED except m9 joint census (OPEN, BACKLOG E15), m16 count wording FIXED, m17 width assumption flagged |
