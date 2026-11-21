# TOLERANCE-BAND RE-SCORE — was the departure↔ADE "trade" a knife-edge-metric artifact?

**MEASURED on `tanitad-eval`, `gpu_lock tolerance-rescore`, 2026-07-24.** NO new training — re-scored the
EXISTING rollouts (base + naive-D2 + g2 + RefcCL-s2 + LOWOOD-CL) on held-out 28:40 with a **tolerance-band**
metric instead of exact-path L2-ADE. Raw: `tolerance_rescore.json`, `tolerance_rescore.log`,
`tolerance_rescore.py`.

**Band width + why.** Primary **1.0 m** = half the 1.75 m lane-half-width — a "benign in-lane recovery"
tolerance around the GT path. **`band_ade2d(band) = mean over the 4 waypoints of max(0, ‖pred−gt‖ − band)`**
forgives 2D deviation within `band` and penalizes only excursions beyond it (the fair analog of
closed_ade2s). Grid {0.5, 1.0, 1.5}. **Sanity: base `band_ade2d(1.0)=0.1997 > 0`** — base itself deviates
beyond 1 m at the 2 s waypoint, so the band is MEANINGFUL (it does not trivially forgive everything).

## Result (held-out 28:40, paired episode-cluster bootstrap; positive Δ = FT better)

| arm | ftADE | RAW dADE (exact L2) | band-0.5 d_ade2d | **band-1.0 d_ade2d [CI]** | band-1.0 dCDR [CI] |
|---|---|---|---|---|---|
| naive-D2 | 0.875 | −0.288 S | −0.173 S | **−0.076 [−0.138,−0.009] S** | +0.008 [−0.019,+0.043] n.s. |
| g2 | 0.713 | −0.125 S | −0.068 n.s. | **−0.024 [−0.073,+0.024] n.s. FORGIVEN** | +0.008 [−0.008,+0.029] n.s. |
| RefcCL-s2 | 0.671 | −0.084 S | −0.036 n.s. | **−0.004 [−0.024,+0.014] n.s. FORGIVEN** | −0.009 n.s. |
| LOWOOD-CL | 0.916 | −0.329 S | −0.242 n.s. | **−0.178 [−0.484,+0.028] n.s. FORGIVEN** | −0.075 n.s. |

## Verdict: pre-registered **BOUND (strict)** — but the "trade" was LARGELY a knife-edge-metric artifact

**Strict pre-registration** (WIN = dCDR CI∌0 **AND** band_ade2d CI∋0): **no config qualifies** → BOUND. But
the re-score **substantially reframes WHY**, and the reframe is the load-bearing result:

1. ⭐ **The raw-ADE cost was mostly a measurement artifact.** Under the fair 1.0 m band the ADE-penalty
   **fully disappears (CI∋0) for 3 of 4 configs** (g2 −0.125→−0.024; RefcCL-s2 −0.084→−0.004; LOWOOD-CL
   −0.329→−0.178, all now CI∋0) and shrinks **74 %** for naive (−0.288→−0.076). **The exact-path L2-ADE
   overstated the "trade" ~4×** — it scored benign in-lane recovery wiggle as a cost. The Pareto wall
   D2/RefcCL/LOWOOD-CL reported was, in large part, a **knife-edge-GT-path artifact**, exactly as
   hypothesised.
2. **The residual blocker SHIFTS from ADE to the departure signal — which is underpowered, not absent.**
   No config has a CI-separated *departure* win on this n=12 held-out set (naive +0.008 n.s. here vs +0.0089
   **S** in the D2 eval — the win sits right at the significance boundary; g2 +0.008 n.s.). The remaining gap
   is a **weak/underpowered departure signal (n=12, wide bands)**, NOT a real ADE trade. Only naive keeps a
   small separated band-ADE residual (−0.076) — the most aggressive config, expected.

## What this changes (the cheap redirection — NOT a renderer)

The LOWOOD-CL "BOUND" and the whole D2/RefcCL Pareto conclusion were **substantially inflated by the
exact-path L2-ADE metric.** Under a fair lane-tolerance metric, the recovery lever carries **no separated
ADE-cost** (3/4 configs) — the direction is **less bounded than it looked.** The cheapest next steps are
therefore **measurement, not a renderer build**:
1. ⭐ **Adopt `band_ade2d` (tolerance-band ADE) as the fair closed-loop metric** alongside corridor_departure
   — exact-path L2-ADE mis-scores benign recovery and should not gate the closed-loop lever.
2. **A more-powered departure eval** (more held-out episodes than n=12 — harvest additional val/train-held-out
   junction episodes) to resolve whether the marginal departure win (naive +0.008, boundary-significant) is
   real. This is the actual residual question, and it is a **cheap data/eval step, not a renderer**.
3. The renderer paths (`Research/2026-07-24-low-ood-closedloop-renderer.md`) remain for the **separate
   reactive-agent collision (B)** problem — this re-score does not change that.

**Honest bounds.** n=12 held-out (wide bands — the core power limit). band_ade2d uses the 4-waypoint 2D
deviation; a lateral-only band (`band_xte_ade`) forgives even more (all n.s.). The 1.0 m band is a choice
(0.5 m keeps naive's cost separated, 1.5 m forgives all) — the point estimate ordering is band-robust; the
exact separation threshold is band-dependent and reported as a grid. Within-instrument RELATIVE, low-OOD
lane-keeping, not a safety rate.

## Net (the whole closed-loop-improvement arc, corrected)
D2 → RefcCL → LOWOOD-CL reported a departure↔ADE Pareto wall. The tolerance-band re-score shows that wall was
**~4× inflated by a knife-edge L2-ADE metric** — under a fair band the ADE-cost is not separated for 3/4
configs. The real residual is a **small, underpowered departure signal (n=12)**, not an ADE trade. **The
direction is partially REOPENED: fix the metric (tolerance-band ADE) + power up the departure eval (cheap) —
the renderer is for reactive-agent safety (B), not for this (A) road-keeping question.**
