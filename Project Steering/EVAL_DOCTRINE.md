# EVAL DOCTRINE — the three tiers (BINDING, 2026-08-09)

**Origin:** PI, 2026-08-07: *"if the model is consuming at eval the future gt data, then
its not really an eval, eval must be without gt."* Measured basis: registry §1.12 — with
recorded future actions removed, v1.6/v1.7 lose lateral skill almost entirely (S-curve
reproduction 97.9 % → ~5 %; hold-action arm 0.0 %).

| tier | condition | what it measures | may be quoted as |
|---|---|---|---|
| **T0** | teacher-forced: predictor consumes recorded future actions | WM fidelity, readout quality, attribution of decode-side changes | "prediction quality" — ⛔ NEVER "driving performance" |
| **T1** | action-closed loop: predictor consumes the decoder/planner's own actions; perception context fixed at t0 (`taniteval` closed-loop pipeline) | driving competence within the WM's imagination — **the PRIMARY offline eval** | "closed-loop (imagination) driving" |
| **T2** | perception-closed loop (AlpaSim/NuRec re-render) | true closed-loop driving incl. scene interaction | "closed-loop driving" — NOT YET PROVISIONED |

**Rules.**
1. Every registry results block states its tier. Pre-doctrine blocks are stamped
   retroactively (§1.10/§1.11 = T0; §1.12 = T1).
2. A capability claim ("drives", "handles", "improves driving") requires T1 or better.
   T0 supports only prediction/attribution claims.
3. The four binding metric families apply at every tier; the S-curve reproduction rate
   and the lag/response instruments are part of the T1 standard battery.
4. T0 remains mandatory for attribution (it is how the decel-ramp was assigned to the
   readout, not the roll) — demoted in meaning, not removed.
5. Model cards and HF pushes carry the tier of every quoted number.
