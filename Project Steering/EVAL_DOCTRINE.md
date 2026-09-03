# EVAL DOCTRINE — the three tiers (BINDING, 2026-08-09)

> ⛔⛔ **AMENDED 2026-09-02 BY THE PI — T1 IS NOT A CLOSED LOOP.** Verbatim: *"the fact
> that the predictor is consuming the output of the planner of the World model based
> system is also open loop because the trajectory of the model is not affecting the new
> ego data (this will be fed from the eval ego data). Closed loop means that the
> trajectory is controlling the vehicle in the simulation, this can be done e.g. in
> AlpaSim or in a real test vehicle."*
>
> **This document manufactured the error.** The T1 row's *"may be quoted as"* cell said
> `"closed-loop (imagination) driving"`, and that licence is how the word reached the
> registry, the reports and the model cards. T2 was ALREADY defined as the real thing,
> so T1's name was stealing T2's word.
>
> ⇒ **T0 and T1 are both OPEN LOOP. T2 is the only closed loop.** ⛔ **CORRECTED
> a few hours after this banner was first written:** I asserted here that T2 was
> not provisioned and that no closed-loop number existed. Both are FALSE — see
> RETRACTION_LOG #14. ⭐ **We DO have closed-loop numbers**: the AlpaSim/NuRec panel of 2026-08-03 (PROGRAM_OVERVIEW §5.0.1) — 9 rollout starts x 50 ticks in a reconstruction RENDERED ON THE JETSON THOR, 437 paired windows, four families, `stack/experiments/alpasim-gsplat/`. ⚠️ But the scene has **no reactive agents**, so safety-grade metrics (collision, off-road) remain out of reach — and **neither refav1 nor refcv3 has been through that harness**. Read
> "closed loop" in any TanitAD document dated before 2026-09-02 as "self-action open
> loop". Glossary: `VOCABULARY.md`; retraction: `RETRACTION_LOG.md` entry #13.


**Origin:** PI, 2026-08-07: *"if the model is consuming at eval the future gt data, then
its not really an eval, eval must be without gt."* Measured basis: registry §1.12 — with
recorded future actions removed, v1.6/v1.7 lose lateral skill almost entirely (S-curve
reproduction 97.9 % → ~5 %; hold-action arm 0.0 %).

| tier | condition | what it measures | may be quoted as |
|---|---|---|---|
| **T0** | teacher-forced: predictor consumes recorded future actions | WM fidelity, readout quality, attribution of decode-side changes | "prediction quality" — ⛔ NEVER "driving performance" |
| **T1** | **self-action OPEN loop**: predictor consumes the decoder/planner's own actions; perception context fixed at t0. ⚠️ The trajectory never reaches the ego data, which keeps arriving from the recording — so the model is NOT driving anything | trajectory quality when the model supplies its own actions — **the PRIMARY OFFLINE eval** | "self-action open-loop (imagination)" — ⛔ NEVER "closed-loop", that word belongs to T2 |
| **T2** | ⭐ **THE ONLY CLOSED LOOP.** The trajectory CONTROLS the vehicle and the sim re-renders, so the next observation is a consequence of the model's own output (AlpaSim/NuRec, or a real test vehicle) | true closed-loop driving incl. scene interaction | "closed-loop driving" — ⭐ **PROVISIONED AND USED** (AlpaSim/NuRec on Thor, 2026-08-03); the old "NOT YET PROVISIONED" was stale. ⚠️ No reactive agents yet |

**Rules.**
1. Every registry results block states its tier. Pre-doctrine blocks are stamped
   retroactively (§1.10/§1.11 = T0; §1.12 = T1).
2. ⛔ **SHARPENED 2026-09-02.** T1 is OPEN loop, so it cannot support a claim that the
   model DRIVES. T1 supports **open-loop trajectory-prediction** claims. A claim that the
   model drives, handles a situation, or completes a route requires **T2** — which is not
   provisioned, so no such claim is currently admissible. T0 supports only prediction and
   attribution claims. *(The old wording read "requires T1 or better", which licensed exactly
   the driving claims the PI's ruling forbids.)*
3. The four binding metric families apply at every tier; the S-curve reproduction rate
   and the lag/response instruments are part of the T1 standard battery.
4. T0 remains mandatory for attribution (it is how the decel-ramp was assigned to the
   readout, not the roll) — demoted in meaning, not removed.
5. Model cards and HF pushes carry the tier of every quoted number.
6. **A census over `windows_*.pt` imports `taniteval.dump_census`; a bare glob is a
   defect.** *(C126: two double-banked dump pairs made every glob-derived "27 arms"
   census count 25 distinct models as 27 — the dashboard and LEADERBOARD generator
   double-counted one model per pair until 2026-08-18. `dump_exclusions.json` is the
   machine-readable truth; prose corrections cannot reach a glob.)* Pod-side censuses
   FILE-SHIP `dump_census.py` + `dump_exclusions.json` with the job.
