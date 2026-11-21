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
> ⇒ **T0 and T1 are both OPEN LOOP. T2 is the only closed loop, and it is not
> provisioned — so the programme has published NO closed-loop number.** Read
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
| **T2** | ⭐ **THE ONLY CLOSED LOOP.** The trajectory CONTROLS the vehicle and the sim re-renders, so the next observation is a consequence of the model's own output (AlpaSim/NuRec, or a real test vehicle) | true closed-loop driving incl. scene interaction | "closed-loop driving" — ⛔ **NOT YET PROVISIONED, so we have published NO closed-loop number** |

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

## Amendment 2026-11-20 — the T1-kin executor carries a heading bound (PI, decision card D16-D)

**PI, verbatim:** *"Approve all four"* (decision card D16, `E:/Projects/TanitAD-artifacts/DECISION_CARD_V8F_HANDOVER_2026-11-20.md`,
sha256 e40c02ca88334fab, ~05:37Z, relayed by the v8proof coordinator), confirmed to the MM first-hand: *"Yes, all four"*.

1. **Executor of record.** From 2026-11-20 the T1-kin tier runs through `stack/scripts/v8f_t1_eval_hb.py` with
   `--kappa-max 0.2`: frame by frame, the executed heading may turn by at most 0.2 1/m times the distance travelled
   (a 5 m turning radius). Positions and speeds pass through unchanged. The pinned `stack/scripts/v8f_t1_eval.py` is not edited;
   the wrapper replaces its `hermite_frames` for one run and restores it.
2. **Why.** The pinned executor holds the heading only while the plan tangent is below 1 mm per 0.2 s, so near a standstill
   the executed heading follows plan jitter. MEASURED (card ADD67, row 122 at bb061af6, T1-kin, UE_s0/UE_s1, 141 episodes):
   heading offset below 2 m/s 52.9 deg without the bound and 6.6 deg with it; all four families and the unsafe rate better
   under the programme's convention, no family worse in four settings.
3. **Every T1-kin number states its executor:** `T1-kin/hb0.2` (the tier of record) or `T1-kin/old` (before this date).
   A comparison across the two executors is invalid unless it is an overlap read on the same arms and windows.
4. **One overlap read, old and new executor:** the coordinator's card ADD68 for the 10k line, and the N1DW deciding cells
   (D1-HO) for the v8f-mini line. Cards frozen before this date keep their pre-registered executor as the read of record and
   report the bounded read beside it.
5. `--kappa-max none` reproduces the old executor exactly (the same function object); it exists for overlap reads only.
   Contract: `stack/tests/test_v8f_t1_eval_hb.py` (identity where the bound cannot act, the analytic step, the seam, pass-through
   of positions and speeds, restore; deleting the bound line turns it red).
