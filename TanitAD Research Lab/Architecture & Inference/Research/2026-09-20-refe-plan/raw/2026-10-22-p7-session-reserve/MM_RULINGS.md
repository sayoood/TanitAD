# MM rulings on `SPEC_P7.md` §6 (2026-10-22, Master Mind)

These rulings are made under the PI's *"follow your recommendation"* (D-SESSION-RESERVE, decided 2026-10-22). The PI may overrule any of them **before the first P7 training input exists**. After that point, a change voids the reserve.

## Q1 · Reserve unit: **R-A**, as registered. R-B is NOT used, and the two are never combined.

**Why R-A was kept.**
- It is the only draw whose constants were fixed before any closure measurement (`SESSION_RESERVE.md` §1, order of events).
- R-B's constants were set after R-A had been drawn.

**Why R-B fails (MEASURED, `raw/reserve_vehicleday.json`, `SESSION_RESERVE.md` §5).**
- It has **5 independent vehicle-days**, from 4 vehicles in 2 cities.
- §4 requires session-grouped K = 4 folds and a cluster bootstrap. Five units can carry neither.
- **This is structural, not bad luck in the draw.** navtrain's 103,037 bank scenes sit in **46 vehicle-days**, about 2,240 scenes each. Any reserve of about 10 % of scenes closed at the vehicle-day level therefore has only about 4–5 units. Reaching about 10 units would cost about 20 % of P7's training scenes, which is a larger cost than the headline can afford.

**What R-A cannot answer (MEASURED: navtest shares 0 of 20 vehicle-days and 1 of 9 dates with navtrain; all 16 R-A sessions keep a same-vehicle same-date sibling in training).**
- ⇒ **HYPOTHESIS:** the seen→unseen gap measured on R-A, and any pessimism κ fitted on R-A, are **lower bounds** for navtest.
- P7.2's prereg must say this.
- It must report R-A's gap next to P7's navtest gap.
- It may not treat a κ fitted on R-A as calibrated for navtest.

**Cheapest instrument for that gap, added now at zero data cost (to be pre-registered in P7's prereg before step 0):** a **sibling-dose diagnostic** inside R-A.
- The 16 reserve sessions differ in how many same-vehicle-day sibling scenes stay in training.
- At the P7 read, regress each session's generalisation gap (PDMS and the four families, reserve vs a seeded same-size seen subset) on log(1 + sibling scenes). Report the slope with a session-cluster bootstrap.
- **Descriptive only:** extrapolating to zero siblings (navtest's case) lies outside the measured range and decides nothing.
- A slope separated from zero is the evidence that would justify a vehicle-day reserve in P8.

⛔ **Not available on banked artifacts (checked against `D-REFE-SESSION-EXPOSURE-1`).**
- No P2 stratum is session-unseen but vehicle-day-seen. The 21 held-out sessions are all session-seen through M-PDM, and navtest, the only session-unseen set, is vehicle-day-unseen and already used up.
- Splitting C2's banked navtest tokens by stratum to choose P7's design would feed navtest into design, so it is refused.

## Q2 · Attribution: **(a) accept the bundle**, claiming only "P7 vs P6". Run (c), the S2 graft pair, alongside it if compute allows. Run (b), the head-off twin at +6.2 A40-days, only if P7 reads ADOPT and the head becomes the claim.

The agent's recommendation is accepted.

## Q3 · Reserve size: **16 sessions stays.**

- Going to 24–32 sessions costs another 5–10 % of training scenes for only about 1.22–1.41× precision (√1.5, √2).
- The reserve fits one or two scalars. It does not judge P7; navtest does.

## Q4 · Held-out exclusion: **CONFIRMED.**

- The 21 SFT-held-out sessions were read by S1, Stage 0 and C3.
- Putting them in the fit set would place a design-time look inside it.

## Q5 · D9 metric cache on HF: **YES.**

It is backed up at P6 resume and built for P7 from day 1. This is the 3-C bank lesson: an input that lives on one pod is a single point of failure.

## Q6 · Seed replicate (+6.2 A40-days): **DEFERRED to P7's read; the compute is the PI's call.**

- It is required before any lever claim whose margin over P6 is under about 3 × F (F = 0.33 PDMS, `RESULT_P2REP`, one draw), so under about 1.0 PDMS.
- Above that margin, the training-variance question (`H-ESTIM-SEED-1`) is still named in the report. It is not silently dropped.

## Q7 · The 0-GPU wiring: **AUTHORISED now.**

The work is split into scoped deliverables, so that P7 is launch-ready when P6 reads:
- **P7-WIRE-RESERVE:** exclude the reserve from every log-keyed input. It must cover the bank's rank-0 and rank-1 rows, the D9 cache, the aux label store, image zips and calibration rows. Exclusion is by session, and every input must contain only keys whose log is resolved.
- **P7-WIRE-HEAD:** the P7.1 head, plus its `launch_gate` entries for G-DVB and G-LIVE.

**What both deliverables must meet:**
- Each needs a deliberate-regression arm that turns the gate RED.
- The derived tuple count is measured by the filter and pinned before step 0.

## Conflicts reported by the spec (§6)

| | conflict | ruling |
|---|---|---|
| (i) | bank tuple count: 189,888 vs 167,300 | **167,300 is authoritative.** It is MEASURED on the built bank and pinned in `launch_gate.P6_BANK_TUPLES`, and the trainer preflight reads it. `SPEC_P6`'s figure is a pre-build figure. The doc correction goes with P7-WIRE-RESERVE. |
| (ii), (iii) | the aux spec's `launch_gate` status and `train.py` anchors are stale | P7-WIRE-HEAD works from the **tip**, never from the aux spec's line anchors, and corrects them in its package. |
