# PRE-REGISTRATION — MM-E2: the three-armed g_tac producer cross-agreement

**Written** 2026-08-28 ~03:05, before any compute · **Register row** D-DATA-GTAC-b
(OPEN) · **Owner at execution** DataFlyWheel (label producers are its domain);
design pinned by the Master Mind so the run cannot drift · **Tier** label-space
instrument validation (no model, no GPU).

## Why this exists

HEAD carries **two g_tac producers built blind to each other** plus a third arm
that abstains — and they already disagree on CORRIDOR_OFFSET (2-vs-1, register
row). "Which producer does the programme keep" is a decision, so the experiment
that decides it is pre-registered, with the two failure modes that already burned
us built in: **raw agreement between two ~63–66 % firers is a base-rate artefact
(C136)**, and **a probe tuned on what it scores manufactures results (2026-08-22
rules)**.

```yaml
hypothesis: MM-E2 (executes the open register row D-DATA-GTAC-b)
arms:                       # label PRODUCERS over IDENTICAL windows
  A: tanitad.data.g_tac_geom          # curvature-relative (kepler, 9e61f1a7c)
  B: tanitad.data.tactical_goals      # raw |peak_lat| >= 1.0 m (v6.1-era)
  C: the v7 emitter (s2_geom_emit_v7) # CORRIDOR_OFFSET: abstains by design;
                                      # LON axis + junction turns comparable
data: the parity census clip set kepler's raws already cover (n=2,400 clips,
  69,447 windows) where pose data is reachable; if only a subset is reachable
  on the executing box, the subset is stated and the run is labelled
  NON-PARITY-SUBSET — admissible for producer-vs-producer, never for coverage.
held_constant: [windows, 2-6 s band, pose source, no threshold changes to ANY
  producer — this run SCORES producers, it never tunes them]
controls:
  chance_kappa: shuffled-clip pairing per axis — MUST read ~0, else instrument
    bug, read NOTHING (committed)
  base_rates: per-token marginals printed beside every agreement number
  reference_arm: junction turns vs Alpamayo's own turn labels (the only
    independent reference in reach; the DataFlyWheel's v7 calibration measured
    precision 70.6 % / recall 61.3 % on it — arms A and B are scored against
    the SAME reference, same clips)
primary_reads:
  1. per-axis Cohen's kappa A-vs-B (chance-corrected; RAW agreement is banned
     from the verdict line)
  2. CORRIDOR_OFFSET co-firing vs the independence product (A-rate x B-rate)
  3. A-vs-reference and B-vs-reference precision/recall on junction turns
```

## Committed outcomes

| outcome | criterion | consequence |
|---|---|---|
| **SAME-QUANTITY** | axis kappa ≥ 0.6 AND reference scores within each other's CIs | keep ONE producer (the reference-calibrated one), delete the other from the write path — duplicate-producer resolution |
| **DIFFERENT-QUANTITY** | kappa < 0.4 with clean chance control | both stay, but the emitted TOKEN NAMES must diverge so downstream cannot conflate them; neither supervises the contested token |
| 🔶 **GREY** | 0.4 ≤ kappa < 0.6 | no keep/delete decision; report numbers, escalate to the PI |
| **REFEREE-SPLIT** | kappa high but reference scores differ beyond CI | agreement is shared BIAS, not correctness — both producers' thresholds go back for calibration (the C136 lesson realised) |

⛔ **CORRIDOR_OFFSET is NOT decidable by this experiment alone.** No independent
reference for "deliberate lane offset" exists in PhysicalAI (no lane graph —
settled, five probes). Cross-agreement can only show the derivers copy each
other's base rate. The cheapest real reference is a **human audit of N=50 fired
windows** (frames are already banked in the alpamayo-label-build package for
spot-check tooling); until someone runs that, CORRIDOR_OFFSET stays
`NOT_YET_EXTRACTABLE` **regardless of kappa**, and the ⛔ no-supervision line in
the register stands.

**Side work item, same package:** rename the v7 emitter's LOCAL
`tactical_goals()` (name-collides with the module; already made one grep audit
lie — 2026-08-28).

**Cost:** CPU minutes, 0 GPU. **Banking:** agreement matrices + kappas + base
rates as raw JSON beside the RESULT; register row updated in the same turn as
the read.

---

## AMENDMENT A — recorded 2026-08-28 late, before execution

A FOURTH producer exists: the PI-designed CoT side-only route (perception-admitted via the
grounding-box test — exactly the independent-reference class this prereg demanded, arriving
by a different door). Two design updates, neither weakening the original rules:
1. The experiment adds arm D (CoT side-only) where clips overlap. ⛔ The standing fact is
   NARROWED, not retired: *cross-agreement between the GEOMETRIC producers still cannot admit
   CORRIDOR_OFFSET* — geometry is now declared structurally blind to a held offset
   (arc-removal cancels it), so a geometry-vs-geometry kappa on this token measures shared
   artefact by construction. Geometry arms are scored on the OTHER axes only.
2. Arm D enters with its own validation record (one failed check with physics on file) and is
   the only arm permitted to emit the token, per vocab_v7's TACTICAL_GOAL_NEEDS_PERCEPTION.
The lane-detector reference (LAB-RUN-002 design) remains the stronger eventual reference;
urgency lowered, requirement unchanged (chance baseline; the reference must be able to say NO).
