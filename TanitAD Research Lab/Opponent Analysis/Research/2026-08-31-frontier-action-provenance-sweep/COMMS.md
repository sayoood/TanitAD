<title>COMMS — E-LAB-OPP-0831</title>

# COMMS — E-LAB-OPP-0831

## ⛔ ESCALATION 1 — one of OUR novelty claims is on hold, and it must not ship until read

**Recorded 2026-08-30:** *"a **HOLD-ACTION FLOOR the model must beat** — not found in any
driving world model."*

**Status: ⚠️ FLAGGED — AT RISK, NOT RESOLVED.** `2607.15898` (Orbis 2) is *reported* to run
counterfactual trajectory scaling against an unaltered-ground-truth arm. That report is
**RELAYED**; I have not read the paper. Per this package's own admissibility rule, **a novelty
claim may be neither upheld nor retracted on relayed evidence.**

**Action requested:** assign the read of `2607.15898` (banked, in `Library/papers/`) before the
paper restates this claim. ⛔ **An unassigned flag is how an unchecked claim ships** — this is
the one item here that has a deadline attached to a publication, not to a gate.

## ⛔ ESCALATION 2 — a comparability rule for the EvalFlyWheel

**Every EPDMS/PDMS number — ours or a competitor's — must carry the NAVSIM version AND the
changelog date it was computed at, not just the split name.** From the official README,
verified today:

- **2025-04-28 (v2.2):** *"Fixed bug in `openscene_meta_datas` for `navhard` and `warmup`…
  please re-download and use the new data"* — ⭐ **names `navhard_two_stage`, the split this
  programme pinned on 2026-08-29.**
- **2025-09-29:** a fix to *"metric filtering where `multiplicative_metrics_prod` and
  `weighted_metrics` were not correctly excluded by the human filter"* — the
  human-forgiveness machinery itself.
- **Latest release is still v2.2** ⇒ the *version string does not reveal either change.* That
  is the trap: stable version, moved basis.

⚠️ **This does not challenge the pin.** It asks that the pinned target be **re-stated with its
date basis**, so a future comparison cannot silently straddle a scoring break.

## ⛔ ESCALATION 3 — a paper for Architecture & Inference to read in full, not cite

`2606.12987` is the closest published analogue to **three** open gate items at once: a CAN
command channel in our exact `(steer, accel)` form; a *"shared-present anchor"* diagnosis that
is P5's pathology named in a driving model; and a **Δt=4, 1.7 M-parameter jump model**
recovering **1.02× GT motion magnitude** where single-pass models capture *"less than half"*.
⚠️ It is latent diffusion at compact scale on 150 nuScenes scenes and changes several things at
once — which is why it needs a read, not a citation.

## Claims-register touchpoints

- **D-ANTIECHO-PRECEDENT** — ⚠️ **status changed to FLAGGED** on the hold-action-floor sub-claim
  only. The other four probes in that row are untouched. ⛔ No retraction: a retraction on
  relayed evidence would be the same error in the opposite direction.
- **Benchmark portfolio (2026-08-29)** — no GO/SKIP change. The navhard-two-stage pin stands;
  what is added is the **date basis** requirement and the `2608.04896` audit as context.
  ⚠️ The audit is **single-stage navtest** under a named backend condition — it does **not**
  automatically reach our two-stage target, and must not be cited as though it does.
- **V7 gate P2(b)** — no status change; closure is the PI's. F1 supplies the first **verified**
  precedent for a genuine command channel, in our own action form, on nuScenes.
- **V7 gate P4/P5** — no status change. F2 is the first driving-domain evidence that the fix for
  *"restating the present"* is a temporally abstract step.
- ⛔ **Nothing here may be quoted from the RELAYED tier.** Two figures circulating from the
  sweep (GAIA-2's action parameterisation; a 79.6-vs-74.0 EPDMS pair) **failed my own
  verification** and are excluded.

## Handovers

| to | what |
|---|---|
| **Benchmarks & Evals** | the date-basis rule; `2608.04896` with its five scope conditions intact; ⭐ the external precedent for our floor doctrine — *a benchmark whose forgiveness rule lets a blind probe outrank human replay is the eval-side twin of the model-side failure we already measure*. |
| **Architecture & Inference** | full read of `2606.12987`; and R3 (ReSim) as a possible **third P2 lever — action-space coverage** — which is neither the channel (b) nor the target (d). |
| **Data Engineering** | R3 connects to L2D's native **13.8 % STUDENT (learner-driver)** split — a non-expert action source already in a corpus we have adapted. |

## Integration status

**NONE REQUIRED, and none performed.** No code, no benchmark run, no leaderboard, no pinned
target altered. Three items need a human: the assigned read in §1, the rule adoption in §2, and
the full read in §3. All are named here rather than left in a file nobody opens.
