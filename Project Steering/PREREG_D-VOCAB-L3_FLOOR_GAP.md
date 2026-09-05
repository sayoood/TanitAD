# PREREG — D-VOCAB-L3: the turning-window floor gap is the two-level vocabulary, and the L=3 fix must close most of it

**Registered 2026-09-05, BEFORE the L=3 vocabulary is implemented.** Both outcomes are committed
below. Authority: Master Mind decisions `M15` / `M16` (`Decisions/2026-09-05-mm-decisions.md`);
register rows `D-MM-VOCAB-1..4`.

⛔ **This document is falsifiable by its own blob id at staging time.** Corrections are issued as a
separately staged ERRATUM, never as a silent edit — the rule that `PREREG_REFCV4B_HIERARCHY_EVAL`
needed on the same day.

---

## 1. The uncomfortable fact this exists to explain

MEASURED (T1, 14 episodes / 28 windows, `ccos` + Stage-B weights, step 21,109): **both A/B arms sit
3.6–5.9× BELOW the trivial `ha` / `ha0_ext` floors on TURNING windows.** ⛔ **refav1 does not
drive**, and no setting of `lat_logit_bias` should be shipped on this checkpoint.

A planner that is 3.6–5.9× worse than *"keep doing what you are doing"* is not merely weak on
those windows — it is **actively harmful** there.

## 2. The mechanism I am asserting, and why it is not a rescue

`ha0_ext` holds the measured yaw rate, i.e. it **holds the measured curvature** — a *continuous*
quantity read off the ego state. The planner cannot do that: it must select a **sustained**
curvature from `canonical_controls`, which today offers exactly **two** (κ = 0 and κ = 0.08 1/m,
R 12.5 m).

Against a corpus whose |κ| median is **0.00085** (R 1176 m) with the measured vocabulary crossover
at **0.04101**:

* choose κ = 0 on a real curve → cross-track error grows as ≈ ½·κ·v²·t²;
* choose κ = 0.08 on the median turn → **~94× overshoot**;
* `ha0_ext` holds the true κ → ≈ 0 error.

⇒ **On turning windows a two-level vocabulary is expected to LOSE to an ego-extrapolation floor,
by construction.** The floor is continuous; the planner is quantised to two points that bracket
almost nothing the road actually does.

⚠️ **Why this is a prediction and not a story.** It was NOT constructed to explain the floor gap —
it is the same mechanism, already MEASURED independently, that (a) makes only **38.7 %** of real
turns expressible, (b) makes the `lat_logit_bias` lever *hurt* (it forced κ = 0.08 onto windows
~80× gentler: ADE 1.6098 → 2.5025), and (c) makes `LANE_KEEP` the **vocabulary-optimal** token on
**90.6 %** of GT-turn windows. The floor gap is a **fourth** consequence, predicted from the same
constant, and it is registered here before the fix so it can fail.

## 3. Committed outcomes

Arms: the **shipped L=1 vocabulary** and the **approved L=3 vocabulary**, everything else held
constant — same checkpoint lineage, same corpus, same windows, same cost metric, same weights.
⛔ **The turning/straight split must use the MEASURED crossover (|gt_κ| > 4e-2, R 25 m), not the
inherited 1e-3 threshold** — 1e-3 is R 1000 m and counts curves the vocabulary cannot express in
either arm, which is the label-set error that produced the retracted "20.5 % turn recall".

| outcome | condition |
|---|---|
| ⭐ **SUPPORTED** | on turning windows the L=3 arm's deficit to `ha0_ext` shrinks by **≥ 50 %** of the L=1 deficit, paired episode-cluster bootstrap, separated |
| ⚠️ **PARTIAL** | a separated shrink of **< 50 %** ⇒ the vocabulary is *a* mechanism and not *the* mechanism; the remainder is a planner or cost defect and must be pursued as one |
| ⛔ **REFUTED** | no separated shrink, or the deficit grows ⇒ **the vocabulary was NOT the mechanism.** `D-MM-VOCAB-1`'s approval stands on its own oracle-table evidence, but this explanation of the floor gap is withdrawn and the search moves to the cost metric and the seed pool |

**Controls, each of which must read a known value or the panel is void:**
1. **Straight windows** (|gt_κ| ≤ 4e-2) — the vocabulary change touches nothing there, so the
   deficit must move by **≈ 0**. ⛔ A "gain" on straights means the arms differ in something other
   than the vocabulary, and the panel is void.
2. **Zero-lever replicate** — the L=1 arm run again, same flags. Per `H-ESTIM-SEED-1` a separated
   CI from a one-seed arm is **necessary, not sufficient**; the shrink must clear the rig's own
   run-to-run noise floor.
3. **The floors themselves must be BIT-IDENTICAL across arms.** `ha` / `ha0` / `ha0_ext` read no
   vocabulary. If a floor moves, the harness is not holding what it claims to hold.
4. `ha0_ext` is the **INTEGRATOR** (`refav1_arm.hold_ext_controls`), per `M11` / `D-MM-ADJ-1` — the
   closed form differs by **1.862923 m at 15 s**, more than the whole quantity being measured.

## 4. What this does NOT claim

* ⛔ **It does not claim refav1 will drive.** Closing the floor gap is necessary and not
  sufficient; beating `ha0_ext` is a floor, not a capability.
* ⛔ **It does not inherit the oracle bound.** The `cl_oraclegoal` arm read **ADE 7.47** with a
  *perfect* goal — but that arm is **CONFOUNDED**: supplying `goal_field` **skips the seed
  entirely** (`supplied` 142/142 vs `tactical_imagined` 142/142), so it is a different pipeline and
  bounds nothing. ⇒ **The prize is currently UNBOUNDED, and de-confounding it is an open work
  item** — supply the oracle goal *through* the seed path so the goal is the only difference. That
  the stream reported this instead of banking the dramatic number is the correct behaviour.
* ⚠️ **Parity.** Changing `canonical_controls` changes the ACTION SPACE. Every banked refav1 number
  is under the OLD vocabulary; each arm records its vocabulary version, and a cross-vocabulary
  comparison is inadmissible unless it says so.
