# PRE-REGISTRATION — W4b: selector recalibration on the frozen unicycle fan

**Registered 2026-08-10 ~13:25Z, BEFORE launch. Both outcomes bound below.**

## Finding that motivates this (MEASURED, registry §1.13 W4 block)

W4 passed both gates (oracle 0.1077, accel MAE 0.774, violations 0.0), but the FROZEN
selector's pick on the re-parameterised fan is near-uninformed: selected ADE **0.7933** vs
0.4056 (its own pick on the old fan). Hypothesis: this is **calibration, not information
loss** — the selection signal is still in the trunk features; only the score head's mapping
to the new fan geometry is stale.

## Arm (ONE lever)

Freeze: trunk, grounding, goal_head, W4 `UnicycleEmission`, the anchors — everything except a
NEW score head. Train ONLY a rescorer: inputs = the same per-candidate offset-head query the
emission taps (plus, optionally as a SECOND pre-registered variant, the candidate's own (a,κ)
sequence — kinematics-aware scoring), output = logits over 256. Rank loss identical to the
trainer's (margin vs GT-nearest winner). ~2,000 steps, ≤2 h pod5. Selection = argmax; no
re-rank, no WM roll (that stays W7's lever — attribution must remain separable).

## Gates (both outcomes bound in advance)

- **G1 (recalibration suffices):** selected ADE on the unicycle fan **≤ 0.45 m** on the same
  881-window grid (within ~10 % of the old selector-on-old-fan 0.4056, against a better
  oracle of 0.1077). ⇒ v5.8f assembly proceeds as W4-fan + recalibrated selector; W7 then
  attacks the remaining sel_gap (0.45→0.11 headroom).
- **G2 (recalibration insufficient):** selected ADE > 0.45. ⇒ The per-candidate conditioning
  does not carry enough selection signal for this fan; selector demotes to top-K pruner and
  **W7 (WM-roll re-rank on the clean fan) becomes the primary selection mechanism.**
  Supporting measurement to bind the pruner role: top-8 oracle on the new fan (reported
  either way; if top-8 oracle ≤ 0.15 the pruner role is viable).

## Measurement contract

Same 881-window grid (episodes<40, stride 8); report selected / oracle / top-{4,8,16} oracle
/ sel_gap via `taniteval.selgap` (episode-cluster bootstrap CI on the gap); LONGITUDINAL +
LATERAL families on the selected trajectory (four-families rule; TACTICAL = selector rank
quality, STRATEGIC n/a with reason). Tier stamp T0 (diagnostic). Evidence class MEASURED with
artifact paths. The G1 comparison (0.45 threshold) is a point-estimate gate; the registry row
carries the CI.

## Status

- [ ] launched (queued behind E4.4 on pod5)
- [ ] result banked (append verdict here + registry §1.13)
