# PRE-REGISTRATION — E6: the hierarchy data-efficiency claim (§3.6)

**Registered 2026-08-10 ~14:30Z, BEFORE any arm runs. Both verdict paragraphs pre-written.**

## The claim under test (PI, 2026-08-08, verbatim intent)

*"Im sure that this concept will make the learning more efficient (with far less data)"* —
the hierarchical tactical/strategic decomposition should reach a given driving quality with
LESS data than a monolithic head of the same capacity. This prereg turns "sure" into a
falsifiable curve.

## Arms

- **H (hierarchical):** frozen trunk + E4 stage-0 (φ_tac/goal fan/selector, as gated by
  e44_gate.json) + E5 goal-conditioned operative readout (g_tac token). Trainable params: P_H.
- **M (monolithic):** frozen SAME trunk + a single readout head, **params-matched to P_H
  within ±5 %** (width-scaled), same losses minus the hierarchy terms, same label budget.
- Grid: **{150, 300, 600} episodes** (nested subsets, fixed seed, episode-level draws from
  the canonical corpus — parity hash carried; the 150 ⊂ 300 ⊂ 600 nesting is mandatory so
  the curve is monotone-comparable).
- Identical steps/optimizer/schedule per cell; 6 cells total (~6 GPU-h + eval).

## Measurement (per cell, all four families, T1 primary)

- T1 ADE (action-closed loop, `taniteval/tools/t1_eval.py` contract) — the headline.
- T1 S-curve reproduction rate (the lateral-capability probe).
- Goal quality: goal FDE@4 s of the selected tactical goal (H only; M reports n/a with
  reason).
- Paired episode-cluster bootstrap on every H-vs-M delta, per cell. No pooling across cells.

## Pre-registered gates

- **G-EFF (the claim holds):** at 150 AND 300 episodes, H's T1 ADE is CI-separated better
  than M's, AND H at 300 episodes is not CI-worse than M at 600 (the "half the data" read).
- **G-NULL (the claim fails):** neither separation holds — the hierarchy's value (if any) is
  not data efficiency at this scale.
- Partial outcomes (exactly one separation) are reported as UNDECIDED — no narrative rescue;
  the next lever is a bigger gap (75-episode cell), not re-reading the same numbers.

## Bound verdict paragraphs

**If G-EFF:** "MEASURED: the hierarchy learns from less data — H@300 ≥ M@600 (paired CI).
The §3.6 claim stands at trunk-frozen stage-0/1 scale; next test is trunk-joint (v1.9)."

**If G-NULL:** "MEASURED: at stage-0/1 scale the hierarchy shows no data-efficiency
advantage over a params-matched monolith. The claim does not transfer from intuition at this
scale; the hierarchy's justification must rest on the OTHER two claimed axes (inference
efficiency, edge-case generalisation — S-rate and tail metrics), or on trunk-joint training.
The concept is NOT abandoned on this null; the scoped-down claim is."

## Dependencies & status

Depends on: E4.4 gate (running), E5.1/5.2 (goal-conditioned readout — not yet implemented).
- [ ] E5.1 implemented → arms buildable
- [ ] launched
- [ ] verdict appended here + registry row
