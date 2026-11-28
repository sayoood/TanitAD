# SPEC addendum A4 — label-side attribution bounds (pre-registered BEFORE they are computed)

**Written 2026-10-04 after A2/A3 were fitted on TRAIN (linear: CV ADE 1.469 vs V0 1.484; gradient-boosted:
CV 1.618 / in-sample 1.485) and before either was scored on EVAL.** POST-HOC. These are **NOT levers** — each uses
the label to restrict the candidate set — they ATTRIBUTE the pick-vs-oracle ADE gap so the next (training) lever is
ranked by measured size (Rule Zero 5).

On EVAL seed 0, emitted pick = argmax of the SHIPPED E9 score within `reach_keep` restricted to (empty → V0):

| id | restriction (label-side) | what it bounds |
|---|---|---|
| B1 | `dir(c; τ_c) == dir(GT)` on every classified window | any lever that fixes turn DIRECTION (incl. a perfectly timed nav/tactical filter) |
| B1t | B1 on GT-turn windows only (V0 elsewhere) | a perfectly TIMED nav filter (the nav token is right-sided on 94 % of informative GT-turn windows) |
| B2 | `|θ(c) − θ_gt| ≤ 15°` | any lever that fixes the turn MAGNITUDE (under/over-turning) |
| B3 | `|L6(c) / L6(GT) − 1| ≤ 0.10` (path length to 6 s, i.e. progress / speed) | any lever that fixes the SPEED PROFILE |
| B4 | B2 ∧ B3 | heading + speed |

Reported per class (turn / straight / gentle / all): ADE, FDE, along/cross (GT-tangent decomposition), the
dir-correct rate, paired episode-cluster bootstrap Δ vs V0 — beside the full ORACLE. Reading rule (fixed now): the
bound with the largest all-window ΔADE names the dominant term; the turn-window column names the dominant term for
route following.
