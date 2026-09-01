<title>SPEC - g_tac producer cross-agreement (D-DATA-GTAC-b)</title>

# SPEC - E-LAB-DATA-0901: do the two `g_tac` producers agree?

**Trigger.** `LAB_BACKLOG` ranked row **7** (P0, unblocked) = register **D-DATA-GTAC-b**:
*"measure cross-agreement of the two blind-built g_tac producers (`tactical_goals.py` vs
`g_tac_geom.py`) and settle the CORRIDOR_OFFSET 2-vs-1 contest"*. Rationale on the row:
*"two producers of the SAME label family are live with unmeasured agreement - a silent fork
in the canonical labels."*

## Hypotheses, both outcomes committed IN ADVANCE

| id | hypothesis | outcome A | outcome B |
|---|---|---|---|
| H-GTAC-1 | The two producers are two implementations of ONE label family, differing by thresholds | measure the agreement rate; settle the threshold contest | **they are not the same family** - the emittable sets are near-disjoint and "agreement" is undefined over most of the vocabulary; the ROW must be re-scoped |
| H-GTAC-2 | Where both emit the same token, the geometric arguments agree | the producers can be reconciled numerically | they disagree numerically too, and one is wrong |

**Pre-committed criteria.**
- H-GTAC-1 -> outcome B iff `|A_tokens & B_emittable| / |A_tokens| < 0.5`.
- H-GTAC-2 supported iff, on scenarios where both emit the same token, the primary argument
  agrees to < 0.5 m.

## Controls - and why they are not optional here

⛔ **Two producers that both always ABSTAIN would score 100 % agreement.** The agreement number
is meaningless without controls that force a known answer:

| id | control | must read |
|---|---|---|
| K1 | constant-speed straight run - NO stop exists | neither producer emits STOP_POINT |
| K2 | clean decelerate-to-zero - a stop exists at a known arc | both emit STOP_POINT (for any producer claiming the token) |
| K3 | sub-threshold creep (v < `V_STOP_MS`) | degenerate: RECORDED, explicitly NOT SCORED, so it can neither inflate nor deflate the rate |

## Method

Both producers take the identical contract - `poses [T,4]` = (x, y, yaw, v), a key/present index,
and the SAME 2-6 s tactical band - verified in source before running, so no adapter is involved
and neither is starved of an input it would normally receive. Eight analytic unicycle
trajectories at 10 Hz (straight, decel-to-stop, creep, two turns, lane change, evade-and-return,
stop-and-go), key index 10. CPU only; no GPU, no pod, no Thor.

## ⚠️ Scope limit, stated BEFORE the run

Analytic trajectories can **detect** a disagreement and can measure agreement **on the
intersection**. They **cannot** settle a threshold contest on the real corpus, because a
threshold is tuned to the corpus distribution. **No threshold claim will be made from this
script**, and any agreement rate reported is a SCENARIO-SET rate, never a corpus rate.
