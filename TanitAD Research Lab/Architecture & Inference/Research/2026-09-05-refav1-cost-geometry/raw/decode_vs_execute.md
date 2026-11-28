# `W_KAPPA` does not change the DECISION — it makes the planner refuse to EXECUTE it

**MEASURED 2026-09-05, zero GPU, from banked dumps.**
Source: `raw/kappa_by_goal_all.txt` (tool `raw/kappa_by_goal.py`, vocabulary imported
from `tanitad.models.vocab_v7`), refav1 `ckpt_ep3` step 21109, p4 panel, 40 windows /
8 episodes, T1 for `cl`.

## The control that makes this readable

⭐ **The decoded goal mix is IDENTICAL in every arm: `LANE_KEEP` 18 / `TURN_L` 9 /
`TURN_R` 13 of 40.** It has to be — the tactical head's goal decode is *upstream* of
the planner's cost and none of these levers touch it (`goal-rule: lat_logit_bias=None
-> plain argmax`, `goal-vocab: goal_kappa_turn=None -> shipped GOAL_KAPPA_TURN=0.08`,
logged identically by every arm). **So every difference below is EXECUTION of a fixed
decision, never a different decision.** That is what makes the comparison clean, and
it is the reason this table is worth more than the recall column alone.

## Realised curvature under each DECODED goal token

| arm | token | med `|k|max` | frac `k != 0` | mean `k^2` |
|---|---|---|---|---|
| `ccos_argmax` | LANE_KEEP | 0.00000 | 0.4444 | 0.002908 |
| `ccos_argmax` | **TURN_L** | 0.08000 | 1.0000 | **0.006698** |
| `ccos_argmax` | **TURN_R** | 0.08000 | 1.0000 | **0.006400** |
| `wk15` (W_KAPPA) | LANE_KEEP | 0.00000 | 0.0000 | 0.000000 |
| `wk15` | **TURN_L** | **0.02066** | 1.0000 | **0.000136** |
| `wk15` | **TURN_R** | 0.08000 | 1.0000 | **0.004617** |
| `combined` (cap+ladder) | LANE_KEEP | 0.00000 | 0.4444 | 0.001161 |
| `combined` | **TURN_L** | 0.08000 | 1.0000 | **0.005797** |
| `combined` | **TURN_R** | 0.08000 | 1.0000 | **0.006400** |

**Change in mean `k^2` under a decoded turn, relative to `ccos_argmax`:**

| arm | TURN_L | TURN_R |
|---|---|---|
| `wk15` | **−98.0 %** | −27.9 % |
| `combined` | −13.5 % | **+0.0 %** |

## What this says

1. ⛔ **`W_KAPPA`'s gains are a DECODE–EXECUTE MISMATCH.** The tactical head decodes
   `TURN_L` on 9 of 40 windows in `wk15` exactly as it does in the baseline, and the
   planner then realises **1/50th** of the curvature. The median `|k|max` under a
   left-turn goal collapses **0.08000 → 0.02066**. The arm has not decided to go
   straight; it has decided to turn and then not turned.
   ⇒ This is the **hierarchy seam** the programme exists to measure, appearing as a
   *lever side effect*: the ADE win at cell 2 is bought by breaking the link between
   the tactical decision and the operative plan.
2. ⭐ **The cap + ladder leaves execution intact.** `combined` realises **0.08000**
   under both turn tokens — the uncapped baseline's own value — and its `TURN_R`
   `k^2` is unchanged to six decimals (**+0.0 %**). It reaches `kamm_over_rate`
   0.0000 *while still executing the decoded turn*, which is the corrected form of
   this package's withdrawn superlative.
3. ⚠️ **The suppression is DIRECTION-ASYMMETRIC, and by a large factor.** `W_KAPPA`
   penalises `k^2`, which is **symmetric in the sign of `k`** — yet left turns lose
   98.0 % of their curvature and right turns only 27.9 %, a **3.5x** ratio in
   suppressed fraction. A sign-symmetric penalty cannot produce a sign-asymmetric
   outcome on its own, so the asymmetry lives in something the penalty *interacts*
   with — the corpus's left turns, the goal seed's own controls, or the `W_VEND`
   term — **not in the curvature cost itself**.
   ⛔ **NOT CLAIMED HERE, and named as a live question:** which of those it is. There
   is a separate `2026-09-05-turn-asymmetry` stream; this is a datum for it, not a
   verdict from me. Reported because it is directly load-bearing for the lever
   ranking, not because it is settled.
4. ⚠️ **`LANE_KEEP` execution also collapses in `wk15`** — `frac k != 0` **0.4444 →
   0.0000**, mean `k^2` → **exactly 0.000000**. Under a lane-keep goal the baseline
   still applies small corrective curvature on 44 % of windows; `wk15` applies none
   at all. That is the same refusal, on the class where it is least visible in a
   recall table (lane_keep recall goes UP, to 1.0, precisely because the arm does
   nothing else).

## Why the recall column alone was not enough

`wk15`'s tactical row reads `lane_keep 1.0 / turn_left 0.0 / turn_right 0.5`, which
invites the reading *"it prefers lane-keeping"*. The goal-token table shows the
opposite: **it decodes turns at exactly the baseline rate and then fails to execute
them.** A decision metric computed from the REALISED trajectory cannot distinguish
*"decided not to turn"* from *"decided to turn and did not"* — and those have
completely different fixes. The first is a goal-head problem; the second is a cost
problem, and it is the second.
