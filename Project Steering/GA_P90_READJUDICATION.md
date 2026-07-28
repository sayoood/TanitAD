# Ga → lateral p90 (PI chose OPTION C). Re-adjudication of every E1 point.

**PI 2026-07-29: "C for 36".** Ga now gates on the **lateral p90**, not the open-loop ADE mean.
Estimator throughout: **paired episode-cluster bootstrap** stored in each frontier
(967 windows / 44 episodes, B=2000). `overlapping_holdout_se` read nowhere.

## ⛔ 0. This re-adjudication is POST-HOC and cannot certify anything

These four arms ran, and their p90s were already visible, before option C was chosen. A criterion
applied after seeing the data **cannot** certify an arm. What it legitimately answers is a question
about the **criterion**: *does changing the statistic change the verdict?* Any arm that looks like a
pass here must be **re-run against a pre-registered Ga-p90** before it counts.

## 1. Result — exactly ONE point changes

| arm | pts | P1 (closed-loop departure) | Ga-p90 (`cross_p90@2s`) | verdict |
|---|---|---|---|---|
| **E1c** | 17 | separated-better from step 500, best **−0.4407** | **+0.379 … +1.362, SEPARATED at every step** | P1 only |
| **E1e-A** | 4 | separated-better ×4, best −0.3911 | +0.172 … +0.199, SEPARATED ×4 | P1 only |
| **E1e-B** | 4 | separated-better ×4 | **@1000: +0.0626 [−0.0014, +0.1757] NOT separated**; 2000/3000/4000 separated | ⭐ **BOTH @1000** |
| **E1f** | 4 | **never** separated | not separated at 2000/3000/4000 | Ga-p90 only (no gain) |

⭐ **E1e-B @ step 1000 is the only point in the entire E1 programme that satisfies BOTH the
closed-loop primary and the guardrail** — `dep_overall` **−0.2378 separated-better**, lateral tail
**not separated**. It **fails** the old Ga (`Ga_openloop_ade2s_ok: False`), so the switch of statistic
is exactly what changes it.

⇒ **Answer to #36: yes, the statistic mattered.** Under the mean, no point ever passed. Under the
p90, one does.

## 2. ⚠️ Why I would NOT call this a deliverable

1. **Post-hoc** (§0).
2. **The interval barely clears zero:** lower bound **−0.0014**, and `p_delta_gt0 = 0.971`. It is
   "not separated" at 95 % by a hair — a 2.9 %-tail result, not a comfortable pass.
3. **It is 1 of 4 steps.** Steps 2000/3000/4000 of the same arm all separate. A lone passing point in
   a series that otherwise fails reads as sampling noise, not a stable operating point.
4. **It is also E1e-B's WEAKEST closed-loop gain** (−0.2378 vs −0.2891 @2000) — i.e. the point that
   passes is the one that did the least, which is the shape you expect if the guardrail is simply
   tracking how much the arm moved.

⇒ **Treat as a CANDIDATE for a pre-registered confirmation, not a result.**

## 3. What the whole table says about the trade

The p90 view sharpens what the mean obscured. **E1c's lateral tail degrades by +0.39 to +1.36 m** —
against a **1.75 m** corridor half-width, that is a large fraction of the lane, and it is separated at
**every one of 17 steps**. The strongest closed-loop arm is also, unambiguously, the one that damages
the safety-relevant lateral statistic most. **λ_replay behaves as a monotone dial on exactly this
trade**: λ=1 (E1c) worst tail / best departure → λ=3 (E1e-A) → λ=8 (E1e-B) mildest tail, and only the
mildest ever touches the guardrail.

## 4. If a confirmation run is authorised, it must pre-register

- **arm:** E1e-B (λ_replay = 8), evaluated at **step 1000**, chosen *because* this analysis flagged
  it — which is precisely why the confirmation must be on data not used to flag it.
- **primary:** `dep_overall` separated-lower (unchanged).
- **guardrail Ga-p90:** `cross_p90@2s` paired delta **not separated-worse**.
- **both outcomes committed in advance**, and the run reported whichever fires.
- ⚠️ **Pre-commit the step too.** Re-scanning all four steps and reporting the best one would
  reproduce the same forking path this document exists to flag.

## 5. Provenance

`pod3:/workspace/e1{c,e,f}/*frontier*.json` · re-adjudicator
`…/incoming/2026-07-29-route-threshold-sweep/code/` (script `readjudicate2.py`, keys verified against
the real frontier schema).
⚠️ **My first extractor used wrong key paths and produced an all-empty P1 column**, which would have
read as "no point passes anywhere" — a false negative from my own bug, caught before it was reported.
The keys that are correct: `EVAL.P1_dep_overall_separated_lower`, `EVAL._dep_overall_delta`,
`M1_lateral_split.openloop.ego.paired_delta_ft_minus_base.cross_p90@2s`.
