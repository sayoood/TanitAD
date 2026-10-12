# PREREG: V-R8-noLW, V-R8 with the listwise selector loss OFF (a PI-gated 12th tiny-rig run)

Written 2026-10-12 by the Master Mind, BEFORE the arm exists (no checkpoint, no number). Motivated by `RESULT.md`
(this dir): V-R8 loses to V0 on turns because its SELECTOR picks badly. The candidate bank holds better turns than
V0's. The selector is trained by the X1 listwise soft-target loss, which replaces the hard selection CE. That target
is lateral-blind (speed profile + terminal heading only) and flat (~33 of 149 rows carry its mass).

## The arm

`V-R8-noLW` = V-R8's argv (`refcv8_ladder/W/argv/V-R8.json`, argv_sha256 `40e5f382…`) with exactly ONE value changed:
`--w-r8-listwise 1.0` → `0.0` (plus its own `--out`). The arm file is `raw/V-R8-noLW.argv.json` (argv_sha256
`dc4c1fa0…`). A diff against V-R8 shows 2 of 189 positions differ (index 182, the weight; index 188, `--out`).
With the weight at 0, `refc_v3_train.py:5658-5661` puts the selection CE (`SEL_V3_WEIGHT`) back automatically. No code
changes. Same seed (0), rig, steps (3000), batch (8) and data as V-R8.

**Compute:** one tiny-rig training run on Thor, about 4.5 h at V-R8r's measured rate. It starts only AFTER the ladder
chain has finished (never next to a training job). Eval runs on the dev box at `--batch 4` with
`ladder_eval_diag.py --save-cands` (about 30 min), the same host and batch as the banked V0 / V-R8 / V-R8d dumps.
**It is a 12th run, and the PI capped the ladder at 11, so it waits on the PI.**

## Readout (fixed now)

Grid: eval139, 1,112 windows, K=8, both sampler seeds. Estimator: paired episode-cluster bootstrap
(`strata_pair.py`). Training-replicate floor: |V-R8r − V-R8| on the same metric and stratum, from Thor's batch-8
dumps (both batch 8, so the floor is internally consistent; it is never paired with a batch-4 dump).

* **P1 (the lever).** `V-R8-noLW base` vs `V-R8 base`, GT-turn windows (turnL ∪ turnR), `cross_abs_6s`:
  separated BETTER on BOTH sampler seeds, AND R = |d| / floor > 1 on both.
* **P2 (it is the selector, not the bank).** On the same windows, `V-R8-noLW oraclebase` is NOT separated worse than
  `V-R8 oraclebase` on `cross_abs_6s` (both seeds). If the bank got worse, a P1 gain would not be a selection gain.

Outcomes:
* **CONFIRMED.** P1 and P2 hold. The listwise term is the turn-selection lever. ⇒ The MM reads `V-R8-noLW` vs V0
  against B-RECIPE's four bars exactly as the ladder reads V-R8 vs V0 (SPEC_WPB_LADDER §6). If it clears all four on
  both seeds, the MM PROPOSES the R8 recipe with `--w-r8-listwise 0.0` for the launch argv. That is a PI decision,
  because it adds a post-hoc arm to the L1 decision.
* **NULL.** P1 is not separated on either seed. The listwise term is not the lever. The next lever, named now: put a
  POSITION term into the selector target (ADE or cross-track next to speed + heading), or sharpen it
  (`list_t` < 1), which needs a CLI knob that does not exist yet (`list_t` / scales are fixed at 1.0 / 1 m / 15°).
* **PARTIAL.** Anything else. Report d, CI and R per seed, and the four families.

Every family is reported (ADE, LONGITUDINAL, LATERAL, TACTICAL; STRATEGIC and distance-keeping are named absent at
this rung). The tier is the ladder's open-loop rows. TINY RUNG, one training seed.

## What this is NOT

Not a ladder rung. It does not change the 11 registered arms, their order, or I-0. It cannot rescue L1 by itself: L1
is scored as registered.
