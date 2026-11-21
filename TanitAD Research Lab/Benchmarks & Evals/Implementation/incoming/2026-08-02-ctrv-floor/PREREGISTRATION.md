# PREREG — E-CTRV: is the driving block's turn/lateral win a floor artifact?

**Written 2026-08-02, BEFORE the driver was run.** Both outcomes are committed
below, and an INSTRUMENT-FAIL branch is included (the C63 lesson: a prereg with
no instrument-fail branch leaves an out-of-range result nowhere to go).

## The observation that motivates it

`taniteval/driving.py:304` — the canonical tier-0 driving-capability block:

```python
FLOORS = ("cv", "holdv0")
```

`cv` extrapolates the last velocity vector linearly; `holdv0` goes straight at
the entry speed. **Both are straight-line predictors — neither can represent a
turn.** Every published verdict that lives on a turning stratum is measured
against a family that is structurally unable to compete there. From
`raw_v4fs-30k-produced.json` (2026-07-28 v4 30k gate), all
`paired_episode_cluster_bootstrap`, orientation `floor − model`:

| claim | published |
|---|---|
| `sustained_turn` (n=142) ADE | model 1.0708 · cv 2.3124 · holdv0 2.0800 → **+1.2416 [0.823, 1.588] separated, favours model** |
| `by_curvature.sharp` (n=144) heading MAE | model 10.001° · cv 28.743° → **+18.74° [15.39, 21.64] separated, favours model** |
| `lat_abs_2s_m` overall | **+0.4574 [0.211, 0.776] separated, favours model** |
| `verdict.where_the_win_lives` | **"lateral only"** |

CTRV (constant turn rate + constant velocity) is the standard third member of
the trivial-floor family, it is admissible under **exactly the same information
budget** (`poses[last]`, `poses[last-1]`, no future), and it is **already
computed on every window** — `stack/scripts/driving_diagnostic.baseline_waypoints`
returns it as `constant_yaw_rate` and `taniteval/rollout.collect` discards it.

It was already MEASURED on other corpora that CTRV is the dominant floor member
(`incoming/2026-07-15-baseline-floor`: CTRV wins **55–58 %** of 26 132 anchors;
CV overstates the floor **4.6× on curves**). That result has never been carried
onto the canonical val40 surface the gate actually decides on.

## Hypotheses

- **H-ARTIFACT** — the turn/lateral wins are substantially a property of the
  floor family. Adding CTRV moves `sustained_turn` / `curv_sharp` /
  `lat_abs_2s_m` toward `tie` or `floor`.
- **H-REAL** — the wins survive against CTRV; the model's path advantage is
  genuine and CTRV adds nothing on this surface.

## Decision rule (fixed in advance)

- Estimator: **paired episode-cluster bootstrap**, B=2000, seed=0, resampling
  unit = val episode. Orientation `floor − model`; positive = model wins.
  `separated` = CI excludes zero. No other estimator is admissible
  (`overlapping_holdout_se` is refused).
- Primary floor: **`ctrv_gated`** (yaw-rate zeroed below 2 m/s — the standstill
  yaw-noise artifact found 2026-07-15). Ungated `ctrv` reported alongside so
  the gate's effect is visible, not assumed.
- **Primary readout: the verdict-flip ledger** — every (stratum, metric) whose
  `favours` label changes between `vs_cv` and `vs_ctrv_gated`.
- **H-ARTIFACT is confirmed** if any of `sustained_turn` ADE, `curv_sharp`
  heading MAE, or overall `lat_abs_2s_m` flips from `favours model` to `tie` or
  `favours floor` on `flagship-30k` (v1, the deployed arm). Otherwise **H-REAL**.
- ⛔ Magnitudes are reported as **direction + separation**; the floor swap does
  not license a re-ranking of arms against each other.

## INSTRUMENT-FAIL branch (mandatory, C63)

The floors are **backfilled** onto banked window dumps, so the whole result
rests on the assumption that window *i* here is window *i* there.
`ctrv_floor.verify_alignment` measures it: the rebuilt `cv` and `gt` must match
the tensors `rollout.collect` persisted, elementwise, `< 1e-4`, with identical
`eid`.

**If alignment fails for an arm, that arm is REFUSED and no verdict is
reported for it** — not "approximately aligned", not "close enough". If it
fails for all arms the experiment returns INSTRUMENT-FAIL and H-ARTIFACT /
H-REAL both stay open.

## What this experiment does NOT establish

1. It says nothing about **closed-loop** capability. Open-loop remains a weak
   claim (arXiv:2605.00066).
2. A floor flip is **not** evidence that the model got worse. The model's
   numbers do not move at all — only the bar it is compared against.
3. CTRV as shipped carries a documented **forward-Euler half-step bias**
   (measured 0.3044 m ADE on a perfect 1 rad arc, see
   `tests/test_ctrv_floor.py`). It is therefore a *conservative* floor: a
   midpoint-corrected CTRV would be stronger, so any H-ARTIFACT verdict here is
   a lower bound on the effect.
4. The re-adjudication uses **banked** per-window predictions. It re-runs no
   model and re-reads no checkpoint, so it inherits whatever those dumps are —
   including their arm labels.
