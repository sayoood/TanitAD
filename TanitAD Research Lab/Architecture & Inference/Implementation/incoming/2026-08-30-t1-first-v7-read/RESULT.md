# RESULT — the programme's FIRST T1 capability read on the v7 line

**Date** 2026-08-30 · **Author** Master Mind · **Tier** **T1 (PRIMARY)** ·
**Evidence class** MEASURED (ours; Thor, RTX-class GPU free during the B1 CPU build)
· **Register rows** D-T1-V7 (the enabling fix), D-T1-V7-READ (this result).

## Why this exists

Every number TanitAD had ever produced was **T0** — a world-model diagnostic that
`EVAL_DOCTRINE` says is *never* driving performance. **No v7 arm had ever been
evaluated at T1**, the tier the doctrine names PRIMARY for any capability claim.
`T1_ON_V7_SCOPING.md` §5 states the consequence plainly: *the programme has no
capability claim of any kind, and cannot acquire one until this adapter exists.*

## What was blocking it, and what was actually done

`T1_ON_V7_SCOPING.md` §6 named two blockers.

* **Blocker 1 — checkpoint shape.** Already closed before today: `load_ext_trunk`
  branches on a `'stack'`-keyed checkpoint through `load_trunk_auto`.
* **Blocker 2 — geometry. NOT closed; it was a COMMENT**, telling the operator to
  pass `--frame-h/--frame-w/--projection/--frame-hfov` *"or T1 is computed in the
  wrong projection and LOOKS VALID."* ⛔ That is the defect, not the mitigation:
  `t1_eval`'s defaults are the deployed **256×256 / f_ref 266 / pinhole** frame
  while every v7 arm is **256×640 / hfov 120 / cylindrical**.

Fixed in `b821ca01f`: `adopt_ckpt_geometry` runs inside `resolve_ext_frames`, BEFORE
the loader. An unpassed field is taken **from the checkpoint**; a passed field that
CONTRADICTS it is a hard exit naming both values. `hfov`/`f_ref` are never both set.
Flagship checkpoints carry no such record, return `None`, and are provably untouched.

**A third blocker surfaced on the first real run and is fixed the same way:** the
context **window**. `t1_eval` defaults to 8; every v7 arm trains at 6, so a flagless
run died in `validate_operative_inputs` (*"window mismatch: got 8, predictor
configured for 6"*). Correct and loud — but it is the geometry defect under another
name, an eval default silently disagreeing with the checkpoint. `window` now adopts
from the checkpoint too, with `W` retained as the flagship fallback.

Every run in this package printed
`ADOPTED FROM CHECKPOINT {frame_h 256, frame_w 640, frame_hfov 120.0,
projection cylindrical, window 6}`. **A run that instead prints the deployed
256×256 pinhole means the guard did not fire and its numbers are inadmissible.**

## The reads

40 episodes / **6,924 windows**, `--grounding-readout`, **episode-cluster bootstrap**
(`taniteval.ci`) — ⛔ never `overlapping_holdout_se`. `ha` = hold-action echo control.

| arm | ade_dense_m cl / ha | fde_last_m cl / ha | LAT_cross_mae_m cl / ha | LAT_heading_deg cl / ha |
|---|---|---|---|---|
| `emao14_30k` (EMA, fixed τ) | 14.069 / **13.879** | 26.297 / **26.131** | 1.310 / **1.124** | 94.63 / 95.13 |
| `o14fut30k` (incumbent) | 14.293 / **14.116** | 26.690 / **26.442** | 1.461 / **1.426** | 99.41 / 96.39 |
| `emao14_30k_tauramp` (EMA, ramp) | 13.864 / **13.806** | 26.076 / **25.951** | **1.072** / 1.170 | 94.19 / 92.06 |

⛔ **In every distance metric, for every arm, the closed-loop arm is worse than its own
hold-action control** — the model's own actions make it worse than freezing the action.
All three `holdv0=LOSES_TO_HOLDV0`. All three `_longitudinal_claim_admissible=false`,
so **no longitudinal number from any of these arms is claimable** under the PI's
2026-08-16 anti-echo rule. Heading MAE ~95° is chance; `speed_bias_mps −10.381`;
`target_speed_acc within_2.0_mps 0.146`. Sole exception: the τ-ramp arm's lateral
cross-track, one metric on one arm.

## ⭐ The result that is not negative

`copy_detector = CLEAN, echo_index 0.0000 vs GT 0.2113` — **on all three arms.**

v1.x's entire failure was **echo**: S-curve reproduction 97.9 % open-loop, **0.0 %
hold-action**, ~5 % closed-loop. These arms do **not** echo. The v7 line has traded
*fake skill by echo* for *honest absence of skill*. That is a real structural change,
and it is precisely what the anti-echo control exists to reveal.

## ⚠️ Scope — binding

These are ~19 M-param **v7-TINY** arms trained on stage **S-W only**, with **every
planner objective at zero** (`--w-o1-ctrl 0 --w-o1-fact 0 --w-o1-scene 0 --w-o2 0
--w-o3 0`). They are a world-model trunk plus a grounding readout and were **never
trained to drive**. ⛔ *"v7 cannot drive"* is **not supported** by this and must not be
quoted from it. The supported claim: **the untrained-planner floor is now MEASURED at
T1**, and the scaled run has a number to beat.

## ⚠️ S-rate is noise at this n — and an unplanned arm proved it

S-rate(masked) cl / ha: EMA-fixed **0.2807 / 0.2105** (above control) · incumbent
**0.0702 / 0.1579** (below) · τ-ramp **0.1404 / 0.2105** (below).

⭐ The τ-ramp and EMA-fixed arms are **T0-INDISTINGUISHABLE** (drift 0.6936 vs 0.6952,
nrmse 0.7408 vs 0.7466, cos 0.7513 vs 0.7524 — all ~20× inside the seed spread), yet
they land on **opposite sides of the same control** at T1. Two models that cannot be
told apart at T0 giving opposite S-rate verdicts is direct evidence that **the S-rate
gap is noise, not signal**. ⇒ the distance metrics, unanimous across all three arms,
are the ones to trust; per C160 no verdict is drawn from the disagreement. *An arm run
for an unrelated question supplied the control this panel lacked.*

## ⛔ CORRECTED 2026-08-30 — this was NOT an instrument gap

The section below recorded `strategic: UNAVAILABLE` as an instrument gap and a work item.
**It is a CORPUS fact.** The deployed val40 carries v7 labels on only **6 of its 40 clips**,
and those 6 are exactly the ones excluded from B1 training for leakage (D-VAL40-NOLABELS).
⇒ val40 is scoreable on **TRAJECTORY/ADE** — what it was built for — and **cannot** serve the
four label-based families; that is also why `tactical` above reads *trajectory-derived*.
**No eval-code work would ever have resolved it.** The fix is a LABELLED held-out split
(`eval_split_v3`, sha `ea8670e041c14ccb`), not an instrument change.

## Instrument gap, recorded not hidden — ⚠️ superseded by the correction above

`strategic` family **UNAVAILABLE** — needs `route_pred`/`route_gt`; a world-model
fidelity pass does not traverse the hierarchy. `_complete=false`. **A missing family is
a WORK ITEM, not a pass** (the four-families rule, clause 5: state the reason and the n).

## Deliverable manifest

| artifact | where it lives |
|---|---|
| `raw/t1_emao14_30k.json` | this package (md5 `632fb91e8af126c2e46fad4d850f0de2`) |
| `raw/t1_o14fut30k.json` | this package (md5 `45cffdd68e414efd861bb123b2afb87c`) |
| `raw/t1_emao14_30k_tauramp.json` | this package (md5 `d1af4e24b993362defb37c53c86a0633`) |
| the guard | `taniteval/tools/t1_eval.py::adopt_ckpt_geometry` (`b821ca01f`) |
| its tests (11, incl. deliberate-regression set) | `taniteval/tests/test_t1_ckpt_geometry.py` |
| per-episode npz dumps | **Thor only**, `/home/nvidia/t1dumps/<arm>/dumps/` — re-analysable with `--analyze-only`, NOT banked (bulk) |
| registry rows | `MODEL_REGISTRY.md` (T1 block), `GOALS_AND_CLAIMS.md` D-T1-V7 / D-T1-V7-READ |

All three JSONs md5-verified against Thor after transfer.
