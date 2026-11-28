# ADDENDUM to PRE-REGISTRATION 6 — a FALLBACK GRID, declared before any number exists on it

**status: PRE-REGISTERED. ⛔ WRITTEN AND COMMITTED BEFORE EITHER GRID PRODUCED ANY STATISTIC.**
**date:** 2026-09-06 · **amends:** `PREREG_ANCHORED_BAR.md` (commit `47872d6`) · **GPU: 0.**

## 1. Why an addendum exists at all

`PREREG_ANCHORED_BAR.md` §2 fixed the fresh grid as **`fresh200`** — `cand[120:320]` of the seed-0
fitlist, 200 clips, disjoint from `fit120`. All 200 `v2ep` episodes were pulled from Thor and are on
the box (7.0 GB, `ok=200 fail=0`). ⛔ **The lead block cannot be built, and the blocker is neither
compute nor design.**

`taniteval/tools/build_lead_block_b1.py --pull` reads the HF token **in place** from
`G:\…\TanitAD\Keys.txt` (CLAUDE.md invariant: never copy, never print, never write to args). ⛔ **The
G: mount will not serve that path.** MEASURED with an **interleaved same-breath control**, which is
what separates "the mount is flapping" from "this file will not hydrate":

| probe | reading |
|---|---|
| `grep -c 'hf_' Keys.txt` | ⛔ **TIMEOUT** (12 s), repeatedly |
| ⭐ CONTROL, same breath: `grep -c '' CLAUDE.md` | **966 lines** — the mount is serving |

⇒ this is the documented **non-global** failure mode, not an outage: metadata resolves, other files
on the same mount read fine, and one path never hydrates. A first build attempt blocked **~25 min**
and exited with `could not read …/Keys.txt`. A mount-guarded relaunch is polling and will run
`fresh200` the moment the file becomes readable.

⛔ **This addendum does NOT relax anything.** It names a fallback grid **now**, with its trigger and
its cost, so that whichever grid is scored was named in a pre-registration **before** it produced a
number — which is the only property that matters.

## 2. ⛔ THE TRIGGER, fixed here

> `fresh200` is the PRIMARY grid. The fallback is used **only** if the mount-guarded builder
> exhausts its wait without `Keys.txt` becoming readable. ⛔ **If `fresh200` builds, the fallback is
> not scored at all** — it is not a second chance and it may not be used to choose between grids.

Whichever grid runs, the report states which, and why.

## 3. THE FALLBACK GRID — `eval141`

The v7.2 **EVAL** split: 141 clips at `run_refcv3_ol/data/eval`, labels
`s2_labels_v7.2_eval.jsonl.gz` (md5 `aa12c948f062181c3297265b51526ec5`), with an **already-built**
lead block `_lead_b1/b1_eval_lead_block.npz` (2026-09-02, `--k 10 --dt 0.2`, the same flags).
⇒ **no HF pull, no `Keys.txt`, nothing blocked.**

* **Disjointness** is asserted by the driver's own GATE 3 (`fit ∩ eval = 0`) and is re-MEASURED here
  with a non-zero control before any statistic is computed.
* Every other knob is held exactly as §2 of the parent pre-registration fixes it.

### 3.1 ⭐ Why this grid is scientifically clean for THIS probe — the argument, not an assurance

The gate statistic is a function of **(human future, lead track, `v0`)** only. Its three sides are
the recorded human future, a `hold_v0` path constructed from `v0`, and a `frozen` path of zeros.
⛔ **No model prediction enters any cell.** The frozen refcv3 checkpoint is loaded only to open the
corpus with the right config and provenance; nothing it outputs reaches the statistic. So this panel
cannot leak model information in either direction — it is arithmetic over corpus geometry.

### 3.2 ⚠️ AND THE EXPOSURE THAT REMAINS, STATED BEFORE IT COULD BE DISCOVERED

There is one path, and it is weak but not zero. If `H-RL-CAP-1` reports **outcome A** on `eval141`
and the PI later adopts `progress_lead_cap "lead"` on that basis, then a **reward-design choice was
made on eval-split data**, and an RL arm trained with that reward would afterwards be evaluated on
the same split. The choice is a binary cap mode selected on geometry rather than on any model's
performance, so the exposure is small — but it is real, and it is the reason `fresh200` is PRIMARY
and this grid is a fallback rather than a convenience.

⛔ **Any claim scored on `eval141` carries this stamp verbatim**, in the register row and in the
report: *"scored on the v7.2 EVAL split because the PRIMARY fresh grid was blocked on an unreadable
`Keys.txt`; a reward-design choice adopted from it would have been made on eval data."*

⭐ **And it is re-runnable.** `fresh200`'s 200 episodes are on the box. The moment `Keys.txt`
hydrates, the same pre-registered hypotheses can be scored on the primary grid with one command, and
the two gridsderive their own anchors independently. The fallback is a way to not lose the turn, not
a substitute for the primary grid.

## 4. What is unchanged

The hypotheses (`H-RL-GATE-STAT-3a`, `H-RL-GATE-STAT-3b`, `H-RL-CAP-1`), the statistic, the anchor
rule, the alpha, the rung, the cells, all three committed outcomes, the mechanism check, the vacuity
classification and all nine controls are **exactly as committed**. Only `PREREG_GRID` moves, and only
under the trigger in §2.

```
    PREREG_HYPOTHESIS_ID: H-RL-GATE-STAT-3a
    PREREG_STATISTIC: S6_wmr
    PREREG_ANCHOR: signflip_episode_null
    PREREG_ALPHA: 0.05
    PREREG_DIRECTION: <=
    PREREG_RUNG: all
    PREREG_CELL: nopert_min
    PREREG_GRID: fresh200_or_eval141_per_addendum_trigger
```
```
    PREREG_HYPOTHESIS_ID: H-RL-GATE-STAT-3b
    PREREG_STATISTIC: S6_wmr
    PREREG_ANCHOR: signflip_episode_null
    PREREG_ALPHA: 0.05
    PREREG_DIRECTION: <=
    PREREG_RUNG: all
    PREREG_CELL: ship
    PREREG_GRID: fresh200_or_eval141_per_addendum_trigger
```
```
    PREREG_HYPOTHESIS_ID: H-RL-CAP-1
    PREREG_STATISTIC: S6_wmr
    PREREG_PAIRED: ship - lead_min
    PREREG_RUNG: all
    PREREG_GRID: fresh200_or_eval141_per_addendum_trigger
```

⛔ There is still **no literal `PREREG_BAR:` line**, and the scoring tool still refuses one.
