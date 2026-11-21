# BANKED — the A2 `W_KAPPA` sweep rungs `wk1` / `wk3` / `wk7`

`Banked 2026-09-06 by the register-repair agent. Previously these rows existed ONLY as prose in`
`Project Steering/Decisions/2026-09-06-mm-decisions.md` (M79). *"An artifact in one document is NOT done."*

| | |
|---|---|
| **evidence class** | **MEASURED (ours)** — re-derived this turn from the run records, not copied from the decisions file |
| **tier** | **T1** (self-action open loop) for `cl` / `ha` / `ha0` / `ha0_ext`; **T0** for `ol` |
| **arm family** | refav1 @ checkpoint step **21,109**, iCEM planner, dev box |
| **n** | **40 windows / 8 episodes** (episode-cluster clusters = 8), stride 16 |
| **estimator** | **episode-cluster bootstrap** (`taniteval/ci.py`), paired form for the deltas the record carries |
| **provenance** | records `raw/rec_wk1.json`, `raw/rec_wk3.json`, `raw/rec_wk7.json` in **this** package (copied from `C:/Users/Admin/refav1_margin/p4out/`); tool `raw/four_family_table.py` of `…/2026-09-05-refav1-cost-geometry/` run unmodified; full output `raw/ff_a2_rungs_and_best_seeds.txt` |
| **cost weights** | `metric=ccos`, `W_JERK=0`, `W_VEND=64.2972`, `W_KAPPA` = the swept variable (`source=CLI override`) — read out of each record's own `cost` block, not from the launcher |

⚠️ **Scope: a 40-window / 8-episode panel.** Nothing here is a corpus-scale claim, and every
delta must be read against the arm's **inference-seed floor** (`D-REFAV1-CG-SEEDFLOOR`;
curvature 0.00200, ADE 0.1035, heading 1.1458), because iCEM **samples** — see `I17`/V3.

---

## 1. The sweep, on the SOUND metric

⭐ Read on **curvature MAE**, not on turn recall: `D-TURNGATE` measured that the **ground truth
itself fails** the `|dyaw| > 0.15` turn gate 3 of 9, so recall is a broken criterion here. The
turn-recall column is printed anyway, marked, because it is half of what the rungs do.

| arm | `W_KAPPA` | ADE (m) | ADE CI95 | **curv MAE (1/m)** | head MAE (deg) | lat κ | turn_L recall (⛔ broken gate) | turn_R recall |
|---|---|---|---|---|---|---|---|---|
| `ccos_argmax` | 0 | 1.3272 | — | ⛔ **0.055369** | 23.4578 | 0.3795 | 0.3636 | 0.7500 |
| **`wk1`** | **1** | **0.9388** | [0.7049, 1.1831] | **0.039406** | 20.3732 | 0.4004 | 0.3636 | 0.6250 |
| **`wk3`** | **3** | **0.9301** | [0.6918, 1.1726] | **0.038737** | 20.1946 | 0.4372 | 0.3636 | 0.6250 |
| **`wk7`** | **7** | **0.8935** | [0.6406, 1.1686] | **0.033522** | 19.0845 | 0.4552 | 0.2727 | 0.6250 |
| ⭐ `wk15` | 15.11245 | 0.8934 | — | ⭐ **0.030982** | 15.2704 | 0.2611 | **0.0000** | 0.5000 |
| `wk151` | 151.1245 | 0.9084 | — | 0.038019 | 15.3785 | 0.0000 | **0.0000** | 0.0000 |
| **FLOOR `ha0`** (perfectly straight) | — | 0.9251 | [0.7024, 1.1783] | **0.040083** | 20.1374 | 0.0000 | — | — |

*(rows `wk1` / `wk3` / `wk7` and the `ha0` floor: THIS package's `raw/ff_a2_rungs_and_best_seeds.txt`.
Rows `ccos_argmax` / `wk15` / `wk151`: the already-banked `…/2026-09-05-refav1-cost-geometry/raw/wkappa_dose.txt`,
which does **not** carry the three rungs — that absence is why this file exists.)*

⛔ **`wk15`'s curvature is quoted at two precisions in two artifacts** — `0.030982` in
`wkappa_dose.txt` and `0.031281` from `rec_best.json`'s sibling read. They are **different arms**:
`wk15` is `W_KAPPA` alone, `best` is `W_KAPPA` + the µ = 0.7 cap. Do not merge the rows.

## 2. The known-value control that licenses the comparison

The four **model-free** arms are bit-identical across all five records read this turn — one
distinct row each, no exceptions:

```
ha        T1 | 0.8888 [0.6026, 1.1928] | speed 0.3058  along 0.6923 | curv 0.076005
ha0       T1 | 0.9251 [0.7024, 1.1783] | speed 0.7217  along 0.6788 | curv 0.040083
ha0_ext   T1 | 0.8772 [0.6195, 1.1437] | speed 0.3058  along 0.6819 | curv 0.077298
ol        T0 | 0.8052 [0.5787, 1.0577] | speed 0.0906  along 0.5927 | curv 0.082614
```

⇒ **one surface, not five.** A panel whose model-free arms drift is comparing different window
grids; this one does not.

## 3. The four families, per family, never pooled

**LONGITUDINAL** (`ha` = 0.3058 speed MAE, `ha0_ext` = 0.3058 — the arms to beat):

| arm | speed MAE | speed bias | speed RMSE | along MAE | accel MAE | tsa<0.5 |
|---|---|---|---|---|---|---|
| `wk1` | 0.7442 | +0.0461 | 1.1462 | 0.6926 | 0.8023 | 0.5100 |
| `wk3` | 0.7453 | +0.0398 | 1.1504 | 0.6919 | 0.8089 | 0.5075 |
| `wk7` | 0.7515 | −0.0379 | 1.1772 | 0.6851 | 0.8098 | 0.5250 |

⛔ **Every rung loses the longitudinal family to `ha` by ~0.44 m/s.** `cl − ha0_ext` speed MAE is
**separated WORSE** on all three — `wk1` **+0.4384 [+0.2558, +0.6565]**, `wk3` **+0.4395
[+0.2571, +0.6584]**, `wk7` **+0.4458 [+0.2584, +0.6792]** — while every `cl − ha0_ext` **ADE** is
**NOT** separated (`wk1` +0.0615, `wk3` +0.0529, `wk7` +0.0162). ⇒ **the rungs tie on position and
lose on speed**, which is the family split ADE hides. ⚠️ **Distance-keeping is
UNAVAILABLE with n = 0** on this panel and the tool says why: *"no lead-agent track supplied"*.
Reported as a stated gap per clause 5 of the four-family rule, not silently dropped.

**LATERAL:**

| arm | head MAE (deg) | yaw-rate MAE | curv MAE | curv bias | cross MAE | cross-final MAE |
|---|---|---|---|---|---|---|
| `wk1` | 20.3732 | 8.5112 | 0.039406 | −0.004043 | 0.4340 | 1.1484 |
| `wk3` | 20.1946 | 8.2470 | 0.038737 | −0.004587 | 0.4215 | 1.1156 |
| `wk7` | 19.0845 | 6.8676 | 0.033522 | −0.005859 | 0.3516 | 0.9245 |

**TACTICAL** (decision quality + goal setting; `ha0_ext` lat κ 0.6277 is the control to beat):

| arm | lat acc | lat κ | lon acc | lon κ | m5 acc | goal FDE (m) | goal bearing (deg) |
|---|---|---|---|---|---|---|---|
| `wk1` | 0.6500 | 0.4004 | 0.4250 | 0.0000 | 0.4750 | 2.2915 | 24.9404 |
| `wk3` | 0.6750 | 0.4372 | 0.4500 | 0.0678 | 0.5250 | 2.2700 | 24.8038 |
| `wk7` | 0.7000 | 0.4552 | 0.4500 | 0.0795 | 0.5000 | 2.1700 | 23.7727 |

per-class lateral recall (n_true: lane_keep 21, turn_left 11, turn_right 8):
`wk1` 0.8095 / 0.3636 / 0.625 · `wk3` 0.8571 / 0.3636 / 0.625 · `wk7` 0.9524 / **0.2727** / 0.625

**STRATEGIC** — ⚠️ **UNAVAILABLE, n = 0**, and the tool states the reason: *"strategic decisions
not present in the scored pass (missing ['route_pred', 'route_true'])"*. The refav1 arm tool does
not emit a route head. **A stated gap with its reason and its n**, per clause 5.

## 4. What these rows do and do not establish

* ⭐ **The point estimates confirm the monotone limb 0 → 1 → 3 → 7 → 15 on curvature AND
  heading** from the arms' own records rather than from prose — which is the claim M79 rested on.
  ⚠️ **But "monotone" is a statement about point estimates here**: one of its four steps
  (`wk1` → `wk3`) is below the inference-seed floor, so the *shape* is supported and the
  *step-by-step ordering* is not. See the floor reading below.
* ⭐ **`wk1` already crosses the straight-line floor** (0.039406 < 0.040083) — the penalty buys
  road-tracking from its first unit.
* ⛔⛔ **THE TURN-RECALL COLLAPSE IS DOSE-DEPENDENT, AND THAT REFUTES AN UNRESTRICTED FORM THAT
  WAS IN CIRCULATION.** *"Every `W_KAPPA` cell reads `turn_left` 0.0000; every non-`W_KAPPA` cell
  reads 0.3636 — no cell is intermediate"* is true of the **2x2x2 FACTORIAL**, whose `W_KAPPA`-on
  cells are **all at 15.11245**. On the `ccos` DOSE SWEEP it is **FALSE**: **0.3636 at W_KAPPA 1
  and 3 · 0.2727 at 7 · 0.0000 at 15.11 and 151**. `wk7` **is** the intermediate cell. ⇒ the
  collapse has a **dose response with a knee between 7 and 15**, which is a different — and more
  useful — object than a binary property of the term. ⭐ **Class `I20`: a FACTORIAL CELL's claim
  quoted about a SWEEP.** ⚠️ A second way the unrestricted form fails: `cos_argmax` is a
  `W_KAPPA = 0` cell and reads **0.0** anyway, because its `cos` metric (not its penalty) kills
  the turn — so the statement is not even monotone in `W_KAPPA` across metrics.
* ⛔ **The turn-recall column is where the price is paid**: the rungs are the part of the sweep
  where the arm still turns, and they are also the part with the worse curvature. ⇒ **This is a
  trade, not a frontier point**, and reading the sweep on either metric alone selects the wrong
  arm.
* ⛔ **No margin here is read against a training replicate** (`I17`/V2 unpriced). The **inference**
  floor is priced: `…/2026-09-05-refav1-close-the-gaps/raw/frontier.txt` gives the **BINDING
  per-metric curvature floor 0.00200** (max over three replicate pairs: ccos 0.00066, combined
  0.00200, Thor wk15 0.00019). Against it:
  * `wk1` → `wk3` = **0.00067 — BELOW the floor** ⇒ ⛔ **not a distinguishable difference on this
    panel and must not be quoted as one.**
  * `wk3` → `wk7` = **0.00522 (2.6×)** and `wk7` → `wk15` = **0.00254 (1.3×)** clear it — the
    second only barely, so the `wk7` vs `wk15` curvature ordering is **weak**, and `wk15`'s claim
    over `wk7` rests on heading (19.08 → 15.27, 3.3× the 1.1458 heading floor) rather than on
    curvature.
  * ⭐ the turn-recall floors on this rig are **EXACTLY 0.00000** (`frontier.txt`), so `wk7`'s
    0.2727 vs `wk1`/`wk3`'s 0.3636 vs `wk15`'s 0.0000 are all above their own floor.

<!-- ARTIFACT-COMPLETE: A2_WKAPPA_SWEEP_RUNGS -->
