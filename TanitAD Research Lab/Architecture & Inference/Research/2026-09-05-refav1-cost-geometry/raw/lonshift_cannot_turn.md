# The panel's best-ADE arm executes **zero** turns — and its lever attribution is still correct

**MEASURED 2026-09-05, zero GPU**, from the banked `lonshift` record and dump.
Two independent probes agree: `four_family_table.py` on `rec_lonshift.json`, and
`kappa_by_goal.py` + `feas_audit.py` on `dump_lonshift`.

⭐ **This is not a correction of a sibling's claim.** `lonshift`'s reported lever
effect — `--a-sustain-mode a0_shift` against `wk15`, LON speed **−0.2263**, along
**−0.1398**, accel **−0.2078**, all separated at 8–52x their floors — is **sound as a
lever effect**, and its statement that the lateral family is untouched is **true
against `wk15`**. What follows is about the **base** that lever was applied to, and it
is visible only when the tactical family's per-class rows are read beside the
lateral family's aggregates.

## What `lonshift` does

| | `lonshift` | `ccos_argmax` (base) | `combined` (cap+ladder) |
|---|---|---|---|
| ADE m | **0.7868** [0.6065, 0.9946] — best in the panel | 1.3272 | 1.0504 |
| `TAC lat_kappa` | **0.0000** (= chance, and = `ha0`'s value) | 0.3795 | 0.4148 |
| lane_keep recall (n=21) | **1.0000** | 0.7143 | 0.7619 |
| **turn_left recall (n=11)** | **0.0000** | 0.3636 | 0.3636 |
| **turn_right recall (n=8)** | **0.0000** | 0.7500 | 0.7500 |
| GT-turn `dir_correct` (n=19) | **0.3684** | 0.5789 | 0.5789 |
| `max\|kappa\|` over 27 windows | **0.0267** 1/m | 0.2000 | 0.1505 |
| `kamm_over_rate` | 0.0000 | 0.2963 | 0.0000 |

Realised curvature under each **decoded** goal token — the goal mix is identical in
every arm (LANE_KEEP 18 / TURN_L 9 / TURN_R 13), so this is execution, not decision:

| arm | TURN_L med `\|k\|max` | TURN_L mean `k^2` | TURN_R mean `k^2` |
|---|---|---|---|
| `ccos_argmax` | 0.08000 | 0.006698 | 0.006400 |
| `lonshift` | **0.01935** | **0.000113 (−98.3 %)** | **0.000746 (−88.3 %)** |

⇒ **`lonshift` decodes a turn on 22 of 40 windows and executes ~2 % of the baseline's
curvature on the left ones and ~12 % on the right.** Its maximum curvature anywhere in
27 windows is **0.0267 1/m**. Its `dir_correct` on true-turn windows is **0.3684**,
*below* the base's 0.5789 — so on the windows that do turn it is not merely gentle, it
is more often pointed the **wrong way**.

## Why this matters, and what it does NOT say

1. ⛔ **On this 2 s panel, ADE is nearly insensitive to whether the arm turns.** The
   best ADE in the programme's refav1 panel belongs to the arm with the **least**
   curvature. Over a 2.0 s horizon a real turn displaces the ego only modestly, so a
   straight-line plan pays little ADE and collects the longitudinal win in full.
   ⇒ **This is the binding four-family rule earning its keep.** A report of
   *"ADE 0.7868, beats `ha0_ext` on FDE"* is true and is **not** a description of
   driving; the tactical family on the same windows reads 0.0000 / 0.0000 recall.
2. ⚠️ **The turn deletion is INHERITED, not caused, by `a0_shift`.** `wk15` — the base
   — already reads turn_left 0.0000, and `D-REFAV1-CG-FACTORIAL` attributes that to
   `W_KAPPA` (main effect on turn_left recall: **−0.3636, to exactly zero, in both of
   its cells**). The sibling's paired comparison is `lonshift` − `wk15`, so it
   correctly reports no lateral change: **both arms cannot turn.** A paired delta
   against a base that already lost a capability cannot see that the capability is
   gone — which is a general lesson about baselines, not a fault in that analysis.
3. ⭐ **The obvious synthesis is NOT `W_KAPPA` + `a0_shift`.** The factorial says the
   cap buys ~74 % of `W_KAPPA`'s ADE and **more** of the safety at **zero** measured
   cost in turns (turn_left effect 0.0000, in floor) and **zero** cost in the
   longitudinal family (`LON_speed` −0.0017 [−0.0078, +0.0034], `LON_accel` +0.0013
   [−0.0035, +0.0064]). `a0_shift` is a **longitudinal** lever and `W_KAPPA` is where
   the turn loss comes from. ⇒ **`--kamm-mu 0.7` + ladder + `a0_shift`, with
   `W_KAPPA = 0`, is the arm that should keep the turns AND the longitudinal gain.**
   ⛔ **UNRUN. Stated as the next experiment, not as a result** — and it is one arm.
4. ⚠️ **NOT CLAIMED:** that `lonshift` is a bad arm. It is the first refav1 arm to beat
   `ha0_ext` on a distance metric with a separated interval, and the a0_shift lever's
   longitudinal effect is the largest measured in this line. The claim here is narrower
   and precise: **its turn execution is zero, that came from its base, and no ADE or
   longitudinal number can reveal it.**
