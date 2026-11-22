# FALLPRE: second count on banked rollouts, with the looks of the pre-roll (definitions, fixed before the tool exists)

A count, no card, no name. Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s), models UE_s0 and UE_s1, the banked rollouts of cards ADD102, ADD104 and ADD105 (arms VD5c, F2, F3, F4, E5c; references RLc and RR). Lead tool `fall_pre.py`, built on the loaders of `export_fall.py` of this folder.

## 1. Why a second count

The first count (`FALLTRACE_DEFS.md`, run on 2026-11-22) is reported as written: all 151 arm-windows where a camera-fed arm falls behind and the label gap form does not are in its class NO-RULE-ACTION (the rule acted at fewer than 5 looks of the window). Its threshold of 5 was set by the lead for a log with a row per step. The log has 6 looks per window, so the class holds 0.89 of the windows where neither falls behind as well, and the count ranks no lever. That is a flaw of the lead's definition, stated here. Two facts of that count stand: in 111 of the 151 the rule changed the plan at none of the 6 looks of the window, and the field `lift` is 0 in every row. The first count did not read the three looks of the pre-roll and did not read the rule log of RLc. This count reads both.

## 2. Looks

Per window and rollout nine looks: three of the pre-roll (file `rule_log_pre_<key>.npz`), three ticks (`rule_log_<key>.npz`) and three later looks (`rule_log_mid_<key>.npz`). Rows are matched to a window on the clip index and `t0`; for the pre-roll rows the tool first finds the one constant offset between their `t0` and the window start that gives every window three rows, and stops when no single offset does. A lead window that does not have exactly three rows in each of the three logs, for the arm or for RLc, stops the tool.

A look is a **cut look** when `present` is 1 and `sat_exec` is above 0 (the rule cut at least one executed step of the plan). Per window and form: `a_pre` (0 to 3) and `a_win` (0 to 6) for the arm X, `r_pre` and `r_win` for RLc, same window, same model seed.

## 3. Classes of an excess window (X falls behind, RLc does not), first match

1. `NO-CUT-BY-X`: `a_pre + a_win` is 0.
2. `MORE-CUT-IN-PRE-ROLL`: `a_pre > r_pre`, and `a_pre - r_pre` is at least `a_win - r_win`.
3. `MORE-CUT-IN-WINDOW`: `a_win > r_win`.
4. `NOT-MORE-LOOKS`: none of the above (X cuts at no more looks than RLc).

## 4. Cause tag of a window of classes 2 to 4, from the cut looks of X (pre-roll and window together), first match

* `FALSE-PRESENCE`: at half or more of the cut looks the label has no lead at the frame of the look (label lead: inside the clip, valid, finite, 0 to 60 m).
* `GAP-LOW`: at half or more of the cut looks that have a label lead, the fed gap is below the label gap by more than 5 m (both at the recorded frame of the look).
* `SLOPE`: at half or more of the cut looks the closing input is below -0.1 m/s (the gap closes) while the label gap rate over the step from the frame of the look to the next frame is at or above -0.1 m/s.
* `OTHER`: none of these.

## 5. What is printed

* Control first; with a failed control one line and nothing else: per arm the number of excess windows and of windows where neither falls behind equal those of the first count (34, 24, 26, 38, 29 and 297, 307, 305, 293, 302), and every lead window has its nine rows.
* Per arm and pooled: the class counts and the cause tags of the excess windows, and beside them the same for the windows where neither falls behind (a class or tag that is as frequent there explains nothing).
* With no class: the mean of the summed `sat_exec` of X minus that of RLc over the excess windows, for the pre-roll and for the window; the share of X's cut looks with no label lead in the excess windows and in all lead windows; the same counts with RR in the place of RLc.
* The pooled counts add arm-windows over five arms; one window can count up to five times. The number of distinct windows is printed.

## 6. Limits

A count on two model seeds and five fits of one head, with the models and the evaluation episodes held fixed: it ranks candidates for the next arm and proves no cause. The label gap is taken at the recorded frame. The cause tags read the label of the join, which is the reference of the rule, not ground truth of the scene. No re-perception.
