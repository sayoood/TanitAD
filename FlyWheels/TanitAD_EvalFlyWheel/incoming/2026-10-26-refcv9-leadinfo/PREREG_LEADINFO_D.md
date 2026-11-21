# PREREG_LEADINFO_D -- is there decel information in the LEAD AGENT that refcv8's lon head does not carry? (DIAGNOSTIC, Master Mind, 2026-10-26 ~07:45 CET)

Sealed before any feature, fit or AUC exists. A DIAGNOSTIC that decides where the next information lever goes; it carries NO bar for any
deliverable and NO quantity from it may be used to fit, tune or choose anything that is later scored on eval139.

## 0. Why (disclosed)
Four post-hoc levers on refcv8 (H2cn, H2cn-L, SELFSHARP, SELCOR; `memory: refcv8-md-fd-frontier`, H-R9-SELCOR-1) move along one missed-decel /
false-decel frontier; the GT-oracle pick from the same fan reads MD 0.075 / FD 0.057. The features used so far (lon softmax, v0, arc deltas,
z_tac PCA) cannot find the right candidate. refcv8's planner does not read its own box head (D12). A decel is usually caused by a slowing lead
agent or a light; this asks whether the LEAD channel carries the missing information at all, using GT as an upper bound.

## 1. Data
eval139, S6 surface, compliant windows, both decel populations exactly as `l2_common.lon_indicators` defines them (GT-decel windows = the MD
population; GT-cruise windows = the FD population). Target `y = 1[GT decel]` on the union. GT agents: the B1 join
`C:/Users/Admin/a40-rescue/b1_train_plus_eval_agents.jsonl.xz` (covers eval139; state the coverage and drop nothing silently: windows without a
join record are counted and reported).

## 2. Feature sets (per window, causal: only frames at or before t0)
* **A (refcv8 today):** softmax of the 6 LON6 logits (`nav_true.tacv6_lon_logits`, battery extras dump_s0) + v0.
* **B = A + GT-LEAD:** the nearest GT dynamic agent in the ego corridor ahead at t0 (ego frame, |y| <= 1.8 m, 0 < x <= 80 m): presence flag,
  range x, lateral y, range-rate and lead speed from its track over the past 1.0 s (finite-difference, causal), and time-gap range / max(v0, 1).
  No-lead windows: flag 0, range 80, rate 0.
* **C = A + SHUFFLED LEAD:** B's lead block permuted across EPISODES (seed 0) -- the control.
* State the corridor/frame convention used and assert it on a control: a synthetic agent placed at (20, 0) m ego frame must be found at
  range 20, lateral 0.

## 3. Estimator
Logistic regression (L2, C = 1, standardised), **5-fold episode-grouped cross-fit** (fold seed 0): every window's P(decel) comes from a model
that never saw its episode. Score = AUC of the out-of-fold P(decel). ΔAUC (B - A) and (C - A) by the paired episode-cluster bootstrap over the
eval139 episodes (B 2000, seed 0) with the out-of-fold predictions held fixed. Report n windows, n episodes, the decel base rate, coverage.
Also the AUC of the raw lon head alone (P(decel classes) from the softmax), for reference to the 0.685 measured earlier.

## 4. Reading (fixed now)
* **LEAD-INFO PRESENT** iff ΔAUC(B - A) >= +0.03 with CI lo > 0 AND ΔAUC(C - A) has CI containing 0. Next: a vision-only lead feature from
  refcv8's OWN box head (P0 post-processing) -- admissible at inference -- measured the same way on eval139 (diagnostic) and, if it carries a
  material share of the GT gain, built into a FRESH-split gate (needs obstacle.offline for the 300 fresh clips: a download, sized first).
* **LEAD-INFO ABSENT** otherwise: the lead channel is not where the frontier breaks; next candidate channel = the traffic-light state
  (the v7.2 GT TL labels as the upper bound), same diagnostic.
* Tier: DIAGNOSTIC on logged eval139 labels, no planner involved -- never "driving performance". GT features are an UPPER BOUND, never an
  inference input (vision-only rule).
