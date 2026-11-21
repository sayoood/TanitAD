# SPEC D0 — how good must the tactical layer be before conditioning SELECTION on it pays? (zero GPU, a design input)

*WP-B (refcv8 tactical conditioning), 2026-10-04. Registered BEFORE any number of this study exists; the script
`code/d0_dose_response.py` refuses to run if this file's sha256 differs from `raw/SPEC_SHA256.txt`.*

**What this is.** A label-side SIMULATION on refcv7-r101-s0 step 50,400's banked fans (launch tree `fec3a0d`, captured by
the route package: `D:/refcv7_route_bin/2026-10-04/{eval_s0g,eval_s1,train_s0}.npz`, md5 `1c48a53e…`, `5f783c84…`,
`53e06ec5…`). It does NOT train anything and it is NOT a lever: every arm re-scores the SAME captured fan with a
synthetic tactical posterior of known quality. Its job is to turn "condition selection on the tactical layer" into a
**number the tactical layer must reach** (R8-4 (i)) before the conditioning can beat refcv7's own pick — the bar the
refcv8 SPEC needs and does not have.

**Tier / evidence class.** OPEN-LOOP, single-shot, logged frames of the held-out eval139 grid (1,112 windows / 139
episodes). MEASURED on the banked arrays; the posteriors are SIMULATED (stated on every row).

## 1. Definitions (literal; `route_metrics.py` of the route package is imported, not re-typed)

* Window classes, GT direction, candidate direction: `rm.gt_class`, `rm.dir_class(rm.terminal_heading(·))` with
  τ = 0.18063741505146028 rad (the run's own). "classified" = turnL/turnR/straight/gentle (the route RESULT's 800).
* V0 = the shipped E9 pick (`sel_idx`); control: argmax of `s_e9` over `reach` must equal `sel_idx` on 1,112/1,112.
* **Calibrated noisy-oracle LATERAL posterior of accuracy q** (3 classes L/K/R): ŷ = y with probability q, else uniform
  over the two other classes; p(ŷ) = q, p(other) = (1 − q)/2; log p floored at log(1e-4).
  q ∈ {1/3 (CONTROL C-U, zero information), 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00}. R = 5 noise draws (seeds
  0–4); per-window Δ is averaged over the draws BEFORE the bootstrap; the across-draw SD of each headline is reported.
* **S-LAT(q, β)**: pick = argmax over `reach` of `s_e9 + β · log p(h_k)`, h_k = the candidate's direction class.
  β ∈ {0.25, 0.5, 1, 2, 4, 8, HARD} (HARD = keep only the argmax class; reach-only where empty), **fitted on TRAIN**
  (`train_s0`, FIT ONLY — the checkpoint saw those clips) by minimum mean ADE over classified windows, per q.
* **Longitudinal CONSTRAINT of relative error σ** (the speed-profile lever, B3): P_k = 6-s arc length of candidate k,
  P_gt = 6-s arc length of GT (windows with `gt_valid[-1]` and P_gt > 0.5 m; else the term is 0).
  P̂ = P_gt · exp(σ ε), ε ~ N(0, 1). **S-LON(σ, λ)**: `s_e9 − λ · |log((P_k + 1) / (P̂ + 1))|`.
  σ ∈ {0, 0.05, 0.10, 0.20, 0.30, 0.50}; λ ∈ {0.5, 1, 2, 4, 8, 16} fitted on TRAIN per σ. R = 5 draws as above.
* **S-JOINT(q, σ)**: both terms; (β, λ) fitted jointly on TRAIN over the two grids (HARD excluded).
* Controls (must NOT gain): **C-U** = S-LAT at q = 1/3; **C-SH** = S-LON with P̂ taken from a seeded derangement of
  windows (σ = 0); **C-CV** = S-LON with the non-oracle P̂ = 6 s × v0 (constant velocity) — not a must-fail control but
  the TRIVIAL predictor any tactical constraint must beat.
* **Real-model rows** (where refcv7's own heads land on the simulated curves): **H-LAT** = S-LAT with the real
  `p_lat` collapsed to 3-way by `rm.LAT_SIDE` (sum of the class probabilities per side); **H-E8** = S-LON with
  P̂ = the chord of refcv7's E8 6-s goal point (`g_tac[:, 2, :2]`) compared with each candidate's 6-s CHORD
  (chord vs chord, stated). Their measured 3-way accuracy / relative progress error on EVAL is printed beside them.

## 2. Metrics and estimator

Per arm, EVAL seed 0 (primary) and EVAL seed 1 (sampler replicate, same windows, same noise draws):
ΔADE vs V0 on all classified / GT-turn / GT-straight; turn direction-correct (Δ); heading within 15° on GT-turn;
|P_pick / P_gt − 1| median (the longitudinal proxy). Paired episode-cluster bootstrap over the eval episodes,
B = 2000, the route package's draws (`rm.make_draws(ep, 2000, 0)`). The interval answers "another draw of EPISODES"
only; the seed-1 column answers the INFERENCE question; nothing here answers the TRAINING question.

## 3. Reading rule (committed before any number)

* **R1 — the lateral bar.** q* = the smallest q at which S-LAT clears ALL of: turn ΔADE CI upper < 0; turn
  direction-correct gain CI lower > 0; straight ΔADE ≤ +0.05 m; all-window ΔADE ≤ 0; same sign on seed 1. q* is
  PROPOSED to the refcv8 SPEC as the minimum lateral 3-way accuracy (on the dense v9 labels, classified windows) the
  tactical head must reach before tactical conditioning of selection can be expected to beat refcv7's pick. If no q
  ≤ 1.0 clears, the lateral conditioning of SELECTION alone cannot beat V0 on this fan, and that is the finding.
* **R2 — the constraint bar.** σ* = the largest σ at which S-LON's all-window ΔADE ≤ −0.303 m (half of B3's
  −0.606 m bound), CI upper < 0, same sign on seed 1. Proposed as the maximum relative 6-s progress error of the
  tactical speed constraint.
* **R3 — instrument check.** If C-U or C-SH shows a gain whose CI excludes 0 on all-window ΔADE, the instrument is
  broken: STOP and report, no bar is proposed.
* **R4 — trivial-predictor check.** If C-CV captures ≥ 50 % of S-LON(σ = 0)'s all-window gain, the speed constraint
  is largely reachable from v0 alone: the R2 bar is then stated RELATIVE to C-CV (the tactical constraint must beat
  constant-velocity progress by the measured margin), never as an absolute.
* **Limitation committed now:** independent, difficulty-blind errors make the simulated posterior OPTIMISTIC relative to
  a real head whose errors concentrate on hard windows (turn onsets, roundabouts). H-LAT / H-E8 show how far below the
  curve a real head lands; a real head that lands BELOW its own accuracy's simulated point is the correlated-error
  penalty, and the proposed bar is raised by that measured gap.

## 4. Also reported (descriptive, no bar): refcv7 baselines for the R8-4 measures on this grid

* (i) TACTICAL accuracy of refcv7's tactical decoder (`p_lat` → 3-way, `p_lon` argmax) against the GT trajectory's
  own 6-s classes (lat: `rm.dir_class`; lon: HOLD v0 ≤ 0.5 ∧ vmax ≤ 0.5 / CREEP vmax ≤ 2 / BRAKE vmin ≤ v0 − 1.5 /
  ACCEL vmax ≥ v0 + 1.5 / CRUISE, speeds from the 8 slots' finite differences, D4's literals) — accuracy, macro-F1,
  per-class recall/precision with n, and the majority-class control. ⚠ A [0, 6] s proxy of the v9 [2, 8] s band.
* (iii) CONSISTENCY: share of fan candidates (within `reach`), of the E9 top-8 and of the pick whose direction class
  equals the tactical head's 3-way argmax, on all classified / GT-turn windows; and the pick's agreement with GT.
