# (prose blocks for RESULT.md — assembled after the numbers land)

## SCOPE-1 — what a T0 selection readout is, and what it is not

Everything measured here is **T0**: a readout of *which already-emitted candidate is returned*,
scored against the logged human future. ⛔ It is **never** a driving claim and it may **never**
be compared against a T1 number. The two `ade_m` figures in this programme are different
objects: this panel's is the **2 s prefix ADE over 4 slots** (0.5/1.0/1.5/2.0 s), and the T1
harness's is its own; `D-RL-VETO-T1-1`'s +0.0362 m is a T1 number and is quoted here only as
the RL arm's exit, never as a comparison against anything in this document.

## SCOPE-2 — the corpus

EVAL split, `run_refcv3_ol/data/eval` + `s2_labels_v7.2_eval.jsonl.gz` +
`b1_eval_lead_block.npz`; **NON-PARITY**, as the base checkpoint itself is (`D-RL-READY-1`).
No cross-arm comparison outside this panel is licensed by anything here.

## SCOPE-3 — `kamm_over` and `peak_g` are LOWER bounds

The emitted fan is free waypoints, so `flyability.friction_load` (the exact, control-rolled
instrument) cannot be applied and the finite difference under-reports by **1.21–1.85×** — on
**both** sides of every comparison, so ratios are robust and levels are lower bounds.

## SCOPE-4 — what "no scene input" means, precisely

MEASURED at source (`raw/no_scene_input.json`; AST call-graph walk **and** a runtime
permutation test, three positive controls):

- The **ranking function** `feasibility + comfort` reaches only `_motion_gate` and `kinematics`
  and reads six ctx keys — `dt`, `a_max`, `kappa_max`, `jerk_max`, `lat_acc_max`,
  `min_motion_m` — **every one a programme constant**. No `v0`, no `lead_path`, no `obstacles`,
  no `gt_traj`, and no dynamic ctx access. Under a full scene garble the gate score and its
  argmax are **bitwise unchanged** (max|diff| 0.000e+00) while `headway` moves 7.5e-01,
  `collision` 1.0 and `progress` 3.06 — so the garble took, and the invariance is a property of
  the components rather than of the test.
- The **candidate set** the gate ranks over is `order[:, :k]` = the model's own `sel_score`,
  masked by `reach_keep` — both scene-conditioned **model outputs**.

⇒ The admissible claim is **"adds no NEW scene input and no new perception"**. Every input the
gate uses, the deployed model already computes. It is admissible under the
vision-only-at-inference rule for that reason, not because it is blind.

## SCOPE-5 — the three variances, and which one the interval answers

| # | variance | applies? | why |
|---|---|---|---|
| V1 episode/window draw | **YES** | answered by the paired episode-cluster bootstrap **and** by an independent, episode-disjoint second draw |
| V2 training run (`H-ESTIM-SEED-1`'s hole) | ⭐ **STRUCTURALLY ABSENT** | zero training steps; both arms are the same frozen checkpoint (md5 asserted). A same-flags/different-seed replicate **cannot differ**, because no seed enters. `H-ESTIM-SEED-1` is closed **by construction** for this lever — the only lever in the programme of which that is true |
| V3 inference sampling | **measured, not assumed** | `refc.py:1720` and `:1501` carry stochastic ops gated by **config**, not by `self.training`. `G-DET` runs the same batch through the same model twice in one process and compares bitwise |
