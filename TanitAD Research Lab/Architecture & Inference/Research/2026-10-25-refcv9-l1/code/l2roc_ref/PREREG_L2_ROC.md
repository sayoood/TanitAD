# PREREG_L2_ROC — is refcv8's cruise false-braking a DISCRIMINATION loss or an OPERATING POINT? (Master Mind, 2026-10-25 ~11:40 Berlin)

Sealed before any AUC is computed. 0 GPU. Follows PREREG_L2 (sha256 4267cd68…) whose exploratory Z6 (eval-tuned, so not
admissible as a result) suggested refcv8 + stage and refcv7 lie on ONE false-decel / missed-decel frontier.

* Data: the banked EMA-48,852 per-frame battery (`C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf`), S6 surface,
  loaded by `l2_common.load_eval_pf` (identity checks K1 of PREREG_L2). COMPLIANT windows only. Inference seeds 0 and 1, each read
  separately (the planner samples; the seed floor is part of the answer).
* Score: dv = v_end − v0, the battery's own `plan_lon_class` quantity (`s6_lon_decomp.py:300`, last-chord speed, dt 1.0).
  Population: GT-decel (positive) vs GT-cruise (negative) windows, GT class from the same function on `g`. Lower dv = more braking.
* Statistic: AUC = P(dv_pos < dv_neg) + ½ P(tie). Arms: refcv8 + shipped stage (`stage_banked`), refcv8 os (no stage), refcv7-50,400.
* Estimator: paired episode-cluster bootstrap over EPISODES (B 2000, seed 0): resample episodes with replacement, recompute both
  arms' AUC on the same draw, report the difference's 2.5/97.5 percentiles. It answers "another draw of episodes" only.
* Controls that must read known values: (C1) GT scored against itself AUC = 1.0000 exactly; (C2) a constant plan (dv ≡ 0) AUC =
  0.5000 exactly; (C3) the arms' FD / MD at the battery threshold reproduce PREREG_L2's ZERO_GPU.json cells to 1e-4.
* Outcomes, committed:
  - **O-SAME**: Δ = AUC(r8+stage) − AUC(r7) has a CI containing 0 at BOTH seeds and |Δ| < 0.02 → D2 is an OPERATING POINT,
    not lost discrimination. Lever = a TRAIN-fitted operating-point shift inside the model (refcv9 target recast: AUC ≥ refcv7's,
    FD ≤ 0.10 at MD ≤ refcv7's 0.559). The locus roll is then deprioritised.
  - **O-WORSE**: CI entirely below 0 at both seeds → genuine discrimination loss. Lever = the SELECTION-vs-CANDIDATE locus roll.
  - **O-BETTER**: CI entirely above 0 at both seeds → refcv8 discriminates better; D2 is purely its operating point (as O-SAME).
  - **MIXED** (seeds disagree, or a CI excludes 0 with |Δ| < 0.02): reported as such; no lever is chosen on it alone.
