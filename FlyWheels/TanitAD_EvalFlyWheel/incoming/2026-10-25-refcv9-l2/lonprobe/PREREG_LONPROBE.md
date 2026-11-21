# PREREG_LONPROBE — does refcv8's z_tac hold a better braking decision than its lon head extracts? (Master Mind, 2026-10-25 19:2x CET)

Sealed before any probe is fitted. 0 GPU. Motive (MEASURED, H-R9-L2-CHANNEL-1, a0a29e2): feeding the GT lon class cuts missed-decel
0.470 -> 0.341 and S6 ADE by 0.150 m (separated), while refcv8's lon head separates decel from cruise only at AUC ~0.685.

* Population: EMA-48,852 per-frame battery, S6, COMPLIANT windows, inference seeds 0 and 1 read separately (`l2_common.load_eval_pf`).
  Positive = GT-decel, negative = GT-cruise (`plan_lon_class` on `g`, the battery's rule). z_tac from the L1 bank
  (`refcv9_l1/bank/ztac_eval_s{0,1}.npz`), joined on (clip_index, ws) exactly as `l1_fit_score.join`; a missing join voids the run.
* Arms (all scored by AUC on the same windows):
  - HEAD: refcv8's own lon head — decel-group logsumexp minus cruise-group logsumexp of `nav_true.tacv6_lon_logits` (l2_common groups).
  - PLAN: the emitted plan's dv = v_end − v0 (with the shipped stage), sign flipped so higher = more braking (the L2 ROC number).
  - PROBE: logistic regression on [standardised PCA_k(z_tac) ‖ v0], CROSS-FITTED over EVAL EPISODES (5 folds, numpy seed 0);
    standardisation, PCA basis, k ∈ {16, 32, 64, 128} and C ∈ {0.01, 0.1, 1, 10} chosen by inner 4-fold episode CV on the training
    folds only. No window is scored by a model that saw its episode.
  - SCALAR: the same procedure on [v0] alone (floor).
  - SHUFFLE (control): PROBE with labels permuted within the training folds — must read AUC within 0.5 ± 0.03.
* Estimator: paired episode-cluster bootstrap over episodes (B 2000, seed 0) for PROBE − HEAD, PROBE − PLAN, PROBE − SCALAR.
  It answers "another draw of EPISODES" only; one training seed; z_tac is identical across inference seeds (L1 K6), so seed 1
  differs only in the HEAD/PLAN arms and the GT population (unchanged).
* Controls: HEAD's AUC reproduces the L2 agent's 0.685 within 0.01 at seed 0 (definition check); PLAN reproduces the L2 ROC's
  0.7445 / 0.7361 exactly; SHUFFLE as above.
* Outcomes, committed:
  - **HEADROOM:** PROBE − HEAD separated > 0 at both seeds and PROBE − SCALAR separated > 0 → z_tac carries a better braking decision
    than the head extracts; the lever is lon-head training (recipe / weight / a clean-split refit), sized by the oracle arm.
  - **NO-HEADROOM:** PROBE − HEAD CI holds 0 or is below → the head already extracts what is linearly decodable; the lever is the
    trunk's input (vision), not the head.
  - PROBE vs PLAN is reported beside: it says whether a z_tac decision would beat the plan's own implicit braking choice.
* A linear probe failing is not evidence about learnability by a nonlinear map (CLAUDE.md); the verdict states the function class.
