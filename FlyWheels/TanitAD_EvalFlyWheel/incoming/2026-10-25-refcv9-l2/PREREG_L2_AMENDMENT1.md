# PREREG_L2 — Amendment 1 (OPERATIONAL ONLY: roll order, early stop, CAL-bank dependency). Sealed before any roll.

Parent: `PREREG_L2.md` sha256 `4267cd684f38a0163fe1d5bf7c96a8b3ec884d48a01a3474d473b557c3bfb06e` (sealed 10:56). No arm, bar, split or calibrator family changes.

## What was seen between the seal and this amendment (calibrator-independent, `result/zero_gpu_nocal/ZERO_GPU.json`, MEASURED, 0-GPU)

* K1 (stage bit-equal to banked S6_pos, both seeds) PASS; K2 (populations 3,251 / 138 / 1,856 / 639 and the R8-CONS-1 literals) PASS.
* Z1 raw head confusion (compliant, seed 0; rows GT class, columns head group decel / cruise / accel): GT-cruise 252 / 1,356 / 248; GT-decel 266 / 280 / 93. Head-level false-decel 0.136,
  head-level missed-decel 0.584 (the head is a WEAKER decel detector than the plan: plan missed-decel 0.470).
* Z2 head-level AUC of the decel-vs-cruise log-odds (GT-decel over GT-cruise) 0.685; ORACLE-TUNED bias frontier: head-FD 0.073 costs head-MD 0.682, head-FD 0.036 costs 0.764.
* Z3 association (banked plan with stage, seed 0): P(plan decel | GT-cruise, head says cruise) = 0.131 on 1,356 windows (seed 1: 0.130) — the same windows carry 178 of the 254 false-decel
  windows. The CONST-cruise-limit PROJECTION (every head call = cruise) is therefore FD ≈ 0.131, above the 0.10 bar. ESTIMATED (association, not an intervention).

## Amendment

1. **Roll order** (operational; replaces PREREG_L2 §10's order): NATIVE-300, IDENT-300, **CONST s0**, CAL s0, ORACLE s0, CAL s1. CONST needs no calibrator and decides O3 vs O4 first.
2. **Early stop, mechanical (`l2_score.py --gate`)**: CAL s1 is SKIPPED iff CAL s0+stage fails at least one of P1, P2, P4, G (the clauses common to PASS and PASS-AT-SHIPPED-BASELINE); a FAIL at one
   seed is a FAIL, so a second seed cannot change it. If CONST s0+stage has its FD lower CI bound above 0.10 the label O4 is stamped immediately, but CAL s0 and ORACLE s0 STILL run (the lever
   as asked, and the ceiling).
3. **Calibrator dependency**: the CAL arms need `bank/calibrator_48852.json`, fitted from L1's EMA-48,852 TRAIN roll. The chain waits for L1's `ZZL1ROLL_train_s0ZZ` marker (or for L1's chain to
   end) BEFORE it queues for the GPU lock; if L1 ends without that bank, CAL is **PENDING (NOTRAINBANK)** and the chain runs the other arms only. Nothing is fitted on any other checkpoint.
4. The ORACLE arm forces the LON6 class of the GT label `lon_v7` where the label is a LON6 hypothesis and leaves every other window native (counted, reported).
