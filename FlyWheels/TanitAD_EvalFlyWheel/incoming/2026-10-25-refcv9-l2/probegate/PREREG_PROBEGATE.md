# PREREG_PROBEGATE -- turn the lon-probe into a driving result (refcv9 L2-P)

Author: TanitAD Research Lab agent (for the Master Mind). Written 2026-10-25 (Europe/Berlin), dev box only; Thor untouched.
Status: SEALED (sha256 in `PREREG_PROBEGATE.sha256`) BEFORE any probe score for the 4,754-window roll was computed. Any later change is a sealed amendment file; no bar is changed after the seal.
Tier: T1 self-action OPEN LOOP on logged eval139 frames (a forced-class intervention on refcv8's own lon head; NOT driving performance, NOT closed loop).

## 0. What was read BEFORE this seal (disclosure; none of it is a probe score for the roll)
* v1 probe numbers, already reported (MEASURED, `lonprobe/LONPROBE.json`, seed 0): AUC HEAD 0.68506, PLAN 0.74454, PROBE 0.76239, SCALAR 0.60062; PROBE-HEAD +0.0773 [+0.0075, +0.1471]. v1 picks (k, C) per fold: [32,.01] [64,.01] [16,.01] [32,1.0] [32,.01].
* Counts: the roll has 4,754 windows over 139 clips; the head's native argmax group over the LON6 hypotheses is DECEL on 941 (0.1979), CRUISE on 2,909, ACCEL on 904 (banked battery logits; the os-only roll's raw logits give the same groups on all 4,754 windows).
  136 episodes carry S6-compliant GT-decel (639) / GT-cruise (1,856) windows (the fit rows, n 2,495); clips with no fit rows: 3 (102 windows), named only by sha12 in kept files.
* L2 results (MEASURED, `result/final/L2_RESULT.json`): BASE-os stage FD 0.1395 / MD 0.4695; ORACLE stage FD 0.1379 / MD 0.3412; ORACLE vs BASE-os MD -0.1283 [-0.1802, -0.0762]; S6 ADE -0.1504 [-0.2564, -0.0631].
* Plumbing runs that computed NO probe score: policy unit test on SYNTHETIC scores with hand-derived literals + two deliberate-regression mutants (`pg_policy_test.py`, 6 tests GREEN); a glue dry-run of `pg_labels.py --dry` with SEEDED RANDOM scores (no probe fitted) and its independent verification; a scorer self-test with the L2 ORACLE trace as a stand-in for both new arms.
  ⇒ DISCLOSED: that self-test showed the privileged ORACLE force itself reads three LATERAL rows (`LAT_curvature_mae_1pm_valid`, `LAT_heading_mae_deg`, `LAT_yaw_rate_mae_radps_valid`) separated WORSE than BASE-os. Clause G below is the brief's, unchanged; the scorer reports the ORACLE rows beside it so a G failure can be attributed to the channel or to the probe (section 8).

## 1. Question and claim
Does using the confirmed probe (cross-fitted logistic on [PCA_k(z_tac) | v0], AUC 0.762 vs the head's 0.685) as a GATE on refcv8's lon-head decel/cruise decision cut missed-decel on the S6 population WITHOUT raising false-decel and WITHOUT degrading the four metric families? PASS would make "train refcv9's lon head toward the probe" the deployable lever; FAIL names the clause and the next lever.

## 2. Inputs and identities (all pinned by sha256 in CODE_PINS.txt, table in section 11)
* Model/ckpt: refcv8 EMA-48,852 (`E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt`, md5 `b4364029b68f29a33eeb4aebbbbd002b`), per-frame nav feed, os-only roll, inference seed 0, windows `step48852ema_pf/windows_s0.json` (md5 `a3a78dba889e70aa69a05271559794ad`, 4,754 windows).
* Head logits used by the policy: `raw_lon_logits` of the L2 ORACLE trace (`bank/trace_oracle_s0.npz`) = the lon head's output in the os-only roll (identical across all three L2 traces, max abs 0.0; differs from the battery's banked logits by <= 0.0116 with 0 argmax-group disagreements).
* z_tac: `refcv9_l1/bank/ztac_eval_s0.npz` (L1's un-hooked os-only roll), joined on (clip_index, ws = t + W - 1) exactly as `lonprobe.load_bank`. v0: the banked extras `v0` (S2 order; equals the S6 `v0` on all S6 rows, max abs 0.0, and the trace `v0_in`).
* GT classes / S6 compliant mask / eid: `l2_common.load_eval_pf(0)` + `l2_roc.dv_of/cls_of` (decel/cruise/accel from the 6 s chord, +-1.5 m/s) -- as lonprobe.

## 3. The probe (v1's recipe, unchanged) and the policy
**Cross-fit.** Fit rows = S6-compliant windows with GT in {decel, cruise} (n 2,495 = 639 + 1,856; 136 episodes). Folds = `lonprobe.folds(eid_fit, 5, 0)` (episode-disjoint). For fold f the probe is `lonprobe.fit_pred(..., use_z=True, seed=0)` fitted on the fit rows of the other four folds (inner 4-fold grid k in {16,32,64,128} x C in {0.01,0.1,1,10}, scored by AUC; PCA and scalers fitted on the training rows only) and predicts a score for EVERY roll window of the held-out episodes of fold f. A clip with no fit rows (no model has seen it) gets fold = (its position among such clips, sorted by episode id) mod 5. The score is the logistic decision function (higher = more decel-like).
**Self-check (abort if it fails):** the scores of the roll windows that ARE fit rows must reproduce v1 -- AUC vs GT within 1e-6 of 0.7623863391 and picks equal to v1's. (Same models by construction; a mismatch means the glue is wrong.)
**tau (label-free).** k = number of roll windows whose head argmax group (over LON6, v7 ids 0,1,3,4,5,7) is DECEL (v7 3,4,5). The probe says "decel" on the k highest scores (ties by window index) -- i.e. the probe's decel rate equals the head's native decel-group argmax rate over all 4,754 windows (k = 941, rate 0.1979). tau is the k-th largest score. No GT label is used to set it.
**PROBEGATE policy (per window).** probe says decel and head group is not DECEL -> force the head's highest-logit class INSIDE the decel group (v7 3/4/5). Probe says not-decel and head group is DECEL -> force the head's highest-logit class inside the cruise group (v7 0/1). Otherwise NATIVE (label -100). Forcing = the L2 oracle hook (one-hot +-20 on that v7 column at `RefCV3Model._r8_apply_force`), via `l2_roll.py --arm oracle --labels-npz`, unchanged. Equal marginal rates imply n_up = n_down exactly (asserted).
**RANDGATE (mandatory control).** The same n_up and n_down. One generator `np.random.default_rng(0)`: first `choice(eligible_up, n_up, replace=False)` where eligible_up = windows whose head group is not DECEL, then `choice(eligible_down, n_down, replace=False)` where eligible_down = windows whose head group is DECEL. Same forced-class rule. It separates the probe's information from "forcing more braking".
**0-GPU verification before launch:** `pg_verify_labels.py` re-implements the policy as a per-window python loop (no shared code) and requires window-by-window equality with the label files, equal n_up / n_down, forced classes inside the right group, all labels in LON6 or -100, and the RANDGATE draw reproduced. The chain refuses to launch unless `LABELS_CHECK.json` and `LABELS_VERIFY.json` both read ALL_PASS and the label files match their recorded sha256.

## 4. Rolls, GPU budget, lock
* Seed-0 rolls: PROBEGATE then RANDGATE, each the full 4,754 windows (`l2_roll.py --arm oracle`, unchanged, sha-pinned; `--require-seal` against L2's own seals). Traces and dump dirs on `F:/refcv9_probegate/`; dumps deleted once the trace is final; small traces copied to `probegate/bank/`.
* ESTIMATED GPU time (from L2's logs, MEASURED walls incl. process start): const_cruise 2,209 s, cal 2,409 s, oracle 2,542 s per 4,754-window roll => 2 rolls 4,400-5,100 s (1.2-1.4 h); cap 1.6 h = 5,760 s of lock-held time. The chain rolls RANDGATE only if (lock-held time after PROBEGATE) x 2.05 <= 5,760 s (i.e. PROBEGATE <= 2,810 s); otherwise it writes the CAP marker and P3 reads PENDING (never PASS).
* Shared lock `gpu_lock.py` (job `refcv9-probegate`): acquire in the detached chain, assert on its `--out` JSON, release in a trap, never break a foreign lock (P4 holds it now; L2-B and L1C are queued; the lock is non-FIFO). OMP_NUM_THREADS=4. Lock wait <= 12 h, else NOLOCK/NO VERDICT.

## 5. Scored population, arms, estimator
* Population: S6 compliant (3,251 windows, 138 episodes); false-decel on GT-cruise (n 1,856), missed-decel on GT-decel (n 639); the shipped stage applied post hoc exactly as L2 (`l2_common.apply_stage`, K1 bit-equal to the banked S6_pos); plan classes by `s6_lon_decomp.plan_lon_class`.
* BASE = L1's un-hooked os-only plans (`BASE-os`, L2's primary comparator); ORACLE = L2's banked `trace_oracle_s0`. REUSE IS PROVEN, not assumed: native300 and ident300 plans are bit-equal to BASE-os on 308 windows (max abs 0.0, selections equal), and the ORACLE trace's 1,082 unforced windows are bit-equal to BASE-os (max abs 0.0, selections equal) -- and K3 re-proves it in situ on the unforced windows of PROBEGATE and RANDGATE. If K3 fails the verdict is INVALID.
* Estimator: paired episode-cluster bootstrap, `pair_read_r8.paired_cell -> taniteval.ci.paired_episode_cluster_bootstrap`, B 2000, seed 0, over EPISODES. It answers "would another draw of EPISODES say this?" ONLY. It is blind to training variance (one refcv8 training seed; H-ESTIM-SEED-1) and to inference variance (one inference seed, 0; L2's BASE-os seed-0-vs-1 floor: FD 0.0022, MD 0.0016, INHERITED context). A separated interval here is NECESSARY, not SUFFICIENT, for a lever claim.

## 6. Bars (S6 compliant, with the stage; all cells vs BASE-os unless stated)
* **P1** PROBEGATE missed-decel separated BELOW BASE: paired cell `missed_decel`, `hi < 0`.
* **P2** PROBEGATE false-decel NOT separated above BASE: paired cell `false_decel`, `lo <= 0`.
* **P3** PROBEGATE - RANDGATE missed-decel separated below 0: paired cell (PROBEGATE vs RANDGATE) `missed_decel`, `hi < 0`.
* **G** (STRICT, as L2): none of the 14 `pair_read_r8.METRICS` rows (S2 four-family set: ADE, LONGITUDINAL, LATERAL, TACTICAL; computed through `l2_common.arm_metrics`) is separated WORSE than BASE (lower-is-better: `lo > 0`; higher-is-better: `hi < 0`), and S6 ADE (`ade_m`) is not separated worse (it is one of the 14, listed separately).
* Reported beside, DESCRIPTIVE (not bars): ceiling share (MD_base - MD_probegate)/(MD_base - MD_oracle) with an episode-cluster bootstrap CI (B 2000, seed 0), oracle numbers recomputed and asserted equal to L2_RESULT (K8); the same share for RANDGATE; fixed/broken windows vs BASE for MD and FD; the descriptive flip table on the S6 population; the rows the ORACLE force and RANDGATE degrade; an ESTIMATED AUC* (two-point linear extrapolation of share against AUC, flagged inadmissible beyond 2x the fitted AUC range).
* Families not computed (named work items): distance-keeping (no lead-agent track in the banked refcv8 dumps, n 0) and STRATEGIC (refcv8 runs --no-strategic, n 0).

## 7. Controls (any failure => verdict INVALID, never PASS)
K0 seals; K1 stage bit-equal banked; K2 population (3,251 / 138); K3 per new trace: trace order = extras order, unforced plan8 max abs vs BASE-os <= 1e-5 and selections equal, hook counters (calls 4,754 = hook_applied + hook_native_fallthrough; hook_applied = oracle_labelled = #forced labels; unarmed/multi/batch>1/navnull = 0), applied logits one-hot (+20) on forced windows and equal raw on unforced, raw logits equal the ORACLE trace's raw; K4 the label files re-derive from the saved scores with `pg_policy`; K5 `LABELS_CHECK`/`LABELS_VERIFY` ALL_PASS + sha256 match; K6 known-null: BASE vs BASE reads delta = lo = hi = 0 on every row, none separated; K7 known-positive: ORACLE vs BASE reproduces L2's `missed_decel` / `false_decel` / `ade_m` cells exactly; K8 BASE-os and ORACLE MD rates equal L2_RESULT's to 1e-12.

## 8. Outcomes (mechanical, written to `result/VERDICT.md`) and the pre-registered next lever (Rule Zero)
* **PASS** = P1 and P2 and P3 and G with all controls GREEN => the deployable lever for refcv9 is lon-head training toward the probe (L3: the cross-fitted decel posterior as an auxiliary target / distillation).
* **FAIL (clauses)** => the failing clauses are named, with this next lever keyed by the first failing clause in the order P1, P3, P2, G (text is in the pinned `pg_score.py::NEXT`): P1 failed with MD point estimate improved ("THIN GATE": a more informative gate, screened at 0 GPU first on banked material); P1 failed with MD not improved ("GATE INERT": stop gating post hoc, attribute the oracle gap per forced class at 0 GPU, train toward the probe only if one class carries the gap); P3 failed ("NOT PROBE-SPECIFIC": margin-conditioned flips + the same RANDGATE); P2 failed ("FALSE-BRAKE COST": up-only margin gate); G failed ("COLLATERAL", or "CHANNEL-LEVEL" when every row it names is also degraded by the privileged ORACLE force: then G as written cannot be passed by any lon-class-force lever and the lever must be paired with a lateral guard).
* **INVALID (control failed)** and **PENDING (an arm not rolled, e.g. CAP / lock timeout)** are never read as PASS or FAIL of the idea; PENDING P3 with P1/P2/G all true is reported as such.
* A FAIL is reported as FAIL, then the next lever is executed or its blocker named by the Master Mind; the chain only names it.

## 9. Limits stated in advance
* The probe is cross-fitted over the EVAL episodes (the only episodes with the z_tac + GT pair banked for this checkpoint): it demonstrates that the information exists and can be used at the head seam; a deployable head would be trained on train episodes (that is what L3 would test). Rates are not extrapolated beyond this population.
* One training seed, one inference seed, one stage table: a separated cell is a statement about episodes only.
* No threshold, bar, population, or comparator is changed after seeing any PROBEGATE score or roll.

## 10. Files
`probegate/`: PREREG_PROBEGATE.md/.sha256, CODE_PINS.txt, pg_policy.py, pg_policy_test.py, pg_labels.py, pg_verify_labels.py, pg_score.py, chain_pg.sh; written by the run: probe_scores_s0.npz, labels_probegate_s0.npz, labels_randgate_s0.npz, LABELS_CHECK.json, LABELS_VERIFY.json, labels.sha256, chain.log, logs/, bank/, result/{PG_RESULT.json, PG_RESULT.md, VERDICT.md}. Heavy: `F:/refcv9_probegate/` (traces, roll json, dumps).

## 11. Pinned code and input hashes (sha256; verified by the chain at start)
```
792522ff0ba96a48fc86ff80cc196df833bcdc9ee4b98dd377525e45b9149a08  code/l2_common.py
4d8029554b7ad040edf7f66831457b1b65b9d87cead4bce20f2299d9e85ba113  code/l2_roll.py
b85f31b5ebd537656ca57ed3e3f9162efeaab66f6c8a7e4059351edbd99067f0  code/l2_score.py
8f0bfb76f0f7aa83283309d8b2b814c54741ccd6a5b95f8996f8788f268f67b3  code/l2_fit.py
359081d637c9b4c8b6da2f40a3f422724d6716a47844f14f821e292bfccef2f2  code/refcv8_eval_remap_devbox_l2.json
51edc7477abafb7da39af7a1a2bd3abc06ebfb1baa682af59c1647c689bdccde  lonprobe/lonprobe.py
5f7cc04733ce18f32d2b4bcb5d84901cc0bd3d7d64a5b5c7142af34c0c540261  roc/l2_roc.py
00ea6c21925a0001c4f4e589b01670570b61da5cb0648d113e419d11d96d82c2  probegate/pg_policy.py
2ede0576616de013991aa2c28a2d13159b917c1157ce1e95547aff5d811500a8  probegate/pg_policy_test.py
07dfa3489ce89303bc52209016e265fa1618c7d24de4adfa6e7fa0968cf40d4f  probegate/pg_labels.py
12d58567eef173e6d326228c3ff5a27e5165d5d527acaeba2c0e8a199c76758f  probegate/pg_verify_labels.py
d36f5b2d908489eec3600a33a50f621c8c1a9cb0dcc53a0cd76c8d96e5e157d3  probegate/pg_score.py
f80ea9af1890789cc91a78206cfdf714c055c24433e51871ef38e73a8d959036  probegate/chain_pg.sh
eb4575bc70887039e1e65f87748f6361eb13f6a545f1ee14d6344cb1428f5fae  bank/trace_oracle_s0.npz
e807c0d7d43ece24e39b6b23d1d5504162222bddca2e24b86c7e7b0d48bd1586  ../refcv9_l1/bank/ztac_eval_s0.npz
d84f17b8b222f5d457c841dfa099f40e1085cb77a27e025563ecaf47283e23d4  ../../refcv8_eval_archive/battery/step48852ema_pf/dump_s0/refcv6_extras.npz
48d04401cf115f1e2e235fb07e3b7b2fd1f79a2b0d2a36d89f33336cf8410aa2  result/final/L2_RESULT.json
```
`CODE_PINS.txt` (these lines, in this order) has sha256 107d65f667e81043199e98da21b833dc3fea692c92453ec264cab0f5ee181088.
