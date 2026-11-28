# RL-S1b: attribution of R0's TTC loss, and the lever (report)

Written by the RL-S1b research agent (Opus) on 2026-10-14. The agent's harness refused the file write, so the Master Mind banked it verbatim from its hand-back.
Pre-registration: `eval/PREREG_RL_S1.md` section 10. Scripts and JSON are in this folder.

**Evidence and estimator.** Every number comes from banked HF artifacts. Their sha256 values match the kill package and the data manifest; the lookup
reproduces the kill package exactly (C0 90.0598 / R0 91.2230 PDMS). Every interval is the paired log-cluster bootstrap (10,000 resamples, seed
20260927). It answers only "would another draw of LOGS say this?". RL-val is a selection tier: one training seed per arm, never an endpoint.

**Arm names.** C0 = C0's BCE-only arm A. R0 = R0's EPO arm B. R0-A = R0's own BCE-only arm A: same seed and flags as C0, differing only in RNG
consumption. R0-A reads the same as C0: PDMS +0.01 [-0.13, +0.14], TTC 0.00, every within-scene AUC delta 0.0000.

## Verdict: the TTC loss is the SCORER's generalisation, not the REWARD's preference

## 1. Changed picks (Q1)

R0 changed its pick on 2,329 of 3,000 sets.

| subset | n | R0's reward prefers R0's pick | mean reward gap (R0 - C0 pick) | median rank of R0 / C0 pick (of 64) | EP gap |
|---|---|---|---|---|---|
| became TTC-failing | 59 | 1 | -0.400 | 46 / 15 | -0.059 |
| stopped failing TTC | 29 | 28 | +0.417 | 16 / 50 | +0.113 |
| became NC-failing | 14 | 0 | -0.757 | 57.5 / 9 | -0.795 |

59 - 29 = 30 = the -1.00 TTC (30 of 3,000).

## 2. Safety heads (Q2)

Within-scene AUC at update 1000, R0 vs C0:

| head | C0 | R0 | delta |
|---|---|---|---|
| TTC | 0.8881 | 0.8818 | -0.0063 [-0.0131, +0.0005] |
| NC | 0.9271 | 0.9186 | **-0.0085 [-0.0173, -0.0001]** |
| DAC | | | -0.0037 |

The TTC logit absorbed progress: its correlation with true EP on TTC-passing proposals moved from -0.257 to -0.083. On the TTC-failing plans R0
picked, R0 predicts p(TTC) 0.935, C0 0.916.

## 3. Hybrid selections vs C0 (Q5)

| hybrid | PDMS | TTC |
|---|---|---|
| safety from R0-A + progress from R0 | +0.95 | **-0.80 [-1.51, -0.14]** |
| the reverse | +0.63 [+0.25, +1.02] | -0.23 (not separated) |

About four fifths of the loss rides on the progress outputs. Stop-gradient on the safety heads can recover at most about 0.2 of the 1.0.

## 4. Candidate rewards (Q3, Q4)

Candidates: (a) R0; (b) guard off; (c1) TTC/NC as hard multipliers; (c2) a fixed penalty of 5/12 or 2/3; (d) lexicographic.

* RL-train labels: 13.6 % of proposals and 50.4 % of scenes carry a TTC failure. For every candidate:
  * the argmax is TTC-failing in 0.07-0.11 % of scenes;
  * the Boltzmann(R/0.02) mass on TTC fails is 0.07-0.11 %.
* Under the real base policy on RL-val, the share of the positive policy-gradient push on TTC fails is 2.06 % for (a). Only (c1) changes it
  materially, to 0.01 %. The pick-level evidence (section 1) says this channel is not the cause.
* First-step EPO gradient, the share of each head's push AGAINST its own label: TTC 17.6 %, NC 34 % under R0. No reward removes it.

## 5. Dose curve (ESTIMATED: logits interpolated C0 -> R0)

| alpha | PDMS | TTC |
|---|---|---|
| 0.25 | +0.60 [+0.27, +0.95] | -0.03 |
| 0.375 | +0.71 | -0.27 |
| 0.5 | +0.84 | -0.47 |
| 1.0 | +1.16 | -1.00 |

At 0.25, TTC at the four checks (u250 / u500 / u750 / u1000) is -0.30 / -0.13 / -0.10 / -0.03, so g5 holds at each. The control also moved:
C0 vs base PDMS -0.57, TTC +0.83 [+0.13, +1.58]. g5 stays versus the control.

## 6. Why the dose and not the other levers

| lever | why not |
|---|---|
| reward levers | the preference they would fix is not there |
| stop-gradient on the safety heads | capped at about 0.2 of the 1.0 (hybrid H1) |
| a stronger KL anchor | pulls toward the base, against the control's TTC gain |
| CPU proxy | inadmissible: it failed its pre-set face-validity check |

The value rule: the largest grid alpha with TTC and NC deltas within half the g5 bound at u1000 and within the bound at every check. It was
stated after seeing the curve, on RL-val (hyper-parameter selection), and is disclosed as such in section 10.

## 7. Implementation

`--epo-lam 0.25`, an existing flag. Reachability, all in `scorer_finetune.py`:

| step | line |
|---|---|
| parse | 534 |
| `lam_pg = a.epo_lam * epo_ramp(...)` | 750 |
| `epo_loss` returns `lam_pg * (...)` | 201 |
| `lB = lB + ep` | 765 |
| `backward()` | 1313 |
| `info["epo"]["lam"]` (G-DVB reads it) | 670 |

`test_rl_s1b.py` (ZZTEST_PASS 18 = 1 unmutated suite + 17 RED mutations) checks:
* 13 EPO / loss segments pinned to R0's launch blob;
* the `rl_reward.py` blob;
* dose identity: grad(0.25) == 0.25 x grad(1.0) bit for bit, and grad(0.0) == 0.0;
* the real `main()` end to end: preflight PASS, `epo.lam` == 0.25, control arm A bit-identical across doses, and B == A at dose 0;
* the gate accepts exactly C1 / R0b / R1b and refuses wrong-dose and wrong-seed variants;
* `refe_rls1` is unchanged;
* G-DVB on the built `epo.lam`.

A vacuous mutation in `test_rl_epo` ("EPO leaks into arm A" only doubles B's dose) is replaced here by one that really feeds a dosed EPO into
arm A.

Re-verified by the Master Mind on the dev box, 2026-10-14:

| suite | result |
|---|---|
| `test_rl_s1b.py` | ZZTEST_PASS 18 |
| `test_rl_epo.py` | ZZTEST_PASS 42 |
| `test_launch_gate.py --static` | ZZLG_STATIC_PASS 552 |

## 8. Caveats

* **Interpolation is not training.** AdamW and the per-arm clip mean the realised dose may differ from 0.25; K3 is the guard.
* **The trainer changed after R0's launch.** `main.fwd` gained the default-OFF aux feed path. Outcome O5 checks the residue: R0b-A vs C0.
* **R0-A vs C0 is a same-seed rerun floor, NOT a training-seed floor.** Only C1 gives the training-seed floor.
* **The 3-seed CPU proxy (`sim_rls1b_proxy.py`) was still running** when this was banked. Its 1-seed run (`proxy_seed0.json`) failed the
  face-validity check set before it ran, so it ranks nothing.
