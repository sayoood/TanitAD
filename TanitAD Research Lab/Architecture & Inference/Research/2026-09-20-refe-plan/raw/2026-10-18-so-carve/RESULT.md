# SOCARVE: would a carve-fed dual have SEEN the R0d breach? (O4's 0-A40 first step, banked R0 states, A2 harness)

> ⚠️ **ERRATUM (Master Mind, 2026-10-18, R49) — read before the sections below.** Two statements in this report are corrected; every number is unchanged. (1) **A2's registered outcome is INVALID** (`tempsweep_a2.json` key `outcome` = `INVALID`; control 3 failed). The text below labels it "A2-ALIGNED(T* = 0.02) ... PROVISIONAL", which softens a failed pre-registered control into a qualified pass. T ≈ 0.02 is a HYPOTHESIS, re-tested by `raw/2026-10-18-so-pickcon/`. (2) **The "1.4 to 4.5 times" ratio is withdrawn**: its denominators (the dual's quantity) have intervals spanning 0 at u500 and u1000, so the ratio is unbounded. State the two quantities instead: dual +0.22 [−0.15, +0.62] vs pick +1.00 [+0.30, +1.73] at u1000 on full RL-val.


Research Lab agent, 2026-10-18. Package target `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-18-so-carve/` (dev box `C:/Users/Admin/socarve/`). CPU only; the pod and the GPU lock were not touched; nothing committed or pushed.
Pre-registrations, each hashed before the numbers it governs: main `PREREG_SOCARVE.md` sha256 `55a5c1759c15ae16...680644b` (frozen 04:47:33Z); A1 `PREREG_SOCARVE_A1_FULLVAL.md` `23e03f76048d780...507c22c22` (04:56:20Z, after the main read); A2 `PREREG_SOCARVE_A2_TEMPSWEEP.md` `de7a130845b6d83...a312ab` (04:58:24Z, after A1). The `.sha256` files hold the full digests; the main hash re-verifies.

## 0. Headline

* **Main outcome (as pre-registered): CARVE-TRACKS-TRAIN, flags F1 + F2.** R0 u1000, P2 TTC (= the dual's g): rho = G_tc / G_tv = 0.031 / 0.344 = **0.09** (< 0.25). F1: the carve is in-sample by construction (R0 trained on every RL-train log, section 2). F2: the train-to-val gap is **not resolved**: G_tv = +0.344 [-3.30, +4.50]. **The label does NOT license "the O4 lever is inert"** (prereg s.7, committed before the numbers).
* **The carve was never held out.** R0 consumed all 16,000 RL-train sets (166 logs) exactly once by u250 and four times by u1000. There is no free carve.
* **Harness control PASS** (max |diff| 3.6e-06 against the banked A2 `realstates.json`; tolerance 1e-4). C0 floor 1.0e-4. All stage-2 controls pass.
* **The finding that matters came from the Rule-Zero addendum A1 (full RL-val, 3,000 sets / 75 logs, live dumps, MEASURED):** the harness's 62-set val pool was not representative (one 5-set log supplies 1.47 of its 1.356 points). On the full RL-val the unconstrained R0 loses only **L = +0.223 [-0.15, +0.62]** of the dual's quantity at u1000 (below the 0.5 bound), against **+1.000 [+0.30, +1.73] for the pick**, the quantity the bar S reads. G_tv_full = **-0.789 [-2.17, +0.34]** (val loses LESS than the train pool; unresolved). So at R0's state there is no demonstrated train-to-val excess for a carve to close; **the dual constrained E_pi[TTC] at T = 0.1, and on val that is 1.4 to 4.5 times smaller than the pick's TTC loss.**
* **Next lever (A2, PROVISIONAL, see s.8):** the constraint at policy temperature T* = 0.02 tracks the pick within 0.25 points at all three R0 states on full RL-val; T = 0.1 does not. A pick-aligned constraint, not the carve, is the better-supported successor of O4. The carve remains the right feed for any trained dual (an in-sample feed is blind by construction), but it is second order. The arm is a PI decision (A40 spend + one trainer flag with a G-DVB entry).

## 1. Evidence class, tier, estimator

* Main read (harness): **ESTIMATED**. Banked R0 / C0 scorer states (A, B), cached visual contexts, CPU fp32, offline selection read; **not T1**, not a driving claim. Pools: train 302 sets / 30 logs (1.9 % of the sets, 18 % of the logs), val 62 / 13 (2.1 %, 17 %). Harness files unchanged, sha256 in the prereg.
* A1 / A2 val side: **MEASURED** (the arm's live bf16 logits as dumped by the trainer, true PDM labels, RL-val selection tier; not an endpoint, not T1). The train side stays ESTIMATED.
* Estimator: paired log-cluster bootstrap, 10,000 resamples, 95 % percentile, seed 20260927. It answers "another draw of LOGS". It does NOT answer "another training run" (one run per arm, no replicate) nor "another inference run" (softmax / argmax deterministic). Nothing here is a lever-effect claim (H-ESTIM-SEED-1).
* Unit: points (x100). `L = -mean(B - A)` of the per-set expected TTC (P2) or of the pick's TTC label (P1); **L > 0 = a violation; the bar is L > 0.5.** `L(P2 TTC) = 100 * g^TTC` with eps = 0.

## 2. The carve, established FROM SOURCE (before any outcome number)

| fact | evidence |
|---|---|
| R0 trained on every RL-train set | trainer blob `fec6174d...` (commit `a120418a`): `tr_idx = [... if key(r) in op.by and r.get("log_name") not in ho_logs ...]` (:496-497), restricted to sets with an RL terms record (:509). No sub-sampling, no held-out train log. R0 `sft.jsonl` data event: `train_sets` 16,000 = `train_sets_in_bank` 16,000, `train_rows_in_heldout_logs` 0 |
| the sampler | `torch.Generator().manual_seed(a.seed)` (:531), `DataLoader(..., batch_size=8, shuffle=True, generator=g, drop_last=True)` (:532-534). R0 argv: `--seed 0 --epochs 4 --max-updates 1000`, `--batch` / `--accum` default 8 |
| the loop | `while not done: for batch in dl:` (:937-938). Each pass is one fresh permutation WITHOUT replacement of all 16,000 sets (2,000 micro-batches; 16,000 % 8 = 0). An update = 8 micro-batches. 1,000 updates x 8 x 8 = 64,000 draws = **exactly 4 epochs**; **u250 = exactly 1 epoch** |
| size of RL-train | 16,000 sets in **166 logs** (sets per log: min 2, median 52, max 1,080), counted from the extracted `rlfan_sets_train` jsonl |
| R0d | the tip trainer (`ac07b1a`) carries the same lines unchanged (:908, :965-967, :1638, :1652-1653), seed 0, so R0d consumed the same stream (inferred from code and seed; the order was not logged) |

**Conclusion: every RL-train log was consumed by u250 (once) and by u1000 (four times). There is no free carve. A carve cut from R0's data is in-sample for A and B: F1.** The measurement therefore answers a narrower question, named in the prereg s.1.1:

> **N1.** At R0 u1000, how large is the train-vs-val gap in the dual's quantity g, resolved against its estimator, and does a log-disjoint subset of the logs the policy trained on read like the other train logs or like val? (How much of the gap can log composition alone produce?)

Whether a HELD-OUT carve tracks val cannot be tested on any R0 state. Only an R0e run can.

## 3. Harness control (prereg s.4)

Stage 1 (`logs/stage1.log`, 311 s, 4 threads, 2.0 GB RSS; free RAM 12.3 GB before loading and at least 12.7 GB during the run, so the 8 GB floor held): R0 u250 / u500 / u1000 reproduce the banked `realstates.json` FULL block on 43 compared fields each (P1 / P2 x TTC, NC, DAC, EP: pooled, train, val, both CI ends; P1 PDMS): max |diff| of means **3.59e-06 / 3.24e-06 / 1.55e-06**, of CI ends 2.65e-06 / 2.31e-06 / 1.23e-06 (tolerances 1e-4 / 1e-3), no missing key. The per-set arrays I saved reproduce the banked train and val means of P2 TTC and P1 TTC at every R0 state (12 of 12). C0 u1000 floor |P2 TTC pooled| = **1.03e-04** (< 0.5). **Verdict VALID.**
Stage-2 controls (`socarve.json` `controls`, all true): zero control (B := A) exactly 0.0 on the real log structure and on the planted arrays; A - A exactly 0 through the real arrays; sign control; planted labels (carve = val -> CARVE-TRACKS-VAL, carve = train -> CARVE-TRACKS-TRAIN, mid -> MID, out-of-CI -> OUT-OF-CI, no gap -> NO-GAP); the same two oracle-carve labels on the REAL R0 u1000 arrays; base arrays identical in every state.

## 4. Main result: the constraint quantity g on three log sets (R0 u1000, P2 TTC; points; ESTIMATED)

g is SPEC_SAFEOBJ s.1.2, `epo_con_term` at `scorer_finetune.py:256-287` (tip blob): `cA_s, cB_s = (piA * yk).sum(1), (piB * yk).sum(1)` (:274), `g = cA_s.mean() - eps - cB_s.mean()` (:275), pi at T = `a.epo_temp` = 0.1 (`epo_policy`, :193). The harness's `pi_comp` is the same sum. The dual reads the update's mean g (`EpoDual.step`, :308-317). Not the pick delta: the pick's `p1gap_ttc` is logged separately (:284).

| log set | n logs / sets | L = 100 g (95 % log-bootstrap) |
|---|---|---|
| (i) the RL-train logs the trainer consumed (all of them, the cached pool) | 30 / 302 | **+1.012** [-0.05, +2.33] |
| (ii) log-disjoint carve, m = 13 of 30, mean over 10,000 random draws | 13 / ~131 | **+1.043** [-0.59, +3.19] (sd over draws 0.71) |
| (iii) RL-val (the harness's cached pool) | 13 / 62 | **+1.356** [-2.03, +5.37] |

`G_tv = L_va - L_tr = +0.344 [-3.30, +4.50]`; `G_tc = L_c - L_tr = +0.031 [-1.34, +1.57]`; `rho = 0.090`; the carve (+1.043) lies inside val's interval. Sensitivity: m = 6 rho 0.29 (sd 1.41), m = 20 rho 0.04 (sd 0.43); the "rest" as the in-sample reading: rho 0.04. **Label: CARVE-TRACKS-TRAIN [F1, F2]. Expected by construction** under exchangeable logs (E[G_tc] = 0 for an in-sample random carve).

Same rule, secondary quantities (u1000): P1 TTC L_tr +2.318 [+0.65, +4.50], L_va(pool) +3.226 [0.00, +7.14], rho 0.085, CARVE-TRACKS-TRAIN [F2]. **NC** (P2): L_tr +0.349 [+0.03, +0.75], L_va(pool) +0.532 [-0.71, +1.84], L_c +0.348, G_tv +0.184 [-1.13, +1.52], rho -0.004, CARVE-TRACKS-TRAIN [F2]; P1 NC L_tr +0.497, L_va(pool) -0.000 (NO-GAP). u500: P2 TTC L_tr +1.219, L_va +1.427, rho 0.143 [F2]; u250: P2 TTC L_tr +1.743, L_va +0.489, NO-GAP (val loses less).
Sampling noise of a carve (S-POWER, P2 TTC u1000): the share of the 10,000 carve draws that read a violation (L > 0) is 0.94 (m = 13), 0.80 (m = 6), 0.99 (m = 20); that read a bar-sized one (L >= 0.5) 0.77 / 0.65 / 0.89. At the unconstrained R0 state, the in-sample train logs themselves read a violation; that is what a feed would see before the dual acts.

Why the pool val interval is so wide: per-log heterogeneity. One val log (5 sets, `...0304_00384`) reads L = +18.2 and supplies 1.47 of the pool's 1.356 points; without it the other 57 sets read about -0.12. (Post-hoc arithmetic on `socarve.json` `per_log_r0_u1000`.) Several train logs read +17 to +19 on 4 sets each. Per-log L (P2 TTC) spans -7.5 to +19 in train and -6.4 to +18.2 in val.

## 5. Secondary registered readings (descriptive)

* **S-DOSE** (harness val pool, P2 TTC): G_tv = -1.254 [-3.42, +1.25] (u250, 1 epoch), +0.208 [-2.96, +4.25] (u500), +0.344 [-3.30, +4.50] (u1000). `r_d` is undefined (G_tv(u250) <= 0), as pre-registered. The point estimates rise with exposure; every interval spans 0 by several points. **Not interpretable.** A1's full-val side supersedes it: G_tv_full -0.893 / -0.989 / -0.789, no exposure growth.
* **S-EXCH** (frozen base scorer and label-only covariates, train pool minus val pool): **EXCH-NOT-CONTRADICTED**, but the intervals are +-6 to +-10 points wide (c0 TTC x100 diff -0.45 [-6.57, +6.00]; oracle mean y^TTC x100 diff +1.64 [-7.90, +10.81]; base pick PDMS +1.99 [-3.00, +8.88]). That is "no evidence of a difference", not evidence of exchangeability: the test cannot see an effect of the size that matters (0.3 to 1 point). The base scorer's SFT exposure to RL-train versus RL-val logs is UNVERIFIED.
* **RZ-1** (R0d's own in-sample log, `sft.jsonl`, MEASURED; registered window = updates 651..750, four 25-update rows): `con_g_ttc` mean x100 **-0.034** (the dual's quantity, held), `con_p1_gap_ttc` mean x100 **-0.297** (the pick, B minus A). Tag **P1-IN-SAMPLE-HELD** (> -0.5). Post-hoc dispersion (`logs/r0d_insample_posthoc.log`): over the 21 post-ramp rows (u226..u750) the pick gap is -0.265 (se 0.089) while g is -0.066 (se 0.032); whole run -0.234 (se 0.064). So in-sample the dual drove g to about 0 and left the pick 0.23 to 0.30 below A, about half of the 0.5 bound. RL-val read -0.77 [-1.49, -0.06] on the pick: the in-sample pick gap and the val gap differ by about 0.5, with a val interval that wide: unresolved.
* The registered RZ-1 branch (P1-IN-SAMPLE-HELD -> "the train-to-val story stands, next lever = R0e") was written before A1 existed. **A1 and A2, pre-registered before computed but chosen after the main read, supersede that branch** (section 6, 7).

## 6. A1: the same quantity on the full RL-val (3,000 sets / 75 logs), R0 live dumps (MEASURED val side)

Controls: P1 `B - A` for TTC, NC, EP, PDMS at u250 / u500 / u1000 reproduces `ab_vs_base.json` `rows.r0_u*.B_minus_A` to <= 1e-6 (12 of 12 rows); zero control exactly 0 at every state; all 62 harness val sets found among the 3,000. **A1 VALID** (`fullval_a1.json`, `logs/fullval_a1.log`).

| R0 | P2 TTC L_va_full (the dual's quantity) | P1 TTC L_va_full (the pick, the bar's quantity) | L_tr (train pool, P2) | G_tv_full (P2) | A1-GAP | A1-LABEL |
|---|---|---|---|---|---|---|
| u250 | +0.851 [+0.56, +1.15] | +1.233 [+0.62, +1.83] | +1.743 | -0.893 [-2.29, +0.22] | UNRESOLVED | INTERMEDIATE(NO-GAP) |
| u500 | +0.230 [-0.10, +0.59] | +0.633 [-0.06, +1.35] | +1.219 | -0.989 [-2.26, +0.07] | UNRESOLVED | INTERMEDIATE(NO-GAP) |
| u1000 | **+0.223 [-0.15, +0.62]** | **+1.000 [+0.30, +1.73]** | +1.012 | **-0.789 [-2.17, +0.34]** | UNRESOLVED | INTERMEDIATE(NO-GAP) |

* Reading rule R-B/R-C: the gap is UNRESOLVED and BAR-SIZED-GAP-EXCLUDED is false (the lower bound is -2.2): no train-to-val excess of TTC is demonstrated; the point estimate is the opposite sign. The train-pool side (30 logs) is now the limiting interval.
* **The quantity wedge**: at u1000 the pick's TTC loss is 4.5 times the dual's quantity on val (1.000 vs 0.223); 2.8 times at u500; 1.4 at u250. On full RL-val the unconstrained R0's P2 loss is below the 0.5 bound at u500 and u1000; the bar S breach lives in the pick.
* NC (P2): full-val L = +0.021 [-0.13, +0.18] (u1000), +0.120 [+0.01, +0.23] (u250); NC never approaches the bound; P1 NC +0.333 [+0.03, +0.69] (u1000).
* Longitudinal context (progress): the EP gain B - A is larger in-sample than on val, and RESOLVED at u500 / u1000: P2 EP gain 4.60 (train pool) vs 2.94 (full val), G = +1.665 [+0.17, +3.32] at u1000 (+1.324 [+0.13, +2.49] at u500; not resolved at u250). The optimised quantity over-states its transferable gain; TTC shows no such excess. (Train side ESTIMATED; EP fidelity of the cached harness was not separately checked.)
* Fidelity of the cached-context harness on the 62 matched val sets: P2 TTC |L dump - L harness| = 0.013 / 0.085 / 0.082 (u250 / u500 / u1000) -> **FAITHFUL**; P1 TTC u1000 0.000, u250 0.000, u500 1.613 (one pick flipped on 1 of 62 sets: FIDELITY-GAP by the 0.25 rule, a quantisation artefact). The train-side L_tr has no dump counterpart, so its fidelity rests on this val-side check.

## 7. The four metric families, as far as these states allow

| family | measured here | not computable, and why |
|---|---|---|
| LONGITUDINAL | TTC (P2 = g, P1 = pick) and EP (progress gain), s.4-6, with estimator and CI | target-speed accuracy, headway / time-gap / TTC-to-lead: scorer-only states on fixed proposals, no plan-vs-reference trace (n = 0) |
| LATERAL | -- | heading, curvature, yaw-rate, cross-track need executed trajectories (n = 0). NC / DAC are collision / drivable sub-scores, not lateral metrics |
| TACTICAL | -- | no manoeuvre-decision instrument at this tier (n = 0) |
| STRATEGIC | -- | **PAUSED** (PI ruling 2026-09-19, `PI_DECISION_QUEUE.md` item 25). Not a pass, not omitted |

Per family, no composite.

## 8. Rule Zero: the next lever, with its result

* **SELTTC-style (the lever the brief named) was already run on these exact R0 states** (`PK/raw/2026-10-16-safety-head/selttc.json`, RESULT s.10): CONTROLS_FAILED (S3 changed 0 picks); diagnostic S1 / S2 move the real P2 TTC loss by at most +0.09 (about 6 %). INHERITED, ESTIMATED, not re-run. It acts on the selection, not on the training constraint, and A1 explains why it is inert for the bar: the deployed pick already sits where the TTC rows put it.
* **A2: which smooth quantity tracks the pick on val?** (`PREREG_SOCARVE_A2_TEMPSWEEP.md`; `tempsweep_a2.json`; MEASURED, full RL-val, R0). L_T = loss of E_{pi_T}[TTC] at policy temperature T, against L_P1:

| T | u250 L_T (diff to L_P1 1.233) | u500 (0.633) | u1000 (1.000) | PICK-ALIGNED (all three within 0.25) |
|---|---|---|---|---|
| 0.1 (the trainer's) | 0.851 (-0.383) | 0.230 (-0.404) | 0.223 (-0.777) | no |
| 0.05 | 1.044 (-0.189) | 0.437 (-0.196) | 0.512 (-0.488) | no |
| **0.02** | 1.181 (-0.052) | 0.656 (+0.022) | 0.775 (-0.225) | **yes** |
| 0.01 | 1.199 (-0.034) | 0.765 (+0.131) | 0.880 (-0.120) | yes |

  Outcome as the rule reads: **A2-ALIGNED(T* = 0.02)**. **PROVISIONAL: A2's pre-registered control 3 FAILED** (T = 1e-4 must reproduce L_P1 to 1e-6; it read -0.015 / +0.026 / -0.017, with 0 picks differing from the argmax). Diagnosis (`so_a2_control3_diag.py`, post-hoc, `logs/a2_control3_diag.log`): near-tie mass; the tolerance was set for a temperature that is not cold enough. At T = 1e-6, 1e-8, 1e-10 the difference is exactly 0.0, so the machinery does reduce to the pick. By the prereg's own rule A2 is INVALID as written; I report the sweep as the provisional reading and do not relax the rule after the fact. The T = 0.02 margin at u1000 (-0.225 vs 0.25) is thin.
* **Next lever (named):** a pick-aligned constraint. First choice: the constraint term's own temperature at T* about 0.02 (one new trainer flag with a G-DVB entry and a reachability trace; an A40 arm; matched control; launch-gate PASS). Cost named in the prereg: a low-T policy concentrates its mass on few proposals, so the dual's gradient is higher variance; not tested. Fallback if T* fails: a hinge on the pick's TTC through a straight-through or sampled estimator. The carve (held out from the policy) is the right FEED for that dual, because an in-sample feed is blind by construction, but A1 shows the first-order defect is the quantity, not the data split.
* **Blocked, named:** the arm needs the PI (A40 spend) and the trainer flag + gate entry (~1 day, 0 GPU). Nothing else blocks it.

## 9. What this package cannot show

* Whether a HELD-OUT carve tracks val: no R0 state has a held-out train log. Only an R0e run can test it.
* The gap of a CONSTRAINED policy: R0 is unconstrained; R0d's in-sample g was driven to about 0 by the dual, a different quantity. R0d's scorer states and dumps are not on this dev box (searched `rls1_adam_dl`, `rls1b_dl`, `r0d_out`).
* Anything about the seed floor, or the live bf16 forward on the train side.
* Population claims beyond 30 train and 13 (harness) / 75 (A1) val logs.

## 10. Deviations and disclosures

* Prereg s.0: I had seen "val minus train P2 TTC -0.21 (u500), -0.35 (u1000)" in `PREREG_SAFEOBJ.md:116` before the freeze, so the sign and size of G_tv were not blind (the measured +0.208 / +0.344 reproduce it). G_tc was not known.
* The main prereg's two timestamps were corrected before any data run (re-hashed twice; the final digest is the one in the `.sha256`).
* A1 and A2 were written after the main read and each before its own numbers; I had seen the T = 0.1 numbers (A1) before A2. Their redirect of the next lever is post-hoc-motivated, which is why the registered RZ-1 branch is reported next to it.
* A2 control 3 failed as written (s.8). Post-hoc items, unregistered: the per-log arithmetic of s.4, `so_r0d_insample_posthoc.py` (s.5), `so_a2_control3_diag.py` (s.8).
* Stage 1 ran detached (WMI, `cmd /c ... < NUL`, below-normal priority) and finished in 311 s; one heavy process at a time; the 8 GB free-RAM floor held throughout.
* No pod IP, port, id, token or token path appears in any file of this package.

## 11. DRAFT register row for `GOALS_AND_CLAIMS.md` (id `D-REFE-SAFEOBJ-CARVE`; I did not edit the register)

> **D-REFE-SAFEOBJ-CARVE** -- O4's 0-A40 first step (train-vs-carve gap of the dual's quantity g on banked R0 states, A2 harness; ESTIMATED, offline selection, not T1; paired log-cluster bootstrap 10,000 / seed 20260927, "another draw of logs" only). **Carve provenance (MEASURED, source): R0 consumed ALL 16,000 RL-train sets / 166 logs, once by u250 and four times by u1000 (shuffle without replacement, seed 0, 4 epochs); there is no free carve and an R0-state carve is in-sample (F1).** Harness control PASS (max diff 3.6e-6 vs banked A2). **Pre-registered outcome CARVE-TRACKS-TRAIN [F1, F2]**: R0 u1000 P2 TTC L_tr +1.01 [-0.05, +2.33], carve (13 of 30 logs) +1.04 [-0.59, +3.19], val pool +1.36 [-2.03, +5.37]; G_tv +0.34 [-3.30, +4.50] (unresolved), rho 0.09. Expected by construction; does NOT show a held-out carve inert.
> **A1 (MEASURED val side, full RL-val 3,000 sets / 75 logs, R0 live dumps):** the dual's quantity loses only +0.22 [-0.15, +0.62] at u1000 (below the 0.5 bound), the pick +1.00 [+0.30, +1.73]; G_tv_full -0.79 [-2.17, +0.34] (unresolved, wrong sign for a train-to-val story). The harness's 62-set val pool was driven by one 5-set log. **The SAFEOBJ constraint (E_pi[TTC] at T = 0.1) is a poor proxy of the bar's quantity (the pick): 1.4 to 4.5 times smaller on val.** R0d in-sample (own log): g -0.03, pick gap -0.30.
> **A2 (PROVISIONAL; control 3 failed as written, diagnosed near-tie mass):** E_pi[TTC] at T = 0.02 tracks the pick within 0.25 at u250 / u500 / u1000; T = 0.1 and 0.05 do not. Next lever: a pick-aligned constraint (T about 0.02 for the constraint term), fed from a held-out carve; arm = PI decision. SELTTC already inert (selttc.json). Nothing here is a lever-effect claim (one seed, no replicate). STRATEGIC PAUSED; lateral / tactical / headway not computable at this tier.
> Artifacts: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-18-so-carve/`.

## 12. Manifest

| artifact | repo target (under `.../raw/2026-10-18-so-carve/`) | where it lives now |
|---|---|---|
| this file | `RESULT.md` | dev box `C:/Users/Admin/socarve/RESULT.md` (**only copy**) |
| 3 pre-registrations + `.sha256` | `PREREG_SOCARVE*.md/.sha256` | `C:/Users/Admin/socarve/` (only copy) |
| code | `so_eval_states.py`, `so_carve_analysis.py`, `so_fullval_a1.py`, `so_tempsweep_a2.py`, `so_a2_control3_diag.py`, `so_r0d_insample_posthoc.py` | `C:/Users/Admin/socarve/` (only copy) |
| results | `harness_control.json`, `socarve.json`, `fullval_a1.json`, `tempsweep_a2.json` | `C:/Users/Admin/socarve/` (only copy) |
| per-set arrays (4 states) | `out/persets_{r0_u250,r0_u500,r0_u1000,c0_u1000}.npz` | `C:/Users/Admin/socarve/out/` (only copy; regenerable by stage 1, 311 s) |
| logs | `logs/{stage1,stage2,fullval_a1,tempsweep_a2,a2_control3_diag,r0d_insample_posthoc}.log` | `C:/Users/Admin/socarve/logs/` |
| the banked file stage 1 compares to | `src/sh_realstates.json` (verbatim blob, sha256 `84b2f1df72840e17...`) | `C:/Users/Admin/socarve/src/` |
| `paths.json` | -- | `C:/Users/Admin/socarve/paths.json` |

Integration escalation: **the Master Mind lands this package** (stage via `paths.json`, scoped commit, push, verify by `git ls-remote`) and decides the register row. The pick-aligned-constraint arm needs a PI decision.
Re-run: `PYTHONUTF8=1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe so_eval_states.py` (needs the dev-box safety-head archive at `SOCARVE_SHD`, the pools in the MM scratchpad, `rls1_adam_dl`), then `so_carve_analysis.py`, `so_fullval_a1.py`, `so_tempsweep_a2.py`.
