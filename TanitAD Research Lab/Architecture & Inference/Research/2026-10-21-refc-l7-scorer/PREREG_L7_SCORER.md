# PREREG L7-SCORER: does a learned per-candidate scorer pick better plans from refc's own fan?

* **Author.** Architecture & Inference agent (scorer9), 2026-10-21, clock read 23:13 +0200 (Berlin).
* **Status. REGISTERED 2026-10-21 23:33 +0200 (Berlin) by the Master Mind**, sealed by the sha256 in `PREREG_L7_SCORER.sha256` beside this
  file, BEFORE any capture roll, any target build on a train split, or any scorer training. Text unchanged from the agent's draft
  except this status line and the title. No L7 number exists; the only rolls that exist (fanceil s0 / s1 on eval139) were made
  for `R8-FANCEIL-1` and no scorer has read them (the target smoke reproduced fanceil's oracle / own-pick means; no scorer was fit on eval139).
* **Design.** `DESIGN.md` beside this file (arms §3.2, targets §3.3, admissibility §4, costs §5, eval §6).
* **Hypothesis `H-L7-1`.** refcv8's fan contains far better plans than it emits (oracle 0.084 m vs pick 0.299 m, S2,
  `R8-FANCEIL-1`), and the best GT-free rule recovers 0.031 m. A scorer **trained** on the fan's own candidates against
  benchmark-grounded targets recovers a materially larger share, on PhysicalAI and on NAVSIM, without a family regression.

---

## 1. Arms

| id | benchmark | generator (FROZEN) | selector |
|---|---|---|---|
| **P-OWN** | PhysicalAI eval139 | refcv8-r101-s0 **EMA-48,852** (`ckpt_ema_48852.pt`, md5 per `fanceil/runs/*/fan_roll_*.json`) | shipped E9 pick `sel_idx` |
| P-BASE, P-MBR32 | same | same | `sel_idx_base`; the frozen MBR32 cell with the model's own `e9_keep` (R8-PICKER definitions, unchanged) |
| **P-L7E** | same | same | L7-E (geometry + v0 only), rule R-IMIT |
| **P-L7F** | same | same | L7-F, R-IMIT (primary) and R-SAFE (secondary) |
| **P-L7S** | same | same | L7-S, R-IMIT (primary) and R-SAFE (secondary) |
| P-L7*r | same | same | each L7 arm retrained with **training seed 1** (mandatory, §5) |
| **N-OWN** | NAVSIM | **A1** = the `H-NT-DECIDE-1` snapshot chosen by `PREREG_NT_DECIDE.md` §5 (whatever its verdict) | R8_LEGAL own pick |
| **N-L7S** | NAVSIM | A1 | L7-NS (L7-S with the six PDM heads), `navsim_v1` aggregate |
| N-L7Sr | NAVSIM | A1 | training seed 1 |

Everything else is fixed now: d_model 128, 2 decoder layers, 4 heads, no cross-candidate self-attention, AdamW lr 3e-4,
wd 0.01, batch 32 windows, ≤ 30 epochs with early stopping on the TRAIN bank's held-out slice (patience 3 on its
validation loss), imitation temperature T = 0.5 m, Huber β 0.1. **No hyper-parameter is chosen on eval139, the navtrain
selection sample or navtest.** A training failure (NaN, a control red) is fixed and re-run from scratch; the fix and the
re-run are reported.

**Masks (every arm, identically):** `e9_rank` semantics — reach ∧ set-speed ceiling ∧ emit, with the shipped empty-row
fallback (`refc_v3.py:1025-1053`). The PI's cap ruling (2026-10-18) is unchanged.

## 2. Data

* **Training banks** (new capture rolls, `DESIGN.md` §3.4): PhysicalAI **train4572** only, 1 window in 4, two inference
  seeds; NAVSIM **navtrain** logs minus the 38 selection-scan logs minus every navtest log (the `H-NT-DECIDE-1` K3 lists),
  2 seeds. Validation slice = 10 % of the TRAIN bank's episodes / logs, disjoint, drawn with seed 20261021.
* **PILOT (permitted, never deciding):** before the full bank, a capture on the **train139** windows (the RETIME
  trainfit window set, `refcv8_pf/trainfit/step30000ema_train139_pf`; TRAIN clips, disjoint from eval139), 2 seeds, may
  train L7-E / L7-F with the §1 settings and be read on eval139 once, labelled PILOT. It changes nothing in this prereg;
  the registered read is the full bank's.
* **PhysicalAI eval** = eval139, 4,754 windows / 139 episodes (S2) and the battery's S6 windows; fans = a capture roll of
  EMA-48,852 at inference seeds 0 and 1 with the scene sources dumped (the fanceil s0 / s1 fans are reused for P-OWN,
  P-BASE, P-MBR32, P-L7E, P-L7F, and must match the new capture bit-for-bit on `anchor_traj` at seed 1 — see K3).
* **NAVSIM dev read** = the registered navtrain selection sample (2,089 tokens / 30 logs, sample sha `9dd68698…`),
  inference seeds 0 and 1.

## 3. Primary metric and bars (committed now; numbers, not adjectives)

### 3.1 PhysicalAI — per arm, at BOTH inference seeds (fans s0 and s1), paired episode-cluster bootstrap over 139 episodes (`taniteval/ci.py`, B 2000, seed 0), FULL-SET means

| bar | definition |
|---|---|
| **P1** | S2 ADE(arm) − S2 ADE(P-OWN): CI **excludes 0 below** at both seeds |
| **P2** | S2 ADE(arm) − S2 ADE(P-MBR32): CI **excludes 0 below** at both seeds (the best GT-free picker; it was −0.0308 [−0.0411, −0.0216] vs own, post-hoc) |
| **P3** | no component of LONGITUDINAL, LATERAL or TACTICAL is separated **worse** than P-OWN at both seeds, and S6 ADE is not separated worse |
| **P4** | the arm's training-seed replicate (P-L7*r) passes P1 and P2 |
| **P5** | the point effect vs P-OWN exceeds both floors: the arm's inference floor \|S2(s0) − S2(s1)\| and its training floor \|S2(L7) − S2(L7r)\| |
| **PV** (visual claim) | S2 ADE(P-L7F or P-L7S) − S2 ADE(P-L7E): CI excludes 0 below at both seeds |
| **STRONG** (ambition line, reported, not the pass bar) | S2 ADE ≤ **0.200 m** at both seeds (closes ≥ 46 % of the 0.215 m pick-to-oracle gap) |

**PASS(arm) = P1 ∧ P2 ∧ P3 ∧ P4 ∧ P5.** The primary rule is R-IMIT; R-SAFE is reported with the same bars and never
replaces a failing R-IMIT after the read.

### 3.2 NAVSIM — navtrain selection sample, inference seeds 0 and 1, `navsim_bars8.pair_record` → paired log-cluster bootstrap (B 2000, seed 0, cluster log_name, sensitivity nuplan_drive)

| bar | definition |
|---|---|
| **N1** | PDMS(N-L7S) − PDMS(N-OWN), mean of the two seeds: separated **above 0** at log_name AND nuplan_drive, **and ≥ +1.0 PDMS** (≈ 2× the larger of REFe P2's training-seed difference 0.33 and TransFuser's published SD 0.56 — the `H-NT-DECIDE-1` band) |
| **N2** | NC and DAC sub-scores not separated worse |
| **N3** | N-L7Sr passes N1 |
| **N4** | no LONGITUDINAL / LATERAL / TACTICAL component separated worse at both seeds |

**PASS(NAVSIM) = N1 ∧ N2 ∧ N3 ∧ N4.**

## 4. Committed outcomes (both directions, fixed before any data)

| outcome | what follows |
|---|---|
| **P-L7S or P-L7F PASS, PV holds** | L7 is a validated selection lever. Next: the SHIP-R8 lane (with RETIME) on the passing arm, then a PI adoption decision; L7 becomes refcv9's selector default and the scorer is the reward for the TOAD test-time search (`refcv7_toad.py`) as the next arm. |
| **PASS, PV fails** (L7-E alone carries it) | Reported as a **non-visual** selection lever (the gain is in geometry + v0 + the model's own scores). It may still ship (it is admissible), but no "the scorer reads the scene" claim is made. Next lever: the trunk-gradient scorer (below). |
| **P1 passes, P2 fails** | L7 beats the shipped pick but not the GT-free MBR32: report both; MBR32 stays the zero-training candidate. Next lever: **on-policy, trunk-gradient scorer** (REFe `--scorer-sees-trunk`; DrivoR lets the scorer's gradient reach the encoder; `D-REFE-ZEROTAIL-3` measured the trunk gradient's gross reach at 58 % of the DAC zeros), needs trainer integration and GPU. |
| **P1 fails** | The learned scorer at this capacity cannot rank the fan from detached tokens. Report every head's held-out AUC and the within-window Spearman (which head failed). Next lever, in order of measured effect: (1) trunk-gradient scorer as above; (2) score the **refined** object (`sel_refined` / `sel_score_emitted`, the live `D-REFC-DDAUDIT-3` defect, 0 params) as an input feature; (3) REFe's set self-attention arm. |
| **NAVSIM PASS** | Request ONE navtest read of N-L7S vs N-OWN from the PI (§6). Committed navtest outcome: ADOPT if PDMS gain ≥ +1.0 separated at log_name, else REPORT. |
| **NAVSIM FAIL** | Report per sub-score which head failed (AUC on the dev sample); next lever is the same trunk-gradient scorer, on A1. |
| **A control reads wrong (§7)** | No verdict is written. The instrument is fixed and the read re-run from scratch. |

## 5. Replicate rule (`H-ESTIM-SEED-1`, and the third variance)

* Every L7 arm is trained twice (training seeds 0 and 1) on the SAME bank. A lever claim needs both to pass.
* Inference variance is answered by the two fan seeds (s0, s1); training variance by L7 vs L7r; the CI answers only
  "another draw of EPISODES / LOGS". Every reported interval names which of the three it answers.

## 6. Navtest rule

**At most one navtest read**, of N-L7S vs N-OWN, only after N1–N4 pass AND the PI's go is recorded. No L7
hyper-parameter, rule or arm is changed after it. No NAVSIM leaderboard submission is part of this prereg.

## 7. Controls (each must read its known value, or the read STOPS)

| # | control | known value |
|---|---|---|
| K1 | CONST scorer | `select()` returns the **lowest emittable index** on every window (literal) |
| K2 | ORACLE scorer (imit = −ADE) | selects the T0 oracle on **every** window; its S2 ADE equals `FANCEIL_R8.json`'s oracle-in-fan for that seed at 1e-6 m |
| K3 | capture identity | the new eval139 capture at seed 1 reproduces `fanceil/runs/s1/dump/fan_extras.npz` `anchor_traj` exactly (or, if Thor's sampler is non-deterministic, within the R8-PICKER K1 envelope, reported) |
| K4 | identity splice | splicing P-OWN through the L7 lane reproduces the banked SHIP-R8 tree exactly (R8-PICKER K8 form) |
| K5 | deliberate regression | `imit` sign flipped ⇒ S2 ADE **worse than RANDOM**, separated |
| K6 | leak guard | a run that passes any `FORBIDDEN_INPUT_KEYS` member (e.g. `v_max_ms`) to the scorer REFUSES; mutation test RED |
| K7 | analytic head | the `comf` head's held-out AUC ≥ 0.95 (its target is an analytic function of the inputs; failing it means the training is broken) |
| K8 | exclusion | train-bank episodes ∩ eval139 = 0; NAVSIM train-bank logs ∩ (38 selection logs ∪ navtest) = 0; a mutation adding one ⇒ REFUSE |
| K9 | estimator | self-pair P-OWN − P-OWN = 0.0 with CI [0, 0] |

## 8. Forbidden

* Fitting, tuning or early-stopping on eval139, the navtrain selection sample, or navtest.
* Changing T, the rules, the bars, the masks or the arm list after any eval read.
* Reading navtest outside §6.
* Feeding the scorer nav, the set-speed value, `z_tac`, `cond`, or any label (`DESIGN.md` §4).
* Calling a single-seed or single-replicate result a lever.
