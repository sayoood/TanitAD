# PREREG_L2 — refcv9 L2: can a TRAIN-fitted calibration of refcv8's longitudinal tactical head remove the cruise false-brake without retraining?

*Research Lab agent for the Master Mind, 2026-10-25 (Berlin) 10:56 by the dev-box clock. Sealed BEFORE any calibrated number exists on any eval window: the sha256 of
this file is in `PREREG_L2.sha256`; `chain_l2.sh` and `l2_roll.py --require-seal` refuse a scored roll if it does not match. Sources:
`Project Steering/PLAN_REFCV9.md` §1 D2, §2 L2, §3; `…/2026-10-15-refcv8-milestone-battery/raw/refcv9_headroom/` (L2_HEADROOM, BRAKE_LOCUS, NAV_BRAKE);
register rows H-R9-L2-HEADROOM-1, H-R9-BRAKE-LOCUS-1, H-R9-ALLOC-ORACLE-1, H-R9-NAV-BRAKE-1, R8-EVAL-NAV-FEED-1, R8-CONS-1.*

## 0. What was found BEFORE this prereg (disclosed — these looks are not blind)

1. **The premise numbers are stale for the SHIP-R8 checkpoint.** PLAN_REFCV9's "0.266 vs 0.091 / stage 0.253 / missed 0.358" are EMA-30k **clip-fed** battery numbers
   (`raw/refcv9_headroom/L2_HEADROOM.json`). Re-read on the banked **EMA-48,852 per-frame-nav** battery (`C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf`,
   banked stage `…/2026-10-20-refcv8-perframe-final/pfpkg/raw/retime_deploy/DEPLOY8_POS_step48852ema_pf_s{0,1}.npz`), same definitions, S6 ceiling-compliant
   (3,251 windows / 138 episodes), GT-cruise 1,856, GT-decel 639 — **MEASURED by `smoke/baseline_explore.py`** (own code; the all-window cells reproduce register row
   R8-CONS-1's literals 0.150 / 0.428 / 0.0965 / 0.548):

   | arm | seed | false-decel (GT-cruise) | missed-decel (GT-decel) |
   |---|---|---|---|
   | refcv8 os | 0 / 1 | 0.1395 / 0.1395 | 0.4695 / 0.4789 |
   | refcv8 + shipped stage | 0 / 1 | 0.1369 / 0.1374 | 0.4695 / 0.4710 |
   | refcv7-50,400 | 0 / 1 | 0.0911 / 0.0938 | 0.5587 / 0.5681 |

   So with the correct per-frame nav the false-brake gap is 0.046 (not 0.175), the stage barely moves it, and **the missed-decel bar 0.358 is 0.11 BELOW the shipped
   baseline's 0.4695**: the committed literal bar (§6 P3) demands a model that dominates on both axes, which no single-axis shift can do. Both readings are committed below.
2. **TRAIN-side only (design inputs):** on the A6 train139 EMA-30k per-frame dump, 29 % of `BRAKE_TO`-labelled windows are trajectory-cruise (223/777): the head's label
   horizon (R8-1: NOW+2…8 s) is longer than the 6 s plan horizon the bars are defined on. A first dev fit with a FOLLOW bias drove it to its bound (−8) — i.e. deleted the
   FOLLOW hypothesis — so the bias is NOT part of the calibrator (§3). The dev fit (`dev30k/calibrator_30k_DEV.json`) is a pipeline check, never a result.
3. **The os-only roll is a NEW sampler draw, not a replay of the banked battery.** Comparing L1's live os-only dump to the banked battery plan: 30/2,052 windows identical,
   28 % differ by ≥ 1 m — the same as banked s0 vs banked s1 (30 identical). "Identity reproduces the banked numbers exactly" is therefore defined at the RATE level
   within the inference-seed floor (K2, K5), and at the BIT level only against the SAME code path (K3).
4. No eval-side calibrated number, head-level confusion or frontier had been computed when this file was sealed.

## 1. Decision locus (MEASURED by source read, launch tree `C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58` = fb47a58)

* Planner-feeding lon head = the refcv6 tactical behaviour decoder: `out = self.tac_decoder_v6(cond, …)` → `out["lon_logits"]` [B, 8] (`stack/tanitad/refs/refc_v3.py:1981`;
  raw logits cached for the LOSS at `:1991`).
* **The intervention seam**: `out, r8_force = self._r8_apply_force(out, b, dev)` (`refc_v3.py:2001`, def `:2094`) runs BEFORE every planner feed. Forcing a class
  (`set_r8_force(lon6=k)`, `:2081`) replaces `out["lon_logits"]` with a one-hot ±20 (`:2099`, `:2104-2107`) — this is exactly how the oracle-class arm (force_lon F1/F2,
  `code/force_lon_reroll_r8.py`) forced the class. A calibration replaces `out["lon_logits"]` at the same seam.
* After the seam, `out["lon_logits"]` reaches EVERY consumer: (a) `r8c.hypothesis_posteriors` → `logp_lon6` (6 hypotheses, masked v7 ids 2 and 6 removed before the softmax;
  `refcv8_conditioning.py:335-341`, call `refc_v3.py:2002`) → candidate ALLOCATION (`joint = lp_l ⊗ lp_o`, top-4 hypotheses floor 2 each + the rest ∝ p;
  `refc.py:2511-2513`, `refcv8_conditioning.py:395-433`; allocated candidates are emittable), the candidate feature `phi` (`refc.py:2572`) and the factorised selection
  score `beta_lon·log p(lon) + gamma_prog·sat_prog + gamma_head·sat_head` (`refcv8_conditioning.py:471-499`, `refc.py:3834`; learned `beta_lon ≈ −0.007` at EMA-30k, so the
  posterior hardly enters the RANKING, the constraint outputs `cons_lon8` are NOT touched by a logit calibration); (b) `v6tac.planner_feeds` →
  `lon_logprob = log_softmax(out["lon_logits"])` over all 8 → `lon_prior` / `tac_lon_prior` (`refcv6_tactical.py:750-751`, `refc_v3.py:2020-2026`; used by
  `refc.py:3534-3557`).
* **Therefore a calibration CAN act without retraining** (it is an eval-time transform of the logits at an existing seam), through ALLOCATION, the lon prior and the
  (tiny) ranking term. It CANNOT act through the constraint heads or the decoder's speed profile given the class. The battery dumps already carry the raw head output:
  `dump_s{0,1}/refcv6_extras.npz` → `nav_true.tacv6_lon_logits` [4,754, 8] (the planner-feeding head; `lon_logits_tac` is the legacy head, a sensitivity only).

## 2. Question, populations, definitions (all imported from the battery, none re-derived)

* **Question.** Does a global per-group logit bias + temperature on the lon head, fitted on TRAIN only, bring the SHIP-R8 system (refcv8 EMA-48,852 per-frame nav + the
  shipped DEPLOY8 stage) to false-decel ≤ 0.10 without a missed-decel penalty, at both inference seeds?
* **Surface/tier.** S6 (0–6 s, instants 1…6 s), T1 self-action OPEN LOOP on logged eval139 frames (not driving performance); ceiling-compliant windows (human ≤ fed ceiling:
  `gt_max_speed_6s ≤ v_max_ms`, reproduced exactly: 3,251 windows / 138 ep).
* **Classes** (`s6_lon_decomp.plan_lon_class`): decel / cruise / accel from the last chord (5→6 s) vs the measured v0, ±1.5 m/s, dt 1 s, applied identically to the GT and to every
  plan. **False-decel** = P(plan class = decel | GT class = cruise); **missed-decel** = P(plan class ≠ decel | GT class = decel). Populations: GT-cruise 1,856, GT-decel 639
  (model-free, identical for every arm).
* **Estimator.** `pair_read_r8.paired_cell` → `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2000, seed 0, over EPISODES, windows finite on both sides. It answers *another draw
  of episodes* ONLY — not another training run (single training seed, H-ESTIM-SEED-1), not another inference run (seeds 0 and 1 are shown separately, never pooled).
  *Separated* = the 95 % interval excludes 0. "Not separated worse" = lower bound ≤ 0 (battery convention).
* **Families (never pooled)** for every arm pair: ADE (ade, fde); LONGITUDINAL (speed MAE, along-track MAE, |along| at 6 s, target-speed ±0.5 m/s, accel MAE, **false-decel,
  missed-decel**, signed along-track at 6 s and mean signed speed error on GT-cruise windows); LATERAL (heading MAE, curvature MAE masked, yaw-rate MAE masked, cross-track MAE,
  |cross| at 6 s); TACTICAL (trajectory-derived lateral / longitudinal decision correct). **Not computed, per family:** LONGITUDINAL distance-keeping (headway / TTC to the lead
  agent) — the banked refcv8 dumps carry no lead-agent track, n = 0, a NAMED WORK ITEM; STRATEGIC — refcv8 runs `--no-strategic`, n = 0.

## 3. The lever (the ONLY thing fitted)

`lc = (l + b) / T` on the 8 `tacv6` lon logits; `b = [0, 0, (masked 0), bD, bD, bD, (masked 0), bA]` on (FOLLOW, CRUISE, BRAKE_TO, CREEP, HOLD, ACCELERATE); θ = (bD, bA, log T).
CRUISE and FOLLOW are the reference group. Fitted by `code/l2_fit.py` by group NLL against the **trajectory-derived class** of the human's 6 s future (the class the bars are defined
on), on **TRAIN rows only**: the A6 train139 EMA-48,852 per-frame-nav roll banked by the L1 agent (`C:/Users/Admin/refcv8_pf/refcv9_l1/bank/train_dump_s0`, inference seed 0,
4,855 windows, rule `t % 5 == 0`; 0 overlap with eval139), S6-valid ∧ ceiling-compliant. Ridge λ ∈ {0, 1e-3, 1e-2, 1e-1, 1} by 5-fold EPISODE cross-validation (fold seed 0; replicate
seed 1, reported). Disclosure: train139 is IN-SAMPLE for refcv8, so the head looks sharper on TRAIN than on eval; the fit can under-correct and nothing can be tuned to compensate.
Eval windows are scored, never used to choose anything. The calibrator is fixed by `bank/calibrator_48852.json` (sha256 recorded in every roll's meta) BEFORE the first calibrated roll.

## 4. Arms (all scored with and without the shipped stage; the DEPLOYED system is "+stage" and the bars read on it)

Stage = the registered per-frame DEPLOY8 table (`refcv8_retime_deploy_gains_pf.json`, sha256 3319b003…, source table 2aacc9ff…) applied by `tanitad.retime_stage.retime_plan` as the
FULL 8-slot call then slots [1,3,4,5,6,7] — MEASURED bit-equal to the banked S6_pos on both seeds (K1).

| arm | definition | role |
|---|---|---|
| **BASE-bat** | the banked battery `os` (+ banked stage), seeds 0 / 1 | refcv8 + stage as registered (rate-level comparator) |
| **BASE-os** | L1's un-hooked os-only roll (`refcv9_l1/bank/eval_dump_s{0,1}`), same code path as the CAL arm | **PRIMARY comparator** (only the hook differs); fallback = the chain's own NATIVE roll if the L1 bank is absent |
| **CAL** | the §3 calibrator at the §1 seam, seeds 0 and 1 | **the lever** |
| **CONST** | LON6 CRUISE forced on EVERY window (the TRAIN-majority labelled class), seed 0 | constant-class control: the limit of the bias family (b_D → −∞) |
| **ORACLE** | the LON6 class of the GT label `lon_v7` forced where labelled and a LON6 hypothesis (else native), seed 0 | ceiling (privileged; never deployable) — the force_lon F1 semantics |
| **IDENT / NATIVE** | hook installed with b = 0, T = 1 / hook not installed, first 300 windows, seed 0 | hook neutrality (K3) |
| refcv7-50,400 | the panel's embedded `b_refcv7_50400_s{seed}` | the other bar comparator |

Estimated cost (ESTIMATED from L1's live os-only roll, 1,250 windows / 1,086 s = 0.87 s/window): 4,754 windows ≈ 69 min per full roll; 4 full rolls (CAL s0, CONST s0, CAL s1,
ORACLE s0) + 2 × 300-window smokes ≈ 4.8 h ≤ 6 h ⇒ both inference seeds for CAL, controls at seed 0 only (a full 6-roll design would be ≈ 7 h).

## 5. 0-GPU readings (descriptive, never bars; ESTIMATED where projected)

* **Z1** head-level confusion on eval S6 windows: head group (argmax of the 6-way softmax mapped to decel/cruise/accel) vs the trajectory class, raw and calibrated; the number of
  GT-cruise windows whose head call flips decel → non-decel and GT-decel windows flipping decel → non-decel / non-decel → decel.
* **Z2** head-level frontier: the head's decel-vs-cruise log-odds as a discriminator on eval (AUC) and the (head-FD, head-MD) pairs under a bias grid with T = 1. **ORACLE-TUNED on
  eval — an upper bound on what any bias can do, labelled as such, never a result and never used to pick b.**
* **Z3** association projection: P(plan decel | GT class, head group) from BASE-os/BASE-bat+stage, applied to the calibrated head groups ⇒ a projected FD/MD. ESTIMATED; association,
  not intervention; reported beside the GPU measurement to show how far the head→plan coupling explains it.

## 6. Bars (mechanical; S6 compliant, "+stage", BOTH inference seeds; CAL vs BASE-os unless stated)

* **P1** FD(CAL+stage) ≤ 0.10 (point).
* **P2** FD(CAL+stage) − FD(BASE-os+stage) separated < 0 AND |Δ| > the inference-seed floor F = max over the BASE pairs of |FD(s0) − FD(s1)|.
* **P3 (the literal PI bar)** MD(CAL+stage) ≤ 0.358 (point).
* **P3L (like-for-like at the shipped baseline)** MD(CAL+stage) − MD(BASE-os+stage) not separated worse.
* **P4** FD(CAL+stage) − FD(refcv7) not separated worse.
* **G (B9-0 guard)** no METRICS row (ADE / LONGITUDINAL / LATERAL / TACTICAL) separated worse for CAL+stage vs BASE-os+stage at both seeds.
* **PASS** = P1 ∧ P2 ∧ P3 ∧ P4 ∧ G.  **PASS-AT-SHIPPED-BASELINE** = P1 ∧ P2 ∧ P3L ∧ P4 ∧ G ∧ ¬P3.  Everything else is **FAIL** and the failed clauses are named. A seed that is not rolled
  is **PENDING**, never PASS.

## 7. Controls (a known value; any failure ⇒ **INVALID**, no outcome read)

| id | control | must read |
|---|---|---|
| K1 | the stage on the banked 8-slot plans | bit-equal to banked `S6_pos`, both seeds (max abs 0) |
| K2 | populations + register literals | 3,251 / 138 ep; GT-cruise 1,856; GT-decel 639; ALL-window refcv8 os FD / MD and refcv7 FD / MD at s0 equal R8-CONS-1's 0.150 / 0.428 / 0.0965 / 0.548 to their printed precision |
| K3 | IDENT-300 vs NATIVE-300 plans | max abs ≤ 1e-5 m, same sel_idx; hook calls = windows |
| K4 | the roll's hook trace (raw → calibrated logits) vs `l2_fit.apply8` on the raw logits | max abs ≤ 1e-4 |
| K5 | BASE-os vs BASE-bat rates (compliant FD, MD) | each within max(0.01, 2·\|s0−s1\|) — else BASE := BASE-os is kept as primary and the shift is reported as an os-only/battery batch effect |
| K6 | CONST potency (positive control: the seam must be live) | FD(CONST) − FD(BASE-os) separated < 0 at seed 0; if not, the outcome is **CHANNEL-INERT** (a finding, not INVALID) |
| K7 | fit controls (`l2_fit.py selftest`): recovery of a known (b, T); identity recovery; a mutated grouping must NOT recover; a shuffled-label fit must not beat the constant predictor out of fold | all as coded |
| K8 | constant-only: the CONST arm's FD/MD are reported beside; a CAL that does not beat the CONST-limited trade-off line is not "a better ranker" | reported |

## 8. Committed outcomes (written before any CAL number)

* **O1 PASS** → L2 = an inference-time calibration constant (b, T) in the refcv9 recipe, foldable into the head's final bias/scale; next: a second training seed (B9-2) and the same
  roll on the refcv9 candidate.
* **O2 PASS-AT-SHIPPED-BASELINE** → report it as that, NOT as PASS: the literal bar is missed; the calibration improves FD at the shipped baseline's missed-decel level only.
* **O3 FAIL, CONST reaches FD ≤ 0.10** → the bias family can reach the false-decel bar by trading missed-decel; the failed clause is the trade-off (P3/P3L/G). The head's call is a
  weak ranker on this axis: next lever = label timing (§9).
* **O4 FAIL, CONST does NOT reach FD ≤ 0.10** → the cruise brake does not live in the lon-head posterior at this seam (it is in the decoder's speed profile / constraint heads / nav
  path): no logit calibration can reach the bar; next lever = §9 plus the constraint-head/nav channels.
* **O5 CHANNEL-INERT (K6 fails)** → the seam does not move the plan at this checkpoint; L2 via the lon posterior is dead; next lever = §9.
* **O6 INVALID** (any other control fails) → the failing control is the next work item.
* A seed or arm not rolled when the report is written is **PENDING**; its clause is not read.

## 9. The alternative lever (pre-registered ONLY as the next arm if §8 is not O1/O2): label timing

Retrain/fine-tune the tactical decoder on lon labels whose horizon matches the plan horizon the bars score (the v9 R8-1 labels are NOW+2…8 s; 29 % of TRAIN BRAKE_TO windows are
trajectory-cruise over 0–6 s): (1) relabel `lon` per frame from the model-free GT future with a 0–6 s horizon (CPU, Data FlyWheel, the 4,572-clip label release); (2) a decoder-only
fine-tune on the frozen refcv8-48,852 trunk (the late-decoder-drift evidence R8-WISEFT-MOD2 says the decoder is where the NavSim loss sits); (3) the SAME §6 bars with a
replicate (B9-2). It needs: the relabel (CPU hours), GPU (Thor after NT A1, or HF Pro; not the 8 GB dev box for a trunk-sized forward), and a PI decision on that compute. It is NOT
started here.

## 10. Execution

`code/chain_l2.sh`: identity checks (this seal, ckpt md5 b4364029…, windows md5 a3a78dba…) → wait for L1's TRAIN bank (`ZZL1ROLL_train_s0ZZ` in `refcv9_l1/chain.log`, the final npz,
4,855 windows) → copy the needed train arrays → `l2_fit.py fit` → `l2_zero_gpu.py` (§5 on the real calibrator) → free-commit gate (≥ 6 GB) → `gpu_lock acquire` (asserted on its `--out` JSON,
never the exit code; release in a trap; only PIDs it started are killed; OMP_NUM_THREADS=4) → NATIVE-300, IDENT-300 smokes → CAL s0 → CONST s0 → CAL s1 → ORACLE s0, each asserted on its
ARTIFACT (final npz with the expected window count; opaque `ZZL2…ZZ` markers) → release → `l2_score.py` (mechanical §6-§8) → `result/`. Nothing is written to E:, F: or G:. No git.

## 11. Deviations

A code fix that changes no definition above (crash, path, dtype) is allowed after the seal and goes into `DEVIATIONS.md` with the reason and the diff summary. Any change to an arm,
a bar, a split or the calibrator family makes the affected reading exploratory.
