# PREREG_V8F_DAG_ORC: O-RECOVER's launch-recovery perturbation on DAG's states, aimed at the crawl rows

Promoted 2026-10-27 by the Master Mind from the draft banked unchanged at `raw/2026-10-27-dag-orc-draft/PREREG_V8F_DAG_ORC.md` (sha256 `02d7a8152c0214c4cb294505c1258adcc87fe224ab425a3da7e2156ccefa90a5`). Written after `RESULT_V8F_DAGGER2` and **before any DAG-ORC gate, run or row, and before N0** (§8): no DAG-ORC checkpoint, rollout, eval or floor value exists, and no `--nav logged` rollout of DAG has run. Claim row `H-V8F-DAG-ORC-1`.
* **MM decision on the control (the draft's open question):** kept as drafted, CTRL = DAG2-0 (one lever, clean attribution). The base gap it carries (DAG2-0 trails DAG on ADE6) is what R-KEEP / R-NOBASE test, and DAG-ORC-SWAP is the registered route if A is attributed but not adopted.
* **N0 runs first** (launch precondition 3; 0 GPU, dev-box CPU). The box-2 arms A, REP (and B) need the PI's go for box-2 time and queue after FLAT-DAG and DAG-OACC.

**Why it runs.** `RESULT_V8F_DAGGER2` read **HARMS**. Its committed next, verbatim: *"DAG2 is not adopted; DAG stays the base. Next: as
PARTIAL, on DAG states"*, i.e. *"the launch-recovery target of `PREREG_V8F_HACC_MC` §7 on the base's states, under its own PREREG"*.
§7's lever is O-RECOVER: an ego-state-perturbation fine-tune. On M2b it read HARMS (`H-V8F-ORECOVER-1`). This PREREG runs it on DAG's
states, with the instruments DAGGER2's read showed were missing: a replicate floor per row, and an adoption test against the base.

**Disclosure of what was seen before writing** (all MEASURED, read from the git mirror at `d64750bd` unless stated):
* `RESULT_V8F_DAGGER2.md` and `read/DAG2_READ.json`'s published rows; `RESULT_V8F_DAG_REREAD.md`; `RESULT_V8F_DAG_HACC.md`;
  `RESULT_V8F_ORECOVER.md`; the register rows `H-V8F-ORECOVER-1`, `-DAGGER-1`, `-DAG-REREAD-1`, `-DAG-HACC-1`, `-DAGGER2-1`.
* DAG2-0's `argv.json`, `summary.json`, `launch_gate_token.json` and `SHA256SUMS.txt` (`E:/Projects/TanitAD-artifacts/v8f-dagger2/`).
  **No `eval_*` file there was opened.**
* The argv and `summary.json` of ORC, ORC-0 and DAG-0, and the box-2 chain log `chain_v8f_dag2.out` (run times only).
* **Not computed:** the floor RF0 (§3). Its inputs exist; its values have not been read.

## 0. What the data say, and why this arm can succeed where DAgger iteration 2 did not

| row (T1-kin, paired, B = 2,000) | DAG − P10 s0 / s1 | DAG2 − DAG2-0 s0 / s1 (DAgger it. 2) | ORC − ORC-0 s0 / s1 (perturbation, on M2b) | DAG-HACC − P10 s0 / s1 (post-map) |
|---|---|---|---|---|
| curvature MAE (1/m) | +0.0612 W / +0.1134 W | +0.0218 W / −0.0301 B | **−0.0537 B / −0.0220 B** | +0.0174 W / +0.0471 W |
| yaw-rate MAE (deg/s) | −0.04 / +8.66 W | +1.36 / +1.20 (n.s.) | **−6.59 B / −3.16 B** | (s1 row removed) |
| L2 curvature MAE, GT ≥ 5.2 m/s | +0.0288 W / +0.0604 W | −0.0102 / −0.0077 (n.s.) | not read paired | +0.0112 W / +0.0280 W |
| M1 crawl share | +0.0470 W / +0.0614 W | −0.0052 (n.s.) / −0.0252 B | not read. Beside only: the head reader's crawl fraction (not M1), unpaired points: ORC 0.1263 vs ORC-0 0.1332, 0.1355 vs 0.1507 | +0.0242 W / +0.0390 W |
| ADE6 (m) | −1.1656 B / −0.9868 B | −0.2247 B / −0.1630 B | +0.1100 W / +0.0355 | −1.039 / −1.092 |
| R-T0, T0 ADE6 (m), bar +0.0604 | | −0.0066 / +0.0309 | **+0.1103 / +0.0849 (above, both)** | |

Sources: `RESULT_V8F_DAG_REREAD.md` §1, `RESULT_V8F_DAGGER2.md` §2, `RESULT_V8F_ORECOVER.md` §1 and §3.2, `RESULT_V8F_DAG_HACC.md` lines 31–33.
B / W = separated-better / -worse. The ORC column is a different base (M2b) and is evidence about the lever, not about DAG.

**Why it can succeed.**
* **It is the only training lever measured to move two crawl rows at both seeds.** On M2b at T1-kin, curvature and yaw-rate MAE were
  separated-better at both seeds. DAgger iteration 2 moved curvature in opposite directions at the two seeds and left M1 and L2 in place.
* **It puts its samples where the crawl is.** The crawl lives in launch windows with v0 median 5.7 / 4.7 m/s (`raw/2026-10-26-lcdiag/DIAGNOSIS.md`,
  INHERITED). The perturbation draws only where the logged v0 < 8 m/s.
  * On the same data order it perturbed **24,720 / 24,830** samples (ORC `summary.json`), every one a speed deficit of 0–50 % with the
    human's future from the true pose as the target. That target is the launch recovery itself.
  * DAgger iteration 2 swapped **14,965 / 14,719** samples (DAG2 `summary.json`), drawn from ticks 1–2 of rollouts at all speeds.
    HYPOTHESIS, not measured: only a small share of them are launch-crawl states. DAG's per-window crawl share is 0.070 at s0
    (`RESULT_V8F_DAG_REREAD.md` §1).
* **The yaw-rate guard that fired on DAG2 is now read against a measured floor** (§3). On M2b this lever improved the same row by
  −6.59 / −3.16 deg/s.

**The known risk, and the designed response.** On M2b the lever cost T0 ADE6 +0.1103 / +0.0849 m, above the +F/2 bar at both seeds.
That is the most likely way this arm fails.
* The response is registered now: a second dose, **B at p = 0.1**. It is trained in the same queue and is read under the same table
  only if A reads HARMS (§7).
* That is the programme's own HARMS mapping: `PREREG_V8F_DAGGER` §7 sends p = 0.3 HARMS to p = 0.1.
* ESTIMATED, assuming the T0 cost scales linearly with p (a HYPOTHESIS): about +0.037 / +0.028 m at p = 0.1, under the bar.

## 1. The arms (box 2; each through the trainer's own launch gate; both seeds)

| arm | what | role |
|---|---|---|
| **A** (DAG-ORC) | DAG2-0's argv, plus `--ego-perturb-p 0.3 --ego-perturb-vscale 0.5 1.0 --ego-perturb-vmax 8.0 --ego-perturb-adrop 1.5 --ego-perturb-yawrate-std 2.0 --ego-perturb-seed <s>`, with its own `--out` | **decisive** |
| **CTRL** = DAG2-0_s | **reused, not retrained.** Banked run `v8f-dagger2/M2b_s{s}_dag20/`, ckpt sha256 `603b9151f5119886…` / `f4234ac11854076b…` | the control: the same 3,000-step fine-tune from DAG_s without the lever |
| **REP** | DAG2-0's argv with `--seed 10` (init DAG_s0) / `--seed 11` (init DAG_s1), with its own `--out` | the replicate: the control's recipe run again (§3) |
| **B** (DAG-ORC-01) | A's argv with `--ego-perturb-p 0.1` | the dose fallback. Read only if A is HARMS (§7); otherwise reported beside |

* **The lever** is O-RECOVER's perturbation at its registered law, exactly as `H-V8F-ORECOVER-1` and ORC Amendment 3 name it. One lever with
  two components (longitudinal and yaw-rate). A separated effect is attributed to the perturbation as a whole, never to one half.
* **Init:** `--init-ckpt` = DAG_s, sha256 `d186fd0bcfa7ee9f…` (s0) / `1d12ec2eb825990d…` (s1), as recorded in DAG2-0's launch-gate token.
  Fine-tune mechanics are DAG2-0's: `--steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p 0 --dagger-p 0`
  plus M2b's argv.

**Is DAG2-0 a valid control? Yes, by its banked argv (MEASURED).**
* `M2b_s{0,1}_dag20/argv.json` holds DAG2-0's argv; A is built from it.
* The pair differs only in `--out`, `--ego-perturb-p` and A's law flags, which is the ORC / ORC-0 pair rule of ORC Amendment 3. The trainer
  refuses the law flags at p = 0, so the control cannot name them.
* Same `--seed`, so the same data order, per-step RNG and scheduled-sampling draws. The perturbation draws from its own seeded generator.
* MEASURED on the banked pairs: the eligibility denominator does not depend on the lever. ORC and ORC-0 read 82,457 / 82,166 eligible, the
  same as DAG2-0.
* **The one difference is the build.** DAG2-0 was trained at `1d4f4fbb`; A will be trained at the launch commit. This is handled in two ways:
  * control K-BUILD (§6);
  * REP is trained at the launch commit, so the floor RF absorbs any build term.

**Why REP uses a different `--seed`, and the same init.** Three reasons, from `H-ESTIM-SEED-1`:
* **It answers the right question.** The bootstrap answers *another draw of EPISODES*. The floor must answer *another training run*. The
  init stays DAG_s, so REP isolates the training run; the two seeds carry the init.
* **A same-seed rerun can read exactly 0.** The trainer re-seeds every step (`v8f_mini_train.py`, `torch.manual_seed(a.seed * 1_000_003 + step)`).
  `PREREG_V8F_DAGGER` §5 notes that its same-seed floor *"reads 0 if the two runs are bit-identical"*. A zero floor says nothing about the
  path divergence that any lever causes. The lever perturbs the gradient at every step, so A leaves the control's path as surely as a new
  data order would. A different seed measures that size, and it never collapses to zero.
* **The same-seed form is already in the bar.** RF0 (§3) is a same-seed, cross-build pair (DAG-0 vs ORC-0, MEASURED identical argv except
  `--out` and the default-off `--dagger-p 0`). The bar therefore covers both ways a rerun differs:
  * same data order, different build;
  * same build, different data order.

## 2. Evaluation and tiers

| where | what | tier |
|---|---|---|
| **dev-box CPU (decisive)** | `v8f_t1_eval run --replan-steps 10 --limit causal --dump-ticks --dump-heads` on eval-141, plain (no post-map). DAG's decisive settings: dead-reckoned nav, logged scene, periodic, executor `plan`. For A, REP and B at both seeds; CTRL's, DAG's and P10's banked runs are reused | **T1-kin**: self-action OPEN loop. ⛔ Never "closed loop" |
| dev-box CPU (K4 only) | one `--limit logged` run per new arm and seed, and one fresh-age T0 eval (`v8f_mini_eval.py run --ages fresh`) as K4's comparator (ORC Amendments 2 and 4) | control |
| box 2 | the chain's standard T0 eval (eval-141, ages cycle) of every checkpoint | **T0**: R-T0 and the T0 four families. ⛔ Never "driving performance" |

**Estimator.** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0, 141 episodes / 2,399 windows, per seed, paired, on the
same windows and matched cells.

**Three questions ride on the interval; each is named:**
* *Another draw of EPISODES?* The bootstrap answers it.
* *Another TRAINING RUN?* The two seeds plus the replicate floors (§3) answer it.
* *Another INFERENCE RUN?* None arises: the planner is deterministic.

## 3. The replicate floors and the bar (every Read-A row, defined now)

**Read-A rows** (`PREREG_V8F_DAG_REREAD` §2's reader, unchanged; each row in its own direction and units):
* PRIMARY: ADE6.
* LONGITUDINAL: speed MAE, along-track MAE, accel MAE, DK-VIOL, DK-HW, DK-TG.
* LATERAL: heading MAE, yaw-rate MAE, curvature MAE, cross-track MAE, L1 lateral-accel MAE, L2.
* TACTICAL: TAC-LAT, TAC-LON.
* STRATEGIC: S-POS@30, S-POS@50, S-POS@80, each against CTRV within a run.
* crawl: M1.

For a row r and seed s, d_r,s(X − Y) is Read A's paired point of X − Y on that row. For S-POS, X − Y is the difference of the two runs'
CTRV contrasts on the same windows.

| symbol | definition | when it is computed |
|---|---|---|
| **RF0_r** | max over s of \|d_r,s(DAG-0 − ORC-0)\|. Banked M2b-init pair: same argv except `--out` and the default-off `--dagger-p 0`; same seed; builds `b43c7a9a` / `853f2fc3`. Causal dumps: `v8f-dagger-t1/dump_DAG0_s*_m10_causal_heads`, `v8f-orecover-t1/dump_ORC0_s*_m10_causal_heads` | **P-0, before any DAG-ORC rollout is read**, 0 GPU. Its output is frozen by sha256 into the reader's config. Its inputs must match their `PINS.txt` sha256 |
| **RF_r** | max over s of \|d_r,s(REP − CTRL)\| | after REP's rollouts, before any A or B row is read |
| **F_r** | ADE6: **F = 0.1207 m** (F_M2b, `MODEL_REGISTRY.md` §17, `H-V8F-SEED-1`). Every other row: **0**, because no floor is registered and the row-equivalent is RF0_r itself | now |
| **bar_r** | **2 × max(F_r, RF0_r, RF_r)**, one value per row, applied at both seeds, inclusive. For every non-ADE row this is **2 × max(RF0_r, RF_r)**, in the row's own units | as its inputs |

**Two choices, and why.**
* **The max is taken over seeds.** Each pair is one draw of \|Δ\|.
  * If Δ ~ N(0, σ²), one draw falls below σ/2 with probability 0.38. A bar of 2 × that draw would then sit under 1σ.
  * The bar takes the max of four draws (two pairs × two seeds). If the four draws share one σ, the chance that all four fall below σ/2
    is 0.38⁴ ≈ 0.02.
* **The F1 floors are not reused.** `PREREG_V8F_ORECOVER`'s F1 floors (0.0124 / 0.0073) belong to the T head's macro-F1. TAC-LAT and TAC-LON
  in Read A are plan-derived agreements (P10 has no T head), so those floors do not apply to them.
* If both floors read exactly 0 on a row, the runs are bit-identical there, and the row clears on separation alone.

**What a row "clears", and what "fires":**
* **clears (R-MOVE, per row):** X − CTRL is separated-better AND the improvement is ≥ bar_r.
* **fires (R-NEW, per row):** all three of:
  1. X − P10 is separated-worse;
  2. DAG − P10 was not separated-worse there, i.e. the row is in the domain D_s below;
  3. X − CTRL is separated-worse with the worsening ≥ bar_r.

## 4. The rows

**The crawl set C** is unchanged from DAGGER2: the rows separated-worse for DAG − P10 (`RESULT_V8F_DAG_REREAD.md` §1).
* s0: curvature MAE, L2, M1.
* s1: curvature MAE, L2, M1, yaw-rate MAE.

**The domain D_s** is every Read-A row not separated-worse for DAG − P10 at s, minus the exempt rows. It has 14 rows at s0 and 13 at s1.

**Exempt** from R-NEW and R-NOBASE, as in DAGGER2: accel MAE and DK-VIOL. They belong to DAG-OACC. They are reported.

| row | contrast, per seed | role |
|---|---|---|
| **R-CRAWL** | every row of C_s is not separated-worse for X − P10 | decisive |
| **R-MOVE** | every row of C_s **clears** for X − CTRL (separated-better AND ≥ bar_r) | decisive |
| **R-TLAT / R-TLON** | the T head's lat / lon macro-F1 at ticks ≥ 1 (the T1-HEADS definition, matched cells), X − CTRL: not separated-worse | guard |
| **R-ADE** | T1-kin plain ADE6, X − CTRL: not separated-worse | guard |
| **R-T0** | T0 own-code ADE6, X − CTRL: point ≤ +F/2 = **0.0604 m** | guard |
| **R-NEW** | no row of D_s **fires** | guard |
| **R-KEEP** | T1-kin ADE6, X − DAG: point ≤ +F/2 = **0.0604 m** | adoption |
| **R-NOBASE** | no row r of D_s has X − P10 separated-worse AND X − DAG separated-worse with the worsening ≥ bar_r | adoption |

**Why R-NEW reads X − CTRL, and adoption reads X − DAG.**
* The guards decide whether the **lever** harms. Every one of them is X − CTRL, the one-lever contrast.
* The adoption rows decide whether X may **replace DAG**. DAGGER2's clause used X − P10 alone, so it compared DAG and DAG2, two training runs,
  on a row with no floor.
* DAGGER2's table would also have adopted DAG2 if it had read FIXED, although DAG2 − DAG ADE6 was +0.0426 [+0.0011, +0.0896] at s0.
* R-KEEP and R-NOBASE close that gap. They use the KEEP form of `PREREG_V8F_DAG_HACC` and `PREREG_V8F_DAG_OACC` (+F/2, here on the point,
  as R-T0 is).
* ⚠ **A disclosed handicap, ESTIMATED by additivity from MEASURED points.** DAG2-0 − DAG ADE6 is +0.268 / +0.132 m:
  * DAG2-0 − P10 is −0.898 / −0.855 m (`RESULT_V8F_DAGGER2.md` §3.1);
  * DAG − P10 is −1.166 / −0.987 m.
* So A starts behind DAG on R-KEEP. A must recover what 3,000 steps of GT-only fine-tuning cost before it can be adopted. If it moves the
  crawl but cannot be adopted, the committed next keeps the swap on (§7).

**Beside (never decisive):**
* X − DAG and B − CTRL on every Read-A row.
* **M2** (X's L2 share on the crawl pairs) and L2 on the GT-defined launch subset (v0 < 6 m/s and GT v(6 s) ≥ v0 + 1 m/s; `PREREG_V8F_HACC_MC` §5).
* **M-LAUNCH**, the lever's mechanism row. It is X − CTRL of the mean (v_own − v_GT) at ticks 1–2 in the GT-launch windows, from the tick
  dumps. The lever predicts it moves toward 0. If R-MOVE fails while M-LAUNCH moves, the speed deficit was repaired and the crawl rows did
  not follow. That outcome points away from the state channel (§8).
* The anti-echo block per run (`longitudinal_claim_admissible`, hold-v0, copy detector), as `RESULT_V8F_ORECOVER` §3.1 reports it.
* The perturbation counts `ego_perturb_n` / `ego_perturb_eligible` / `ego_perturb_nav_kept` per run.

## 5. The four metric families (all reported, per seed, per family, never pooled)

| family | T1-kin (primary): Read A rows, X − CTRL, X − P10, X − DAG | T0 (box 2's eval): X − CTRL, beside | not computable, with the reason |
|---|---|---|---|
| **LONGITUDINAL** | speed MAE (target speed), along-track MAE, accel MAE; distance keeping DK-VIOL / DK-HW / DK-TG (`read_dk`, LEAD windows only) | speed, accel, along-track MAE; headway, time gap, TTC, LEAD-VIOL (`orc_families.py`, unchanged CLI) | none. The anti-echo block is reported beside every longitudinal number |
| **LATERAL** | heading, yaw-rate, curvature, cross-track, L1 lateral-accel, L2 | heading, curvature, yaw-rate, cross-track | none |
| **TACTICAL** | TAC-LAT / TAC-LON (plan-derived decision agreement); R-TLAT / R-TLON (T-head macro-F1, ticks ≥ 1) | lat / lon head macro-F1 and accuracy; goal-head macro-F1, 22 tokens (`v8f_paired_families.py`) | T1-kin goal-head F1 and head accuracy: the paired tool reads eval `run` dumps, not tick dumps (as ORC §3.6). Anchor selection: UNAVAILABLE by design (one path per window) |
| **STRATEGIC** (⛔ **not paused for v8F**, PI ruling 2026-10-17) | S-POS@30 / 50 / 80 against CTRV, per run | S head checkpoint error and strategic rows of the eval JSON | paired S-POS across arms is not computed (it is read against CTRV within a run, as in DAGGER2) |

⚠ **DAGGER2 left the T0 families NOT RUN. Here they are a build item of the reader**: `orc_families.py` pointed at A / CTRL. A missing
family is a work item, never a silent omission.

## 6. Controls (each must pass, or the read is REFUSED)

* **K-GATE.** Each new run (A, REP and B, both seeds) has a launch-gate PASS token binding the commit, the argv sha256 and the data sha256.
  * The gate is `v8f_mini_train.py --selfcheck-only`: G-DVB on every `--ego-perturb-*` flag and on `--init-ckpt` / `--warmup`, and G-LIVE
    with the ego probe on a perturbed batch.
  * As in ORC and DAGGER2, the HMAC is not verified.
* **K-PAIR.**
  * argv(A_s) equals argv(DAG2-0_s) except `--out`, `--ego-perturb-p` = 0.3 and the five A-only law flags, which carry the values in §1.
  * argv(REP_s) equals argv(DAG2-0_s) except `--out` and `--seed` (10 / 11).
  * argv(B_s) equals argv(A_s) except `--out` and `--ego-perturb-p` = 0.1.
* **K-INIT.** Every new run's `built_levers.init_ckpt.sha256` equals DAG_s's (`d186fd0b…` / `1d12ec2e…`).
* **K-REACH.** `ego_perturb_n / ego_perturb_eligible` lies in 0.3 ± 0.02 for A and 0.1 ± 0.02 for B, and is exactly 0 for REP.
  * ESTIMATED binomial 3σ at about 82,000 eligible: 0.005.
* **K-BUILD.** The launch commit's `v8f_mini_train.py`, `v8f_mini_data.py` and `v8f_mini.py` must satisfy one of two conditions:
  * byte-identical to `1d4f4fbb`'s; or
  * every changed hunk sits behind a default-off flag whose bit-identity test passes on the launch commit.
  * The diff is listed in the result.
* **K-P0.** RF0's inputs match their `PINS.txt` sha256 and pass Read A's per-run controls. RF0's frozen sha256 predates every A / REP / B
  rollout's mtime.
* **K-ZERO.** Two structural zeros, each with a deliberate-regression arm:
  * REP − REP reads exactly 0.0 with CI [0, 0] on every row. Feeding REP's dump as CTRL must make RF exactly 0.
  * A reader with the floor term deleted must fail its self-test.
* **K-RC.** The reader, run on DAG2 / DAG2-0 with the floors set to 0 and adoption disabled, reproduces `DAG2_READ.json`'s decisive rows
  with 0 differences.
* **Read A's and DAGGER2's reader controls**, unchanged: K4 against the fresh-age T0 eval, K7 track rebuild, PREREG-TEXT and the rest
  (`raw/2026-10-27-dagger2-t1read/`).

## 7. Outcomes, committed now (arm X = A; X = B only on A's HARMS)

| outcome | condition | committed next |
|---|---|---|
| **REFUSED** | any control fails | fix and re-read; no row is reported |
| **HARMS** | controls pass; a guard (R-TLAT, R-TLON, R-ADE, R-T0, R-NEW) fails at ≥ 1 seed | **A:** not adopted; read **B** under this same table (train B first if the PI dropped it from the queue). **B:** not adopted; the perturbation cannot be dosed under the guards; **NEXT-CRAWL on DAG states** (§8) |
| **FIXED / ADOPTED** | guards held; R-MOVE and R-CRAWL at both seeds; R-KEEP and R-NOBASE at both seeds | adopt X as the base. Re-run Read A on X: DOMINANT there is the dominance claim. Then X with DAG-OACC's lever, if DAG-OACC is adopted |
| **FIXED / NOT ADOPTED** | guards held; R-MOVE and R-CRAWL at both seeds; R-KEEP or R-NOBASE fails at ≥ 1 seed | the lever is attributed, but X cannot replace DAG. Next: **DAG-ORC-SWAP**, under its own PREREG: DAG2's argv plus this perturbation at X's dose, control DAG2 (banked). The swap kept own-state skill (DAG2 − DAG2-0 ADE6 −0.22 / −0.16 m) |
| **PARTIAL / ADOPTED** | guards held; R-MOVE at both seeds; R-CRAWL fails at ≥ 1 seed; R-KEEP and R-NOBASE at both seeds | adopt X. The crawl rows that still fail R-CRAWL go to **NEXT-CRAWL on X's states** |
| **PARTIAL / NOT ADOPTED** | guards held; R-MOVE at both seeds; R-CRAWL fails at ≥ 1 seed; R-KEEP or R-NOBASE fails at ≥ 1 seed | as FIXED / NOT ADOPTED: DAG-ORC-SWAP |
| **PARTIAL / ONE SEED** | guards held; R-MOVE at exactly one seed | not adopted, and never called an effect of the lever (`H-ESTIM-SEED-1`). **NEXT-CRAWL on DAG states** |
| **NO EFFECT** | guards held; R-MOVE at neither seed | own-state levers have now failed twice on DAG (DAgger iteration 2, then the perturbation). The state channel is eliminated as the crawl's lever. **NEXT-CRAWL on DAG states** |

**The table is complete and exclusive (MEASURED, `enumerate_cells.py`, exit 0; output `enumerate_cells.out`).**
* There are 524,288 cells: controls × 5 guards × 2 seeds × (R-CRAWL, R-MOVE, R-KEEP, R-NOBASE) × 2 seeds. Each cell matches exactly one row.
  | row | cells |
  |---|---|
  | REFUSED | 262,144 |
  | HARMS | 261,888 |
  | FIXED / ADOPTED | 1 |
  | FIXED / NOT ADOPTED | 15 |
  | PARTIAL / ADOPTED | 3 |
  | PARTIAL / NOT ADOPTED | 45 |
  | PARTIAL / ONE SEED | 128 |
  | NO EFFECT | 64 |
* Adoption is reachable only with R-MOVE, R-KEEP and R-NOBASE at both seeds.
* With adoption projected away, the 16 (R-CRAWL, R-MOVE) cells map exactly as `PREREG_V8F_DAGGER2` Amendment 1 (FIXED 1 / PARTIAL 11 /
  NO EFFECT 4). The floors and the adoption split move no DAGGER2 cell.
* The A → B cascade is total over 128 combinations. NEXT-CRAWL is total over its 6 cells, and the launch routing over its 4.
* Mutation proof: 13 / 13 mutants exit 1 (`enumerate_cells_mutation.txt`).

**Precedence.** REFUSED is decided first, then HARMS. With the guards held, R-MOVE partitions the rest: both seeds, one seed, or neither.
R-CRAWL then splits FIXED from PARTIAL, and adoption is decided last.

**What the dose cascade does and does not license.**
* B is a second shot at the guards, taken only after A's HARMS. Its success claim is labelled with its dose (p = 0.1) and carries B's own
  floors and seeds. It is never pooled with A.
* If A is not HARMS, B is a dose-response row beside. It decides nothing, and it never promotes B over A.

## 8. NEXT-CRAWL: what follows if the crawl does not move, reasoned from the data and registered now

**What the measurements rank.**
1. **Only the executed-horizon along-track profile has moved the crawl rows.** HACC's post-map, re-fitted on DAG with zero training, cut
   the excess over P10:
   * M1 by 49 % / 36 %;
   * L2 by 61 % / 54 %;
   * curvature by 72 % / 58 %.
   It failed KEEP at s0 (+0.1264 m), so it is not the base.
2. **The state-channel levers have not moved them.** DAgger iteration 2 read R-MOVE fails at both seeds. If this arm also reads NO EFFECT
   or PARTIAL / ONE SEED, the perturbation joins it.
3. **The crawl is born in the loop.** The diagnosis below is on HACC and INHERITED here:
   * no excess when the tick-0 plan is executed whole (m = 30);
   * 89–104 % of the excess sits in plan 2;
   * at equal own speed, the curvature error equals P10's;
   * freezing the cameras keeps 81 % / 124 % of it.
4. **One loop channel has never been intervened on: nav.** In T1-kin the dead-reckoner holds the t0 manoeuvre token and validity, and
   decrements only the distance by the car's own arc (`v8f_t1_eval.py`, `WindowState`, lines 156–220 at `d64750bd`). In a crawl the
   countdown stalls with the car. Past the manoeuvre point, it holds distance 0 under the t0 token. That is a self-consistent "slow for the
   manoeuvre" input. HYPOTHESIS, never measured.

**Order: the cheapest discriminating experiment first (operating rule 5), then the lever with the largest measured effect.**

**N0, the nav probe (0 GPU; registered here in full).**
* **Runs:** the base at T1-kin with `--nav logged`, both seeds, every other setting decisive. The `--nav deadreckon` runs are banked.
* **Read:** the rows of C for nav-logged − nav-deadreckon, the same checkpoint, paired, B = 2,000.
  * The planner is deterministic and the checkpoint is shared, so no training-run term enters, and separation decides.
* **NAVMOVE:** every row of C separated-better at both seeds.
* ⚠ `--nav logged` is an open channel: per the tool's own `open_channels`, it is *"route countdown from the logged human position"*. So
  NAVMOVE cannot separate a route-extent defect from a timing leak. The next arm's read does.

| N0 | DAG-OACC's read: DAG-OACC − DAG on every row of C | committed next |
|---|---|---|
| **NAVMOVE** | any | **ROUTE-EXTENT:** a PREREG for the dead-reckoner fix, which advances the nav row through the logged route's manoeuvre sequence by the car's own arc, as a navigation system does. Zero training, dev-box CPU, control `--nav deadreckon`. Nav is an INPUT at training and inference (PI), and the fix adds no human timing. If it reproduces N0's gain, the defect is route extent; if not, N0's gain was the timing leak and LAUNCH-INTENT follows |
| not NAVMOVE | separated-better at both seeds | **COMPOSE:** DAG-OACC's acceleration term on the base, read on Read A under its own PREREG. It is already the measured launch-profile lever |
| not NAVMOVE | not, or not yet read | **LAUNCH-INTENT:** O's along-track loss over the executed horizon (knots 1–12 at m = 10), up-weighted on GT-launch TRAIN samples (v0 < 6 m/s and GT v(6 s) ≥ v0 + 1 m/s). One flag with its G-DVB entry, DAG init, 3,000 steps, under its own PREREG. The launch subset is label-side (future GT speed), which is admissible for labels; nothing new enters at inference. It is the trained form of the profile change that cut the crawl by 36–72 % at zero training (rank 1 above), and it is aimed at O's own slow launch intent (O's v0 −1.78 / −0.93 m/s at tick 1, INHERITED from `DIAGNOSIS.md`) |

## 9. What this read cannot claim

* **It is not closed loop.** T1-kin is a self-action open loop: the lead follows its logged future and the cameras are logged.
* **A one-seed R-MOVE is not an effect.** Only R-MOVE at both seeds, against floors from two independent replicate pairs, attributes a
  crawl change to the lever.
* **No lateral / longitudinal split.** The lever has two components, and neither half is credited.
* ⛔ No bar, row, floor formula or routing changes after any DAG-ORC row is seen. RF0 is frozen before any DAG-ORC rollout is read.

## 10. Queue, cost, and launch preconditions

**Box 2** (needs the **PI's go for box-2 time**; it queues after FLAT-DAG and DAG-OACC). The chain order is A s0, REP s0, A s1, REP s1,
B s0, B s1, so a stopped chain still leaves complete seed pairs first. Each run goes through its gate.

| item | MEASURED basis | ESTIMATED |
|---|---|---|
| gate | DAGGER2's four gates ran 12:02–12:06 (`chain_v8f_dag2.out`) | ≈ 1 min per run |
| train + standard T0 eval | 44 / 41 / 43 / 43 min per DAGGER2 run (chain timestamps); training 2,251 / 2,377 s (DAG2-0 s0 / s1) and 2,413 s (DAG2 s0) (`summary.json`); ORC 2,258 / 2,254 s against ORC-0 2,249 / 2,257 s (s0 / s1), so the perturbation costs no measurable time | ≈ 43 min per run |
| **A + REP (4 runs)** | | **≈ 3.0 h** |
| **B (2 runs)** | | **+ ≈ 1.5 h (≈ 4.5 h total)** |
| no bank stage | DAGGER2's banks took 51 / 54 min (s0 / s1) | 0 |

**Dev-box CPU.**
* ESTIMATED: the new arms' rollouts take about 37 min for A + REP and 19 min for B. MEASURED basis: DAGGER2's 8 rollouts plus 4 fresh T0
  evals took 37 min.
* P-0 (RF0) reads banked dumps only and needs no rollout. N0 is 2 rollouts, about 10 min.
* About 1.3 GB of dumps (+0.65 GB with B) go to E:. Nothing goes to C:.

**Launch preconditions (committed now).**
1. P-0 is computed and frozen.
2. The reader is built from `dag2_read.py`, with K-RC green and a mutation proof.
3. **N0 MAY run before A's gate** (recommended; it costs no GPU).
   * If it reads NAVMOVE, DAG-ORC is **held**, and ROUTE-EXTENT goes first. The crawl would then ride on a channel this lever does not touch.
   * If not, DAG-ORC launches as written, and NEXT-CRAWL later starts from its LAUNCH-INTENT / COMPOSE branch.
4. **If a read adopts another base (for example DAG-OACC) before A's gate,** this PREREG is not launched as written. It is amended, before
   any run, to that base with a newly trained control, because CTRL = DAG2-0 is valid only for DAG.

## Amendment 1 (A1-2026-10-27): re-based on DAG-OACC, before any DAG-ORC run (launch precondition 4)

Written 2026-10-27 ~23:10 by the Master Mind.

**State when written.**
* No DAG-ORC run, gate or row exists.
* The PI's go of 2026-10-27, verbatim, was *"Yes, queue DAG-ORC on box 2 after the SPEEDREAD pull, through its launch gate."* It was
  asked as ~3 h of box 2. The dev-box waiter that would have started the DAG-based chain was stopped by the Master Mind before it fired
  (`box2_ship/logs/dagorc_waiter.out`). Nothing reached box 2.

**Why.** Launch precondition 4 (§10): *"If a read adopts another base (for example DAG-OACC) before A's gate, this PREREG is not launched as
written."*
* `PREREG_V8F_DAG_OACC` read **LONGITUDINAL FIXED** under its Amendment 1. The read is `raw/2026-10-27-dag-oacc/read/DAGOACC_READ.json`
  (sha256 `9e6bd4f6…`), with 60 / 60 controls PASS. The unamended table gives the same word.
* Its committed next adopts DAG-OACC as the base and sends the open crawl rows to this arm, re-based on DAG-OACC.
* The open rows at both seeds, DAG-OACC − P10, all separated-worse: yaw-rate MAE, curvature MAE, L2, M1.

### 1. The arms, re-based (the COMPOSE form)

| arm | what | role |
|---|---|---|
| **A** | DAG-OACC_s's training argv **verbatim** (`v8f-dagoacc/M2b_s{s}_dagoacc/argv.json`: init M2b_s, 3,000 steps, `--dagger-p 0.1` on the iteration-1 bank, `--w-acc 49.6969`), plus the perturbation at §1's law (`--ego-perturb-p 0.3 --ego-perturb-vscale 0.5 1.0 --ego-perturb-vmax 8.0 --ego-perturb-adrop 1.5 --ego-perturb-yawrate-std 2.0 --ego-perturb-seed <s>`), plus `--ego-perturb-with-dagger`, with its own `--out` | decisive |
| **CTRL** = DAG-OACC_s | **banked, reused.** Checkpoints `67265ec1…` (s0) / `90b01474…` (s1); T1-kin dumps `v8f-dagoacc-t1/` | the one-lever control: A minus the perturbation |
| **REP** | DAG-OACC_s's argv with `--seed 10` / `--seed 11` (init M2b_s0 / M2b_s1), with its own `--out`. Trained new | the replicate floor RF |
| **B** | A with `--ego-perturb-p 0.1` | the dose fallback. Trained **only on A's HARMS, with its own PI go** |

* **Why this form, and not "a newly trained control" from DAG-OACC's checkpoint.**
  * Precondition 4 asked for a new control because DAG2-0 is valid only for DAG. Here the base's own banked run is the control. It
    differs from A in the lever alone, the way DAG-OACC itself was read against DAG.
  * So no control is retrained, and only REP is new.
  * It also removes the adoption handicap that §4 disclosed. A no longer starts behind its base: A and DAG-OACC are trained the same way
    from M2b, except for the perturbation.
  * The alternative was rejected. It fine-tunes 3,000 GT-only steps from DAG-OACC's checkpoint, with a new control and a new replicate:
    6 runs, ~4.3 h, outside the ~3 h asked. It would carry DAG2-0's measured handicap (+0.27 / +0.13 m ADE6 against DAG).
* **The composition is a build item and a LAUNCH BLOCKER.**
  * The trainer refuses `--dagger-p > 0` together with `--ego-perturb-p > 0` (`v8f_mini_train.py`, the `PREREG_V8F_DAGGER` §4
    exclusivity, written for attribution inside the DAGGER arm).
  * A new **default-off** flag, `--ego-perturb-with-dagger`, lifts that refusal for an argv that names it. Semantics, fixed now:
    * the swap is drawn first;
    * the perturbation is then drawn, at its own p and from its own seeded generator, **only on samples the swap did not replace**;
    * no sample is ever both swapped and perturbed.
  * With the flag off, every existing argv must build and train bit-identically (K-BUILD).
  * Proofs before any gate, per the CLAUDE.md rule of 2026-09-10:
    * a G-DVB entry for the flag;
    * G-LIVE on a composed batch: both levers reach the loss gradient, and the ego probe's p = 0 control reads exactly 0.0;
    * a mutation (delete the perturbation's apply on the composed path ⇒ RED);
    * the run record carries `dagger_n / dagger_eligible` in 0.1 ± 0.02, `ego_perturb_n / ego_perturb_eligible` (non-swapped eligible) in
      0.3 ± 0.02, and an overlap count of **exactly 0**. REP reads ego 0 exactly and swap 0.1 ± 0.02.

### 2. What every section now reads

* **"DAG" means DAG-OACC, and "CTRL" means DAG-OACC,** in §3, §4, §6 and §7. This covers R-KEEP, R-NOBASE, the domain D_s and the beside
  rows.
* **R-ADE and R-KEEP now share one contrast**, A − DAG-OACC. R-KEEP keeps its point bar (≤ +F/2 = 0.0604 m), and R-ADE keeps "not
  separated-worse". The same holds for R-NEW and R-NOBASE. Both coincidences are disclosed; no row is dropped.
* **The crawl set C** is the set of rows separated-worse for DAG-OACC − P10. At both seeds that is yaw-rate MAE, curvature MAE, L2 and M1.
* **The domain D_s** is every Read-A row not separated-worse for DAG-OACC − P10, with **no exempt rows**.
  * accel MAE and DK-VIOL were exempt because *"they belong to DAG-OACC"*. DAG-OACC has now cleared them, so they are guarded here.
* **The floors.**
  * RF0 is unchanged: P-0 is frozen at `raw/2026-10-27-dag-orc-read/P0_RF0.json`, sha256 `1608d70b…`.
  * RF_r = max over s of |d_r,s(REP − DAG-OACC)|.
  * F and bar_r are as in §3.
* **K-PAIR.**
  * argv(A_s) equals argv(DAG-OACC_s) except `--out`, the six perturbation flags at §1's values and `--ego-perturb-with-dagger`.
  * argv(REP_s) equals argv(DAG-OACC_s) except `--out` and `--seed`.
  * The bank sha256 is DAG-OACC's.
* **K-INIT:** M2b_s, as recorded in DAG-OACC's launch-gate token.
* **K-BUILD:** against `aeb54468`, the commit DAG-OACC trained at, under §6's two conditions. The new flag's hunks pass a bit-identity
  test with the flag off.
* **K-RC:** the reader, with the floors at 0 and DAG-OACC in the base slots, reproduces `DAGOACC_READ.json`'s Read-A table for
  DAG-OACC − P10 with 0 differences, as well as DAG2's rows as before.

### 3. Outcomes (§7's table, the same conditions in the same order, re-based)

* **FIXED / ADOPTED:** adopt X as the base and re-run Read A on X. DOMINANT there is the dominance claim. ("Then X with DAG-OACC's lever"
  is already done.)
* **FIXED / NOT ADOPTED** and **PARTIAL / NOT ADOPTED:** DAG-ORC-SWAP cannot apply, because A already keeps the swap on. Next is **B (p
  0.1) under this table**: a lower dose that may keep the base. It needs its own PI go.
* **HARMS** of A ⇒ B, as written. HARMS of B ⇒ NEXT-CRAWL.
* **NEXT-CRAWL** (§8) runs on DAG-OACC's states.
  * N0 already read NOT NAVMOVE.
  * COMPOSE is done: DAG-OACC − DAG is not separated-better on the rows of C.
  * So its branch is **LAUNCH-INTENT**, with DAG-OACC's recipe in place of "DAG init".
* **Every other row is unchanged.** The 524,288-cell enumeration stays complete and exclusive, because no condition changed, only its
  base.

### 4. Cost and queue

* A + REP is 4 runs, ≈ 2.8 h. MEASURED basis: DAG-OACC's two runs took 41 and 42 min with their T0 evals (`chain_v8f_dagoacc_r2.out`).
  This is within the ~3 h of the PI's go.
* No bank stage: the iteration-1 bank is reused at DAG-OACC's sha256.
* B is not queued.
* Dev-box CPU: the T1-kin rollouts of A and REP at both seeds. CTRL's and P10's runs are banked.

## Amendment 2 (A2-2026-10-28): the composed perturbation drew from the swap's own stream; fixed, and all four runs re-run, before any DAG-ORC row is read

Written 2026-10-28 ~02:30 by the Master Mind.

**State when written.**
* No DAG-ORC T1-kin rollout, read or row exists.
* r1 (`chain_v8f_dagoaccorc.sh` at `e7ea20d1`, started 01:16 under the PI's go of 2026-10-27) passed all four gates, A and REP at both
  seeds (`box2_ship/logs/chain_v8f_dagoaccorc.out`).
* r1 then trained A s0 (3,000 steps), pulled it and passed its run record. **K-REACH failed:** `ego_perturb_n / ego_perturb_eligible` =
  19,161 / 76,898 = **0.2492**, outside 0.3 ± 0.02. The swap share was 0.1013 and the overlap was exactly 0.
* The chain stopped, as designed (`ZZDAGOACCORC-CONTROL-FAIL`). REP s0, A s1 and REP s1 never ran. The refused run is never rolled out
  or read.

**The defect.**
* `DaggerSwap._seed_of` (`stack/tanitad/data/v8f_dagger.py`) and `EgoPerturb._seed_of` (`stack/tanitad/data/v8f_mini_data.py`) are the
  same formula, `(seed × 1,000,003 + step) mod 2^62`. The registered argv sets both seeds to s.
* Both levers read their selection uniform from row 0 of `torch.rand(k, B)`. The two rows are therefore identical (verified:
  `torch.equal` is True).
* So a sample eligible for both levers is swapped iff u < 0.1 and perturbed iff 0.1 ≤ u < 0.3. That gives P(perturbed | not swapped) =
  0.2 / 0.9 = 0.222, against 0.3 for a sample only the perturbation may touch. The measured 0.2492 lies between the two.
* Amendment 1 §1 fixed the semantics as *"drawn, at its own p and from its own seeded generator"*, which means independent of the swap.
  The build met that in form but not in fact: two generators with the same formula and the same seed are one generator.
* The composition tests missed it, because their fixture did not combine equal seeds with overlapping eligibility. K-REACH caught it on
  the first real run, which is the job the PREREG gave it (*"the gate does not cover the loop's draw"*).

### 1. What changes

* **The trainer.** On the composed path only, the perturbation's generator takes a domain-separated seed, so its stream is independent of
  the swap's at equal seeds. Every other path stays bit-identical: flag off, either lever alone, plain M2b. The commit, the recorded field
  and the proofs are listed in §3.
* **The runs.** All four runs, A and REP at both seeds, are re-run from their gates at the fixed commit by the r2 chain.
  * Every r1 artifact is renamed `<name>_r1_kreach_stop`, on box 2 and on the dev box. Nothing is deleted.
  * Gates are re-run because the commit changed, and the launch-gate tokens bind to the new commit.

### 2. What does not change

* The arms, the registered argv (including `--ego-perturb-seed <s>` and `--dagger-seed <s>`) and the K-REACH bands.
* The one-lever rule, the reader (`2e5f9f17…`), the crawl set, the domain, the floors and §7's outcome table.

### 3. Proofs, before r2 launches

| check | result | file (under `raw/2026-10-28-dag-orc-a2/`) |
|---|---|---|
| the fix: `EgoPerturb` takes a `stream`; 0 = the original formula, so every non-composed path is unchanged; the composed stage sets `EGO_PERTURB_COMPOSED_STREAM = 1`, a sha256-mixed seed; recorded as `levers.ego_perturb_stream`; a G-DVB read-back and a resume guard | trainer `1c888280…`, data lib `c117e1cf…`, tests `faad3be2…` | `fix/NOTES.md`, `fix/diff.patch` |
| the new tests on `e7ea20d1` (the shared stream) | **RED**: P(perturbed \| not swapped) 0.0000 at 0.5 / 0.5 and 0.221 / 0.224 at 0.1 / 0.3; the two selection rows equal; a 200-step run at box 2's shape reads 1,561 / 6,345 = 0.2460 | `fix/logs/new_red_on_old.log` |
| the same tests on the fix | **GREEN**: the same 200-step run reads 1,860 / 6,345 = 0.2931, swap 720 / 7,380, overlap 0, `levers.ego_perturb_stream` 1 | `fix/logs/new_green2.log`; the run record `r2/fixtures/compose_fix_real/A_summary.json` |
| the flag-off golden (`ego_dagger_compose_golden.json`, unchanged) | passes | `fix/RESULTS.md` |
| the seven v8F suites, one file at a time | 36, 80 (+2 skipped), 95 (+2 skipped), 57, 7, 56 and 13 (+32 skipped) passed, 0 failed | `fix/RESULTS.md` |
| the mutation sweep | **27 / 27 killed**, including `M17b`: the stream set, gated and recorded as 1 but not mixed into the seed | `fix/RESULTS.md`, `fix/mutation.txt` |
| the r2 chain (aside, the predecessor check, `STREAM_PATTERN`, `levers.ego_perturb_stream` = 1 on A and 0 on REP) | 66 / 66 tests, 223 / 223 mutants killed | `r2/logs/selftest_chain_r2.out` |

* The two skips in `test_v8f_dagger_train` need `raw/2026-10-26-dagger-train/`, which the fix's tree did not export. They are re-run on an
  archive of the landed tip before r2 launches.

### 4. Cost

* r1 used box 2 from 01:16 to 02:10 (0.9 h). r2 is about 2.8 h (four runs, the r1 basis). That is about 3.7 h in all, against the ~3 h the
  PI's go was asked as. The 0.7 h overrun is disclosed here.

### 5. The class

*Two seeded generators with the same formula and the same seed are one generator.* Two levers can be independent in a specification and
still be coupled in a build. The test that exposes this has to reproduce the registered argv's seeds together with overlapping
eligibility; a fixture with either one alone passes.
