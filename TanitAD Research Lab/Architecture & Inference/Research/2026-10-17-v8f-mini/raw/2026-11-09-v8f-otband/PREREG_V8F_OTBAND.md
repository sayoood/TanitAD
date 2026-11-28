# PREREG_V8F_OTBAND: does a per-band trajectory loss, with O's loss detached from T, and a detached strategic token into T, improve the 6 s plan inside its bands? (stream B draft 2026-11-08; frozen by the Master Mind 2026-11-09, before any OTBAND training run exists)

Status: FROZEN (prefix sha256 in the Log). Code: commit `8e9499e0f7c201945a856b51e698b465767b6d01`, branch `agent/v8f-otband-20261109`, which is `1faa1a06` plus `otband.diff` (sha256 `bc10a0e1`) applied unchanged. Style and rules follow `theory/PREREG_V8F_LADDER.md`. Every number carries its evidence class; every eval number carries its tier.

## Why
PI ruling (`REQ_HORIZONS.md` R-EGO / R-T2O / R-OUT; `SPEC_HWM.md` section 9.1): all three layers read the current measured ego state and plan nothing for each other; the tactical layer emits goals and tactical constraints (up to 6 s) that are taken as actions conditioning the operative layer (T's code through `e_code`, as built); the planner output is a trajectory or trajectories up to 6 s. SPEC section 9 adds that a lower level's loss puts 0.0 gradient on the upper level. M2b ran with `detach_upper` off, so O's loss trained T through the code.
The band baseline (`horizon_review/band_eval/BAND_BASELINE.md`, MEASURED, T1-kin, DAG-OACC s0 / s1 against P10) says where the error sits: operative ADE 0.219 / 0.225 m, tactical ADE 3.254 / 3.368 m; the operative band holds 10 x (-0.003) of the 36.03 m sum behind the 30-knot ADE gap (about 0.1 %). The tactical lateral reads (yaw-rate MAE 30.4 / 37.2 deg/s, curvature MAE 0.185 / 0.260 1/m) are worse than the kinematic floor (P10: 21.5 deg/s, 0.056 1/m).
This card asks, with the architecture unchanged (one O decoder, 30 knots): (1) does weighting the loss by band, with O detached from T, move the bands; (2) does the detached strategic token add to T and through its code to O.

## Arms (S's output held at M2b s{k} by freezing S, `adapter` and `cond`; T and O trainable; `--detach-upper` in every argv; box-2 argvs in `argvs_otband.txt`, choices in `IMPL_CHOICES_OTBAND.md`)
| arm | loss | token | what it answers |
|---|---|---|---|
| **OB-CTL (control)** | today's 30-knot term (`--w-traj 1`) | none | the matched reference: same detach, same freeze, same data and steps |
| **OB-BAND** | `--w-op 1/3`, `--w-tac-traj 2/3`, `--w-traj 0` | none | the band-loss effect (against OB-CTL) |
| **OB-BAND-TOK** | as OB-BAND | detached strategic token into T | the token effect (against OB-BAND) |
* **Replicate OB-BAND_rep:** OB-BAND s0's flags and init, training seed 2: the run-to-run floor of the retrain (`H-ESTIM-SEED-1`). A difference between two arms is a lever effect only if it exceeds this floor on the same row.
* Seeds 0 and 1 (init M2b s0 and s1) for the three arms; 7 runs in all. `--steps-mult 2` (ESTIMATED 2 epochs). A 200-step pilot of OB-BAND s0 measures the time per step before any full run; the chain refuses a projected total above 72 h on box 2 and that case goes to the PI.
* **Token shuffle control (no training):** OB-BAND-TOK evaluated with the token shuffled across episodes (`D.attach_stok(ds, dir, seed)`; a seeded cyclic derangement).
* The token content is stream C's `stok` (UNVERIFIED until its dump exists; assumptions in `IMPL_CHOICES_OTBAND.md` section 5). OB-BAND-TOK cannot launch before that dump: a named blocker; the other arms do not wait.
* Both weights sum so that, where the bands agree, the band loss equals the 30-knot ADE weighting (10/30 and 20/30): the arms differ in the added FDE at knot 10 and the replacement of FDE at 30, not in the knot weighting. Small expected effect: stated here, so a SPLIT-NEUTRAL outcome is not read as a surprise.

## Read (T1-kin: self-action open loop, m = 10, causal bucket, logged scene; 141 episodes / 2,399 windows, per seed)
Instrument: `v8f_t1_eval.py` dumps with the band slicing of `band_eval/band_lib.py` (knots 1-10 = 0.2-2.0 s operative, knots 11-30 = 2.2-6.0 s tactical), the same as the baseline. Estimator: `taniteval.ci.paired_episode_cluster_bootstrap` (arm minus DAG-OACC, and arm minus arm, same windows), B = 2,000, seed 0. It answers "would another draw of EPISODES say this?" only; "would another TRAINING run say this?" is answered by the replicate row and the two seeds side by side; the planner is deterministic, so the inference-variance question has nothing to read. Rows are never pooled across families or bands.
Four families per band (BINDING, CLAUDE.md 2026-08-02), each with its CI:
* **PRIMARY:** ADE and FDE per band (FDE at 2 s and at 6 s).
* **LONGITUDINAL:** speed MAE, acceleration MAE, along-track MAE, distance keeping (headway abs error, time-gap abs error, DK-VIOL share).
* **LATERAL:** heading MAE, curvature MAE, yaw-rate MAE, cross-track MAE.
* **TACTICAL:** TAC-LAT and TAC-LON accuracy (whole-plan labels, as the baseline states), plus T's own decision reads (goal, lat, lon, constraints against the labels; dumps with `--dump-heads`).
* **STRATEGIC: PAUSED** (PI item 25, `PI_DECISION_QUEUE.md`, 2026-09-19): reported as paused with this ruling, not as a pass, not omitted. The token is an input to T, not a strategic result.
Beside the families (no bar): **trajectories, plural** (R-OUT): `forward_topk` decodes O once per top-K tactical manoeuvre code (K = 3, eval only, no parameter): per band, the best-of-K ADE (minimum over the K plans) next to the top-1 ADE, and the share of windows where the GT's manoeuvre is among the K. Train terms (`ade_op`, `ade_tac`, `op`, `tac`) from `train_log.jsonl` (T0, teacher-forced mix: a curve, not a result).

## Committed bars (against OB-CTL of the same seed, paired; ESTIMATED from the baseline's own CI widths, committed before any run)
Δ(x) = arm minus OB-CTL on x with its paired 95 % CI; "up" = the upper bound. OB-CTL shares the arm's init, detach, freeze, data and steps, so this is the difference that isolates the band loss. Lower is better for every error row. Beside every row, with no bar: arm minus DAG-OACC of the same seed, paired, same estimator, the read against the adopted reactive base.
* **B1 operative non-regression:** up Δ(ADE_op) ≤ +0.03 m, up Δ(FDE_op) ≤ +0.05 m (baseline widths ±0.007 and ±0.02 m); up Δ(speed MAE_op) ≤ +0.03 m/s; up Δ(cross-track MAE_op) ≤ +0.02 m; up Δ(yaw-rate MAE_op) ≤ +0.5 deg/s.
* **B2 tactical primary:** up Δ(ADE_tac) < 0 and up Δ(FDE_tac) < 0.
* **B3 tactical longitudinal:** up Δ(speed MAE_tac) ≤ 0 and up Δ(along-track MAE_tac) ≤ 0; distance keeping not worse in either band: up Δ(DK headway) ≤ +0.2 m, up Δ(DK time gap) ≤ +0.03 s.
* **B4 tactical lateral (the named weakness):** up Δ(yaw-rate MAE_tac) < 0 and up Δ(curvature MAE_tac) < 0; P10's values beside.
* **B5 tactical decisions:** TAC-LAT and TAC-LON accuracy not below OB-CTL's by more than 0.02.
* **B6 effect floor:** a difference between two arms is a lever effect only if it also exceeds the OB-BAND_rep difference on that row.
* **Token (OB-BAND-TOK against OB-BAND):** TOKEN-USED needs all of: up Δ(ADE_tac) < -(replicate floor); the shuffled-token read of OB-BAND-TOK worse than its true-token read, paired CI of the difference excluding 0 on ADE_tac; B1 holds.

## Committed outcomes (per seed; both seeds must agree)
* **BAND-GAIN:** OB-BAND against OB-CTL: B1 and B2 hold and exceed the replicate floor, and no row's up Δ exceeds its bar.
* **BAND-NEUTRAL:** B1 holds, B2 does not exceed the floor, no row worse than its bar.
* **BAND-COST:** B1 fails, or any tactical row is worse than its bar.
* **TOKEN-USED / TOKEN-IGNORED:** by the token rule.
* **B4** is read on its own and named in every outcome.
* **OB-CTL against banked DAG-OACC:** reported as a row, not a bar. The two differ in init, the detach, the freeze and the step count, so this row attributes nothing to any one of them.

## What each outcome triggers (committed now, Rule Zero: a failed arm leaves the next arm behind)
* **BAND-GAIN:** if B1 also holds against DAG-OACC of the same seed, the band-loss argv becomes the O/T training recipe of record and HWM-PLAN (`SPEC_HWM.md` section 4) runs on it. If B1 fails against DAG-OACC, that regression is named and BAND-COST's first lever runs.
* **BAND-NEUTRAL (the expected result for a reweighting of equal knot weights):** the next lever is signal, not weights, cheapest first: (1) `--w-tac-traj` x3 with `--w-op` unchanged (the tactical band gets 3x its share), (2) the v10 2-6 s labels for T (stream A's release by md5, `--v10-md5`), (3) tactical-band plan penalties (`--w-pen`) aimed at B4.
* **BAND-COST (an operative row fails):** (1) `--w-op` x3, (2) unfreeze `cond` (O's embeddings adapt), (3) report the seed spread against the replicate before reading it.
* **B4 fails with B1, B2 holding:** a lateral term on knots 11-30 only (yaw-rate and curvature next to ADE; `second_diff_02` and `consistency_penalties` exist as instruments), as the next arm in the same package.
* **TOKEN-IGNORED:** (1) read the token directly (linear probe of `stok` against the goal and lat labels, T0) to separate "T ignores it" from "it carries nothing"; (2) if it carries information: raise `e_stok`'s learning rate and train T longer with the token on; (3) if it carries nothing, the blocker is stream C's content, named with the probe numbers.
* **TOKEN-USED:** the token arm becomes the T of record; the shuffle control is attached to every later eval of it.
* **Top-K:** if best-of-3 ADE_tac beats top-1 ADE_tac by more than the replicate floor on both seeds, multi-trajectory output is worth a trained form (a mixture head over the K codes): proposed to the Master Mind, no head is added here.
* **Blockers named at freeze:** OB-BAND-TOK needs stream C's `stok` dump, `stok_meta.json` and the `stok` keys in `v8f_t1_eval.py`; it launches after them, and the other five runs do not wait. Launch gate: `launch_gate.py` has no profile for the mini trainer (AUDIT_MODEL F12), so every arm runs under the per-run token route of `theory/ladder/box2/`: pin the argv, run the trainer with `--selfcheck-only` at this commit, assert the BUILT levers against the argv, write a token bound to commit, argv and data, and refuse to train without it. The v10 labels are a later lever only (BAND-NEUTRAL, item 2); they need the release md5 in `ACCEPTED_V10_TRAIN_MD5` and its 4,572-clip base as `--v9-train`.

## Controls (the read is INVALID if one fails; each has a test in `stack/tests/test_v8f_otband.py` and a deliberate-regression mutation, `mutate_otband.py`)
1. **Default identity:** the default argv builds the pre-change model bit for bit (state dict, loss terms, total, `forward_infer`, gradients).
2. **Weights:** each band term moves the total by exactly its weight; each band term's gradient reaches only its own knots.
3. **Purity (`--detach-upper`):** O's `op` and `tac` terms put exactly 0.0 gradient on S, T and the token embedding while O's head receives gradient; removing the code detach or the checkpoint detach goes red. Without the flag the same term reaches T (the earlier behaviour, tested).
4. **Code edge:** changing T's code changes O's knots in both bands; zeroing it changes them; cutting the `e_code` wire goes red.
5. **Token edge:** the token reaches T's logits, T's code and O's plan; shuffling across episodes changes them; it is read at S's tick frame (sample-and-hold); the shuffle store is a derangement.
6. **Gate:** every flag has a declared-vs-built entry; a lever declared and not built is refused; `oband_probe` runs in G-LIVE on the real batch.
7. **Freeze identity:** with `--freeze S T adapter cond` S's and T's weights and the shared embeddings equal the init checkpoint bit for bit (tested; the arms freeze S, `adapter`, `cond` only).

## Implementation choices
`theory/otband/IMPL_CHOICES_OTBAND.md` (sha256 sidecar) records the band loss scales, the freeze groups, the token interface assumptions, the top-K mode, the md5 acceptance and the argvs.

## Log
* 2026-11-08 draft written by stream B after the PI ruling removed the tactical trajectory head, the hand-over and the continuity term. Not frozen. No OTBAND training run, token dump or v10 release existed at this time.
* 2026-11-09 frozen by the Master Mind: prefix (every byte before `## Log`) 11,910 bytes, sha256 `fd81c66e1459da16bddd44c1501394d30cc81d036f54ef9c4ad28e4594eb6c37`. Changes from the draft: the bars read against OB-CTL of the same seed, with DAG-OACC beside them, because the draft's bars named DAG-OACC while its outcomes named OB-CTL; the DAG-OACC row states the four differences; the recipe of record also needs B1 against DAG-OACC; a pilot and a 72 h limit; the v10 labels stated as a later lever; the token route stated. No OTBAND training run, token dump or per-run gate existed at this time.
* 2026-11-09 amendment 1, before any run, the code commit of record is `202849da013b5d74c0b0086dd9865fcda6836018` (`8e9499e0` plus one commit). `oband_probe` now reads, on the real batch, every O term of the arm's own loss under `--detach-upper`: `op` and `tac` on band arms, `ade` and `fde` on OB-CTL, which the probe skipped before; the selfcheck records it for any arm with `--detach-upper`. This makes control 3 true on the real batch as the card writes it. The training path, the bars and the arms are unchanged. MEASURED: `test_v8f_otband.py` 49 pass on CPU, including a regression where the probe is told detach and the model does not detach (it refuses).
