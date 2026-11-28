# PREREG_V8F_OBD: on DAG-OACC's recipe, does the PI's detach (R-T2O) keep the base's gains, and does the band loss add its operative gain there? (Master Mind, frozen 2026-11-10 before any OBD run)

Status: FROZEN 2026-11-10, before any OBD run, argv launch or box-2 contact; the sha256 of this file is in `PREREG_V8F_OBD.sha256`. Code of record: commit `cdbff971fa6dad4d20b937fe8abda6cb9f622adb` on `agent/v8f-otband-20261109` = the OTBAND commit `202849da013b5d74c0b0086dd9865fcda6836018` plus the probe change of Controls item 3 (2 lines and one test file, nothing else; pushed and `ls-remote` verified 2026-11-10). Style and rules follow `theory/PREREG_V8F_OTBAND.md`. Every number carries its evidence class; every eval number carries its tier.

## Why
* **The adopted base does not meet the PI's ruling.** `REQ_HORIZONS.md` R-T2O and `SPEC_HWM.md` section 9: a lower level's loss puts 0.0 gradient on the upper level. DAG-OACC (`v8f-dagoacc/M2b_s{k}_dagoacc/argv.json`) runs with `detach_upper` off, so O's loss trains T through the code (`PI-V8F-2026-10-25c`). Its recipe with the detach has never been measured.
* **OTBAND read (H-V8F-OTBAND-1, MEASURED, T1-kin, both seeds):** on a logged-only retrain from M2b, the band loss improves the operative band above the replicate floor (ADE -0.017 / -0.015 m) and moves no tactical row; the replicate floor on tactical ADE is 0.287 m. Beside, the banked ladder puts DAG-OACC's tactical lead on the DAgger rows (tactical ADE -0.82 / -0.62 m) and its operative and acceleration lead on the acceleration term (operative ADE -0.062 / -0.079 m). The OTBAND arms carried neither, so the band loss was never read on the base it has to beat.
* **A hint, not evidence:** OB-CTL (detached, S frozen, 9,582 steps, logged rows only) minus DAG0 (not detached, 3,000 steps, logged rows only) reads tactical ADE -0.107 / -0.095 m, not separated, and operative ADE -0.016 / -0.030 m, separated: the detach did not visibly cost on the logged-only recipe. The pair differs in four ways at once, so it attributes nothing.

This card asks, with the architecture unchanged: (1) does `--detach-upper` keep DAG-OACC's gains on DAG-OACC's own recipe; (2) does the band loss add its operative gain on that detached base.

## Arms (box 2, init M2b s{k}; DAG-OACC's argv verbatim, only the listed flags added; box-2 argvs in `argvs_obd.txt`)
| arm | added to DAG-OACC s{k}'s argv | seeds | what it answers |
|---|---|---|---|
| **OBD-NULL** | nothing (only `--out`) | 0 | code-parity control at the code of record: box-2 training at a fixed seed is deterministic (memory `same-seed-replicate-is-bit-identical`), so it must reproduce DAG-OACC s0's checkpoint bit for bit |
| **OBD-CTL** | `--detach-upper` | 0, 1 | the detach's effect on the base (against DAG-OACC of the same seed) |
| **OBD-BAND** | `--detach-upper --w-traj 0 --w-op 0.3333333 --w-tac-traj 0.6666667` | 0, 1 | the band loss on the detached base (against OBD-CTL of the same seed) |
| **OBD-BAND_rep** | OBD-BAND s0's flags at `--seed 2` | 2 (init M2b s0) | the run-to-run floor of the band arm (`H-ESTIM-SEED-1`) |
| **OBD-NULL_rep** | OBD-NULL s0's argv at `--seed 2` | 2 (init M2b s0) | the run-to-run floor of the base recipe at this commit, seed only |
* Seven runs, 3,000 steps each (MEASURED: DAG-OACC s0 trained in 2,276 s on box 2). ESTIMATED total about 4.5 h of training plus ship, gates and pulls; the chain refuses a projected total above 12 h and that case goes to the PI.
* **Why OBD-NULL_rep and not the banked DAG-ORC pair:** that pair (DAG-OACC's argv at training seed 10 / 11) also carries a later trainer file (`v8f_mini_train.py` sha256 `1c888280` against `30819a6f`), and it differs from DAG-OACC at init s0 by tactical ADE +0.602 [+0.413, +0.798] m and tactical yaw-rate MAE +24.4 deg/s (MEASURED, T1-kin, OTBAND read amendment 4). A floor that large would let any detach cost through at s0; a seed-only pair at this commit says how much of it is the seed.
* No `--freeze`: DAG-OACC trains the whole model, and the card moves one lever per comparison.
* **Floors (rule r4 of the OTBAND read, unchanged):** the floor of a row is |delta| of a same-recipe pair on that row, read at seed 0 and applied to both seeds. For OBD-CTL against DAG-OACC (or OBD-NULL, see the NULL rule): F = |OBD-NULL_rep - OBD-NULL_s0|. For OBD-BAND against OBD-CTL: G = |OBD-BAND_rep - OBD-BAND_s0|. The DAG-ORC pair is reported beside, labelled as seed plus trainer version.

## Read (T1-kin: self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene; 141 episodes / 2,399 windows, per seed)
Instrument: the OTBAND read's dump driver and readers (`theory/otband/read/`: `run_dumps.py`, `read_otband.py`'s band rows via `read_lib.py`, `heads_read.py`, `topk_read.py`), every compared arm dumped in one environment (CPU, 2 threads), windows and model-free arrays asserted equal to the banked DAG-OACC dump. Estimator: `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0; it answers "another draw of EPISODES" only; "another TRAINING run" is the floors above; the planner is deterministic. Rows are never pooled across families or bands. Bands: operative knots 1-10 (0.2-2.0 s), tactical knots 11-30 (2.2-6.0 s).
Four families per band (BINDING, CLAUDE.md 2026-08-02), each with its CI:
* **PRIMARY:** ADE, FDE.
* **LONGITUDINAL:** speed MAE, acceleration MAE, along-track MAE, distance keeping (headway abs error, time-gap abs error, DK-VIOL counts).
* **LATERAL:** heading MAE, curvature MAE, yaw-rate MAE, cross-track MAE.
* **TACTICAL:** TAC-LAT and TAC-LON accuracy (plan-derived), T's own decisions (T-LAT-F1, T-LON-F1, goal macro-F1 at tick 0 and ticks 1-2).
* **STRATEGIC: PAUSED** (PI item 25, `PI_DECISION_QUEUE.md`, 2026-09-19): reported as paused with this ruling, not as a pass, not omitted.
Beside (no bar): top-3 one-shot reads (best-of-3 minus top-1 per band, GT manoeuvre pair in top-1 / top-3); T0 train terms (a curve, not a result).

## Committed rules (before any OBD run)
Δ(x) = arm minus reference on row x, paired 95 % CI. The bar rows are the 11 metric rows of each band plus TAC-LAT and TAC-LON accuracy: 24 cells per seed. Lower is better for every error row, higher for accuracy.
* **A cost** on a cell at a seed: Δ separated on the worse side and |Δ| above that cell's floor.
* **A gain** on a cell at a seed: Δ separated on the better side and |Δ| above that cell's floor.
* For accuracy cells the floor is the same-recipe pair's |Δ| on that accuracy, computed by the same reader.
* A seed's label is read per seed; **an outcome needs both seeds to agree**, else it is SEEDS-DISAGREE.

## Committed outcomes
* **DETACH-KEEPS:** OBD-CTL against DAG-OACC, floor F: no cost on any of the 24 cells.
* **DETACH-COSTS:** at least one cost; every costing cell is named with its Δ, CI and floor.
* **BAND-ADDS:** OBD-BAND against OBD-CTL, floor G: operative ADE is a gain, and no cell is a cost.
* **BAND-NEUTRAL:** no cost, and operative ADE is not a gain.
* **BAND-COSTS:** at least one cost; the cells are named.
* **SEEDS-DISAGREE** (either question): reported per seed; the read is repeated with a third training seed per init (`--seed 10` / `11`) for the arms concerned before any outcome is quoted.

## What each outcome triggers (committed now, Rule Zero)
* **DETACH-KEEPS and BAND-ADDS:** OBD-BAND's argv becomes the O/T training recipe of record (it meets R-T2O); HWM-PLAN (`SPEC_HWM.md` section 4) and the top-K card build on it.
* **DETACH-KEEPS and BAND-NEUTRAL or BAND-COSTS:** OBD-CTL's argv becomes the recipe of record; the band loss leaves the recipe, with its logged-only operative gain recorded as not carried to the DAgger base.
* **DETACH-COSTS:** the base cannot meet R-T2O at this recipe without the named cost. Next levers, cheapest first, under this card's rules: (1) T's own loss weight `--w-tac` x2 (the detach removes O's gradient into T, so T's own term carries T alone), (2) T's own DAgger term on the DAgger rows (already present; read its weight), (3) report the costing cells to the PI: R-T2O is the PI's ruling, and accepting a measured cost of it is the PI's decision.
* **NULL fails parity** (OBD-NULL's checkpoint differs from DAG-OACC s0's): the code of record changed the training path. OBD-NULL_s1 is then trained, every row "against DAG-OACC" is read against OBD-NULL of the same seed instead, and NULL minus DAG-OACC is reported as a code-drift row. The rules above are unchanged.

## Controls (the read is INVALID if one fails)
1. **NULL parity, read on tensors:** every model tensor and every optimizer-state tensor of OBD-NULL_s0's `ckpt_last.pt` equals DAG-OACC s0's bit for bit (shape, dtype, bytes), or the NULL fallback above applies (that is not INVALID; it is named). Whole-file sha256 cannot be the test: the checkpoint embeds the argv's sha, and the argv carries `--out` (`v8f_mini_train.py:3518-3523` at `202849da`, 2 lines lower at the code of record; found by the chain builder), so two files of identical weights always differ. The package's `NULL_PARITY.json` records both, and the chain decides on tensors (`OBD_PARITY_BASIS=tensors`).
2. **Per-run token route** (`theory/ladder/box2/`, as OTBAND): pin the argv, run the trainer with `--selfcheck-only` at the commit, assert every BUILT lever against the argv (detach, band weights, DAgger p, bank sha256, acceleration weight, seed), write a token bound to commit, argv and data, refuse to train without it.
3. **Purity on the real batch:** under `--detach-upper`, every O term of the arm's own loss, the acceleration term included, puts exactly 0.0 gradient on S, T and the token embedding while O's head receives gradient; a deliberate regression (detach removed) goes red. **Found by the chain builder (`PURITY_ACC_FINDING.md`):** at `202849da` the launch probe (`oband_probe`, `v8f_mini_train.py:1601`) reads `op`/`tac` or `ade`/`fde` and never the acceleration term (`v8f_mini.py:1190`, `:1239`); MEASURED on CPU the term is pure (0.0 on S and T, 16.46 on `head_o`), but an acceleration-only leak would pass the gate. The code of record is therefore `202849da` plus the probe change (`keys = (*keys, "acc")` when `w_acc > 0`) and its test `stack/tests/test_v8f_oband_acc.py` (5 cases, three red without the change): commit `cdbff971fa6dad4d20b937fe8abda6cb9f622adb`. MEASURED on CPU before freeze: `test_v8f_oband_acc`, `test_v8f_otband` and `test_v8f_ego_dagger_compose` give 89 passed, 1 failed on that tree and 84 passed, 1 failed on the unpatched `202849da` export. The one failure is the same test on both (`test_flag_off_reproduces_the_unmodified_tip_bit_for_bit`: summary.json carries the T2 bank's `t2_bank_n`, `_p`, `_rows` outside its allowed key set, while weights, logs and counters match bit for bit), so the change adds no failure. The full suite runs in each run's launch gate (G-SUITE) on box 2. The probe reads the logged probe batch, not a batch with DAgger rows: the detach is applied at O's input whatever the batch, so this is stated, not tested.
4. **DAgger bank identity:** each arm's bank sha256 equals DAG-OACC's (s0 `197a68bb`, s1 `36c7aea9`), read back from the built run.
5. **Read controls** (the OTBAND read's): window identity per load; checkpoint sha256 against the bank manifest before and after each dump; T-head identity control 0 [0, 0]; top-K repeat difference 0.0 m.

## Log
* 2026-11-10 draft written by the Master Mind after the OTBAND read (H-V8F-OTBAND-1). No OBD run, argv or chain exists at this time. Box-2 ordering after the measured-speed (EGOWM) runs is the PI's call.
* 2026-11-10 same day, before freeze and before any run: OBD-NULL_rep added (a seed-only floor at this commit, because the banked DAG-ORC pair carries a trainer change and a 0.60 m spread at init s0); the launch package exists (`theory/obd/box2/`, UNIT 1,033 ok, dry run 136 ok, no box-2 contact); NULL parity moved to tensors; the acceleration term's purity probe named as a code precondition.
* 2026-11-10 FROZEN by the Master Mind before any OBD run: code of record `cdbff971fa6dad4d20b937fe8abda6cb9f622adb` named (the probe precondition of control 3 is met); NULL parity is read on tensors, so the chain launches with `OBD_PARITY_BASIS=tensors`. Box-2 ordering after the EGOWM runs stays the PI's call.
