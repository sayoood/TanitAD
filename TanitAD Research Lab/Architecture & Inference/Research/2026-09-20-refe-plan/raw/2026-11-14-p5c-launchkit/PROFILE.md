# PROFILE.md: the P5-C launch-gate profiles (`refe/launch_gate.py`, `refe/gate_checks.py`)

Card = `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c/PREREG_P5C.md` (frozen sha256 in `PREREG_P5C.sha256`), amendment `PREREG_P5C_A1.md`. Line numbers are of `PREREG_P5C.md`. Evidence class of every value: **card value = INHERITED from a frozen card** (the card itself cites its source); the check column is **MEASURED by `test_p5c_profile.py` / `test_traj_label_table.py`** (logs in `logs/`).

## Shape
- **Three profiles, one per arm**, cloned from `refe_p6` (so `refe_p7` is untouched): `refe_p5c` = **C1**, `refe_p5c_c0` = **C0**, `refe_p5c_c0r` = **C0r**. The arm IS the profile, so C1 launched without its table flag cannot read as a clean C0 (the flag's default is OFF).
- `p5c_argv(arm, base="p7")` is the registered argv template; `p5c_refusals` makes the argv's FLAG SET and every non-path lever equal it. `argv(C1) - argv(C0) = {--traj-label-table, --out}` and `argv(C0r) - argv(C0) = {--seed, --out}` (card :375), asserted by `test_p5c_profile.py argv_delta` and RED on any other difference.
- B-P6 fallback (card 2.1): `p5c_profile(arm, "p6")` (no `--ego-yaw-table`, base pin = P6's `968a469d...`). Built and tested, **NOT registered** in `PROFILES` (register it only if P7's verdict is REFUSED/INVALID).
- **Pins the MM fills** (all `None` in the source; the profile REFUSES while unfilled, test `pins_refuse`): `P5C_BASE_SHA256`, `P5C_LABEL_TABLE_SHA256`, `P5C_RANK_POLICY` (top of the P5-C block in `launch_gate.py`).

## Levers
| argv lever | C1 | C0 | C0r | card line | G-DVB / gate check | RED arm |
|---|---|---|---|---|---|---|
| `--backbone vitl16 --scorer-sees-trunk --nav-mode command --yaw-wrap off --labeller online --label-chunks 2 --score-w 0.1` | same | same | same | :113-114 (as P7) | `required_flags` (refe_p6) + template equality; existing G-DVB rows (`nav_mode`, `yaw_wrap`, `labeller`, `label_chunks`, `score_w`) | each moved in `levers` |
| `--init-from <base>` (DATA flag) | named | named | named | :106, :384 | existing `init_from` row (declared = built `warm_start(path)`) + NEW `p5c_init_rows`: file sha256 = `init_from_sha256` pin | wrong sha256 -> red row (`dvb_policy_pin`); flag dropped -> `levers` |
| `--epochs 0 --steps 2000 --epoch-unit scenes` | same | same | same | :109, :385 | existing `steps` row (re-derived) + cosine `T_max`; G-CLOCK `expect_schedule {epochs 0, steps 2000, epoch_len 92,896}` | `--epochs 25` / `--steps 1999` in `levers` |
| `--lr 5e-5` | same | same | same | :108, :386 | existing `lr` row (`opt.param_groups[0]["lr"]`) | `--lr 2e-4` in `levers`; profile registers 5e-5 (mutant) |
| `--batch 4 --accum 64 --weight-decay 0.01 --amp bf16 --tf32` | same | same | same | :110-111 | existing `batch`/`accum`/`weight_decay`/`amp`/`tf32` rows | `levers` |
| `--w-field 0.0` | named | named | named | :115 (P7 names it) | `required_flags` | -- |
| aux head OFF: no `--aux-drv*`, no `--aux-drv-labels` | forbidden | forbidden | forbidden | :115, :387 | `forbidden_flags += AUX_HEAD_FLAGS` | `--aux-drv`, `--aux-drv-labels` in `levers`; mutant drops the forbid |
| `--ego-yaw-table` (B-P7) | named | named | named | :114, :389 | `required_flags` (P7_YAW_REQUIRED) + data flag (token fingerprints the file) + existing `ego_yaw_table` row | dropped in `levers`; B-P6 forbids it (`b6`) |
| `--seed` | 0 (named) | 0 (named) | **1** | :116; C0r :62, :375 | `required_flags` + existing `seed` row (`torch.initial_seed()`) | wrong / dropped seed in `levers`; C0r-seed-0 mutant in `argv_delta` |
| `--traj-label-table <T>` (DATA flag) | **required** | **forbidden** | **forbidden** | :118, :383 | existing `traj_label_table` row (every TargetBank construction received it) + file sha256 = `traj_label_sha256` + NEW row: table `meta.rank_policy` = `traj_label_rank_policy` | table dropped at a call site -> `label_table_not_passed` (test_traj_label_table.py); policy flipped -> reader refuses (`rank_policy`) |
| `--targets .../train_grow_cmd_3c_p7` (DATA flag) | named | named | named | :112, :376 | G-CLOCK: 150,603 tuples, `targets_sha256 92af8d66...` (P7_BANK_TUPLES / P7_TARGETS_SHA256, shared with refe_p7), `--targets` bank name registered; content recomputed (`required` because `p5c_rules`) | other bank in `levers` |
| `--images --calib --metric-cache` (DATA flags), `--label-workers`, `--workers 2 --log-every 1 --ckpt-every-min 20 --compile` | named | named | named | :113, :117 | presence + non-path value equal to the template (`label-workers` is the pod's own derivation, presence only) | -- |
| `--out <dir>` | `.../p5c_c1` | `.../p5c_c0` | `.../p5c_c0r` | :388 | no `ckpt_last.pt` in `--out` at MINT time (`p5c_refusals`; not applied to `verify`, so a supervisor relaunch is unaffected) | `fresh_out` |
| `--resume` | allowed, optional | same | same | :388 | the supervisor's relaunch (same as P7's ARGS) | -- |

## Not carried from `refe_p7` (decisions in HANDOFF.md)
- G-RESERVE (it audits five inputs including the head's label store, `--aux-drv-labels`, which P5-C does not have; the bank is P7's reserve-free bank pinned by content).
- P7's `test_session_reserve.py`, `test_p7_integration.py`, `test_p7_chain.py` in G-SUITE (they test refe_p7's own chain/reserve). Added instead: `test_traj_label_table.py --fast` (`ZZTRAJLABEL_PASS`), `test_p5c_profile.py --fast` (`ZZP5CPROF_PASS`), plus refe_p7's `test_p7_head_wiring.py --static` and `test_ego_yaw.py --fast`.

## Card G-LIVE / post-training items: built by HANDOFF_2 (2026-11-14)
| card | where | what is checked | RED arms (all tested) |
|---|---|---|---|
| 10.2 (f) warm-start identity | `p5c_gate.step0_facts` (taken in `instrumented_run`'s build capture at the trainer's first `zero_grad`: warm start applied, nothing trained), `judge_warm_identity`, `gate_checks.p5c_live_rows` | every `state_dict` tensor (dtype, shape, sha256 of the bytes) == the base file's (read as `train.warm_start` reads it), optimiser state empty, no `ckpt_last.pt` in `--out`, the base FILE's sha256 == the profile pin `init_from_sha256` | gate arms `warm_not_loaded`, `warm_perturbed`; wrong checkpoint; unpinned base; stateful optimiser; resume file; missing / extra tensor |
| 10.2 (h) lever isolation | `p5c_gate.isolation_facts`, `judge_isolation`; argv half = `test_p5c_live.py argv_isolation` | argv(C1) - argv(C0) = {`--traj-label-table`, `--out`}; on real `TargetBank` rows (live train bank vs a flag-off shallow copy) the items are equal in every slot except item[3]; one model forward is identical; score loss and its gradient are bit-identical; the REAL `wta_loss` differs; the control (raw label on the same forward) reads EXACTLY 0.0. C0 / C0r: the live bank must carry NO table | `label_lookup_removed`, `label_table_not_passed`, ego slot moved with the table, forward differs, control non-zero; 4 source mutants of `p5c_gate.py`, 1 of `train.py` |
| 10.4 G-ADAM | `refe/p5c_postgate.py` | every registered optimiser tensor has state: step == 2,000, finite NON-ZERO `exp_avg_sq` (card) AND `exp_avg` (brief), state covers EXACTLY the trainable count; frozen trunk bit-identical to the base; traj head moved; C0's traj head differs from C1's; the probe's own controls (exact 0.0 / no state) | state deleted, moment zeroed, wrong step, count +/-1, moved trunk tensor, untrained head, identical arms |
**Card vs brief:** the card says non-zero `exp_avg_sq`; the brief says first moment. The gate requires both. **Finding (MEASURED, `test_p5c_live.py`):** the WTA loss is an L1, so the gradient w.r.t. a proposal is a sign: a small label shift changes the loss VALUE but may leave the proposal gradient bit-identical; (h) therefore judges the traj-loss value (diff != 0), not its gradient.
The three new test scripts are in `refe_p5c*` G-SUITE (`test_p5c_live.py --fast`, `test_p5c_postgate.py --fast`, `test_p5c_chain.py`); `test_p5c_profile.py` pins that.
