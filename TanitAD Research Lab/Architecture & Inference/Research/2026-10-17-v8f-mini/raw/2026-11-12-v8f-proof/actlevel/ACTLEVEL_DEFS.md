# ACTLEVEL_DEFS: what `--w-act-level V` does, every decision with its file:line

Note: `F:/TanitAD-scratch/v8proof/PREREG_V8F_PROOF_ADD37_N6.md` sections 2 and 5 (lever, controls v1 to v5). Tree: `F:/TanitAD-scratch/v8proof/actlevel/work/tree_level` (a copy of `actcons/work/tree_patch`; the sha256 of the three changed files in the base were checked before the work: trainer `529e6a0d...`, act_cons `ad537369...`, test `356e84ba...`). Line numbers are of tree_level. Evidence class: MEASURED by reading source and by the checks in `checks/`, `MUTATIONS_ACTLEVEL.json`, `tests.log`; tier of any later read: T0 (diagnostic on logged frames).

## The lever

A TRAINING-ONLY term inside the ACTCONS block. No module, no parameter, no state-dict key, no config field. `act_cons_term(..., level=True)` also returns `level` and `n_level`; the trainer adds `V * level` to the total next to `W * cons`. Default `--w-act-level 0.0`: `level=False`, the code path and every float are the base tree's.

## The term (`stack/tanitad/models/act_cons.py`)

* Reference rollout with gradient only when `level` is true: `with torch.set_grad_enabled(bool(level)): ref, _, _ = st.roll_pairs(...)` (`:139-140`). With `level` false this is the former `torch.no_grad()` (same ops, no graph). The consistency read of the reference stays under `torch.no_grad()` (`:146-147`), so the consistency term uses the reference detached in both cases (test `test_consistency_term_is_unchanged_by_the_level_flag`: value and gradient equal with level on and off within 1e-5).
* Valid pairs = the consistency term's (`valid`, `:127`: label at frame s, gap 0 to 60, frame_ok at s, ego speed >= 2 x max). Level entry per pair and read step k in (5, 10): `level_label_mask` (`:95-110`): presence > 0.5, finite gap, `0 <= gap <= 60`, `frame_ok` at frame `s + k` (`f = s + k`, `:103`). `use = ok & valid` (`:156`). Labels at invalid entries are replaced by 0.0 before the subtraction so a NaN label never enters the graph (`torch.where`, `:106`).
* `e = |gap_reference(s, k) - gap_label(s + k)|` in metres with `gap_reference` = the lead head read of the reference states (`_gap_m`, `:88`, head parameters detached through `functional_call`, the same call as the consistency read) (`:161`). `level = sum(e * use) / n_level` (`:163`). `n_level = use.sum()`. No valid entry: `torch.zeros(())` with no graph and `n_level` 0 (`:165-166`; the all-rows-invalid early return `:132-133`).
* Start state under `torch.no_grad()` as before (`:124-125`); gradient reaches the predictor and the action embedding only (test, probe).
* Result dict: `term`, `n_valid`, `n_rows` as before; `level` and `n_level` only when `level` is true (test `test_level_off_returns_the_old_key_set_and_record`).

### Frame rule, verified from source (not assumed)

`HJEPAStack.roll_pairs` (`hjepa.py:956-969` of the pinned file) returns `pred[S, B, k, D]` for start s with `tgt = z[:, s + 1 : s + k + 1]`: rolled step j (0-based) is the state of frame `s + 1 + j`. The read slices of `act_cons.T_READS` are `[0:5]` and `[5:10]`: five states ending at step 5 and at step 10, i.e. frames s + 1..s + 5 and s + 6..s + 10; the head returns the gap at the LAST of the five (`_gap_m`, `o[:, 0, 1]`), so the read at k = 5 is the gap at frame s + 5 and at k = 10 the gap at frame s + 10. The same convention in `F:/TanitAD-scratch/v8proof/imaghead/imag_dump.py` (header lines 3-6: window frame 4 = t, rolled-out states from frame 4, the true future states at 1.0 s are frames 5..9 and at 2.0 s frames 10..14, read at the end frame of each five): its 1.0 s read is scored against the label at frame 9 = 4 + 5, the 2.0 s read at frame 14 = 4 + 10. Test `test_step_k_of_roll_pairs_is_frame_s_plus_k` pins the source facts; the planted test shows that labels one frame early or late do not give |c|.

## Trainer wiring (`stack/scripts/v8f_mini_train.py`), mirrored from `--w-act-cons`

| item | where |
|---|---|
| G-DVB entry `w_act_level` (kind `data:act_level`) | `:130` |
| flag `--w-act-level` (float, default 0.0) | `:306` |
| `bad_act_cons`: V finite and >= 0 refused otherwise; V > 0 with W == 0 refused (also when no record exists); declared == built (`rec["w_act_level"]`); V == 0 with a level key in the record refused | `:1280-1283`, `:1297-1301` |
| built record: `AC.build_record(..., a.w_act_level)` adds `w_act_level` and `level_reads` only when V > 0 (`act_cons.py:64-76`) | `:3312` |
| live probe: the level term's own backward (`run_level`) reports `level`, `n_level`, `level_grad_norm_predictor` (> 0), `level_grad_norm_action_embed` (> 0), `level_grad_norm_lead_head` (exactly 0.0), `level_grad_norm_state_projection` (exactly 0.0), `level_grad_abs_sum_outside_predictor_and_embed` (exactly 0.0), the all-invalid control (`control_all_invalid_level` exactly 0.0, no graph, `control_all_invalid_n_level` 0); a violation is an `AssertionError`, which the caller turns into `ZZDVB-REFUSEDZZ act-cons` | `:2076-2135` |
| selfcheck `levers.w_act_level` (only when V > 0, read from the built record) | `:3446-3447` |
| loop: `T["total"] = T["total"] + a.w_act_level * ac_r["level"]`, log keys `act_level`, `act_level_n`, not-applied guard `ZZACT-LEVEL-NOT-APPLIEDZZ`, run totals `lv_tot` | `:3659-3670`, `:3492` |
| summary keys `w_act_level`, `act_level_n_valid`, `act_level_mean` (only when V > 0) | `:3745-3747` |

With V == 0 no new key appears in the log, the selfcheck, the summary, the probe record or the build record (checks: `checks/a4_level_smoke.json` `V0_control`, tests).

## Controls built (note section 5)

* v2 (gradient): probe values above; test `test_level_gradient_reaches_predictor_and_action_embed_only`, `test_probe_reads_level_gradients_and_goes_red_when_the_head_is_attached`, all-invalid control `test_level_does_not_use_the_offset_and_all_invalid_control_reads_exactly_zero`.
* v3 (identity): `checks/v3_identity.py <old_tree> <new_tree>` (two subprocess trainers, CPU, tiny synthetic world, W = 0.001, 4 steps, every numeric key of every log row compared exactly, explicit `--w-act-level 0.0` too). Result `checks/v3_identity.json`: pass, max_abs_diff 0.0 over 4 rows x 37 keys (MEASURED here).
* v4 (planted value and mutations): `test_planted_level_reads_abs_c_and_the_frame_rule_is_s_plus_k` (kinematic predictor and head, labels = hand-computed reference read + c at frames s + k: level = |c| within 1e-5 for c = +2.5 and -2.5; the same labels one frame early or late: not |c|); `MUTATIONS_ACTLEVEL.json` (6 of 6 red, unmutated green before and after).
* v5 (log): smoke `checks/a4_level_smoke.json`: `act_level` finite in every logged row, `act_level_n` above 0 in 100 percent of the logged rows (4 rows on the tiny run; the pod gate reads this on real data).

## Departures from note N6 (each named)

1. **Valid pairs include the consistency term's speed bound** (ego speed at frame s >= 2 x max = 3.0 m/s). The note says "every valid (sample, start s) pair of the consistency term", which carries that bound; the level term therefore does not see standstill starts. No change made; named so the lead can decide.
2. **Level labels at invalid entries are set to 0.0 before the subtraction** and multiplied by the validity mask (a NaN label must not reach the graph). The value is the same as masking after.
3. **A start s + k beyond the window is not possible** with the proof window (F = 16: s + k <= 14), the label mask indexes frame s + k directly; a window shorter than 11 frames is already refused by `check_buildable`.
4. **The probe computes the consistency and the level gradients in two separate backward passes** (consistency term as before, then the level term alone) so that the level's exact zeros are read on the level alone; the consistency gradient is not changed by the level flag (test).
5. **`level` is a keyword argument of `act_cons_term` (default False)**, not a separate function, so that the reference rollout is computed once. The two existing source-count lines in `test_v8f_act_cons.py` (lines 268 and 379) now name the call with `level=a.w_act_level > 0.0`; nothing else of that file changed.

## REVIEW items for the lead

* R1: the level term makes the reference rollout carry a graph: memory and time per step rise (second backward pass through the 3 start rollouts). Not measured here (CPU, tiny model); the note's budget estimate says "a second backward pass".
* R2: the level labels (frames s + k, k = 5, 10) are never the start frames (starts 0, 2, 4) on the 16-frame window, so the level term reads a label the consistency term does not (it reads frame s only).
* R3: the level is not divided by anything else than the entry count: its scale is metres, the consistency term's scale is metres too; V = 0.001 and 0.01 against W = 0.001 are as the note's arms table says.
* R4: the identity check compares loss and every numeric log key; it does not compare weights after the steps (the weights follow from the same floats; `checks/a4_level_smoke.json` compares weights of V versus V = 0 and reads a non-zero difference 3.9e-4).
* R5: test `test_the_loop_lines_of_the_level_exist_once` and the smoke mutation are what turn red when the add line `V x level` is removed (m1); no pytest runs a training step.
* R6: new test file `stack/tests/test_v8f_act_level.py` is a new pin; `test_v8f_act_cons.py` changed (hash differs from the one in `arms_actcons_c0.txt`).

## Arms: tokens

* argv (appended to the arm line of the dose arms, after `--w-act-cons 0.001`): RLVa `--w-act-level 0.001`, RLVb `--w-act-level 0.01`.
* expectation strings (gate_extra renders whole floats as ints; 0.001 and 0.01 are not whole, so they render as the float repr, as `levers.w_act_cons=0.0001` does in `arms_actcons_c0.txt`): `levers.w_act_level=0.001` and `levers.w_act_level=0.01`; together with `levers.w_act_cons=0.001 levers.act_cons_max=1.5`. Measured here: the selfcheck of the smoke run with `--w-act-level 0.01` holds `levers.w_act_level` = 0.01 (JSON). The gate parser itself was not run here.

## Not verified

No real data, no pod, no real checkpoint: the tiny synthetic world only. The magnitude of the level on a trained base (here 17 to 23 m on an untrained tiny head), the share of valid level entries on real clips, the cost per step and the pod dry gate are first seen on the pod. The check script `v3_identity.py` was run with the old tree `actcons/work/tree_patch` only.
