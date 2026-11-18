# BUILD_TFAR: the T-reads-far arm, built and NOT launched (2026-11-18, build agent for the MM)

Card: `F:/TanitAD-scratch/strat1/cards/treadsfar/PREREG_V8F_TFAR.md`, FROZEN, sha256 `816d4357041fe7e7c03022d768900a67ec80b9bc8b633d85cd0d399b178cb4cd` (re-hashed at the start of this build, equal to `PREREG_V8F_TFAR.sha256`). Build list: `BUILD_GAPS_TFAR.md` (same folder).
Base: `aedf8716` (CKPT120 fix3). Tree: `F:/TanitAD-scratch/strat1/tfar/tree/` = `git archive aedf8716 stack taniteval` + the 12 files below (`tfar.patch`, LF, 12 files, 317,568 bytes, against `aedf8716`; sha list `pod/tfar_tree.sha256`). Nothing committed, nothing pushed, no GPU, no pod touched.
Not merged: `4dedac45` (S1d kind) and `f95de67a` (NAVRB2). The patch touches only `v8f_mini.py` and `v8f_mini_train.py` of the existing files, so it is conflict-free with the S1d commit.

## 1. Manifest (12 files, all in `tree/`, all under the MM's landing)

| file | status | what |
|---|---|---|
| `stack/tanitad/models/v8f_mini.py` | changed | the lever: constants `T_FAR_*`, `NAV_STRAT_T_MODES`, key prefixes in `NEW_LEVER_KEY_PREFIXES`; 5 config fields (declared, validated); `e_ck_far = Linear(4, d)` and `e_nav_s_t = Linear(5, d)` built LAST, zero-initialised, the init draw forked (building them consumes no draw of the global stream); `run_t(..., far=None, far_ok=None)`: far term ADDED to T's checkpoint token only (O's `_ck_cond` call untouched), route-text term ADDED to T's nav token; `_forward`: ONE `torch.rand(B)` draw shared by the 3 checkpoints and the far point, S's far point detached under `detach_upper`, re-expressed with `transform_ckpt(pose_s, pose_t)`, variants zero / roll (train only); `set_t_far_eval` zero / set / oracle (evaluator-only model attribute, never a batch key); groups `TFAR` / `NAVST`; `built_levers()` |
| `stack/scripts/v8f_mini_train.py` | changed | 7 flags + 7 `FLAG_CHECKS` (G-DVB) entries; `make_config`; `bad_tfar` (refusals below); optimiser groups `tfar` / `tfar_nd` / `navst` / `navst_nd` at lr x mult; G-LIVE declares `e_ck_far.weight` dead for TZ (and requires the exact zero); `tfar_probe` (G-LIVE, details below); `ckpt120_probe`: T-ISO-FAR asserted only when the layer is off, T-reads-nav_s allowed when TN; init-ckpt fresh-key rule; WM-only default list; `param_names` (registration order) written into a lever-ON checkpoint only |
| `stack/scripts/tfar_check.py` | new | study checker (imports `ckpt120_check` and `strat1_check`, edits neither): kinds TF R3 TZ TN TFnt TFd TFc H2c H2, seeds s0 s1 rep s10 s11; mints / checks `tfar-gate-token-v1` bound to commit, argv sha256, every data sha256 (RC-A120 `0f122570...`, nav_s `0f7431a1...`, nav, init ckpt), tree manifest, trainer, checker, G-SUITE result; `gate`, `post`, `geval`, `hyg`, `suite`, `mint`, `run` |
| `stack/scripts/tfar_reach.py` | new | K-REACH-e: Adam-state zero-grad gate scoped to `s_far_out.*`, `e_ck_far.*` (TZ: bias only), `e_nav_s_t.*`, `e_nav_s.*`; names ids from the checkpoint's `param_names` (exact); positive control + a planted-dead negative control per new weight |
| `stack/scripts/v8f_mini_eval_tfar.py` | new | T0 wrapper on `v8f_mini_eval_far.py` (both pins checked): `--far-token real|donor|zero|oracle`, dumps `eval_<tag>_tfar.npz` (T lat / lon / goal logits, `t_far_in`, `t_far_ok`, `donor_index`, `oracle_valid`) |
| `stack/scripts/v8f_t1_eval_tfar.py` | new | T1-kin wrapper on `v8f_t1_eval_far.py`: refusals vs cfg; `t_far_in` / `t_far_ok` per tick with `--dump-ticks` (equals S's raw far point re-expressed into T's frame); `v8f_t1_eval.py` is not touched |
| `stack/tests/test_v8f_tfar.py` | new | 30 tests: pins, off = `aedf8716` bit for bit incl. `built_levers()` keys, step-0 identity and RNG, gradients with exact-zero controls, T-ISO-UP / T-ISO-O / K-KEYS / G-CLOCK-TFAR (planted poses), TF / variants, TN, G-DVB refusals, 5 end-to-end trainer arms with K-REACH-e on the real checkpoint, mutations |
| `stack/tests/test_tfar_check.py` | new | 15 tests (many parametrised): argv per arm, good selfcheck per arm, every probe key / argv lever / built-lever mutation RED, post identity, hyg, geval, token + runner refusals |
| `stack/tests/test_tfar_reach.py` | new | 9 tests: healthy, dead layers, exact-zero weights (TZ), expectations both ways, blind-gate and positive-control fail-closed, CLI |
| `stack/tests/test_tfar_eval.py` | new | 6 end-to-end tests on the synthetic world: T0 variants, refusals, layer-off run bit-identical to the far wrapper, T1 `t_far_in`, TN |
| `stack/tests/v8f_mini_tip_snapshot_tfar.txt` / `.sha256` | new | the `aedf8716` `v8f_mini.py`, the reference of the bit-identity tests (sha256 `a77265ce...`, equals `git show aedf8716:...`) |

## 2. Card requirements to where they are checked

| card | where |
|---|---|
| zero-init 1,280-parameter layer, off = `aedf8716` bit for bit | `test_defaults_off_is_the_tip_bit_for_bit_with_the_h2_levers_on`, `test_layers_on_at_step_0_...`, `test_building_the_layers_consumes_no_draw_of_the_global_stream` |
| a new lever key only when the lever is on (the NAVRB2 `nav_rb` lesson) | `built_levers()` emits `t_far*` / `nav_strat_t` only with the layer built; asserted in `test_v8f_tfar.py` and in `tfar_check.gate_problems` (a key with the lever off is a problem) |
| T must not train S: gradient exactly 0.0 | `test_T_ISO_UP_...`, `tfar_probe` (non-zero weights, positive control on the new layer); mutation: cutting the stop-gradient turns it RED |
| G-DVB entry per flag, refusals | `FLAG_CHECKS`, `bad_tfar`: `--t-far on` without `--s-far on` / without `--detach-upper`; a layer built iff declared; non-default input / variant / tf / lr-mult with the layer off; lr-mult vs the group's lr; `--nav-strat-t on` without `--nav-strat on`; WM-only |
| G-LIVE: gradient on the new path, exact-zero control (token off), scrambled-far-point must-fail arm | `tfar_probe` (`control_far_weight_grad_input_zero == 0.0` with the bias alive; `control_nav_s_t_weight_grad_nav_s_zero == 0.0`); the must-fail arm is the R3 kind (`--t-far-variant roll`) and the TZ kind, minted and read by `tfar_check.py` (kinds R3, TZ) |
| K-RNG, G-CLOCK-TFAR, K-KEYS, T-ISO-O | `tfar_probe` + tests (mutations: second Bernoulli, skipped transform) |
| post-training zero-grad gate | `tfar_reach.py` (planted HEAD-DEAD controls), run per arm by `pod/tfar_chain.sh` |
| ORACLE diagnostic row | `v8f_mini_eval_tfar.py --far-token oracle` (RC-A120 eval split, flagged windows) |
| mutations: delete the add, non-zero init, cut the stop-gradient, drop the key from the readback | `test_mutation_deleting_the_add...`, `..._a_nonzero_init...`, `..._cutting_the_stop_gradient...`, `..._dropping_the_key_from_the_readback...`; also second draw, skipped transform, make_config line, optimiser group |

`tfar_probe` keys read by the gate (each with its required value): `zero_init`, `step0_identical`, `rng_state_identical` (True); `far_weight_grad`, `t_iso_up_positive_control` (> 0); `control_far_weight_grad_input_zero`, `t_iso_up_s_grad` (EXACTLY 0.0); `control_far_bias_grad_input_zero` (> 0); `t_reads_far`, `t_iso_o_ck_token_identical` (True); `clock_max_abs_err` (<= 1e-6); `keys_read_labels` (empty); TN: `nav_s_t_weight_grad` (> 0), `control_nav_s_t_weight_grad_nav_s_zero` (EXACTLY 0.0).

## 3. Where the build differs from BUILD_GAPS_TFAR (read these before landing)

1. **Replays.** BUILD_GAPS names `g_box_overfit._build_model` and `refcv7_loader` as needing a mirror. In this tree `g_box_overfit.py` contains no `v8f_mini` / `V8FMini` reference (count 0) and `refcv7_loader.py` is not in the tree: they build the refcv models, not v8f-mini. Nothing was mirrored. G-EVAL is `tfar_check.geval` (strict load through the evaluator build path, a cfg that drops `t_far` / `nav_strat_t` is REFUSED). UNVERIFIED on the pod.
2. **Init draw forked.** The card says "no RNG draw"; `nn.Linear(...)` itself draws, so both layers are built inside `torch.random.fork_rng`. The global stream after the build is the off model's (tested).
3. **`ckpt120_probe` edited.** Its T-ISO-FAR clause asserts the far point does not reach T. With the layer on and non-zero (a resumed run) that is false by design, so the assertion applies only with `--t-far off`; the same for T reading `nav_s` with TN. T-ISO-UP / T-ISO-O replace it in `tfar_probe`. At a fresh start the old clause still reads True (zero weights).
4. **`param_names` in the checkpoint** (lever-ON arms only): the shape alignment of `zerograd_gate` is ambiguous for the [d] biases; a flag-off checkpoint keeps its key set.
5. **Orchestration of TZ.** `e_ck_far.weight` is declared dead in G-LIVE (required to read EXACTLY 0.0, bias alive), and K-REACH-e expects the bias only.
6. **`tfar_check.py` renders H2_s10 / H2_s11** (comparators, no TFAR flag named) by widening CKPT120's seed table for one call only (its tests pin the card's seeds, and its own file is unchanged).

## 4. NOT built (stated plainly)

* **The reader** (`tfar_read.py`: the FB stratum from labels, D-LON-1/2, D-TAC-1/2, T-READS, R3 / TZ must-fail rules, floors, the 24-cell table, the STRATEGIC reuse, the echo share, the 4-seed rule). The task named the eval wiring; the wrappers above produce every array the reader needs (T0 variants, T1 `t_far_in`, `eval_<tag>_tfar.npz`), but the reader itself is a separate deliverable beside the CKPT120 reader (`F:/TanitAD-scratch/strat1/read/ckpt120/c120_read.py`).
* **TFa** (the appended-token form, a card branch): needs its own model code and its own card step.
* No pilot (200 steps), no mint, no G-SUITE on the pod, no pod copy: the kit is staged only on the dev box.
* The matched-continuation branch kinds TFc / H2c render and gate, but need the banked H2 checkpoints at `<inputs>/tfar/h2_init_<seed>.pt` (not staged).

## 5. Tests and the regression

* First pass of the four new files (partial run killed at ~138 tests, 0 failures printed; box under heavy load, F: I/O): `tmp/new3.log`. Earlier passes found and fixed 7 test or probe defects (a probe that ran on an already-woken model, an assertion order, a stale text pin).
* Regression vs `aedf8716`: `tmp/regress.py` (145 files: every test naming `v8f_mini` / `V8FMini` / `launch_gate` / `ckpt120` / `strat1` / `tfar` / `v8f_t1_eval` / `zerograd`, the 4 new files first), serial, CPU, `wait_commit.py 6 120` before each; failing ids are re-run on a pristine `aedf8716` export. Result file `tmp/regress_summary.json`, last driver line `ZZ REGRESS DONE n_regressions N`.
  **RESULT: PENDING at hand-back** (the driver was running; see `HANDOFF.md`). Bar: n_regressions == 0.
* Text pins I kept intact on purpose: the `run_t` call strings in `test_v8f_flat.py`, the `decay` / `nodecay` lines in `test_v8f_ckpt120.py`, the `make_config` s_far line; the mutation tests that pin trainer text should be re-read if the trainer is edited.
* Before landing: run `test_text_encoding_is_explicit` (memory), and the ~69 launch_gate-reading tests (they are in the regression list).

## 6. Pod kit (staged in `tfar/pod/`, `bash -n` ok, LF, NOT copied to the pod, NOT launched)

* `tfar_gsuite.sh`: G-SUITE for the landed commit on a tree = `/root/strat1/new11` (aedf8716) + the files of `tfar_tree.sha256`; two waves, memory guard 30 / 52 GB; result `gsuite_result_tfar.json`; sentinel `ZZ TFAR GSUITE DONE|FAIL ZZ`. MM sets `COMMIT` and `NFILES` (12 now).
* `tfar_chain.sh`: arms H2:s10 H2:s11 (skipped when the CKPT120 chain trained them) TF:s0 TF:s1 TF:rep R3:s0 TZ:s0 TN:s0; mint -> run -> post -> `tfar_reach.py` -> `ZZ ARM <tag> DONE ZZ`; ends `ZZ STRAT1 TFAR DONE|FAIL ZZ`. MM sets `COMMIT`. `ARMS` overridable for the conditional arms.
* `queue_tfar.sh`: waits on `PRED_LOG` / `PRED_PATTERN` (the end line of whatever runs before TFAR, set by the MM; unset = refuse), then `flock /root/gpu_queue.lock` the chain.

## 7. MM steps

1. Read `tmp/regress_summary.json` (`n_regressions` 0 and the base-failure list); decide on `tfar.patch`.
2. Land `tfar.patch` on a branch from `aedf8716` (`agent/v8f-tfar-<date>`), set `COMMIT` / `NFILES`, copy `pod/` and the tree files to the pod (sha256 list), run `tfar_gsuite.sh`.
3. PI go on Q1-Q3, then set `PRED_LOG` / `PRED_PATTERN` and launch `queue_tfar.sh`. The reader (section 4) is a separate work item.

## 8. Deletions (own temporary data)

`F:/TanitAD-scratch/strat1/tfar/tar/t.tar` (465,776,640 B, re-creatable git archive of `aedf8716`). The rest of `tmp/` is listed for deletion in `HANDOFF.md` once the regression has been read.
