# T-EGOCUT trainer lever: HANDOFF (written 2026-10-27 ~06:46, build agent)

## State
Lever BUILT and TESTED in `E:/Projects/TanitAD-artifacts/egocut_build/tree/` (an LF `git archive b43c7a9a` export, pruned). NOT landed, NOT launched, nothing committed. Parked unless DAgger (p = 0.1) reads NO EFFECT or HARMS.
Tests on the final bytes: `test_v8f_tegocut.py` 56 passed; `test_v8f_dagger_train.py` 97; `test_v8f_mini.py` 29; `test_v8f_ego_perturb.py` 82; the four widened additions-only pin suites, selected tests (saved-pre-change / 200-step / pre-amendment): 22 + 8 passed (flat, mixo, oacc, window_start; whole suites NOT run).

## The detached job (mutation proof)
* PID 79132 (cmd.exe, WMI-created, launched 06:43:59). Script: `tree/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-27-tegocut-train/mutate_tegocut.py` (55 mutants, anchors checked exactly-once, texts compile).
* Log: `egocut_build/logs/mutate.log` (KILLED / SURVIVOR per mutant, final line "N of M mutants killed"); `logs/mutate.out`; end marker `logs/mutate.done` (rc 0 = all killed and R77 clean). Work copy: `egocut_build/mut` (delete after).
* Expected runtime ~50-70 min (5 group baselines + 55 mutants, one pytest at a time).
* The three the brief names: M01 (swap line deleted), M02 (flag defaults on; M02b config default), M04 (c trainable).
* A SURVIVOR is a test gap: add a test to `test_v8f_tegocut.py`, re-run `--only <name>`; a "(baseline red)" entry means a group baseline failed (environment, not the mutant).

## R77
`egocut_build/R77_SHA256_before.txt` = sha256 of the 10 deliverables taken BEFORE the proof. After the job: `cd tree && sha256sum -c ../R77_SHA256_before.txt` must say OK for all, and `mutate.log` must end "delivered files changed during the run: none".

## Next step (landing, when the PI/Master Mind says the arm runs)
1. Read `mutate.log`; if all killed, copy the 10 deliverables (manifest below) into the repo via the coordinated lander. `egocut_build/PATCH_tegocut_vs_b43c7a9a.diff` is the diff of the 6 modified existing files against the tip.
2. The saved pre-change refs (`pre_tegocut_*.py.ref`) must be the TIP at the time of landing if the trainer/model moved since b43c7a9a (the default-off identity test compares against them).
3. K-BUILD / K-DATA / K-TOOL / the reader / the chain stages from PREREG section 6 are NOT part of this build.

## Manifest (all under `egocut_build/tree/`)
* `stack/tanitad/models/v8f_mini.py` (T_EGO_MODES :131, `tensor_sha256`, `check_t_ego_c`, cfg field :210, buffer :637, `_t_ego_const`/`set_t_ego_const` :666/:673, the ONE run_t line :727, `_load_from_state_dict` :868, `built_levers` :972)
* `stack/scripts/v8f_mini_train.py` (FLAG_CHECKS :96, `--t-ego` :156, `bad_t_ego` :491, init record :592, `read_init_t_ego_c` :601, gate call :649, `t_ego_probe` :1224, main wiring :1722/:1777/:1798)
* `stack/tests/test_v8f_tegocut.py` (new), widened pins in `stack/tests/test_v8f_{flat,mixo,oacc,window_start}.py`
* `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-27-tegocut-train/{pre_tegocut_v8f_mini.py.ref, pre_tegocut_v8f_mini_train.py.ref, mutate_tegocut.py}`
* Support only (exports of the tip, not deliverables): the rest of `tree/`, `logs/`, `run_mut.cmd`.
