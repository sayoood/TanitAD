# BUILD_NAVRB2: the ROUNDABOUT channel (part 1 of 2; part 2 = HANDOFF.md steps 1-5)

Built 2026-11-17 on `4dedac45` (branch `agent/v8f-strat1-s1d-20261117`; card section 0.1 says why not the arch-inf tip). Nothing committed, nothing pushed, no GPU, no pod, nothing under `E:/Projects/TanitAD/stack`; `launch_gate.py`, `stack/tanitad/models/__init__.py`, `v8f_t1_eval.py`, `v8f_mini_eval.py` untouched. Card: `PREREG_V8F_NAVRB2.md` (sha256 in `PREREG_V8F_NAVRB2.sha256`, written before any arm could train; no-fog ruling added 2026-11-17).

## 1. Patch and files
`navrb2.patch` (sha256 fa67441e..., LF, 7 files) against `4dedac45`: modified `stack/tanitad/models/v8f_mini.py`, `stack/tanitad/data/v8f_mini_data.py`, `stack/scripts/v8f_mini_train.py`; new `stack/scripts/build_navtext_nav_rb.py`, `stack/tests/test_v8f_navrb2.py`, `stack/tests/v8f_mini_tip_snapshot_navrb2.txt` / `.sha256`. `git apply --check` PASS on a 4dedac45 export; applied files equal `tree/` modulo CR (autocrlf on this box). Regenerate: `git diff --no-index orig/<f> tree/<f>` per file (prefix fix as in this run).

## 2. Table (G-CLOCK-RB)
`table/navtext_nav_rb.npz` sha256 `f4c4ab7a8abf2498e532d648874df2549b7b83292b31766ad80e581d009bcebc` (single copy). Inputs: `read/inputs/navtok_table.npz`, `navtext_nav.npz` (alignment), `join_eval141.npz`, `eval_grid_windows.json` (sha256 in the report). MEASURED:
* train 29,836 of 461,518 steps on (433 of 4,572 clips); 0 differ from the NAVRB rule; every used row has age >= 0.
* eval 1,334 of 14,237 (17 clips); 235 grid windows / 16 episodes at NOW (= `RESULT_NAVRB.md`).
* regression arm (state read 5 steps = 1 s ahead): FAILS on both splits (train 1,892 steps differ, 27,591 rows used before their time; eval 76 / 1,254; 236 windows). The builder refuses to write when the arm passes or the true arm fails.

## 3. Reachability (file:line in `tree/stack/`)
| hop | site |
|---|---|
| argparse | `scripts/v8f_mini_train.py:218` (`--nav-rb` and the 4 flags after it), FLAG_CHECKS `:124` |
| config | `make_config` `:470` -> `V8FMiniConfig.nav_rb` (`tanitad/models/v8f_mini.py:266`) |
| data | join `train.py:3489` `D.nav_rb_arrays` (`tanitad/data/v8f_mini_data.py:375`); window item `:799`; collate `:850` |
| model | `e_nav_rb` built `v8f_mini.py:912` (zero, forked RNG); `_nav_tok` `:917`, called in `run_s` `:996`, `run_t` `:1014`, `run_o` `:1040` |
| loss / backward | the S, T and O terms through the nav token; optimiser group `navrb` `train.py:3723`; G-DVB `bad_navrb` `:1772`; G-LIVE `navrb_probe` `:1819` and live_check NAVRB group |
(MEASURED by grep in `tree/` at hand-back; the numbers move with any edit.)

## 3b. Detected information: no-fog ruling (PI 2026-11-17, "Drop fog for now")
The card's section 5.2 applies it in the read only: weather on clear / rain / snow, fog windows unscored with n printed, fog logit masked before the argmax. The trainer and label table are unchanged (S1R keeps the banked table, parity with S1). The reader (part 2, with the evaluator wrappers) implements the mask.

## 4. Tests (`tests/test_v8f_navrb2.py`, CPU, `CUDA_VISIBLE_DEVICES=-1`, tree on PYTHONPATH, `tanitad.__file__` inside `tree/`): 28 passed
* Z0 flag off == the 4dedac45 model (state dict, init, loss terms, gradients, forward_infer bit-equal); mutation (channel built unconditionally) RED.
* Z1 flag on at zero init == off (every output and term), global RNG unchanged; mutation (no fork_rng) RED.
* Z2 `e_nav_rb.weight` gradient > 0 on the channel, EXACTLY 0.0 on the zero channel; a trained weight moves forward_infer; mutation (forward term deleted) RED; a built channel with no batch key is a KeyError.
* Key guard admits `nav_rb` only for a built model; mutation (guard line deleted) RED.
* Data: sha / alignment / value refusals, zero variant, item rows equal the nav's frames.
* Builder: hand-rule values, G-CLOCK-RB with its regression arm, no read after NOW (garbage probe; future mutant RED), misalignment refused, no situation import and no ego word in the rule; real table reproduces 1,334 / 235 / 16 and a wrong expected count is refused.
* Trainer end to end (synthetic): N1R selfcheck PASS, probe numbers, NAVRB group grad > 0, weight trained off zero, group lr_mult 10, checkpoint records; N1Z keeps the weight exactly 0.0 (declared dead, NAVRB grad 0.0); an off arm keeps its key sets; G-DVB refusals (off with a named lever, wrong sha, ego perturbation); mutations (join line deleted, item rows dropped) RED.
* DAgger: `dagger_probe` passes with `nav_rb` and the swap holds it as the same object; mutation (partition removed) RED.
* **Existing suites: NOT RUN (UNVERIFIED).** HANDOFF step 3.

## 5. MM steps (after part 2)
1. Freeze the card (sha256 recorded) and rule on card section 0 (base commit, lr-mult 10, cost).
2. Commit by plumbing on a new branch from `4dedac45` (e.g. `agent/v8f-navrb2-20261117`): hash-object the `tree/` blobs (LF), scratch index, revert guard, CAS update-ref, push, `ls-remote` check.
3. Pod G-SUITE on that commit (memory headroom first), compare against 4dedac45.
4. Ship `navtext_nav_rb.npz` + report to the pod inputs; mint each arm; chain under `flock /root/gpu_queue.lock` (HANDOFF step 4).
Cost: 4 arms about 0.63 L40-h (E); conditional arms in card 4.2 go back to the PI.

## 6. Deletions (own temp, literal paths)
* `F:/TanitAD-scratch/strat1/navrb2/tree` (first partial export of 5e7083d8, 95,122 KB, re-creatable from git), before re-export.
* `F:/TanitAD-scratch/strat1/navrb2/tmp` (pytest basetemp, TMP, apply check; 64,313 KB).

## 7. Manifest (nothing in the repo; the MM stages)
| artifact | where |
|---|---|
| card + sha256 | `F:/TanitAD-scratch/strat1/navrb2/PREREG_V8F_NAVRB2.md`, `.sha256` |
| patch | `F:/TanitAD-scratch/strat1/navrb2/navrb2.patch` |
| work tree (LF) | `F:/TanitAD-scratch/strat1/navrb2/tree/` |
| originals | `F:/TanitAD-scratch/strat1/navrb2/orig/` |
| table + report (single copy) | `F:/TanitAD-scratch/strat1/navrb2/table/` |
| patch scripts | `F:/TanitAD-scratch/strat1/navrb2/patch_model.py`, `patch_trainer.py` |
| handoff | `F:/TanitAD-scratch/strat1/navrb2/HANDOFF.md` |
All on F: only.
