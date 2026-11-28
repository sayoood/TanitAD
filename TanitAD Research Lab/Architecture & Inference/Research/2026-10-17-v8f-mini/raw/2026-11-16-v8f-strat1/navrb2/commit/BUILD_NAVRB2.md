# BUILD_NAVRB2: the ROUNDABOUT channel (parts 1 and 2)

Built 2026-11-17 on `4dedac45` (branch `agent/v8f-strat1-s1d-20261117`; card section 0 says why not the arch-inf tip). Nothing committed, nothing pushed, no GPU, no training. Pod: only `/root/strat1/navrb2/` written (5 files, section 6). Nothing under `E:/Projects/TanitAD/stack`; `launch_gate.py`, `stack/tanitad/models/__init__.py`, `v8f_t1_eval.py` (sha `180044eb`) and `v8f_mini_eval.py` (sha `008f48b7`) are byte-untouched (the evaluator wrappers pin both and refuse on a difference). Card: `PREREG_V8F_NAVRB2.md`, FROZEN, sha256 in `PREREG_V8F_NAVRB2.sha256` (`e238ba8d...`).

## 1. Patch and files
`navrb2.patch` (LF, 14 files, regenerated against `4dedac45` by `tmp/mkpatch.py`; the work tree `tree/` is the source of truth: land its blobs by `hash-object`, LF). Changed (5): `stack/tanitad/models/v8f_mini.py`, `stack/tanitad/data/v8f_mini_data.py`, `stack/scripts/v8f_mini_train.py`, `stack/scripts/strat1_check.py`, `stack/tests/test_strat1_s1d.py` (its pinned mutation strings and the KINDS assertion follow the checker text: `kind` to `bk`, KINDS prefix). New (9): `stack/scripts/build_navtext_nav_rb.py`, `v8f_t1_eval_rb.py`, `v8f_mini_eval_rb.py`, `navrb_geval_identity.py`; `stack/tests/test_v8f_navrb2.py`, `test_strat1_navrb2_check.py`, `test_v8f_navrb2_eval.py`, `v8f_mini_tip_snapshot_navrb2.txt` / `.sha256`. `pod/navrb2_tree.sha256` lists the 14 blobs.

## 2. Table (G-CLOCK-RB)
`table/navtext_nav_rb.npz` sha256 `f4c4ab7a8abf2498e532d648874df2549b7b83292b31766ad80e581d009bcebc` (single copy) + `navtext_nav_rb_report.json` (sha256 `18604801...`). MEASURED (part 1): train 29,836 of 461,518 steps on (433 of 4,572 clips); eval 1,334 of 14,237 (17 clips), 235 grid windows / 16 episodes at NOW; the regression arm (state read 1 s ahead) FAILS on both splits.

## 3. Checker kinds (`strat1_check.py`, HANDOFF step 1)
`N1R` / `S1R` / `S1dR` / `N1Z` (bases N1 / S1 / S1d / N1). An R kind renders its base kind's argv byte for byte and APPENDS exactly `--nav-rb on --nav-rb-npz <t> --nav-rb-sha256 <s> --nav-rb-variant real|zero --nav-rb-lr-mult 10`; the argv sha256 of C0 / N1 / S1 at s0 / s1 stays the a4357e79 value (test). G-DVB / G-LIVE read (`rb_gate_problems`): five flags named, `levers.nav_rb == on`, `info.nav_rb.sha256` == argv sha, variant, `navrb_probe` pass with `step0_identity True`, `keys_read_has_nav_rb`, `e_nav_rb_grad > 0`, zero-channel control EXACTLY 0.0, `live.groups_grad.NAVRB > 0`; N1Z: variant zero, `n_on 0`, NAVRB grad exactly 0.0 and `e_nav_rb.weight` in `live.declared_dead`; a STRAT1 kind refuses any nav_rb flag, lever, probe or info. Post: `e_nav_rb.weight` moved off 0.0 (N1Z exactly 0.0; absent in non-R kinds). G-EVAL (`geval`) also compares the built `nav_rb` lever, and a cfg that drops the channel must be REFUSED. G-CLOCK-RB is read from `navtext_nav_rb_report.json` (`rb_report_problems`); `--nav-rb-npz` is data-bound (`DATA_FLAGS_RB`, kept out of `DATA_FLAGS` so no STRAT1 fixture moves); the token carries `nav_rb_report_sha256` for R kinds only. Tests: `test_strat1_navrb2_check.py` (render, gate with about 20 refusals, post, report, binding and token, and 12 mutations each turned RED: zero-control clause, step-0 identity, gradient, sha, N1Z dead clause, variant, render, base map, post weight, post non-R, report regression arm, data flag).

## 4. Evaluator wrappers (HANDOFF step 2; pinned tools untouched)
* `v8f_t1_eval_rb.py` (pattern `v8f_t1_eval_far.py`): `--nav-rb off|on` refused unless equal to the checkpoint's cfg, `--nav-rb-npz/-sha256` (eval split), `--nav-rb-variant follow|real|zero`; rows <= t0 real / donor / zero by the tool's own rule, rows after t0 HELD from t0 in dead-reckoning (the item builder's logged read is overwritten; a test shows what the unwrapped read would see), logged mode keeps the logged rows; `--dump-nav` adds `nav_rb_in`. Flag off: nothing patched (test).
* `v8f_mini_eval_rb.py` (T0, STRATEGIC / detected information): own flags taken out of argv, `D.load_join` patched and restored; `--nav-withheld` zeroes the channel under `follow`; `--nav-shuffle` with `follow` is REFUSED (the tool swaps only the nav array; the donor read is the T1 wrapper's).
* **G-EVAL identity**: `test_g_eval_identity_...` asserts, at batch 32 (T1 default), 64 (T0 default) and 256, that a model with `e_nav_rb.weight` exactly 0.0 gives bit-equal `forward_infer` outputs to the same weights with the channel off, chunked by the evaluator's batch. `navrb_geval_identity.py pair|compare` does it on a real checkpoint through the real evaluators (`zon.pt` / `zoff.pt`, then `compare` of the two dumps) at the tool's default batch. **UNVERIFIED on real data**: that pair is not in the chain (it needs the eval cache); run it after N1R_s0 (commands in the script's docstring).

## 5. Tests run (CPU, `CUDA_VISIBLE_DEVICES=-1`, `tanitad.__file__` inside `tree/`)
* `test_v8f_navrb2.py` 28 passed (part 1). `test_strat1_navrb2_check.py` + `test_v8f_navrb2_eval.py`: 36 passed together. `test_strat1_s1d.py` + `test_strat1_token.py`: pass with the NAVRB2 checker (after the retargeting in section 1).
* **Regression** (HANDOFF step 3): `tmp/regress.py` runs `test_v8f_strat1`, `test_strat1_token`, `test_strat1_s1d`, `test_v8f_dagger`, `test_v8f_dagger_train`, `test_v8f_ego_perturb`, `test_v8f_recover_wiring`, `test_v8f_kindx_s1`, every `test_v8f_mini*` and every launch_gate-reading test (82 files), serially with `wait_commit.py 6 120` before each, and re-runs every failing file on a pristine `4dedac45` export (`tmp/reg_base`). Result file: `tmp/regress_summary.json` (key `n_regressions`; bar 0). Its state at hand-back is in HANDOFF.md. The pod G-SUITE (section 6) is the binding full-suite comparison.

## 6. Pod files (only `/root/strat1/navrb2/`; staged, NOT launched; no IP, port or id in any file)
| file | role |
|---|---|
| `navrb2_gsuite.sh` | full G-SUITE, CPU, 8 shards in 2 waves, 2 threads, memory guard + watchdog (pattern `ckpt120_gsuite_fix3.sh`); tree `/root/strat1/navrb2/tree` vs `new8` (4dedac45), base junit `new8`; result `gsuite_result_navrb2.json`; sentinel `ZZ NAVRB2 GSUITE DONE|FAIL ZZ` in `logs/navrb2_gsuite.log` |
| `navrb2_chain.sh` | 4 base arms under the GPU lock: mint, run, post, zerograd gate on `e_nav_rb.weight`, `ZZ ARM <tag> DONE ZZ`; ends `ZZ STRAT1 NAVRB2 DONE|FAIL ZZ` |
| `queue_navrb2.sh` | waits for `ZZ DSCALE10K BLOCK3 (DONE|FAIL) ZZ` in `/root/dscale10k/block3.log`, then `exec flock /root/gpu_queue.lock bash .../navrb2_chain.sh` |
| `navrb2_tree.sha256`, `navrb2_inputs.sha256` | the 14 tree blobs; `navtext_nav_rb.npz` + report |

Both run scripts carry `COMMIT=__MM_FILLS_THE_NAVRB2_COMMIT__` and refuse until it is set (the token binds the commit). The zerograd call (`--allow '.*' --control 'e_nav_rb\.weight'`) is UNVERIFIED against a real v8f_mini checkpoint (a checkpoint that needs `--param-names` fails loud; fix then).

## 7. MM steps, in order
1. Commit by plumbing on a new branch from `4dedac45` (e.g. `agent/v8f-navrb2-20261117`): `hash-object -w` each of the 14 blobs from `tree/` (LF), scratch index from the `4dedac45` tree, `update-index --cacheinfo`, `write-tree`, revert guard, `commit-tree`, CAS `update-ref`, push, `ls-remote` check. Land `PREREG_V8F_NAVRB2.md` + `.sha256` (frozen) in the same or an earlier commit, before any mint.
2. Pod: build `/root/strat1/navrb2/tree` = `cp -a /root/strat1/new8` + the 14 files, `sha256sum -c navrb2_tree.sha256`; copy `navtext_nav_rb.npz` and `navtext_nav_rb_report.json` into `/root/strat1/inputs/strat1/` (`sha256sum -c navrb2_inputs.sha256` there). Set `COMMIT` in both run scripts.
3. Run the G-SUITE (`nohup setsid bash /root/strat1/navrb2/navrb2_gsuite.sh ...`); on DONE (0 regressions) launch `queue_navrb2.sh`. Pod order (MM ruling 5): CKPT120 fix3, C0REP, DSCALE10K block 3, then these arms.
3b. After N1R_s0: the G-EVAL identity pair on a real checkpoint (section 4) before the read.
Cost: 4 base arms about 0.63 L40-h (E); conditional arms (card 4.2) go back to the PI.

## 8. Deliverable manifest
| artifact | where |
|---|---|
| card + sha256 (FROZEN) | `F:/TanitAD-scratch/strat1/navrb2/PREREG_V8F_NAVRB2.md`, `.sha256` |
| patch (14 files) | `F:/TanitAD-scratch/strat1/navrb2/navrb2.patch` |
| work tree (LF) | `F:/TanitAD-scratch/strat1/navrb2/tree/` |
| originals of part 1 | `F:/TanitAD-scratch/strat1/navrb2/orig/` |
| table + report (single copy) | `F:/TanitAD-scratch/strat1/navrb2/table/` |
| pod scripts + sha lists | `F:/TanitAD-scratch/strat1/navrb2/pod/` and the pod `/root/strat1/navrb2/` (copies) |
| regression driver + results | `F:/TanitAD-scratch/strat1/navrb2/tmp/regress.py`, `regress_summary.json`, `regress_*.log` |
| handoff | `F:/TanitAD-scratch/strat1/navrb2/HANDOFF.md` |
All on F: (plus the 5 pod copies); nothing in the repo; the MM stages.

## 9. Deletions (own temp, literal paths)
* `F:/TanitAD-scratch/strat1/navrb2/tree` first partial export of 5e7083d8 (95,122 KB) before re-export (part 1).
* `F:/TanitAD-scratch/strat1/navrb2/tmp` pytest basetemp / TMP / apply check (64,313 KB, part 1).
* Part 2: the contents of `F:/TanitAD-scratch/strat1/navrb2/tmp/` (basetemps `bt*`, `t`, `pa`, `reg_base`, junit xmls, logs) are deleted after the regression result is read; the result file and the scripts are kept. Logged in HANDOFF.md.
