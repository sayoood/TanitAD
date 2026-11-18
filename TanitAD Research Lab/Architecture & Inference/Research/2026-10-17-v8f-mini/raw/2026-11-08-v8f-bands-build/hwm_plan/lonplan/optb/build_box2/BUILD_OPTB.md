# BUILD_OPTB: ACTRSP Option B lever (build complete, dry run green)

Nothing launched, committed or pushed. No GPU, no box 2. Frozen spec `AMEND_OPTB.md` (sha256 `7cb0245c...`), written before any new-flag test ran.

## Escalation (headline)
1. The lever attaches to `a13f6f16` (branch `agent/v8f-actrsp-20261116`), not to the repo tip `5e7083d8`: the tip trainer has no `--wm-only`. Commit `optb.patch` plus the package on a branch from `a13f6f16`.
2. Done: the lever, mutations MR11-MR15, the regression, the checker, the argvs, the chain package and its dry run (`DRYRUN ok=140 failed=0`). Not done, by design: pilot, gate tokens, launch (box 2 untouched).
3. One change outside the four flags: `HJEPAStack.built_rollout` (the G-DVB ladder reader) reads on a float64 copy. Reason (MEASURED): on a trained WM the float32 reader gave level-3 k-step weight 0.99985 against the declared 1.0 (tolerance 1e-4), which would refuse every fine-tune gate. A fresh init reads 1.0 either way.
4. `optb.patch` was re-derived on 2026-11-18 from `tree/` against the three base files (diff -u, a/ b/ labels): 776 lines, sha256 `b015bc1de456b2c7...`. It differs from the 4212b0e2 patch by exactly one line, the MM's one-line change of the `freeze_all_but_wm` call (one-argument call when `--wm-freeze-proj` is off).

## What was built (`optb.patch`, 776 lines, sha256 `b015bc1d...`)
Files: `stack/tanitad/models/hjepa.py`, `stack/tanitad/models/v8f_mini.py`, `stack/scripts/v8f_mini_train.py`, new `stack/tests/test_v8f_optb.py`. Edited tree: `optb/tree/`.
Flags (all default-off, named in argv per `AMEND_OPTB.md` section 1): `--wm-rsp-readout {idm,fixed_k}` (default idm), `--wm-rsp-probe`, `--wm-rsp-probe-sha256`, `--wm-init-from-ckpt-wm {off,on}`, `--wm-freeze-proj {off,on}`.

## Hops (file:line in `optb/tree/`; grep the names)
| hop | where |
|---|---|
| argparse (5 flags, defaults idm, "", "", off, off) | `v8f_mini_train.py` build_parser, after `--wm-rsp-w` |
| FLAG_CHECKS entries (a flag without one is refused by the gate) | `v8f_mini_train.py` FLAG_CHECKS, after `wm_rsp_w` |
| make_config applies readout, probe, sha, freeze | `make_config`, line before `wm_rsp_w=` |
| config fields and refusals (fixed_k needs w > 0, freeze on, probe + 64-hex sha; probe without fixed_k refused) | `v8f_mini.py` V8FMiniConfig, `__post_init__` |
| probe load, sha256 check, K = 10, shape check; `w = RA.W / RA.sd`, `s_cal` | `v8f_mini.py` `load_rsp_probe`, called in `WMAux.__init__` after the stack is built |
| level-1 LevelConfig gets `rsp_readout="fixed_k"` | `WMAux.__init__` level-1 kwargs |
| buffers `rsp_probe_w` (float32), `rsp_probe_s` (float64), non-persistent | `HJEPAStack.set_rsp_probe` |
| loss: dispatch, two contrasts, endpoint read, constant s, `mean (G-1)^2` | `HJEPAStack.response_loss` -> `_response_loss_fixed_k`; added to `level/L1` by the existing block in `HJEPAStack.loss` |
| backward | unchanged `T["total"].backward()` in the WM-only loop |
| checkpoint start: all tensors incl. `wm.*`, strict, read back; strides compared | `load_wm_only_init(init_wm=True)` |
| freeze | `freeze_all_but_wm(model, freeze_proj)` before the optimiser |
| G-DVB | `bad_wm_rsp` (readout, contrasts 2, live weights vs the re-read file, s_cal), `bad_wm_freeze` (requires_grad of `wm.proj.*`, proj equals the checkpoint's), `bad_wm_only` (init record), `wm_rsp_readback` in `selfcheck.json` |

## Tests (CPU, one pytest at a time, `wait_commit.py 6 120` before each)
`stack/tests/test_v8f_optb.py`: **23 passed** (22 in one run of 141 s; the one test whose string match was wrong, `test_gate_refuses_fixed_k_without_freeze_or_without_init_wm`, was fixed and re-ran green). Content: defaults equal the earlier build; state-dict keys and values equal with and without the probe buffers; gradient reaches `predictors.0` and `act_embed` and reads EXACTLY 0.0 on `idm_heads`, `proj`, levels 2-3 and the adapter output; control with the flag off has no term key and reads 0.0; C9' (float64): matched constant gives loss < 1e-24 and gradient < 1e-12, B = 0 gives exactly 1.0, a doubled constant gives G = 0.5 and loss 0.25, history-carrying states and autocorrelated actions leave the identity at < 1e-24, delta scales out, an opposite world gives loss 4, geometry 1 call, 2 contrasts, 9 anchors, K 10, context 1, endpoint read (a W = 0.875 world pins it); config, sha-mismatch (one changed byte), wrong K and wrong width refusals; the trainer's own gate on the synthetic recipe for the RSP arm and the NULL arm (readback, `live.c3` pass, proj frozen, `init_wm` read back equal), refusals (no probe, idle probe flags, no freeze, no init, wrong sha, other strides, MR11 and MR12 shapes); a real 2-step run: `wm.proj` bit-equal to the checkpoint and more than 10 other `wm.*` tensors moved.
## Census, mutations, regression (MEASURED, CPU, one pytest at a time)
* Census (`census_gate.py`): every new flag is traced argparse -> make_config -> model -> loss/gate; hops table above.
* Mutations (`mutation_results_optb.json`): MR11-MR15 all RED, each named must-fail test failed, no control test went red.
* Regression vs base `a13f6f16` (`regress_compare.json`; 34 of 89 files had failures on the new tree, compared per test against the base tree): 1 test failed on new and passed on base, `test_v8f_mrate_train.py::test_the_gate_refuses_a_checkpoint_of_another_shape_and_a_missing_freeze`. Cause: `freeze_all_but_wm` was called with two arguments when `--wm-freeze-proj` is off. Fixed by the MM (one-argument call); `test_v8f_mrate_train.py` + `test_v8f_optb.py` = 36 passed. Regressions after the fix: 0. The other failures also fail on the base (23 on base, 24 on new; the 24th is that one test).
* Launch-gate-reading set: no pin or `launch_gate.py` edit was made, so the profile fingerprints are expected unmoved: UNVERIFIED (not run).

## Dry run (`pkg/tests/DRYRUN_RESULT.log`): `DRYRUN ok=140 failed=0`
Command: `cd pkg && PATH=/usr/bin:/bin:$PATH OPTB_TEST_TMP=<dir>/optb_dry_run_x bash tests/test_optb_dryrun.sh` (ssh/sftp/scp are poisoned; nothing reached box 2; no BREACH file). Unit test `tests/test_optb_check.py`: `UNIT ok=119 bad=0`. Fixes made to the package to get green (no rule weakened):
* `stages/optb_inputs.sh` had CR line ends (25): converted to LF (section 10 hygiene). `optb_check.py` is uniformly CRLF (1020 of 1020 lines); it is Python, not tested for LF, left as is.
* `tests/plant_trainer.py` docstring (the stand-in trainer's needle text) lacked three checker NEEDLES (`"--wm-rsp-readout"`, `"--wm-freeze-proj"`, `def load_wm_only_init`): the ship step refused the throw-away commit. Added the three strings; the checker's NEEDLES are the frozen build's.
* `optb_chain.sh`: a refusal of any arm made the chain stop before chassis 1. MM ruling 2 (`MM_RULINGS_OPTB.md`) and `AMEND_OPTB.md` section 2.4 (seeds chassis 0 {0,2,4}, chassis 1 {1,3,5}) say only a chassis-0 failure stops before chassis 1. Added `REFUSED0` (chassis 0 refusals only) for that stop; a chassis-1 arm refusal (FTRSP_s3) lets the other independent arms run.
* `tests/test_optb_dryrun.sh` section 5: the slow-pilot case used 6.0 s/step, forked from ACTRSP. For OPTB, 10 x 4,000 steps x 6.0 s = 66.7 h < 72 h (`AMEND_OPTB.md` section 2: 4,000 steps, 10 arms; 72 h rule = `optb_check.LIMIT_H`), so the permit correctly passed. Changed the three occurrences to 7.0 s/step (77.8 h).

## Package (`pkg/`, 30 files in `PACKAGE_MANIFEST.sha256`, sha256 of the manifest `147d02d3...`)
Chain, stages, checker, ship/refresh/hfbank, waiter, launcher, frozen argvs and probes, tests. Build-time forge scripts (`fork_*.py`) and the dry-run log are outside the manifest.

## MM steps to commit and ship (not run by this agent)
1. Plumbing commit from `a13f6f16` on `agent/v8f-optb-20261117`: `hash-object -w` each file; scratch index from the `a13f6f16` tree; `update-index --cacheinfo` for `stack/tanitad/models/hjepa.py`, `stack/tanitad/models/v8f_mini.py`, `stack/scripts/v8f_mini_train.py` (blobs from `tree/`), new `stack/tests/test_v8f_optb.py`, and the package under a repo folder of the MM's choice (full path <= 200 chars); `write-tree`, revert guard, `commit-tree`, CAS `update-ref`, push, verify with `git ls-remote`. Check that `git apply --check optb.patch` on `a13f6f16` yields the same four blobs.
2. G-SUITE on the landed commit (not on the dev box); the launch-gate-reading tests if any pin moved.
3. Copy `pkg/` to `C:/Users/Admin/optb_pkg`; check with `sha256sum -c PACKAGE_MANIFEST.sha256`.
4. `bash optb_waiter.sh snap`, then `launch_optb_chain.ps1 -Commit <40 hex> -HfRepo <owner/name>` (WMI, 0x08000200), then `bash optb_waiter.sh wait` in the background. Read `PILOT.verified.json` (stop if s/step > 1.0 or `max_memory_allocated` > 6.5 GB); pin the trainable-parameter count from the pilot's `selfcheck.json`.

## Cost (ESTIMATED): 14-16 h box 2 + 0.8 h pilot; read about 22 h dev-box CPU.

## MM rulings needed: see `AMEND_OPTB.md` section 6 (LR, seeds, NULL freeze, land base, shipping inputs).

## Manifest (all on F:, `F:/TanitAD-scratch/actrsp/optb/`; ONE place, bank before the window ends)
`AMEND_OPTB.md`, `AMEND_OPTB.sha256`, `MM_RULINGS_OPTB.md`, `optb.patch`, `BUILD_OPTB.md`, `HANDOFF.md`, `mutation_results_optb.json`, `regress_compare.json`, `dryrun4.log`, `deletions_log.txt`, `readout_K10_R35_s{0,1}.pt`, `tree/` (edited export), `pkg/` (with `PACKAGE_MANIFEST.sha256`, `tests/DRYRUN_RESULT.log`).

## Deletions (literal paths, bytes in `deletions_log.txt`)
`F:/TanitAD-scratch/actrsp/optb/tmp` 990,240,632 (own: pytest basetemps, gate_reuse dirs, three dry-run trees), `.../optb/basetree` 463,155,913 (re-creatable from git `a13f6f16`), `.../optb/base` 512,246 (three base files, re-creatable; patch refreshed first), `pkg/fork_check2.old_unused`, `pkg/optb_check.before_stage2`, `pkg/__pycache__`, `dry_s1.log`, `dryrun3.log`. `C:/Users/Admin/AppData/Local/Temp/optb_dry_run_a` was already absent. Kept: `tree/`, `pkg/`, `optb.patch`.
