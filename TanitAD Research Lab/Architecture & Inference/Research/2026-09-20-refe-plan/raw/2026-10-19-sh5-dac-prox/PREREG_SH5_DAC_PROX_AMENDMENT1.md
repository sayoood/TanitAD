# PREREG SH5-DAC-PROX: AMENDMENT 1 (2026-10-19). The G-LIVE probe runs under the trainer's autocast; the ctx loads without a host-side stack; `fit` and `analyze` share one analysis-set definition; the pre-flight and `fit` body get tests

Amends `PREREG_SH5_DAC_PROX.md` (sealed 2026-10-19T13:39:48+02:00, sha256 `687e08c8468c16b6904d968a2c0316c76066841244718789f62c3f927d0a0839`). Sealed at 2026-10-19T14:25:19+02:00. **CODE ONLY.** No registered number, bar, rule, arm, recipe, control, decision, conversion, outcome or committed next arm changes; sections 1-12 and 14-15 of the prereg stand verbatim, and section 13's pin table is superseded by the table below.

## 1. What happened (in order)

1. 13:39:48 the prereg was sealed. 16 code files pinned. 69 CPU tests green; the false-fire simulation (section 7) and the design calculations (sections 3.3, 7, 8) were already banked.
2. After the seal, as the prereg allows, I wrote the label array (`prep --labels`, `raw/prep_labels_report.json`; the rebuilt `sh5_inputs.npz` reproduced byte for byte) and ran a real-data alignment check (`raw/postseal_alignment_check.txt`): 53,221 unsafe of 316,224; P(label unsafe | a key-time clearance < 0) = 1.0; AUC_F of P2's row 0.826694 (the figure already used in section 7); the LOCAL restriction gives 1,444 sets / 816 tokens / 38 logs.
3. I then ran a **CPU dry run of the trainer on the real inputs and labels with a RANDOM ctx** (30-50 real tokens, 5-fold `S` + one `plant` learner, 1 epoch; no ctx exists yet) to prove the real-shape plumbing before the Master Mind spends a GPU slot. **It found a defect** (2.1). I also read the rest of the GPU-path code for the same class and found two more (2.2, 2.3).

## 2. The defects and the exact changes

**2.1 G-LIVE crashed on the real ctx dtype (the finding).** The cached ctx is **bf16** (`NpyCtx.load` returns bf16 tensors) and the real run trains and predicts under **CUDA bf16 autocast** (`Trainer._ac`). `Trainer.glive` called `sh5_branch.glive`, which ran `kv` / `forward` OUTSIDE that autocast: a bf16 input met an fp32 `Linear` (`RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float`). Every CPU test carried an **fp32** ctx, so all 69 passed. On the GPU, `fit` would have crashed at its first step, before any optimizer step (a loud failure, not a wrong number; it would have cost a lock-held start and a second round trip, and the fix under a held lock). **Change:** `sh5_branch.glive(..., ac=None)` runs the probe inside `ac()` (default = a null context = the sealed behaviour, so the fp32 tests are unchanged, tested equal); `Trainer.glive` passes `ac=self._ac`; `Trainer._ac` builds `torch.autocast(device_type=cuda|cpu, dtype=bf16, enabled=bool(rc.autocast))`, which is identical to the sealed behaviour for cuda + autocast and for autocast off, and adds a CPU bf16 emulation that only the tests use (the real device is cuda).

**2.2 `NpyCtx.load` doubled the host RAM.** The sealed version appended every token of the shard to a host list and then `torch.stack`ed it: a host peak of **two shards (about 5.2 GiB for the 2.6 GiB default)** on a box whose RAM gate is 3 GB and whose available RAM was 5.9 GB when I measured it. **Change:** the device tensor is preallocated and filled one token at a time (host peak one token, 3.75 MiB); same bytes, same order (the bf16-bits round-trip test is unchanged and passes; a new assertion forbids `torch.stack` in `load`).

**2.3 `analyze` and `fit` could disagree on the analysis set.** `fit` kept the tokens whose ctx was BUILT (the manifest's `done` flags); `analyze --keep-tokens` kept ALL the manifest's tokens, so a partial build would have made the OOF arrays and the analysis data differ (a loud shape error, but only after the GPU work). **Change:** one helper, `ctx_keep(all_tokens, manifest_path)`, used by both (test: `TestCtxKeep`). `analyze --keep-tokens` now takes the ctx manifest itself, as READY.md already says.

**2.4 Tests that were missing (no behaviour change).** `TestGLiveBf16` (3) and `TestBf16Path` (4), including the **RED** that the sealed call crashes on a bf16 ctx (twice: at the function and, by source mutation, at the trainer); `TestFitBodyOnCpu` (3): `_fit_locked` end to end on CPU on a bf16-bits ctx FILE at the registered model size (d 256): the E = 0 C-ID pre-flight for both seeds passing `sh5_gates.check_cid`, the partial-fold PRE pre-flight report (`preflight_PRE_s0.json`, which had no test and whose helper `_fl_pairs` I also exercised on the real arrays: all-strata AUC of P2's row 0.9007), and a failed G-LIVE taking no step and writing no OOF file; `TestCtxKeep` (2); a no-`stack` assertion in the ctx round-trip test; `test_sh5_guard.py` (4: below). `run_tests.py` now lists `test_sh5_guard.py`. One older test (`test_within_token`) had already been corrected before the seal for the asymmetric synthetic world (section 7); nothing else in the tests changed.

## 3. What does NOT change, and why the earlier evidence still holds

- The false-fire simulation (section 7) ran the Trainer, stacker, bootstrap and decision functions with `autocast=False`, `MemCtx`, `train()` / `predict()`: none of the changed lines (`glive`, `NpyCtx.load`, `ctx_keep`) is on its path, and `_ac()` with autocast off is a disabled autocast before and after. Its numbers are valid for the amended code; it was not re-run. The design calculations (A-TEST, plant calibration, MDE) never touched this code.
- Every registered constant (`sh5_lib.py`) is unchanged; `sh5_lib.py`, `sh5_admit.py`, `sh5_gates.py`, `sh5_ctx_build.py`, `sh5_gpulock.py`, `sh5_synth.py`, `sh5_design_calc.py`, `sh5_cost.py`, `sh5_sim_falsefire.py`, `test_sh5_lib.py` keep their sealed pins.
- The CPU fp32 path is bit-identical (`test_fp32_default_is_unchanged`).

## 4. Disclosure: what the post-seal runs read

The alignment check printed counts, one conditional probability and the AUC of P2's own row. The dry run trained 6 learners for 1 epoch on **random** ctx with real labels, saved nothing, and printed only "finite" and "covered" flags; **no SH5 prediction was inspected and no number from it enters any decision.** No ctx exists, no GPU job has run, no result exists.

## 5. Evidence

`raw/tests_summary_amend1.json` (all test files, with counts), `raw/tests_summary_sealed_state.json` (the sealed state's record, kept for provenance), `raw/logs/run_tests_amend1.log`.

## 6. Still UNVERIFIED (carried to the hand-back)

- `sh5_ctx_build.py run` end to end: it needs the GPU lock and >= 8 GB RAM (L1's gate for the same job class); I verified on CPU that its imports and the signatures of `REFePlanner`, `build_scenarios_for_log` and `FrameResolver` resolve, that its gate and lock logic pass their tests, and that its pattern is L1's step 3; **the C-FID early stop (rc 5) is the guard against a wrong ctx**.
- `fit` on CUDA: the bf16 path is emulated on CPU; the first real GPU chunk's G-LIVE is the first real-GPU check, and any further defect found there is amended BEFORE any number exists.
- At 14:0x the dev-box GPU lock was held by a foreign job (`refcv8-trainfit`, since 13:34:10); the lock wrapper waits for it (`--max-wait`) and never removes it.

## 7. Root-cause class (for `RETRACTION_LOG.md`, which this agent does not edit)

*A test world that differs from the production path in the one dimension the test never exercised.* The ctx was fp32 in every test and bf16 in production; the CPU had no autocast and the GPU does. It is the sibling of "built, tested, unreachable from its caller": there the mechanism was never run, here it ran in a world that was not the real one. The rule adopted: **every pinned test world exercises the production dtype path** (a bf16-bits file read through the production reader, under the production autocast), and a dry run on the real inputs precedes any GPU slot.

## 8. Re-pinned code (the guard accepts exactly these new pins, from exactly these old pins) and the checkers

| file | pin before | pin after |
|---|---|---|
| `code/sh5_branch.py` | `0f93dd4021b41e71310a93731dbb52245e5194f09a79d38c514ffc82c1b1e825` | `8d7be1129ee4981f885c7f7aed852f3a869bde79ebacd5b556c3ad91be364357` |
| `code/sh5_train_eval.py` | `828ca9a89199068049cf31c357cadb2b2665eb2c9b3239a84484a435e43d057d` | `9185ec60442af108db627bc8aa9355da53483a9149adabdb35b8f9cf125c0937` |
| `code/test_sh5_branch.py` | `971fe270dcb49a948d6608036a6315fdcc7961f26ffc670f43094fd3879798a9` | `96d4d93033c3b3328ec0ea5ebc110bdd4ab4f2df61b16155068d7e033a905f1a` |
| `code/test_sh5_ctx.py` | `32cd8e0491101299d9a6b5a3edf0948ee6cb0499cab869852b2e913387445fa6` | `8c2d0fcf3d2b8ad7fcecad33fb336b0eb9d3c2c9001cd95aecb2dde688953e3f` |
| `code/test_sh5_pipeline.py` | `7fd71187529d6d6a438ca32976cb8f50f45f6f9073ba763da0ef0ea2b96895b1` | `3fabf7c5fe8b52af79cdf6e4a2a7bf7468e7108359e64e4c1c11ba6a2da9055c` |
| `code/run_tests.py` | `df9f0a9bb0b87a79dce37d89ec2a651646052c67fdc77ec977eb8f5f3a72f02e` | `75037822455173079691f732673f18d22a1ac5bb17a904c13438e39a37c4d319` |

`sh5_guard.py` (a checker, sealed state sha256 `4f68d9b44f83e4f0d503cf68d4469cde2d251b35816cc43c9c4b94e23ba625da`) now also reads Amendments: `PREREG_SH5_DAC_PROX_AMENDMENT<k>.md` + `.sha256` + `CODE_PINS_AMENDMENT<k>.json` ({"from", "to"}), in order; a file moves only from its CURRENT pin and only if it was already pinned; a missing record, an edited amendment or a broken chain refuses (`test_sh5_guard.py`, with RED cases). New checkers: `seal_amendment.py`, `test_sh5_guard.py`.

| checker (not pinned) | sha256 now |
|---|---|
| `code/sh5_guard.py` | `ac7b35e788ffe228ee1e580517e28aa083b7a30bd546a48cc93550787431286f` |
| `code/seal.py` | `bf08e3ca19cf7aefa9b96477652f0aa57bc85e6cdb860bc976eb6423e26118a8` |
| `code/seal_amendment.py` | `cc221c30c8676cae90fb46936f5f33d9087e45206db837062c563585d7d5d4ff` |
| `code/test_sh5_guard.py` | `ef2d66166e77f61c99c732b8e9e853ff240d90d32850ddf76bfa3fbe9999b511` |
