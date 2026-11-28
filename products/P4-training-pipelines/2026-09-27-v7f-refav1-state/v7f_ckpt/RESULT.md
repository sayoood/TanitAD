# RESULT — F3: a v6 resume is now EXACT (and a second resume defect, F3b, found and fixed on the way)

**2026-09-27 · TrainingFlyWheel (P4) · v7F G-CKPT sub-agent · CPU only (CUDA hidden) · STAGED, never committed.**
Working copy `C:/Users/Admin/v7f_ckpt/` (a byte copy of the v7F merge: `train_v6_staged.py` sha256 `8c212e5f…`,
`launch_gate.py` `f1534182…`, `v6.py` `4ebeca6c…`). Every number below carries its evidence class and artifact.

## Headline
1. **G-CKPT (v7f dev-box REHEARSAL, through the real `launch_gate.py run` CLI): S-W PASS · S-T PASS** (was FAIL/FAIL).
   MEASURED — `raw/gate_after_{sw,st}/evidence/G-CKPT.json`: the resumed run draws the uninterrupted run's steps
   101–102 (S-T `[[1,21],[1,10]],[[1,6],[1,18]]`, S-W `[[0,25],[0,38]],[[2,8],[2,63]]`), and the final state is
   BIT-identical — model 261/261 (S-T) · 259/259 (S-W) digests equal, optimiser state 243/243 · 258/258 equal, lr after
   the schedule replay == the saved lr (`9.48335631477948e-08`). G-LIVE / G-DVB / G-EVAL still PASS; G-HYG still FAILs
   on F2 only (the parent's `@strict_fields`, not in this copy); the token is INCOMPLETE by design (rehearsal).
2. **Tests (MEASURED, CPU, same tree, same env incl. `TANITAD_R3_REF_TRAINER` / `TANITAD_V6_PRE_R1R4` = the
   `c36b6ddd` files):** the parent's set (`test_launch_gate_v7f.py` + the 6 merge files) **256 passed BEFORE → 281
   passed AFTER, 0 failed, 0 skipped** (256 + 5 new gate tests + 20 in the new `tests/test_v6_exact_resume.py`); the
   existing resume/ckpt tests (10 files) **270 passed before and after**. Logs `raw/pytest/`.
3. **F3b — NEW, MEASURED, and it touched EVERY v6 resume on the default schedule:** `train()` replayed the cosine LR
   schedule by `sched.step()` ON TOP of the optimiser's LOADED (already decayed) `lr`; `CosineAnnealingLR` is
   recursive, so the whole resumed remainder of a run trained at **`(1 + cos(π·n/T)) / 2` of its schedule** — a resume
   at 50 % of `--steps` halved every later LR, at 90 % it left 2.4 %. MEASURED on the unit rig (resume at 3 of 5):
   lr after step 4 = `3.2991502812526298e-06` resumed vs `9.549150281252631e-06` uninterrupted, ratio **0.34549 =
   (1+cos(0.6π))/2** (`raw/probe_before_after.json`). ⚠️ **The gate could not see it: its optimiser param-group
   comparison EXCLUDED `lr`.** The live v6F S-W run WAS resumed (INHERITED: the trainer's own comment at the
   `step_s_interval` block, "the first ~900 steps after a resume" on the live v6F log) — **which `n`, and so what
   factor its post-resume segment trained at, is UNVERIFIED**: read its `train_log.jsonl` `lr` rows around the
   `steps_this_process` reset against the closed form. `--trunk-lr-factor` runs (LambdaLR, closed form) were never
   affected.
4. **The gate's OLD G-CKPT protocol could not pass for a CORRECT trainer** — MEASURED, and the gate is changed in the
   same patch: it resumed a run that had FINISHED at `--steps n` with `--steps n+extra` and compared it with an
   uninterrupted `n+extra` run. With the trainer fixed, the draws matched but 101 (S-T) / 86 (S-W) tensors still
   differed, because run 1's own first n steps are on a different cosine schedule (S-T, step 50: lr
   `5.000000000000004e-05` at T=100 vs `5.1539752927808516e-05` at T=102; loss 2.3970 vs 2.4110 —
   `raw/gate_oldprotocol_{sw,st}/`). New protocol: an INTERRUPTED run launched with the FINAL `--steps`, killed right
   after its step-n checkpoint, resumed with the SAME argv, compared with that argv uninterrupted.

### ⇒ WHAT THE PARENT MUST APPLY (integration — escalated, not a README request)
`python code/apply_f3_exact_resume.py --stack C:/Users/Admin/v7f_merge/stack` (it needs
`code/fix/stack/tests/test_v6_exact_resume.py` next to it; every anchor must match exactly once or NOTHING is written).
It changes exactly four files:

| file | change |
|---|---|
| `stack/scripts/train_v6_staged.py` | F3 exact-resume payload (capture at the checkpoint, restore before the first step) + F3b `replay_lr_schedule` |
| `stack/scripts/launch_gate.py` | G-CKPT (v6): interrupted-run protocol; judge compares lr, final optimiser state, requires the stream keys and `resume_rng.mode == "restored"`; arms `v6_resume_drops_rng`, `v6_resume_lr_compounds`, `v6_ckpt_strips_rng` |
| `stack/tests/test_launch_gate_v7f.py` | the G-CKPT pin flipped to a positive PASS (S-T) + S-W positive + 3 arms + a source mutation; **anchored on the G-CKPT bullet line alone and the G-CKPT test functions — no G-HYG region** |
| `stack/tests/test_v6_exact_resume.py` | NEW, 20 tests |

PROVEN (`raw/apply_proof.json`): the script applied to fresh copies of the `8c212e5f…`/`f1534182…`/`428a647b…` files
reproduces my working files **byte-for-byte**, and it applies cleanly to the CURRENT merge copy
(`C:/Users/Admin/v7f_merge/stack`, fresh copy taken 2026-09-27T16:23:45Z; its trainer and gate were still the
pre-F3 bytes, so the result equals my working files; its test file already carried the parent's F2 edits,
`7e999fd6…`, and the 3 test anchors still matched once each — the patched file compiles).

## 1. What was wrong, and what the fix is

**F3 (the assigned defect), MEASURED by the gate** (INHERITED from `…/v7f_gate/RESULT.md` and re-read from
`…/v7f_gate/raw/merge_rehearsal_{st,sw}/G-CKPT.json`): the S-T resume drew `[[1,5],[1,23]],[[2,10],[2,7]]` — the
uninterrupted run's FIRST batches — instead of its steps 101–102; 101 tensors differed at the end (S-W 86). Cause:
`train()` re-seeds torch / `random` / `rng` / `gen` from `--seed` at every launch, and `_save_ckpt` wrote
`{stack, opt, step, config}` only.

**The fix (`train_v6_staged.py`).** Three additive checkpoint keys, captured FIRST in the save block (the state at
the end of the saved step, before anything the save block does — an uninterrupted run need not save there), and
restored immediately before the step loop, after every construction-time consumer of the seeded streams:

| key | carries | why |
|---|---|---|
| `rng_state` | torch CPU; torch CUDA per device (when training on CUDA); python `random`; the uniform sampler's `rng` (`make_sampler`); the shared `gen` (O4/T3/F-10 samplers, `sample_random_deltas`, `v6_loss_step`, O5 crop, O9); `spec_gen` (seed+7); X4's generator (seed+11); numpy's global stream | every stream the step loop consumes (grep of every `Generator()` / `random` / `manual_seed` in the trainer). numpy is consumed by nothing today — carried so a future consumer cannot silently break exactness |
| `data_pos` | step / next_step; the stream FINGERPRINT (n_windows, batch, eps_per_batch, seed, sampler, t5, t3, domain mix, accum knobs, device type); the T3 curriculum's applied alpha AND the unrounded `prog` its weights were computed at | a resume compares fingerprints and RECORDS any difference; T3's live weights are a function of the unrounded prog of the step they were last recomputed at |
| `loop_state` | the SIGReg row bank (it enters the O6 LOSS); the O6/X4 spectrum pools, references, last readings; the o6 trend series; the observer monitor | every rolling buffer that carries state across steps |

NOT carried, by construction: the LR schedule (replayed — see F3b), the EMA ramp (reads the absolute step), the O13
projection (re-seeded per call), the X2 seam dump (no RNG — `taniteval/seam_dump.py` has no random call).
`config.json` gains `resume_rng` ON RESUME ONLY (`restored` / `restored-stream-mismatch` / `absent-replayed`); a
fresh launch's config is unchanged.

**Pre-F3 checkpoints — RESUME, LOUDLY (chosen and tested):** refusing would strand every banked v6 checkpoint; a
replayed stream re-draws in-distribution batches, it does not corrupt the model. The trainer prints a banner and
records `resume_rng.mode = "absent-replayed"` in config.json AND in every checkpoint it writes; the NEXT checkpoint
carries the payload, so the replay happens at most once. F3b's fix needs no payload, so even a legacy resume now
gets the right LR (tested).

**F3b (`replay_lr_schedule`).** Before replaying, every group's `lr` is reset to its `initial_lr`; the replay then
performs the identical sequence of float operations the uninterrupted run did, so the lr of step n+1 is BIT-equal
(MEASURED: gate `resume_lr` saved == after_replay on S-W and S-T; unit rig lr rows equal).

## 2. Evidence — before / after

### 2a. The unit rig (`tests/test_v6_exact_resume.py`; CPU; tiny S-W; 3 synthetic episodes; batch 2; interrupted at N=3, resumed to 5) — MEASURED, `raw/probe_before_after.json`
| | pre-F3 trainer (`8c212e5f`) | F3 trainer |
|---|---|---|
| resumed draws (steps 4–5) | `(1,47),(1,6),(0,36),(0,59)` = the uninterrupted run's steps 1–2 | `(0,47),(0,44),(0,56),(0,30)` = its steps 4–5 |
| loss step 4 / 5 | 1.69579 / 1.69936 (uninterrupted 1.94922 / 2.25317) | 1.9492214918136597 / 2.2531685829162598 — equal |
| lr after step 4 | 3.2991502812526298e-06 (uninterrupted 9.549150281252631e-06; ratio 0.3454915 = F3b) | equal |
| final model / optimiser | 86 of 259 tensors / 172 state entries differ | 0 / 0 |

**Fresh-run numerics unchanged** (MEASURED, same file): pre-F3 vs F3 trainer, fresh 5-step runs on 4 configs (nav
O4 · nav-off O4 · nav uniform sampler · nav SIGReg bank): draws (10/10), per-step loss, per-step lr, final model
(259 / 247 / 259 / 259 tensors) and optimiser state all bit-identical; only the checkpoint gains the three keys.

### 2b. The gate (`launch_gate.py run --profile v7f --stage devbox --cpu-only --rehearsal on`, the gate agent's launch argv files)
| | S-T | S-W | artifact |
|---|---|---|---|
| BEFORE (gate agent, pre-F3 trainer, old protocol) | FAIL — replayed first batches; 101 tensors differ | FAIL — 86 differ | `…/v7f_gate/raw/merge_rehearsal_{st,sw}/G-CKPT.json` (MEASURED by the gate agent) |
| trainer fixed, OLD protocol | FAIL — draws now correct, stream keys present; 101 tensors differ (schedule T=100 vs 102) | FAIL — 86 differ | `raw/gate_oldprotocol_{st,sw}/` (MEASURED) |
| AFTER (fixed trainer + new protocol) | **PASS** | **PASS** | `raw/gate_after_{st,sw}/` (MEASURED) |

## 3. Tests

| run | BEFORE (pre-F3 bytes of the 3 files, no new test) | AFTER (final bytes) | log |
|---|---|---|---|
| the parent's set: `test_launch_gate_v7f.py` `test_declared_vs_built.py` `test_tactical_label_reach_v6.py` `test_v6_effective_weights.py` `test_v7f_r1r4.py` `test_v7f_r2_nav_fixes.py` `test_v7f_r6_strategic_off.py` (+ `test_v6_exact_resume.py` AFTER) | **256 passed** | **281 passed, 0 failed, 0 skipped** | `raw/pytest/{before_merge_set,after_merge_set_plus_new}.txt` |
| existing resume/ckpt tests: `test_ckpt_trajectory` `test_dinov3_seed` `test_eval_speed_ckpt` `test_refa_v1_ema_resume` `test_replay` `test_seam_dump_import_guard` `test_v6_chain` `test_v6_ladder_edges` `test_v6_staged` `test_resume_continues_the_data_order` | **270 passed** | **270 passed** | `raw/pytest/{before,after}_resume_set.txt` |

`test_launch_gate_v7f.py`: 52 → 57 tests (1 pin flipped to a positive, +S-W positive, +3 arms, +1 source mutation).
`test_v6_exact_resume.py` (20): the v7F-path positive with LITERAL draws; the checkpoint layout; 6 configs (nav-off
default path · uniform sampler / python `rng` · SIGReg bank · spectrum pools · T3 curriculum · all monitors together:
SIGReg bank + O6 pool + `spec_gen` + X4 pools + X4 generator + observer monitor); 7 arms (below); the legacy
checkpoint; fresh-run invariance; a recorded stream mismatch; the payload needs NO unpickling global (MEASURED: a
pre-F3 v6 checkpoint already needs `torch.torch_version.TorchVersion` for `weights_only=True` — from the config's
provenance — and a post-F3 one needs exactly that and nothing more); `_save_ckpt` refuses an overwriting extra key.

### Regression arms (each must go RED — and does)
| where | arm | how it goes RED |
|---|---|---|
| unit | `drop="all"` (no restore) | resumed draws == the uninterrupted FIRST batches, literally `(1,47),(1,6),(0,36),(0,59)`; 86 tensors differ |
| unit | uniform sampler, no restore | draws == its first batches `(1,55),(1,7),(1,64),(1,53)` (the python `rng` stream) |
| unit | F3b re-introduced | draws exact, lr ratio == 0.3454915028125263 (to 1e-9), model differs |
| unit | `loop_state` dropped + SIGReg bank | step-4 LOSS differs (the bank enters the O6 loss) |
| unit | `loop_state` dropped + spectrum pooling | step-4 pooled spectrum over 1 step instead of 2 |
| unit | T3 prog not restored | the sampler's weights digest differs at the first resumed draw |
| unit | SOURCE mutation (restore call → no-op, compiled as its own module) | draws replay |
| unit | legacy checkpoint (keys stripped) | resumes, banner printed, `absent-replayed` recorded, draws replay, lr still exact, next ckpt carries the payload (a POSITIVE test of the chosen behaviour) |
| gate | `v6_resume_drops_rng` (S-T) | FAIL "restarts at every launch"; draws == first batches |
| gate | `v6_resume_lr_compounds` (S-W) | FAIL naming F3b; after_replay / saved == saved / 1e-4 |
| gate | `v6_ckpt_strips_rng` (S-W; pre-F3 checkpoint layout) | FAIL: "carries no RNG state" + replay + `'absent-replayed'` |
| gate | SOURCE mutation of the restore call | FAIL "restarts at every launch" |
| gate | `v6_resume_drops_opt` (existing) | still FAILs on `resume_opt_state_vs_saved` |

**On-disk SOURCE mutations** (`raw/mutations/mutation_record.json`; the trainer restored byte-identical afterwards,
sha256 asserted): M1 restore call deleted → 7 of 8 POSITIVE tests FAIL; M2 the F3b lr reset deleted → 7 of 8 FAIL;
M3 the payload not saved → 8 of 8 FAIL (the one that survives M1/M2 is the checkpoint-layout test, which does not
resume).

## 4. NOT done / open (stated, not narrowed)
* **Thor stage NOT run** (no data here): the non-rehearsal G-CKPT (real corpus, real nav, CUDA RNG state) is
  UNVERIFIED. The CUDA path is written (`torch.cuda.get/set_rng_state_all` when `--device cuda`) but never executed
  here (CPU only by rule). Bit-identity on GPU additionally needs deterministic kernels; the claim here is CPU.
* **S-T in the unit tests:** the unit rig is S-W (S-T needs a predecessor); S-T exactness is covered by the gate
  rehearsal (PASS) only.
* **O7/O8/O9/O10 auxiliary heads (found while auditing, NOT fixed — outside F3):** they are trained (their params are
  in the optimiser) but are not in `stack.state_dict()`, so a resume of such an arm RE-INITIALISES them while loading
  their Adam moments. v7F sets none of them (all four weights default 0). Next lever: save/restore their
  `state_dict()` beside `stack` (same additive-key pattern), with a resume test per head.
* **Historical impact of F3b is UNVERIFIED** (see headline 3). Every resumed v6 run on the default schedule is exposed.
* The gate agent's `v7f_gate/RESULT.md` / `DESIGN.md` still describe G-CKPT as FAIL and the old protocol — not my
  package; the Master Mind should amend them when landing this.
* `GOALS_AND_CLAIMS.md` / `RETRACTION_LOG.md` NOT edited (single-committer rule; steering files are the Master
  Mind's). Suggested entries: F3b as a retraction-class finding — *"a check that excluded the one field the defect
  lived in"* (the lr-excluding param-group comparison) — and the old G-CKPT protocol as *"an instrument that could
  not tell a correct trainer from a broken one"*.

## 5. Deliverable manifest
All repo paths are under `products/P4-training-pipelines/2026-09-27-v7f-refav1-state/v7f_ckpt/` (STAGED, never
committed; each verified by blob comparison — `raw/staging_verification.json`).

| artifact | where | copies |
|---|---|---|
| `RESULT.md` | repo | 1 |
| `code/apply_f3_exact_resume.py` — THE deliverable (sha256 `c3487de5e05b02c3…`) | repo + `C:/Users/Admin/v7f_ckpt/pkg/code/` | 2 |
| `code/fix/stack/scripts/train_v6_staged.py` (sha256 `fa5ef2ba23d5f714…`, CRLF like the merge copy) | repo + `C:/Users/Admin/v7f_ckpt/stack/scripts/` | 2 |
| `code/fix/stack/scripts/launch_gate.py` (`b232aeadaf977bd7…`, LF) | repo + `C:/Users/Admin/v7f_ckpt/stack/scripts/` | 2 |
| `code/fix/stack/tests/test_launch_gate_v7f.py` (`30fd870985ccd7ea…`, LF; on the PRE-F2 base — apply the SCRIPT to the parent's F2 copy, do not copy this file over it) | repo + `C:/Users/Admin/v7f_ckpt/stack/tests/` | 2 |
| `code/fix/stack/tests/test_v6_exact_resume.py` (`cdfe34964d99077b…`, NEW; the script copies it) | repo + `C:/Users/Admin/v7f_ckpt/stack/tests/` + `…/pkg/code/fix/` | 3 |
| `code/diffs_vs_pre_f3/*.diff` (3, review aids; the script is the deliverable) | repo | 1 (regenerable) |
| `raw/gate_after_{sw,st}/` — the FINAL-bytes CLI rehearsal: `cli.log`, `evidence/G-*.json`, `rehearsal_verdicts.json` (token SUMMARY) | repo + `C:/Users/Admin/v7f_ckpt_gate/runs/after_{sw,st}/` | 2 |
| `raw/gate_oldprotocol_{sw,st}/` — trainer fixed, OLD protocol (G-LIVE + G-CKPT) | repo + `C:/Users/Admin/v7f_ckpt_gate/runs/probe_oldprotocol_{sw,st}/` | 2 |
| `raw/gate_launch_argv_{st,sw}.json` (the gate agent's launch argv files, re-used) | repo | 1 (+ the gate agent's copy) |
| `raw/probe_before_after.{json,py}` — pre-F3 vs F3 trainer: fresh-run invariance (4 configs) + the resume | repo + `C:/Users/Admin/v7f_ckpt/work/` | 2 |
| `raw/discovery_F3b_rng_fix_only.txt` — how F3b was found (RNG fix alone: step-4 loss equal, step 5 diverges) | repo | 1 |
| `raw/mutations/` — `mutation_record.json`, `mutate_source.py`, 3 trimmed pytest summaries | repo + `C:/Users/Admin/v7f_ckpt/work/mutations/` (full logs) | 2 |
| `raw/pytest/*.txt`, `raw/run_suites*.{py,log}` | repo + `C:/Users/Admin/v7f_ckpt/work/suites/` | 2 |
| `raw/apply_proof.{json,py}` — byte-for-byte reproduction + the apply onto the CURRENT merge copy | repo + `C:/Users/Admin/v7f_ckpt/work/` | 2 |
| `raw/bank.py`, `raw/staging_verification.json` | repo | 1 |
| gate run dirs (smoke dirs, checkpoints), the superseded first after-run `after_v1_*` (trainer `6dc78141…`, before the two cosmetic/robustness edits — same verdicts) | `C:/Users/Admin/v7f_ckpt_gate/runs/` | **C: only** (scratch) |
| the rehearsal TOKENS (INCOMPLETE, test-key-signed, fake commit `7f7f…`) | `C:/Users/Admin/v7f_ckpt_gate/runs/*/INCOMPLETE_*.json` | **C: only, by design** |
| the HMAC TEST key | `C:/Users/Admin/v7f_ckpt_gate/tmp_keys/gate.key` | **C: only, by design** (outside the tree; the default key path never touched) |
| the pre-F3 reference copies (`pristine/`), the `c36b6ddd` reference files for the env vars | `C:/Users/Admin/v7f_ckpt/pristine/`, `C:/Users/Admin/v7f_ckpt_gate/ref/` | C: only (reproducible from the merge / git) |

