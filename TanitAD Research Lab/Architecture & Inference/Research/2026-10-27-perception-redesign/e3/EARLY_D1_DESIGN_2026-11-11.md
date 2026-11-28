# Early D1: pause E3 at step 30,000, run D1, resume E3 as the same run

Tip read: `66681a601d78275a40b6eb748221f2b676e817f1` (branch agent/arch-inf-20260803, ls-remote after fetch, 2026-11-11). All `file:line` below are at that sha (blobs exported with `git show`).
Paths: T=`stack/scripts/refc_v3_train.py`, LR=`stack/tanitad/train/lr_schedule.py`, SUP=`stack/ops/sup_refcv7.sh`, CH=`stack/ops/r9e1_gate3/r9gate3_chain.sh`, ST=`.../r9gate3_study.sh`, LG=`stack/scripts/launch_gate.py`, MK=`stack/scripts/r9_e3_make_argv.py`, NOTE=`E3_STUDY_NOTE_2026-11-11.md`.
Evidence classes: MEASURED = read from code at the tip (nothing was run); INHERITED = from a doc; HYPOTHESIS = my inference.
Read-only: nothing edited, staged, run on Thor/box 2, or trained.

## Short answers
1. RESUME: YES, with no new flag. Relaunching the SAME argv into the SAME `--out` resumes `ckpt.pt` (model, opt, step, data position, EMA/B1, RNG). The LR position is a pure function of the step. Missing: nothing for CPU-exact; on GPU it is not bitwise (stated by the trainer).
2. PAUSE: NO stop-at / pause flag, marker or supervisor state exists. The supervisor treats any early trainer exit as a crash and relaunches (<= MAX_RELAUNCH=4 per CH:130). Smallest correct addition is a one-shot supervisor pause (details in section 2), landed BEFORE E3 launches.
3. TOKEN: the existing E3 token admits the resumed relaunch unchanged (same commit, tree, argv, data) provided nothing under `stack/` or `taniteval/` is landed between launch and resume. Resume adds no flag, so the argv sha does not move.

## 1. RESUME (exists and works)

| item | restored? | evidence |
|---|---|---|
| mechanism | `if ck.exists():` load `<out>/ckpt.pt` (no flag; resume wins over `--init-from`) | T:11188, 11195-11219 (MEASURED) |
| model | yes, strict `load_state_dict` | T:11198 |
| optimizer | yes, `opt.load_state_dict` (incl. per-group lr state; lr is re-applied every step from `initial_lr` x sched) | T:11199; apply_lr_schedule call T:11811 |
| step counter | yes, `step = state["step"]`; loop is `while step < args.steps` | T:11200, 11786 |
| LR schedule position | yes by construction: `lr_factor(step,args)` is "a pure function of (step, args)"; E3 is `--lr-schedule wsd`, so step 30,000 resumes at the stable factor 1.0 | LR:90-96, LR:62-68 |
| data order | yes: `ResumableEpochSampler`, `data_pos` {epoch,batch,n,seed,batch_size}; mismatch of n/seed/batch REFUSES | T:9162-9254, T:11201-11204, ckpt write T:12094-12101 |
| EMA + B1 guard | yes; a resume without EMA when `--ema-decay` set REFUSES (and the converse) | T:5429-5460, 11693-11716 |
| P10 identity | yes, refuses mismatch | T:5492-5506 |
| RNG (torch CPU+CUDA, numpy, python, model-owned generators) | yes, restored just before the first step; in-training eval is stream-neutral so a checkpoint written before an eval resumes on the right stream | T:5538-5565, 11752-11757, EVAL_STREAM_NEUTRAL T:5567, eval restore T:12127/12242/12315 |
| gate evidence | G-CKPT already proves, at smoke scale, that model + opt + sampler position + data_order.resumed_at_step survive a resume | LG:8147-8215 (MEASURED by reading) |

Would a resumed E3 differ from an uninterrupted one? On CPU the trainer claims exactness (tests `test_resume_rng.py`, `test_resume_continues_the_data_order.py`, `test_lr_schedule_wsd.py` exist at tip). On CUDA: "atomicAdd kernels are not bitwise-deterministic run to run, so GPU bit-exactness is not claimed" (T:5558-5560). So the resumed E3 is the same run statistically, not bit-identical; this is the SAME property the supervisor's crash-relaunch already has. Also: workers' own streams are inert because datasets draw no RNG (T:5560-5562, trainer's own claim, INHERITED from its comment).

Details that matter for the pause point:
- `ckpt.pt` is written at `step % save_every == 0` (default 500, T:14108; E2 argv has 500, so E3's does) BEFORE `ckpt_full_30000.pt` and before the eval (T:12086-12107). So at step 30,000 `ckpt.pt` and `ckpt_full_30000.pt` have the same step and keys (G-CKPT checks key-set equality, LG:3779-3799). The resume source can be either; `ckpt.pt` is what the trainer reads (T:11188). Nothing needs copying.
- Order inside the step-30,000 iteration: ckpt.pt -> ckpt_full_30000.pt -> eval (E2 argv `--eval-every 500`, 30000 % 500 == 0) -> `step` loop continues (T:12086-12130). A stop between ckpt_full and the eval's metrics row would lose the step-30,000 eval row for good (resume starts at step 30,000 and the next eval is 30,500). So pause after the eval row is flushed, not on the appearance of the file.
- Killing later than 30,000 loses nothing important: resume restarts from `ckpt.pt@30000` (as long as the kill is before step 30,500, the next save) and redoes <= ~499 steps. `metrics.jsonl` is opened `"a"` (T:11616), so redone steps leave DUPLICATE rows for the same step; `final_step_from_metrics` / `run_is_complete` use max step so they are unaffected (SUP:244-263), but analyses that read rows must de-duplicate. A pause within the first ~49 steps after 30,000 (before the log-every-50 row at 30,050) leaves no duplicates (HYPOTHESIS on timing: E3 ~9.4 s/step per NOTE, so 50 steps ~ 8 min).
- `config.json` is rewritten on every launch with `argv_effective` (T:11297, LR:362-372); E3's resumed launch writes the same argv, so D1's source-config check stays valid.

## 2. PAUSE (missing)

What exists:
- Trainer: no stop-at, no pause file, no SIGTERM handler (grep `signal.|SIGTERM|atexit` in T finds none; comment-only hits). A trainer exits only when `step == args.steps` (T:11786, summary T:12322).
- Supervisor: after the trainer exits, if not `run_is_complete` it logs "ARTIFACT CHECK FAILED ... relaunching into the SAME out dir" and loops, up to MAX_RELAUNCH (SUP:540-565, 484; CH:130 sets 4). Done-marker = trainer's `summary.json` + `zerograd_gate` record (SUP:194-204, 267-290, G5 SUP:59-81). There is no paused state.
- Chain: one arm at a time under the Thor GPU lock fd 9 (CH:113-141); the lock is released after the supervisor returns (CH:141); the verdict is read from artifacts: done+zerograd => `-DONE`, ZEROGRAD_FAIL => `-ZEROGRAD-FAIL`, rc 5/GATEREFUSED => `-REFUSED`, ELSE `-FAILED` (CH:142-145). So a plain kill of E3's supervisor+trainer is logged as `r9e3_V-FAILED`, then the loop continues to the next arm and finally `ALLDONE` (CH:146-148). It does not delete the run dir, write ZEROGRAD_FAIL.json, or write a done-marker, but "FAILED" is the status line, and anything reading CHAIN_STATUS as truth is misled.
- ARMS is read ONCE at chain start (CH:22) and `r9gate3_restart.sh` writes it from argument 2 (restart:16,29). NOTE says `"V V_d1"` in one restart is unsafe because the pinned driver runs for all arms and D1 needs E3's `config.json` (NOTE Q3.4).

Manual route that needs no code (usable only as a fallback): at the right moment kill the SUPERVISOR first, then the trainer by `train.pid` (SUP:51, 56-58 rule; trainer pid file SUP:517). Cost: chain logs `V-FAILED` + `ALLDONE`; the GPU lock is freed when the chain exits; you must then start the D1 chain and the E3 resume by hand. Also: `torch` DataLoader workers (6) must be gone and `nvidia-smi` must show 0 compute apps before D1's gate, because the chain STOPs `GPU-NOT-EMPTY` otherwise (CH:115).

SMALLEST CORRECT ADDITION (one-shot supervisor pause; no trainer change, no argv change):
1. SUP poll loop (SUP:521-538): add an optional manifest variable `PAUSE_AT_STEP` (e.g. 30000) and a pause latch `${OUT_DIR}/PAUSED.json`. While the trainer runs, poll (shorter than POLL_S, e.g. 15 s, only while waiting for the pause) for: `${OUT_DIR}/${FULL_CKPT_FMT % PAUSE_AT_STEP}` exists AND `metrics.jsonl` contains an eval row at step == PAUSE_AT_STEP (so the step-30,000 eval is not lost) AND `PAUSED.json` is absent. Then SIGTERM the trainer by `TRAIN_PID`, `wait` it, write `PAUSED.json` (step, ckpt sha256, utc), and `exit 7`. Do not run the zerograd gate, do not write the done-marker.
2. SUP startup: `PAUSED.json` present => the pause is already spent: do not pause again (this is what lets the SAME manifest resume E3 without pausing a second time). A resume is then an ordinary supervisor start on the same manifest (SUP:484 loop, resumes `ckpt.pt`).
3. CH:142-145: add `elif [ "$RC" = 7 ] && [ -e "$OUTD/PAUSED.json" ]; then ev "$A-PAUSED" ...` BEFORE the `-FAILED` branch, and keep `flock -u 9` after it (CH:141 already releases first).
4. Order the arm list so the chain does the whole thing itself: `ARMS="V V_d1 V"`. Re-entry of the third item works as written: not done (CH:91), pinned evidence present (CH:94-96), the existing token is reused when `verify` returns MATCH (CH:100-104), lock (CH:113-115), env manifest rewritten identically (CH:120-132), supervisor resumes (SUP:484+). HYPOTHESIS (not read in detail): variables like `PASSED`/`TRAINED` tolerate a repeated arm; no code reviewed assumes uniqueness. A test must pin that.
   - V_d1 needs, BEFORE the chain reaches it: its own dev-box G-SUITE-PINNED evidence (CH:94 waits at most `PINNED_WAIT_S`=14,400 s, then SKIPS the arm, CH:95-97) and its own OVF records (below). `restart` launches the pinned driver for all arms in the list at once (restart:43), so the D1 arm's pinned run would fail without `config.json` (NOTE Q3.4). So either (a) do not put V_d1 in the first restart and instead write `$R/gate3/ARMS` = `V V_d1 V` afterwards ... NOT POSSIBLE: ARMS is read once at CH:22 (MEASURED), or (b) run the pinned evidence for V_d1 by hand from a prepared `config.json` before the first restart (path-map mechanism, NOTE Q3.3), or (c) accept two chain invocations (first `V`; the pause makes it end with `-PAUSED` + `ALLDONE`; then the D1 chain; then a third "resume" invocation that runs the supervisor directly, see Plan step 8). (c) needs the least new code (CH:142-145 only, and SUP).
5. Files to change (all inside `stack/`, so they move the tree sha): `stack/ops/sup_refcv7.sh`, `stack/ops/r9e1_gate3/r9gate3_chain.sh` (+ its repo copy, shipped to `$R/kit` by restart:28), plus a test. `r9_e3_make_argv.py` and `launch_gate.py` need NO change if `PAUSE_AT_STEP` is a manifest variable (manifest `runs.d/<ARM>.env` is generated by CH:120-132, so add `PAUSE_AT_STEP=30000` there for the non-D1 arm only; the manifest is not bound by the token).

Alternative considered and rejected: a trainer-side `--stop-after-step` flag. It changes the argv (new argv sha, a new G-DVB entry is mandatory, `LG` refuses an unregistered flag per the launch-gate rule in CLAUDE.md) AND the D1 argv must equal E3's except the 4-flag delta, so D1 would then carry the flag too or the gate's `wsd_branch_reasons` refuses (LG:3544-3556: only `--out --steps --wsd-branch-from --wsd-branch-decay` and exempt `--save-at` may differ). Not smaller.

Landing constraint (MEASURED): the token binds `tree_sha256` over `stack/` + `taniteval/` (LG:2149-2170, 11146-11190), the OVF records bind the closure of the model/harness, and a closure-touching land voids them (NOTE Q1; memory `land-closure-change-voids-bound-records`). Therefore the supervisor/chain change must land BEFORE E3's gate, records and token are minted. A land after E3 launches would void E3's token and forbid the resume.

## 3. GATE / TOKEN

- Token binds: kind, HMAC, verdict PASS, `commit`, `argv_sha256` and the argv itself, `tree_sha256`, and a data manifest (fingerprint of every non-output path flag; `--out` and `--eval-window-dump` are output flags and excluded) (LG:11146-11190, LG:169, LG:2109-2122). No expiry field exists (grep for expir/ttl/max_age: only subprocess timeouts).
- Every supervisor launch re-verifies (SUP:34-38 G2, loop SUP:494-505) and `exec` re-verifies again (LG:11219-11235). "A relaunch after a crash is a launch." A resume after D1 is the same event.
- Resume flags: none. The resume source is `<out>/ckpt.pt`, found by the trainer itself (T:11188), so the argv sha is unchanged and `GATE_ARGV_FILE` stays the one gated argv (SUP:121-151). => The existing `PASS_<c12>.json` MATCHes on resume. The chain even reuses it on re-entry (CH:100-104).
- What would break MATCH: any change under `stack/` or `taniteval/` (tree sha), a different commit passed, a rewritten argv file, a data-file/dir change in the E3 inputs (directory fingerprints hash relpath+size only), a different HOST key (HMAC key per host, `load_key`).
- D1 needs its OWN token under profile `refcv9-full-d1` (`r9_profile_of_arm`, ST:51-59; profile LG:1946-1957), its own OVF records (4 GPU bind runs; D1's argv sha differs; NOTE Q2), bound at the same REC commit. Not reusable from E3 (INHERITED from NOTE Q1/Q2, plus MEASURED LG binding by argv sha at 7048-7062 per the NOTE; I did not re-read those lines).

## 4. D1 side

- D1 argv = E3 argv with exactly 4 flags changed: `--out`, `--steps 36000`, `--wsd-branch-from <E3 out>/ckpt_full_30000.pt`, `--wsd-branch-decay 6000`; `--save-at` dropped (MK:155-172; profile LG:1946).
- Gate: `wsd_branch_reasons` (LG:3522-3560): file name must be `ckpt_full_30000.pt`; `<ckpt dir>/config.json` must exist and its `argv_effective` equal D1's argv outside the delta; `profile_argv_rules` runs it in every gate stage (LG:3395). Trainer: `branch_preflight` (LR:253-270) re-checks the same, `check_branch_checkpoint` (LR:286-309) needs keys model/opt/step/data_pos and a STABLE-phase step with `--steps == step + M`; `stage_branch` copies the source to `<D1 out>/ckpt.pt` with md5 check and a `wsd_branch_source.json` record (LR:312-359); a fresh `--out` is required (LR:335-341), and a relaunch of D1 into its own out dir resumes its staged ckpt.pt.
- The source checkpoint's sha256 is bound as a DATA input of D1 (`--wsd-branch-from` is a path flag and not an output flag; LG:2109-2122, fingerprint LG:2237-2245). So D1's REAL gate and token must be minted after `ckpt_full_30000.pt` exists, and the file must not change afterwards. The dev-box pinned run may use a path-map to a stand-in (NOTE Q3.3, INHERITED; whether a missing file is tolerated by every devbox check is UNVERIFIED there and here).
- Blockers on a D1 restart while E3 is unfinished (all MEASURED):
  a. GPU lock: held by the chain for E3's whole training (CH:113-141), and every gate step takes the SAME lock (NOTE Q3, INHERITED; the supervisor-side lock is `$R/runs.d/<ARM>.lock`, SUP:155, 209). Needs the E3 supervisor gone and the GPU empty (CH:115).
  b. ARMS read once (CH:22) and `ALLDONE` is only reached at the end of the loop (CH:146-148).
  c. E3 run dir state: not done => the chain does not skip it (CH:91); `ZEROGRAD_FAIL.json`/done-marker absent after a pause, so a later supervisor start is legal (SUP:182-204).
  d. D1 final gate also needs `$R/gate3/zerograd_controls.env` (CH:84) and its own `runs.d/r9e3_V_d1.argv.json`+pi_cost approval (NOTE Q5.1).
  e. G-EVAL/G-LIVE on Thor need the GPU (CH:106 -> thor_gate final): impossible during E3 training, possible in the pause window.

## 5. Smallest end-to-end plan, with tests BEFORE E3 reaches step 30,000

Ordered (O = new work, E = exists):
1. (O) Land the pause: `sup_refcv7.sh` (`PAUSE_AT_STEP`, latch `PAUSED.json`, rc 7), `r9gate3_chain.sh` (`-PAUSED` branch; CH:142-145), `PAUSE_AT_STEP=30000` in the manifest heredoc (CH:120-132) for non-D1 arms only. Must land before E3's records/token are minted. Do not touch `refc_v3_train.py`.
2. (E) Generate argvs: `r9_e3_make_argv.py` -> `r9e3_V.argv.json`, `r9e3_V_d1.argv.json`, PI-cost approvals (NOTE Q5.1).
3. (E) Bind OVF records for E3 AND D1 while the GPU is free, at one REC commit (NOTE Q1/Q2; 4 Thor GPU runs). Run the D1 bind `--dry-run` first (NOTE says the D1 bind before the checkpoint exists is UNVERIFIED).
4. (E) Restart for E3 only: `R9_STUDY=e3 bash r9gate3_restart.sh <SHA> "V" <REC12>`. Token `PASS_<c12>.json` for V is minted by the chain.
5. (E, while E3 trains, CPU only) dev-box pinned evidence for `V_d1` from a copy of E3's `config.json` with a path-map (NOTE Q3.3).
6. (O/E) At step 30,000 the supervisor pauses (step 1), chain logs `r9e3_V-PAUSED`, then `ALLDONE`, lock released. Operator verifies `PAUSED.json`, `ckpt_full_30000.pt` sha256 and `nvidia-smi` empty.
7. (E) D1: `R9_STUDY=e3 bash r9gate3_restart.sh <SHA> "V_d1" <REC12>`; gate under `refcv9-full-d1`, 36,000-step training (~94 h at 9.4 s/step per NOTE; the PI's ~17 h applies to the 6,000 decay steps; the stage_branch start is step 30,000, so the arm trains ~6,000 steps, ~15.7 h ESTIMATED from the 9.4 s/step K0 rate), `V_d1-DONE`.
8. (E, thin) Resume E3: start the supervisor on the SAME manifest `runs.d/r9e3_V.env` under the Thor lock (same command the chain uses at CH:113-139 with `PAUSED.json` present, so no second pause), or re-run the chain with `ARMS=V` (CH:91 not done, CH:100-104 token MATCH, then CH:113+). `verify` MATCH is the precondition; if the tree changed, STOP, do not edit the token.
9. (E) At step 73,280 the normal completion path (zerograd gate on `ckpt.pt`, done-marker) applies unchanged.

Tests that prove each step before step 30,000 (all CPU / tiny):
- T1 pause trigger (step 1): extend the style of `stack/tests/test_sup_refcv7_zerograd.py` (a fake PYTHON/trainer that writes metrics rows, a `ckpt_full_<N>.pt` and an eval row, then sleeps). Assert: supervisor SIGTERMs by train.pid, writes `PAUSED.json`, exits 7, writes no done-marker, does not relaunch; deliberate-regression arm: delete the pause block and require RED (the relaunch counter reaches 2). Cases: file present but no eval row => no pause yet; `PAUSED.json` pre-existing => no second pause; pause step 0 / unset => behaviour byte-identical to today.
- T2 chain classification (step 3): drive the CH:142-145 tail on a fake `$OUTD` with `PAUSED.json` + rc 7 => event `-PAUSED`, not `-FAILED`; and the existing `-DONE`/`-FAILED` cases unchanged (`test_r9gate3_study_e3.py`, `test_r9gate3_study.py` style).
- T3 resume = continuation: tiny CPU run of `refc_v3_train` with `--lr-schedule wsd --save-at k --save-every k --steps 2k`, kill after step k (as T1 would), relaunch the same argv, compare losses from step k with an uninterrupted run (pattern of `test_resume_rng.py`, `test_resume_continues_the_data_order.py`); also assert the LR at step k+1 equals the uninterrupted run's and that the resumed `config.json` `argv_effective` equals the first launch's.
- T4 token across the pause (step 3/8): with a tiny profile run `launch_gate.py verify` before launch, after the supervisor pause and after the resume: all three MATCH; then touch one byte under `stack/` and require REFUSED (the negative control that proves the guard is live).
- T5 D1 from the paused run (step 7): feed the T3 run's `ckpt_full_k.pt` + `config.json` to `branch_preflight`/`stage_branch` (LR:253-359) and `wsd_branch_reasons` (LG:3522) with a miniature profile; assert a 4-flag-delta argv passes and one extra differing flag refuses.
- T6 duplicate arm: a dry run of the chain loop with `ARMS="V V_d1 V"` (stub gate/supervisor) proving the second V re-enters via CH:91-104 and the first-V pause latch is not re-fired; or skip by using the two-invocation route (plan step 8).
- Also before E3 starts: confirm on a pod/Thor copy (not here) that killing the trainer leaves no orphan dataloader workers holding VRAM (CH:115 would STOP otherwise); this cannot be proven CPU-only (HYPOTHESIS).

## Missing pieces (summary)
1. A supervisor pause (`PAUSE_AT_STEP`, `PAUSED.json`, rc 7) and a chain `-PAUSED` branch; none exists (SUP:540-565 relaunches; CH:142-145 has no paused state).
2. Both must land before E3's token/records are minted (the tree sha covers `stack/ops`).
3. D1's own OVF records, token, argv, pinned evidence, `zerograd_controls.env` (NOTE Q5).
4. A resume invocation after D1 (re-run supervisor/chain on the same manifest; ARMS is read once, CH:22).
5. Tests T1-T6.
6. Unverified: orphan GPU processes after the kill; D1 bind before the checkpoint exists; tolerance of devbox checks to a missing source checkpoint; arm-repeat in the chain loop.

No gate check is relaxed by any of this: the pause changes only supervisor/chain behaviour, the token and the profile rules stay as written.
