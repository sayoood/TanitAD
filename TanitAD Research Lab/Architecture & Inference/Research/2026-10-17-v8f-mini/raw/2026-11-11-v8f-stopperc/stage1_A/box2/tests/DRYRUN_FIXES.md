# STOPPERC dry run: changes to the package under test (2026-11-12)

The dry run changed `stopperc_chain.sh` and `stopperc_check.py`. Old shas are those of the package as received.

| file | old sha256 | new sha256 |
|---|---|---|
| `stopperc_chain.sh` | a62733d59614bc84738952773c8736ac6b396acbf32f6a21692c5bf2e868e4f9 | f69d10973c2c263b074f0ff90c24129a89b4693fddea4bd10434cbe7b033a51f |
| `stopperc_check.py` | a135a0eae000ef799fea83e0716d0b22732f9d008d32dc9ad865fd19af665004 | 4e3dca4197e1c963a25c74bf630bbe5987b1c6daa83078d35903c6bedab54f3f |

Unchanged: argvs_stopperc.txt (b005ea0b...), make_argvs_stopperc.py, both job .cmd files, and every STOPBAL file (stopbal_check.py 70e260a6...5212 is still the pin).

## 1. REAL BUG (fixed): a W that rounds to 0.0000 passed the W smoke
`wsmoke_stage` accepted any `^[0-9]+\.[0-9]{4}$` W, including `0.0000`. The checker refuses a non-positive `STOPPERC_W` (`stopperc_check._w_final`), but only when the first A argv is built, i.e. AFTER NULL_s0 and NULL_rep2 (about 80 min of box 2). Found by dry-run scenario 3 (goal term 1e-5 gives W 0.0000: the chain ran on to A_s0).
Fix (chain, line 86): after the regex, require `float(W) > 0` or `die 20` (STOP rc 14 at the W smoke, no run spent). The dry run now stops there (rc 14, no `done/wsmoke.done`, no run on the box).

## 2. TEST HOOKS (not bugs; both inert on a real launch)
A throw-away repo cannot carry the code-of-record sha and a fake BAL checkpoint cannot hash to the pinned 5169ea95, so two hooks were added, following the base checker's own pattern (`STOPBAL_FAKE_*`, gated on the fake-box switch):
* `stopperc_chain.sh` line 11: `COMMIT` is replaced by `$STOPPERC_TEST_COMMIT` ONLY when `STOPPERC_DRY=1` (stub transport; the sourced common script refuses DRY unless both ssh and sftp carry the stub marker). A real launch has DRY unset. The dry run proves the default commit is 3e264bbf (scenario 6, `nocommit`: ship folder keyed by it, refusal rc 8) and that a stub without DRY refuses (rc 16).
* `stopperc_check.py` `bal_pin()`: `STOPPERC_TEST_BAL_SHA` replaces the pinned BAL-NULL_s0 sha ONLY while `STOPBAL_ROOT` (the base's fake switch) is set. Without the hook the fake BAL checkpoint is UNKNOWN and the chain stops rc 13 (scenario 2, `unknown`): the real pin is exercised too.

Unit tests re-run after both edits: `36 passed`.

## 3. REAL BUG (fixed 2026-11-13, MM): G-LIVE8 built V8FMini without its prior (found on the REAL box, after A_s0 trained)
`eval_build_problems` called `M.V8FMini(M.V8FMiniConfig(**ck["cfg"]))`. At `3e264bbf`, `V8FMini.__init__(self, cfg, prior, cons_mu=None, cons_sd=None)` (`stack/tanitad/models/v8f_mini.py:707`) needs the prior. A_s0 trained 3,000 steps on box 2 (trainer rc 0, 15:52-16:30), and the post check then failed: *"G-LIVE8 the eval loader cannot build the ON checkpoint: TypeError: V8FMini.__init__() missing 1 required positional argument: 'prior'"*. Chain STOP rc 5 at 16:31:38.
Why no test caught it: the dry run sets `STOPPERC_SKIP_EVALBUILD` (stand-in trainer, no real checkpoint), and no unit test called `eval_build_problems`. The README listed it under NOT verified ("`eval_build_problems` is untested against a real ON checkpoint"). A check that is never run is a check that does not exist.
Fix: build exactly as the eval loaders do, `M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))` (`3e264bbf` `stack/scripts/v8f_t1_eval.py:510`, `v8f_mini_eval.py:262`), then a strict load.
Evidence (MEASURED on box 2 against the REAL A_s0 `ckpt_last.pt` with the shipped `3e264bbf` tree, `D:\tanitad\stopperc\repo\stack`): the fixed route builds and loads strictly (8 `stop_*` tensors). The mutant without the prior reproduces the job's exact TypeError. A control with one `stop_head` tensor removed is refused by the strict load (RuntimeError).
Tests: `test_eval_build_passes_a_well_formed_on_checkpoint_with_the_loaders_prior` and `test_eval_build_red_arm_a_missing_head_tensor_is_refused` (a stub `v8f_mini` with the real constructor signature). Mutation: putting the prior-less call back turns both RED with the job's message. Unit tests: `38 passed`.
Re-run: the old A_s0 output on box 2 was moved aside to `D:\tanitad\runs\v8f_stopperc\A_s0_postfail_20261113` (ckpt sha256 `6871633f...e2b5`). The trainer resumes from an existing `ckpt_last.pt` ("resume wins", `v8f_mini_train.py:243`) and appends to `train_log.jsonl`, so re-running into the old folder would have reloaded step 3,000. A_s0 retrains from scratch. A same-seed retrain is expected to reproduce that sha (box-2 training is deterministic at a fixed seed); a mismatch is reported, never hidden.

## 4. REAL BUG (fixed 2026-11-13, MM): G-LIVE6 compared EVERY A arm's own step-0 ratio with the smoke's W (refused a correct A_s1)
The card fixes ONE W: *"W is fixed at the gate smoke so that the term equals T's goal-BCE term at step 0, never tuned"*. The chain writes one `W.txt` from the A_s0 smoke, and every A argv carries it. G-LIVE6 nevertheless required `stop_aux_probe.w_stop_aux_for_goal_equal` of EACH A arm to round to the argv W. That holds on A_s0, which has the smoke's seed, init and probe batch. It cannot hold on A_s1, which starts from `M2b_s1` with another probe batch. Real box: A_s1's gate read **0.06278651564882848 (probe_n_pos 2, n_has 57)** against W **0.0533** and refused it (chain STOP rc 4, 17:36:00), after A_s0 had banked.
Why no test caught it: every arm-A unit test ran at `A_s0`, and the dry run's stand-in trainer emits the same probe values at every seed. A seed-dependent clause had no seed-varying input. (Same family as section 3: a path the dry run cannot exercise is a path that has not been tested.)
Fix: `WSMOKE_TAG = "A_s0"`. G-LIVE6 compares with the argv W on that arm only. On every other A arm the reading must be finite and > 0, and the read reports it beside. It never refuses.
Evidence (MEASURED on the REAL selfchecks): the fixed checker's full `gate_problems` gives `[]` on A_s1's real gate selfcheck (pulled from box 2), and the old checker gives exactly the one G-LIVE6 line from the STOP. On A_s0's banked real selfcheck (probe 0.05327 -> 0.0533) the fixed gate gives `[]`; with A_s1's ratio planted into A_s0 it is still refused.
Tests: `test_live6_compares_w_on_the_smoke_arm_only`, `test_live6_other_seed_still_needs_a_finite_positive_reading`. Mutants: always comparing -> RED; dropping the finite check -> RED. Unit tests: `40 passed`.
For the read: at s1 the stop term at step 0 is 0.0533 / 0.0628 = 0.85 of the goal term (1.00 at s0 by construction). Reported beside, as the card fixes W once.
