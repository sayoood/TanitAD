# DAG-ORC box-2 queue stages and chain (PREREG_V8F_DAG_ORC section 1 / 10) -- BUILT, NOT INSTALLED, NOT LAUNCHED

Everything is under `E:/Projects/TanitAD-artifacts/dag_orc_chain/`. Nothing was committed, pushed, installed into `C:/Users/Admin/box2_ship/`, or sent to box 2. No GPU. Scratch (disposable): `F:/TanitAD-scratch/dagorc_*`.

## What is here

| path | what |
|---|---|
| `stage/TanitAD Research Lab/.../2026-10-17-v8f-mini/code/v8f_mini_box2_queue.cmd` | the queue with the new stages (tip 3b7b404e + **20 inserted lines, 0 removed**; sha256 in MANIFEST). `queue.diff` is the diff, `build_queue.py` regenerates it byte for byte from `tip/queue_3b7b404e.cmd` |
| `stage/stack/tests/test_v8f_{orc_chain,flat,oacc,kind,mixo,lora_ar2}.py` | the six queue-reading tests that enumerate labels, each edited to declare the one new label `dagorcarm` (`apply_test_edits.py`) |
| `box2_ship/chain_v8f_dagorc.sh` (+ `.in`, `make_chain.py`) | the chain. Generated from the template; every pinned sha256 is COMPUTED from a banked artifact. `make_chain.py --check` = byte-equal to a fresh regeneration |
| `box2_ship/gate_dagorc_assert.py`, `v8f_dagorc_chain_checks.py` | per-arm gate assertions + token; run record (K-GATE/K-INIT), K-REACH, ONE-LEVER |
| `box2_ship/v8f_guard_dagorc.sh`, `wake_v8fdagorc.sh`, `run_chain_v8f_dagorc.cmd` | guard (DAG-OACC's, line for line but one comment), waker, launcher (AFTER_OUT baked in = DAG-OACC r2 chain's log) |
| `selftest_dagorc.py`, `fakebox.py`, `mutate_dagorc.py` | the proof (sections S A B C D E), the fake box 2 for section E, the mutation runner |
| `reach_synthetic.py`, `reach_real_ckpt.py` (+ `.json`, `fixtures/`) | reachability through the REAL tip trainer: synthetic world (gate + short training + trainer mutation) and the REAL DAG_s checkpoints |
| `pytest_sweep.log` | one pytest file at a time on the composed tree: all 15 queue-reading files + `test_v8f_ego_perturb.py`, `test_v8f_dagger_train.py`, `test_v8f_init_ckpt_o_interface.py` |
| `chain_dry_run.txt`, `selftest_output.txt`, `mutation_output.txt` / `mutation.json`, `MANIFEST.txt` | the records |

## The stages (INSERTIONS ONLY)

Three arms ride the existing `gatedag / traindag / evaldag` stages through `:dagname` / `:dagarm` hand-off lines and one new block `:dagorcarm` (so the pinned-argv + `fc /b` + launch-gate-token machinery is reused, not copied). All three init from DAG_s (`M2b_s<S>_dag`), `--steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --dagger-p 0`:

* `dagorc` (A) = DAG2-0's banked argv + `--ego-perturb-p 0.3` + ORC's five law flags (vscale 0.5 1.0, vmax 8.0, adrop 1.5, yawrate-std 2.0, seed = S)
* `dagorcrep` (REP) = DAG2-0's argv with `--seed 10` / `11`, p 0, no law flag
* `dagorc01` (B) = A with p 0.1
* an extra argument (a dose) is REFUSED rc 2.

## Reachability (CLAUDE.md "built, tested, and unreachable from its caller") -- no launch blocker

Traced: argparse `--ego-perturb-*` (train.py 125-136) -> `FLAG_CHECKS` `ego:` entries (87-88) -> `make_ego_perturb` (309) -> `bad_ego_perturb` (315; called 576) -> `ego_perturb_probe` (called 1527; controls EXACTLY 0.0) -> `b = ego.apply(b, step)` in the loop (1602) -> `model.loss` (1624) -> backward (1629). `--init-ckpt` and `--dagger-p 0` combine with p > 0 (the trainer refuses only `--dagger-p > 0` together with p > 0). Mutated, not inspected: `reach_synthetic.py` deletes the loop line in a COPY of the trainer -> the gate still PASSES and `ego_perturb_n` reads 0, i.e. **the gate does not cover the loop**, which is why the chain enforces **K-REACH after every run** (A/B n/eligible within p +-0.02; REP exactly 0 with eligible > 0). Every one of the 30 flags the six production argv use has a `FLAG_CHECKS` entry (`reach_real_ckpt.py` R5). The REAL DAG_s checkpoints load into the models these argv build (strict, read back, sha256 = pin) and the config check refuses width / grid / history / doctored-cfg (R3). Both checkpoints carry `cfg.o_interface = 'full'`, so the F1 hunk on the tip is live for them.

K-BUILD: the tip trainer differs from 1d4f4fbb (the commit DAG2-0 trained at) only by the F1 `INIT_CKPT_MATCH` `o_interface` hunk (landed 793a78e); data / model blobs identical. The chain records the diff in `$ART/K-BUILD.txt` for the reader to judge.

## Chain (PREREG section 10 order, one job at a time, SAFE MODE)

wait for AFTER_OUT (REQUIRED, no default) -> take BOX2_BUSY atomically -> refresh box 2 from the **REMOTE** tip with a small export (<= 300 MB asserted) and REFUSE a tip without the stages / levers -> pin checks (key-file sha256, v9 md5, DAG_s ckpt/argv/summary on box 2, DAG/DAG2-0 argv/summary here) -> the four A/REP gates (each asserted HERE, token written only after) -> A s0, REP s0, A s1, REP s1 (train, eval, hash, pull behind the 1 GiB floor, run record, K-REACH) -> B s0/s1 (unless `DAGORC_SKIP_B=1`) -> pins again -> DONE. Pulls go to `E:/Projects/TanitAD-artifacts/v8f-dag-orc/`.

## Proof (all CPU, no box-2 contact)

* **Self-test** (`python selftest_dagorc.py`, `selftest_output.txt` / `selftest_result.json`): **150/150 checks pass, 230/230 deliberate regressions caught, 0 missed** (sections S A B C D E, 344 s). S: all 245 stage invocations of the 50 pre-existing stages print byte-identical output on the tip queue and the staged queue, and 4 real (non-dry) runs behave identically; the diff is insertions only. A/B: the three arms render exactly DAG2-0's banked argv + the registered change; a real (non-dry) queue run on a fake box pins the training argv and runs the pinned line. C: the gate asserter on DAG2-0's REAL gate record + ORC's REAL ego probe + the real tip trainer's gate (133 doctored negatives), the run record, K-REACH (band edges, the loop-line-deleted run), ONE-LEVER. D: syntax, regeneration, pins vs the banked artifacts, dry-run (+5 doctored queues), `--check-tree` (+7 doctored trees), the predecessor rule (9 verdicts), 9 early stops each with ZERO box-2 calls, wiring pins. E: the chain end to end against the fake box: all six runs in PREREG order with every pull, token, run record and K-REACH line; `DAGORC_SKIP_B=1`; and 13 faults (wrong init checkpoint, dead ego probe, REP applied the perturbation, A never applied it, argv.json not the planned argv, job rc 1, wrong DAG_s on box 2, wrong v9 md5, wrong key-file sha, a pull with one wrong byte, an eval that rewrote the checkpoint, a remote tip without the stages, a remote tip whose trainer lost the loop's `ego.apply`): each STOPs with the named marker, releases BOX2_BUSY and leaves every later run untrained.
* **Mutation proof** (`python mutate_dagorc.py`, `mutation.json`, `mutation_output.txt`): 83 single-defect mutants of COPIES (20 queue, 25 asserter, 12 run-record/K-REACH, 21 chain, 5 guard/waker/launcher); **83 killed, 0 survived, 0 anchor errors, 0 harness errors; control green** (queue copy S,A,B,C,D and ship copy C,D,E[subset] both pass). 10 chain mutants (K-REACH not enforced, tip levers not required, v9 md5 / key-sha / DAG_s pin not a stop, pulled-checkpoint hash, eval-rewrote-checkpoint, busy marker not released, stage list, ...) are killed by section E ONLY, i.e. by running the chain, not by reading its text. The regeneration guard (D3) is off for them. The FIRST full run (`mutation_run1_output.txt`) had 71 killed / 1 survived / 11 harness errors: the survivor (w01, a waker marker renamed) exposed a substring-match pin, and the harness errors were rig bugs (a chain that waits forever hung the rig; D6 index errors; an unlink); all repaired, then the whole set re-run.
* **pytest** (`pytest_sweep.log`, one file at a time on a composed tree built by `git archive 3b7b404e` + the staged queue + the six staged tests): all 15 queue-reading test files + `test_v8f_ego_perturb.py` (82) + `test_v8f_dagger_train.py` (97) + `test_v8f_init_ckpt_o_interface.py` (7): **1,519 passed, 0 failed**. (`test_v8f_lora_ar2.py` first failed 2 + 14 errors only because my composed tree lacked `2026-10-17-refav1-hj-mini/code/pull_encode_pooled.py`; with it added: 64 passed. Both runs are in the log.) Negative control (`pytest_unedited_tests_negative_control.log`): the SIX tests UNEDITED (tip versions) against the staged queue each fail on exactly one assertion, the new-label set (`dagorcarm`), and nothing else, so the six edits are necessary.
* **Reachability** (`reach_synthetic.json`, 29/29: the tip's REAL trainer gate accepts the queue's argv for A / REP / B at both seeds, refuses 4 doctored ones, a short real training gives `ego_perturb_n/eligible` ~ p for A and B and exactly 0 for REP, and the same run with the loop's `ego.apply` line deleted passes the gate and reads n = 0; `reach_real_ckpt.json`, 29/29: the real DAG_s checkpoints, the 30 flags' G-DVB entries, the refusals).

## What the proof does NOT show (read before trusting it)

* Section E drives the REAL chain, the REAL queue lines, the REAL gate asserter / run record / K-REACH against a FAKE box 2 (ssh/scp shims over a local tree). The fake box's gate record is DAG2-0's REAL gate selfcheck with ORC's REAL ego probe spliced in (the real tip trainer's own synthetic-world gate is covered by `reach_synthetic.py`), its run summaries are DAG2-0's real summaries relabelled, its checkpoints are stand-ins, and the v9 md5s are read from the local copies. It proves the chain's DECISIONS (order, stops, pins, enforcement), not that box 2's python will run.
* The first real gate on box 2 is the first time the production-sized trainer gate runs on these argv; the ego probe fixtures for A/B come from ORC's gate, the REP fixture from the tip trainer.
* The asserter's `token` mode re-runs `check`, so `DAG-ORC gate: PASS` prints twice per gate; the waker therefore counts the `gate token written for` lines (4 = the A/REP gates), not the PASS lines.
* REP naming `--ego-perturb-seed 0` is invisible to the TRAINER's own gate (0 is the default); only the asserter's line check refuses it (doctored negative in section C).

## Install / launch (NOT DONE -- needs the Master Mind and the PI's go)

1. Land `stage/` (the queue + the six tests) on the remote tip of `agent/arch-inf-20260803` and push. The chain exports from `git ls-remote origin`, so a tip without the stages makes it STOP at the refresh (tested: `E_oldtip`, `E_noapply`). Verify: `bash chain_v8f_dagorc.sh --check-tree <checkout of the tip>` -> rc 0.
2. Install, never overwrite: `cp -n box2_ship/{chain_v8f_dagorc.sh,gate_dagorc_assert.py,v8f_dagorc_chain_checks.py,v8f_guard_dagorc.sh,wake_v8fdagorc.sh,run_chain_v8f_dagorc.cmd} C:/Users/Admin/box2_ship/` (`oaf_lib.sh`, `v8f_oaf_chain_checks.py`, `mk_v8f_job.py`, `waitjob.sh` are already there and are sha-pinned / used as-is). Two live chains use that directory.
3. Launch on the PI's go: the `.cmd` launcher detached (WMI `cmd /c ... < NUL`; the shell tool kills children otherwise); `bash wake_v8fdagorc.sh` as the script-polling waker. AFTER_OUT is the DAG-OACC r2 chain's log: GO after its DONE or after a STOP of its own once BOX2_BUSY is released; HALT on a guard stop or a held BOX2_BUSY.
4. E: has ~6 GB free; the chain estimates ~0.6 GB for six runs and refuses (ZZDISK-FLOOR-STOP) if a pull would leave < 1 GiB.
