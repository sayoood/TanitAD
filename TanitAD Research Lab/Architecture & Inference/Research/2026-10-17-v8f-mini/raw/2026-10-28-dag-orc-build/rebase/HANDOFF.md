# HANDOFF - DAG-ORC re-based on DAG-OACC (instrument build), written 2026-10-27 ~23:55

Everything is BUILT; four detached CPU jobs are still producing the proof artifacts. Nothing is installed into box2_ship, nothing launched, no git write. Read `NOTES.md` (what changed where) and `LAUNCH.txt` (install + launch) first.

## Running now (detached, `nohup`; they survive the agent)
| job | log | ends with | pass condition |
|---|---|---|---|
| chain self-test + chain/asserter/queue/waiter MUTATION proof (`chain/selftest_chain.py`) | `chain/logs/selftest_chain_try2.out` | writes `chain/selftest_chain_result.json`, last line `SELFTEST: n/n tests pass; MUTATION: k/k mutants killed` | rc 0: 0 failures AND 0 survivors. At 23:53: 62 results, 0 FAIL, 0 SURVIVOR. |
| six repo label tests on a scratch overlay of the tip (`chain/run_label_tests.sh`) | `chain/logs/label_tests_driver.out`, `label_tests_edited.out`, `label_tests_negative.out` | `ZZLABEL-DONE` | edited tests PASS; the negative control (unedited tests, patched queue) goes RED on the label sets |
| reader lane A (`read/run_lane_a.sh`): selftest -> dry-run -> K-RC leg 2 (`--rc-oacc`, 0 differences) -> smoke stand-ins -> K-RC leg 1 (`--rc-dag2`) | `read/logs/lane_a.out`, `read/logs/final_<step>.out` | `ZZLANEA-DONE` or `ZZLANEA-STOP <step>` | selftest 330/330; RC_OACC.json PASS 0 differences; RC_DAG2.json PASS; SMOKE.json written |
| reader MUTATION harness (`read/dag_orc_mutation.py --workers 2`) | `read/logs/mutation_final.out` | `read/mutation.json` | CONTROL green, every mutant KILLED (about 100 mutants, each = a full reader selftest, so ~1 h) |

The reader file `read/dag_orc_read.py` is FROZEN at sha256 `e4868c15493288fef078e1452246c5b994d13bf967b37d94e37823cdf04864af` (the only change since the first RC_OACC run was the smoke stand-in for REP, which crashed on a missing `t1_logged`). `RC_OACC.json` / `DRY_RUN.json` / `RC_DAG2.json` are valid only if they carry this reader sha: lane A regenerates all three on this sha. If anyone edits `rebase_reader.py`, regenerate with `python rebase_reader.py`, then rerun lane A AND the mutation harness.

## Exact next steps for the fresh agent
1. `tail` the four logs; a job without its end marker and with no live process (`powershell Get-CimInstance Win32_Process | ? CommandLine -match 'selftest_chain|dag_orc|run_label'`) died: rerun it alone (they are independent). Rerun commands: `python chain/selftest_chain.py` ; `bash chain/run_label_tests.sh` ; `bash read/run_lane_a.sh` ; `python -I read/dag_orc_mutation.py --workers 2 --out read/mutation.json` (CPU only, `CUDA_VISIBLE_DEVICES=-1`; check `python C:/Users/Admin/orcread/pytest_runs.py` shows < 2 before the label tests).
2. Any FAIL / SURVIVOR / RED: fix the generator (`build_asserter.py`, `build_queue.py`, `make_chain.py`, `build_small.py`, `rebase_*.py`), regenerate in the order `build_asserter.py` -> `make_chain.py` (re-pins the asserter sha) -> `build_small.py`, rerun the affected job. Never hand-edit a generated file.
3. `python make_manifest.py` (refreshes `MANIFEST.txt`); put the final counts into `NOTES.md` section 3.
4. Hand to the Master Mind: counts, install list (`LAUNCH.txt` section 1), the launch command (`LAUNCH.txt` section 2), the open items below.

## Open items (not coded / not decidable here)
* The trainer's `--ego-perturb-with-dagger` and its composed-batch G-LIVE fields are another agent's build: the asserter checks the swap probe + the ego probe as they exist, the composition flag built True on A / absent on REP, and `ego_dagger_overlap_n` (optional in the probe, REQUIRED exactly 0 in the run record). Add clauses if the new trainer adds a composed-batch probe, then `make_chain.py`.
* The queue stage `:dagoaccorcarm` (+19 lines) and the six label-set tests must be landed (stage dir) before a real launch; the lever grep REFUSES a tip without `add_argument("--ego-perturb-with-dagger"`.
* No fake-box end-to-end of the post-gate real-run branch (`gate_one` / `runone`); its decision points and the queue's non-dry gate/train path are tested (self-test sections A, K, R).
