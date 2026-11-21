# DAG-ORC re-based on DAG-OACC (Amendment 1, A1-2026-10-27): build notes

Built by the instrument subagent, 2026-10-27. CPU only; box 2 never contacted; no git write; NOTHING installed into `box2_ship` (the Master Mind installs after review, list in `LAUNCH.txt`).
Spec = `AMEND1.md` (raw-byte sha256 `f2f9615748451fd7d1b832ab1b04c45f4285621298f9f35a0cad9e9b75725ac1`, a copy sits in `read/AMEND1.md`). Base: the DAG-ORC build in `E:/Projects/TanitAD-artifacts/dag_orc_chain/` and `dag_orc_read/` (pristine, untouched).

## 1. Layout

| dir | what |
|---|---|
| `read/` | the re-based reader `dag_orc_read.py`, its self-test `dag_orc_selftest.py`, mutation harness `dag_orc_mutation.py`, `RC_OACC.json`, `RC_DAG2.json`, `DRY_RUN.json`, `SMOKE.json`, `mutation.json`, `logs/` |
| `rebase_reader.py`, `rebase_selftest.py`, `rebase_mutation.py` | the AUDITABLE RECORD of every edit: exact-count replacements applied to the pristine reader / self-test / mutation files (an anchor that does not occur exactly the stated number of times aborts) |
| `chain/box2_ship/` | the files to install (see `LAUNCH.txt`) |
| `chain/` | `make_chain.py`, `build_asserter.py`, `build_queue.py`, `build_small.py` (generators; every pin COPIED from a banked artifact), `world.py` + `selftest_chain.py` (self-test and mutation proof), `queue.diff`, `stage/` (the patched queue and the 6 patched repo tests), `chain_dry_run.txt`, `logs/` |

## 2. Where "DAG" became "DAG-OACC"

### 2.1 Reader (`rebase_reader.py` lists all 67 replacements; by group)
* **Identity and pins.** `PREREG_ID` / AMEND1 pin (raw-byte sha256, coded phrases `AMEND_PHRASES`, checked by `amend_problems()`), `banked_pin_problems()` (DAGOACC_READ.json sha `9e6bd4f6...`, DAG-OACC argv/summary/ckpt/token/bank/M2b pins are READ from the banked artifacts and asserted equal, never typed).
* **Slots.** `CTRL` = DAG-OACC (banked, reused, read **light**: `slot_needs(arm, e, light=("CTRL",))` - no logged lane, no fresh T0); `DAG` and `CTRL` are ONE aliased slot; `REP` = DAG-OACC's argv with `--seed 10/11`; `B` is read only if present (not queued). `banked_ctrl_dag_manifest()` now returns DAG-OACC from `v8f-dagoacc-t1/runs.json` arms.DAGOACC.
* **Contrasts.** R-ADE and R-KEEP share one contrast, A - DAG-OACC; R-NEW and R-NOBASE too (both coincidences disclosed in the read output, no row dropped). `RF_r = max_s |d_r,s(REP - DAG-OACC)|`; RF0 unchanged (`P0_RF0.json`, sha `1608d70b...`).
* **Crawl set / domain.** `CRAWL_SET` = {yaw-rate MAE, curvature MAE, L2, M1} at BOTH seeds; `EXEMPT = ()` (accel MAE and DK-VIOL are guarded now); `DOMAIN_N = {0: 15, 1: 15}` (D_s has no exempt rows). `domain(dag_p10, exempt=None)`.
* **K-controls.** K-PAIR (A = DAG-OACC's argv except `--out`, the six perturbation flags and `--ego-perturb-with-dagger`; REP = except `--out` and `--seed`; bank sha256 = DAG-OACC's); `k_registered` (adds `--w-acc`, bank sha, dagger-seed, `--dagger-p 0.1`, the compose flag); `k_init(want=None)` default `M2B_CKPT_SHA` (M2b_s, from DAG-OACC's launch-gate token); `k_build` against `aeb54468` with a FOURTH file (`stack/tanitad/data/v8f_dagger.py`: a key file of DAG-OACC's gate that the DAG-based build did not list); **`k_reach(arm, ...)` rewritten**: A swap `dagger_n/dagger_eligible` 0.1 +- 0.02 AND ego 0.3 +- 0.02 (non-swapped eligible) AND `ego_dagger_overlap_n` exactly 0 AND the built flag True; REP swap 0.1 +- 0.02, `ego_perturb_n` exactly 0, flag absent.
* **K-RC.** Two legs: `--rc-dag2` (DAG2's rows as before) and `--rc-oacc` (`read_core` mode `rc-oacc`: DAG-OACC - P10 Read-A table vs `DAGOACC_READ.json`, floors 0, 0 differences). `rc_evidence_ok(path, oacc=False)` binds the evidence to the reader's own sha256 (a changed reader invalidates it). `compose_seed(..., pre_a1=False)` keeps the pre-A1 composition for the DAG2 leg only.
* **Outcome routing** (`AMENDED_NEXT`): FIXED/NOT ADOPTED and PARTIAL/NOT ADOPTED -> "B (p 0.1) under this table" with its own PI go (DAG-ORC-SWAP cannot apply); HARMS of A -> B; HARMS of B -> NEXT-CRAWL on DAG-OACC's states, LAUNCH-INTENT branch with DAG-OACC's recipe; FIXED/ADOPTED -> adopt X as the base and re-run Read A on X. The 524,288-cell enumeration (`ref/enumerate_cells.py`) is unchanged and stays complete and exclusive.
* **CLI.** `--amend1`, `--rc-oacc`, `--rc-oacc-evidence`.

### 2.2 Gate asserter `gate_dagoaccorc_assert.py` (`build_asserter.py`)
`finite` / `wacc_problems` / `swap_problems` are DAG-OACC r2's functions BYTE FOR BYTE (extracted by AST, `ARMS[arm]` -> `SWAP_CFG[arm]` the only edit); `argv_of` / `flags` / `argv_sha256` / `drop` are DAG-ORC's. NEW: `ego_problems` (perturbation reaches the loss, every control exactly 0.0, built record, composition flag built on A and absent on REP, optional `ego_dagger_overlap_n` == 0), `expected_from_ref` (A = ref with `--ego-perturb-p 0` -> `0.3` + the five law flags right after it, `--ego-perturb-with-dagger` appended at the END; REP = `--seed` -> 10+S), `check` / `write_token` / `main`. Arms `dagoaccorc` / `dagoaccorcrep` (no B arm).

### 2.3 Run-record checks `v8f_dagoaccorc_chain_checks.py`
`run-record` (adds bank sha, w_acc, dagger_p 0.1, composition flag), `reached` (K-REACH as above; reads the counts at the top level OR under `levers`/`built_levers`/`totals`), `one-lever`, `pin-sha`, `argv-sha`.

### 2.4 Chain `chain_v8f_dagoaccorc.sh` (template + `make_chain.py`)
* Pins: DAG-OACC argv/summary/ckpt sha (SHA256SUMS), M2b init sha (token), iteration-1 bank sha (argv.json), w_acc `49.6969` and its file sha, BUILD_REF `aeb544683b1c...` (token commit), sha of the reviewed asserter and checks (`lib_check` STOPs on any difference), the INSTALLED `oaf_lib.sh` / `v8f_oaf_chain_checks.py` shas. EXPORT = the EXPORT block of the INSTALLED `chain_v8f_dagoacc_r2.sh`.
* No `waccdag` stage (w_acc is banked; box 2's `%RUNSM%\dagoacc\w_acc.txt` is asserted to hash to the banked file), no B runs, no `DAGORC_SKIP_B`.
* `RUNS=("dagoaccorc 0" "dagoaccorcrep 0" "dagoaccorc 1" "dagoaccorcrep 1")`; four gates first, then the four runs; K-REACH ENFORCED after each (`ZZDAGOACCORC-CONTROL-FAIL`).
* **LEVER greps include `add_argument\("--ego-perturb-with-dagger"`, `"ego_perturb_with_dagger"`, `"ego_dagger_overlap_n"`:** a tip without the flag stops at the lever check, BY DESIGN, before anything is shipped to box 2.
* Markers/log names renamed `dagoaccorc` (chain out, guard out, current file, tar, TIP file, busy text) so none of the DAG-based installed files is overwritten.

### 2.5 Queue (`build_queue.py`, `queue.diff`, `stage/`)
Insertion-only (+19 lines, 0 removed) against the tip queue `bf404bfe...` (identical at tip `9ba6b6a1`): one doc line, two `:dagname` acceptances, two `:dagarm` hand-offs, and a new `:dagoaccorcarm` that CALLS the untouched `:dagoaccarm` (bank sha, w_acc, DAGP) and rewrites `DAGF` in the same layout with the perturbation and the flag. **This stage is NOT in the brief's list** (the brief named chain / asserter / reader); it is unavoidable because the existing `dagorcarm` takes DAG-based inputs. It must be landed (with the 6 label-set tests, `stage/stack/tests/`) before a real launch.

### 2.6 Small files
Guard (comment line only), waker (`dagorc` -> `dagoaccorc`, gate-line grep), runner `.cmd`, and the NEW waiter launcher `run_dagoaccorc_after_speedread.{cmd,sh}` + `wake_dagoaccorc_start.sh` (`build_small.py` lists each replacement). The waiter starts the chain only when `ZZSPEEDREAD-PULL-DONE|-STOP` is in `F:/TanitAD-scratch/speedread/pull.out` AND `BOX2_BUSY.txt` is absent; `AFTER_OUT=C:/Users/Admin/box2_ship/logs/chain_v8f_dagoacc_r2.out`.

## 3. Things the Master Mind must know / decide
1. **The trainer flag is not here.** `--ego-perturb-with-dagger` is the other agent's build. The asserter/checks read the interface the amendment names (`ego_perturb_with_dagger`, `ego_dagger_overlap_n` in the run record; the probe's `ego_dagger_overlap_n` is OPTIONAL and, when present, must be 0). **The composed-batch G-LIVE self-check fields of that trainer are unknown to me**: the gate asserts the swap probe and the ego probe as they exist today (both are required to pass, with every control exactly 0.0). If the new trainer adds a composed-batch probe, add its clauses to `ego_problems` and regenerate (`make_chain.py` re-pins the asserter sha).
2. The trainer's loop order at the tip is `ego.apply` THEN `dag.apply`; the amendment fixes swap FIRST. That is the trainer agent's change; the chain's lever grep for those two lines is deliberately prefix-only (`b = ego.apply(b, step`).
3. EXPORT is DAG-OACC r2's list; if the new trainer imports a NEW module, re-run the export closure (the gate would fail with an ImportError before any training).
4. Dry runs against the unpatched tip are impossible by design (`--dry-run` needs `V8F_QUEUE=<patched queue>`; `--check-tree` needs a stub-patched tree). Both are exercised in `selftest_chain.py`.
5. Not covered by a fake-box end-to-end: the post-gate branch of the real run (`gate_one` / `runone` bodies on box 2). Its decision points are tested directly (gate asserter on 4 real-record worlds, run-record, K-REACH, token) and the queue's non-dry gate/train path runs on a stub box (section R of the self-test); `oaf_lib.sh` is the proven library of the FLAT-DAG, DAG-OACC and DAG-ORC chains.
6. Pre-registration hygiene: B is not queued; a lower dose needs its own PI go.
