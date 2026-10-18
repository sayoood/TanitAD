# P6 launch-gate dry run at `a5a4ee8`: diagnosis and fixes for the three failed checks (2026-10-18)

**Author:** diagnosis agent for the Master Mind, 2026-10-18.

**Code base:** the diagnosis was made at the mirror tip `5448a05`. The pod's shipped P6 tree is `a5a4ee8`, which is an ancestor of `5448a05`; `test_p6_chain.py` and `p6_pod.sh` are byte-identical between the two.

**Rebased onto `c2d4042` (SO-PICKCON).** PICKCON also edited three of my 11 files: `gate_checks.py`, `launch_gate.py` and `test_launch_gate.py`.
* All three were 3-way merged (`git merge-file`, current = c2d4042, base = 5448a05, other = mine) with **0 conflicts**.
* PICKCON's hunks are scorer-side only (`epo_con_temp`, `scorer_effective`, G13 `temp_reach`) and do not touch mine.
* Their diffs in `out/diffs/` are against **c2d4042**, with base blobs `gate_checks.py 2c3f922e`, `launch_gate.py a8822b43` and `test_launch_gate.py 7db5f699`.
* The other 8 files are byte-identical between 5448a05 and c2d4042, so their 5448a05 bases still hold. `make_out.py` asserts this.
* Per-file `base_commit`, `base_blob` and `c2d4042_blob` are in `out/base_blobs.json`.
* The test battery ran on a fresh **c2d4042** export plus the 11 files, against a pristine c2d4042 BASE.

**Repo home of this file:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-18-p6-gate-diagnosis/RESULT.md`. Below, `PK` = `.../2026-09-20-refe-plan`, and `D` = `PK/raw/2026-10-18-p6-gate-diagnosis`.

**Pod work:** reads only, plus CPU diagnostics. No training, chain or gate was run there. The scratch files are removed.

**Overall:** each check has a root cause and a fix, verified as far as the dev box can verify it. The pod-side proof is the re-run in section 4.

| check | root cause (class) | fix | verification |
|---|---|---|---|
| G-CLOCK | the registered count was derived from the OLD bank (TRAPS: a derived constant that did not follow its input). 167,300 is correct for 3-C; there is no build defect. | per-bank registration; the old number is not edited; `PREREG_3C.addendum6.md` | arithmetic + `wc -l` + the trainer's own bank event agree; new checks and 2 mutations in `test_p6_levers.py` |
| G-SUITE | an environment leak into a unit test: the gate runs its suite inside the chain's env, and the test's hand-kept scrub list missed `P6_CBANK` | the sandbox drops the chain's whole namespace by prefix; hermeticity checks added | reproduced RED with `P6_CBANK` set (rc 1 at `:841`); fixed: ZZP6CHAIN_PASS 349 under the full dry-run env |
| G-LIVE-LAB | per-set label **latency** (one job per 64-proposal set) sits near the 0.89 s micro-batch budget on the 8.5-CPU pod. More workers do not help; NFS is not it. | SPEC_P6 §3.1's latency split, built: `--label-chunks 2`, and label workers `CORES-2` (= 6) | paced probe on the pod: stall 2.3-3.8 % → **0.41 %**; pool-alone 9.65 sets/s; threshold untouched |

## 1. G-CLOCK: `step-0 bank has 167300 tuples != registered 189888`

### Where 189,888 comes from
* It is registered at `PK/refe/launch_gate.py:226` (5448a05): `"n_tuples": 189888` in profile `refe_p6`.
* `PK/refe/gate_checks.py:1512-1518` checks it against the BUILD capture.
* Its source is `PK/SPEC_P6_PAPER_FAITHFUL.md:63`: *"bank | 189,888 tuples over 103,037 scenes | MEASURED"*. It recurs at `:112`, at `:149` (D8) and at `:256` (G-CLOCK).
* That is the PRE-3C bank `train_grow_cmd`: 103,037 rank-0 + 86,851 rank-1 rows. `PREREG_3C.md` §3C-PASS says *"P6 re-gates on it (new data-manifest sha ⇒ new launch token)"*, but it never named this bank-derived value.

### 167,300 is right for the 3-C bank
All readings were MEASURED on the pod, 2026-10-18 (`D/raw/gate_a5a4ee8/`).
* **Arithmetic:** 103,037 (rank 0, byte-identical, sha256 `daa23b96…`) + (86,851 − 22,588 dropped twins) = **167,300**. The 22,588 = 44,409 − 21,821 kept.
* **`wc -l`** of the bank files: 103,037 and 64,263.
* **The trainer's own `bank` event** in the G-LIVE-MEM run reads `n_tuples` 167,300 and `per_rank` {0: 103,037, 1: 64,263}.

`n_scenes` is 103,037, so epochs 25, steps 10,075 and epoch_len 103,037 are all unchanged. There is no build defect.

### Fix
* `launch_gate.py` gains `P6_BANK_TUPLES = {"train_grow_cmd": 189888, "train_grow_cmd_3c": 167300}`. The profile now carries `n_tuples_by_bank`.
* New `gate_checks.clock_bank_tuples` resolves the count from the `--targets` directory name. An **unregistered bank FAILS**, so one bank's number never vouches for another.
* 189,888 is untouched and remains the pre-3C bank's number. `p6_pod.sh`'s default `CBANK` is still that bank.
* The count is registered by **`PREREG_3C.addendum6.md`**, sha256 in `PREREG_3C.addendum6.sha256`. It was written before any P6 training data exists; it discloses that only the gate's own instrument runs have touched the bank.
* Tests in `test_p6_levers.py`:
  * `profile_p6` re-derives both entries from their sources and resolves 5 `--targets` paths.
  * `gate_p6_rows` runs per-bank PASS / wrong-count FAIL / unregistered-bank FAIL on the REAL built bank.
  * New mutations that must go RED: the 3-C entry carrying 189888, and a lookup that ignores the bank.
* **Doc item, not edited:** SPEC_P6 `:63/:112/:149/:256` still describe P6's bank as 189,888. That is true of the pre-3C bank and stale for the PI's 3-C choice.

## 2. G-SUITE: `test_p6_chain.py: rc 1, 'ZZP6CHAIN_PASS' NOT seen`

### Mechanism (file:line at 5448a05)
1. `gate_checks.check_suite` gives each suite script `env = dict(os.environ, …)` (`:1479`). The suite therefore inherits the CHAIN's environment.
2. The orchestrator's `post_merge/pod/gate_dryrun.sh` exports `P6_CBANK=$OUT`, the REAL 3-C bank.
3. `test_p6_chain.Sandbox.run` copied `os.environ` (`:571`) and popped a hand-kept list of 20 names (`:579`). That list predates a5a4ee8's `P6_CBANK` knob, so the knob passed through.
4. The sandboxed chain hit a5a4ee8's guard (`p6_pod.sh:288`): `REFUSED: P6_CBANK is set … would rebuild (rm -rf) it`. Case 4b failed at `test_p6_chain.py:841` (`assert rc == 0`). This matches the pod's 16.8 s failure.

### Discriminating runs (dev box, `D/raw/devbox_g_suite/`)
* 5448a05 with `P6_CBANK` set: rc 1 at `:841`.
* The same files without it: **ZZP6CHAIN_PASS 341**.
* In a path containing spaces and `&`: also 341.

The class is an environment dependence inside a unit test: the test assumed that the inherited env carries no chain knobs, and that is false whenever the gate runs it from inside the chain. It is not NFS, chown, CPU count or the chain code.

⚠️ **Hazard:** without a5a4ee8's guard, this unit test would have `rm -rf`'d the real 3-C bank, about 13 h of pod CPU.

### Fix (`test_p6_chain.py`)
* `sandbox_env()` drops the prefixes `P6_ STUB_ NUPLAN_ DRIVERL_ TANITAD_GATE_`.
* It also drops the paths the chain exports: `REFE_BACKBONE_ROOT`, `REFE_NUPLAN_DB_ROOT`, `REFE_NAVTRAIN_YAML`, `REFE_PAUSE_FILE`, `CUDA_VISIBLE_DEVICES` and the HF token names.
* It keeps `PATH`, the temp dirs and `REFE_GATE_NESTED`.
* New case 4i′ (+8 checks, 341 → 349):
  * the scrub itself is checked as a unit;
  * a `P6_CBANK` leaked into `os.environ` gives a passing chain, a canary bank untouched, and no call that names it;
  * a contrast arm hands the same value to the chain, which is REFUSED. That proves the probe can see a leak.
* **Verified:** under the full dry-run env (`P6_CBANK`, `P6_STOP_AFTER=launch_gate`, `OMP_NUM_THREADS=4`, …) the result is **ZZP6CHAIN_PASS 349**, rc 0. The clean env also gives 349.

### The Master Mind's datapoint
* **No live HF access.** The `403 … cannot write` lines come from `TH.FakeApi(deny=…)` negative controls (`test_p6_chain.py:805-812`). They appear verbatim in the PASSING 2026-10-15 log (`PK/raw/2026-10-15-p6-prereg-final/test_p6_chain.log`).
* **The dev-box failure is a different assertion:** `:1020`, the `nice` stub was never called. MEASURED cause: `pickcon/lane_runner.py` puts `<Git>/bin` first on PATH. That `bash.exe` is Git's launcher, and it prepends `/usr/bin`: under it `type -p nice` resolves to `/usr/bin/nice`, while `<Git>/usr/bin/bash.exe` resolves to the stub.
* My first battery reproduced it with the same prefix (`battery_aborted_wrapperbash/`).
* **Fix:** a new `bash_exe()` picks the MSYS bash when PATH finds the launcher. It is a no-op on Linux.
* ⚠️ Eleven other `test_*.py` use `shutil.which("bash")` and are exposed to the same runner choice; they were not changed.

## 3. G-LIVE-LAB: `GPU stall 10.3 % of a window > 5 %`

### What the gate saw (`evidence_G-LIVE-LAB.json`, `mem_metrics.jsonl`)
* Steady windows: 10.33 / 4.79 / 1.48 % at `--label-workers 4` (the old rule `CORES-4` with CORES = 8 on the 8.5-CPU quota; the old pod gave 7). The first window, 1.2 %, includes start-up and is not judged.
* Window 1 had a 6.5 s stall in 63.3 s, a queue depth of 0.28 sets and 0.694 CPU-s/set, against 0.60-0.61 in windows 2 and 3.
* Throughput (7.553 ≥ 6.735) and identity (64/64) passed.

### Measurements
Taken on the idle pod, CPU only, with `p6_cpu_probe.py pool` on real parts at the trainer clock (57/64 s per micro-batch), beside a 1-core trainer stand-in. Raw: `D/raw/pod_probe/`, `D/raw/nfs_parts_latency.txt`; scripts in `D/code/`.

| lever | result |
|---|---|
| NFS reads of parts | 400 cold files at training-distribution keys: p50 10 ms, p99 22 ms, max 72 ms. Against a 890 ms budget this is **not the cause**. |
| more workers, 1 job/set | W4 3.19 / 2.30 / 2.25 %; W6 3.68 %; W8 3.77 %. **More workers do not help.** |
| 2 jobs/set (`--label-chunks 2`) | W4 1.89 / 1.95 %; **W6 0.41 %**; W8 0.15 / 0.05 % |
| pool-alone sets/s (gate bar 6.735) | W4c1 8.02; W4c2 6.80 (1 % margin); **W6c2 9.65**; W8c2 10.69 |
| trainer stand-in's CPU share | W4c1 99.97 %; W4c2 99.97 %; **W6c2 97.0 %**; W8c2 92.6 % |
| memory | cgroup `failcnt` 0, swap 0. Not implicated; pinned memory was not probed further. |

### Root cause and fix
**Root cause:** one PDM job per set costs about 0.6-0.7 CPU-s on a single core. That is about 75 % of the period, so jitter crosses it, and a wider pool cannot shorten one job. This is the same diagnosis as `PK/eval/P6_ON_11VCPU.md` §3 (lever 3).

**Fix:** build SPEC_P6 §3.1's *"a set may be split into proposal chunks across workers for latency"*.
* `online_labeller.OnlineLabeller(chunks=k)`: k contiguous chunks per sample, a sha per chunk, and reassembly in order. A sample with any unlabelled chunk is masked whole and counted once.
* The trainer flag `--label-chunks`, default 1. Every other caller is unchanged.
* A new G-DVB row, `label_chunks`.
* The G-LIVE-LAB judge FAILS a lab reading taken at a different chunk count than the launch argv.
* `diag_p6_labeller_identity.py --chunks`. The gate's D1(a) bit-identity versus the OFFLINE unchunked labeller is therefore proven ON the chunked path.
* The `refe_p6` profile requires `--label-chunks 2`.
* In `p6_pod.sh`: `LC=${P6_LABEL_CHUNKS:-2}`, `LW = CORES-2` (= 6 here, 9 on an 11-CPU pod), and the lab key `w$LW-c$LC`.
* `p6_cpu_probe.sh` and `test_p6_cpu_probe.py` carry the same LW formula.
* Labels are per proposal by source: `navsim_pdm_targets.score_plans` computes `ep_j = raw_j/max(raw_0, raw_j)` against PDM-Closed alone. The synthetic backend is tested bit-identical at chunks 1, 2 and 4.
* **The 5 % bar is unchanged.**

**Residual risks, stated:**
1. The pdm-backend identity is proven only by the pod's D1(a). If it fails, the gate FAILS closed; the fallback is `P6_LABEL_CHUNKS=1` plus a one-line profile change.
2. The step-time cost of W6 on the real trainer is UNMEASURED; the stand-in loses 3 %. Compare G-LIVE-MEM's non-stall window time with a5a4ee8's, about 57 s.
3. The emulation is milder than the real run (W4c1: 2.6 % emulated against real windows averaging 5.5 %), so the margin under 5 % is smaller than 0.41 % suggests.

**Disclosed, not changed:** the bar is "max over ≥ 2 steady windows of a 4-step run". With 3 windows a single bad one decides. That is a noisy statistic but not a mis-derivation, so no threshold item is raised.

## 4. Re-run recipe
1. **Land on `c2d4042`** the 11 changed files (plus `test_rls1_queue.py` as file 12, section 6), `PREREG_3C.addendum6.md` + `.sha256`, and this package (`out/paths.json`).
   * The three PICKCON-overlapping files are already merged onto c2d4042, so the files in `out/` can land as whole files.
   * Revert guard: each file's blob at the landing base must equal its `base_blob` in `out/base_blobs.json`. If the tip has moved past c2d4042 on any of the 11, re-merge first.
2. **Ship:** run `python refe/launch_gate.py ship-manifest --pkg <pkg> --commit <new 40-hex> --out <pkg>/SHIP_MANIFEST.json`, then replace `/workspace/refe-plan-p6`. The new `p6_pod.sh` md5 starts `0a2eefc566ca`; if `P6N_P6POD_MD5` is set, update it.
3. **Orchestrator `--from e`** is enough, because steps a-d are done and the bank is unchanged:
   `SSH=… P6N_SHIP_COMMIT=<new> P6N_HF_TOKEN_FILE=<pod path> bash post_merge_to_p6gate.sh --from e`. Add `--no-launch` for a dry run only.
   * The ARGS change (`--label-workers 6 --label-chunks 2`) gives a new ARGS sha, so lab, mem, egr and the gate re-run by themselves.
   * The cache and lanes stages stay done. No env or argv needs setting.
   * Overrides: `P6_LABEL_CHUNKS`, `P6_LABEL_WORKERS`.
4. **What to read in the gate:**
   * G-CLOCK should show `bank_registration: bank 'train_grow_cmd_3c'`.
   * G-SUITE should show `ZZP6CHAIN_PASS 349`.
   * G-LIVE-LAB should show `lab.chunks 2`, identity 64/64, throughput ≥ 6.735, and the steady stall windows.
   * G-LIVE-MEM's step times should be compared with a5a4ee8 to check the starvation risk.

## 5. Tests (dev box, ≤ 2 concurrent, full form)
`TESTS: see TEST_RESULTS.md` (filled when the detached battery ends; HANDOFF.md).

## 6. `refe/test_rls1_queue.py`: the battery's one REGRESSION row is a sandbox race on a starved box, not the fix

This section was added on 2026-10-18 by the rls1-queue diagnosis agent. The evidence is in `D/raw/rls1q_flake/` and the code in `D/code/rls1q/`.

### Verdict

The REGRESSION row is **NOT caused by the P6 fix**. The battery recorded FIX rc 1 after 2.7 s at `test_rls1_queue.py:184`, *"returned before S1 was over"*, and BASE `ZZTEST_PASS 254` after 704.5 s. Four facts show the fix is not involved.

* **The failing scenario reaches none of the 11 changed files (MEASURED).**
  * Scenario `s1` runs the queue functions that `helpers()` and `fn_any()` extract from `refe/rls1_pod.sh`.
  * Four files are byte-identical in FIX and BASE: `rls1_pod.sh`, `test_rls1_queue.py`, `test_rls1_chain.py` and `rls1_checks.py` (sha1 `a9d17c08`, `299f1a34`, `f6137ff0`, `52ad5bc3`).
  * The module-load import closure of the test is `gate_token`, `hf_egress`, `onpolicy_relabel_pdm`, `p2r_*`, `rls1_checks`, `select_bank_subset`, `test_p2r_chain` and `test_rls1_chain`. None of them is one of the 11.
  * `launch_gate` is imported only later, inside `rls1b_checks` and `rls1c`.
* **FIX passes the same scenario when it is simply run again (MEASURED).**
  * `s1` passed 8/8 on an idle box (`probe_fix_q1.log`).
  * It passed 10/10 with a 0.6 s injected slow start, running concurrently with BASE (`probe_fix_d06.log`).
  * A battery-style full FIX run got past line 184 and was still running at a 300 s timeout.
* **BASE fails the identical assertion on demand (MEASURED).**
  * When the fake S1's bash reaches its lock 2.5 s late, BASE fails `returned before S1 was over` 3/3, and FIX fails it 3/3 in the same way (`probe_{base,fix}_d25.log`). The scenario takes 2.4–2.6 s, matching the battery's 2.7 s.
  * The same scenario also flaked on **BASE** 1/10 on a second assertion, *"RL-S1 held chain.lock while S1 was alive"* (`probe_base_d06.log` i=2). That run's queue log is correct: `ZZRLS1_QUEUED` at :58, S1 `DONE` at :01 and `QUEUE_LOCK` at :02, with no `YIELD`. So the failure was a sampler artefact, not an overlap.
* **The Master Mind's "600 s hang with an EMPTY log" was the test's normal runtime.**
  * The test prints only its final token, and a full pass takes about 700 s (BASE took 704.5 s in the battery).
  * Run it with a timeout of at least 1,500 s.

### Mechanism (the test harness; file:line at c2d4042)

1. **No check that the fake S1 is alive.**
   * `Queue.other()` (`:111`) Popens the fake chain and returns at once. The scenario then sleeps 1.0 s (`:179`) and starts the queue (`:181`). Nothing checks that the fake chain HOLDS its `chain.lock` by then.
   * If its bash needs more than ~1 s to reach `flock -n 8`, or dies, `queue_wait` (`rls1_pod.sh:116`) sees nothing ahead. It then takes `chain.lock`, settles for `RLS1_SETTLE_S`=1 and returns rc 0 after ~1.2 s, so `:184` fires.
2. **The sampler's S1 witness is unsound.**
   * `observe()` (`:122`) try-locked **S1's lock first and OUR lock second**.
   * The two reads are not atomic: RL-S1 can rightly take its lock between them just after S1 exits.
   * A try-lock of S1's lock also reads "held" when it collides with the queue's own `ahead()` try-lock of the same file.
   * Either way, a row can read `s1=1 ours=1` without any overlap.

### The box was starved while this ran (MEASURED, `envmon.tsv` and the logs)

* **Around 10:38–10:42:**
  * C: had **0.17 GB free**, and the pagefile is on C:.
  * Free virtual memory was **6.0 GB** and free physical memory **0.7 GB**.
  * One probe died with `OSError 28 No space left on device`.
  * A fake chain's bash died with rc **3221225773 = 0xC000012D, STATUS_COMMITMENT_LIMIT**.
* **At 10:51**, during the final re-runs, MSYS bash fork failures appeared in the test's own output:
  * `child_copy: cygheap read copy failed ... Win32 error 299`
  * `fatal error in forked process - couldn't create signal pipe, Win32 error 5`
* **Neither the disk nor the memory is this diagnosis's own use.** Its only footprint was one 409 MB tree copy, which was deleted.
* ⚠ **The box state at the battery's 09:49 failure was not recorded.** That the same exhaustion slowed or killed the fake S1 there is **UNVERIFIED**, but it is the only mechanism found that produces that signature on byte-identical code.

### Fix: the test only, staged as file 12 (`refe/test_rls1_queue.py`, base blob `6c40ecbf`)

No assertion was removed or loosened. The following are unchanged:
* the 3.5 s bound and the log-order assertion;
* the invariant;
* every existing mutation.

The changes:
1. **`other()` returns only once the fake chain HOLDS its lock.**
   * The fake chain writes a marker after its `flock`, outside `data/`, so no chain function reads it.
   * If the chain dies first, the test fails as `SANDBOX, not a queue result`.
2. **`rc()` calls `sandbox_alive()`.** A fake chain that died with rc ≠ 0 fails the scenario as SANDBOX and is never read as a queue result.
3. **`observe()` reads OUR lock first, then S1's liveness from S1's OWN log** (START logged, no DONE). This is sound:
   * S1 holds its lock from before the sampler starts until after it logs DONE.
   * So "no DONE" at the second read means S1 was held at the first read too.
   * A row `s1=1 ours=1` is therefore a real overlap.
4. **New scenario `s1_slow_start`, plus a mutation that must go RED** (the check that would have caught this). In the scenario the fake S1 needs 4 s to reach its lock. In the mutation the handshake is off.

### Re-runs (battery-style `run()` env; the corrected tree is FIX plus this file)

| run | result |
|---|---|
| attempt 1: fix v1, 10:33–10:38. C: had 0.17 GB free and commit was exhausted | **r1:** rc 1 at 43.5 s, `SANDBOX, not a queue result: the fake refe_p2rep chain exited rc 3221225773` (0xC000012D) in `replicate_past`, so the handshake made a silent death loud. **r2:** rc 1 at 36.0 s, an `s1` log assertion (no `QUEUED`) that nothing labelled; `sandbox_alive()` was added for exactly this case (`fixq_attempt1_r*.out`). |
| attempt 2: fix v2, 10:50–10:53. Commit headroom was 4.8–10 GB and the log shows MSYS fork errors | r1 rc 1 at 132.4 s and r2 rc 1 at 134.4 s. **Every queue scenario passed in both runs**, including `s1`, `s1_slow_start`, `both`, `replicate_*`, `stale`, `p6`, `gpu` and `instance`. Both runs then failed in `kmon_scenarios` `healthy` (`:456`, no `kill monitor READY`), a scenario this edit does not touch (`fixq_v2r*__test_rls1_queue.py.log`). |
| `kmon healthy` alone, BASE and FIX+fix concurrently, 5 runs each, 10:53–10:55, same starvation | **BASE 4/5 FAIL, FIX+fix 5/5 FAIL, on the same assertion.** The cause is the environment, not the tree. The fake trainer lives for 4 s, which loses the race against a slow fake-monitor start (`probe_*_kmon_healthy.log`). |
| targeted runs of the edited code | `s1` passed 3/3 with a 2.5 s injected slow start; it had failed 3/3 on both trees. `s1_slow_start` passed 2/2 and `both` 3/3. The handshake-off, S1-not-ahead and lock-from-start mutations went **RED 3/3**, and the unmutated scenarios stayed GREEN under the same starvation (`mutcheck_fixq.log`). |
| full form in a quiet window | ⛔ **NOT RUN (blocked).** `code/rls1q/rerun_when_quiet.py` is ready. It waits for 3 polls in a row with commit headroom ≥ 20 GB and physical free ≥ 6 GB, then runs two pairs, each FIX+fix and BASE concurrently, and writes `C:/Users/Admin/p6gate/rls1q_diag/rerun/RESULTS.json` and `DONE`. I launched it detached three times via WMI: twice through `cmd /c`, once as python.exe directly. **Each time the whole process tree vanished silently after its FIRST 30 s poll**: no traceback, and cmd never logged its `EXIT` line (`rerun/env.tsv`). Something outside it kills it; the cause is UNVERIFIED. ⇒ Launch it from the Master Mind's own scheduler once the sharded `zgwire` pytest has ended. Expected tokens: fixq `ZZTEST_PASS 258` (254 + 3 for `s1_slow_start` + 1 for the handshake mutation) and base `ZZTEST_PASS 254`. A full pass needs about 700 s, so give it a timeout of 1,500 s or more. |

**What is starving the box:** another stream's sharded `pytest` (`zgwire/res/{base,mine}_r_shard`, 9+ processes, 1.5–3.8 GB private each, started 10:50:23), plus a `b8carve` test. ⛔ **No full-form result from this box is meaningful while they run**, on ANY tree.
**Follow-up (not done here):** `kmon healthy` needs the same handshake as `s1`. The fake trainer in `finish` mode should wait for the fake monitor's READY before its 4 s, in the same way `other()` now waits for the lock.

### Open: a HYPOTHESIS, not tested here, about production code that is not one of the 11 files

* In `rls1_pod.sh:120`, `if why="$(ahead)"` treats a FAILED command substitution the same as "nothing ahead". The `return 0` is decided in such an iteration, together with one GPU-idle poll.
* So a fork failure at the decisive poll could release RL-S1 while S1 is alive but between GPU phases. The GPU check fails closed; the lock check would fail open.
* The cheapest test is to replace `ahead` with a function that `exit 1`s on its N-th call and see whether `queue_wait` returns.
* This is a pod-side robustness question. It is not a P6 blocker and is not changed here.

### Integration

* ⛔ **`runner/make_out.py` does not know about file 12.** Its `CHANGED` list has 11 entries, so re-running it rewrites `paths.json` and `base_blobs.json` WITHOUT `test_rls1_queue.py`.
* Either land from the current `out/paths.json`, or first add `test_rls1_queue.py` to `CHANGED` and to `REBASED` (`c2d4042`), and copy it into `wt/fix/refe/` and `trees/fix_full/`.

## Master Mind landing note (2026-10-18)
* **Landed: the 11 fix files** (G-CLOCK per-bank tuples + `PREREG_3C.addendum6.md`, G-SUITE namespace scrub, G-LIVE-LAB `--label-chunks 2`), with the full battery in `raw/battery/` (`TEST_RESULTS.md`, per-test logs, `summary.json`).
* The battery's one REGRESSION row (`test_rls1_queue.py:184`) is **environmental, not the fix** (section 6): the failing scenario's import closure contains none of the 11 files, FIX passes it 8/8 on an idle box, and BASE fails it 3/3 under the same injected 2.5 s slow start.
* ⛔ **NOT landed: file 12, the `test_rls1_queue.py` race fix.** Its targeted runs pass and its mutations go RED, but it has no clean FULL-FORM pass yet (the box was starved; `kmon healthy` fails on BASE 4/5 under the same starvation). It lands only after `code/rls1q/rerun_when_quiet.py` passes; staged at `C:/Users/Admin/p6gate/out/` until then.
* Open, not a P6 blocker: the `rls1_pod.sh:120` fail-open HYPOTHESIS (section 6) and the `kmon healthy` handshake follow-up.
