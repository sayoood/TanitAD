# P2-s1's launch gate REFUSED on G-SUITE: a test that passes on the dev box and fails on the pod (Master Mind, 2026-10-15)

## What happened (MEASURED, pod)

- **The refusal.** At 2026-10-15 00:39:59Z, `ZZP2REP_REFUSED stage=launch_gate`.
  - Every check passed (G-HYG, G-DVB, G-LIVE, G-EVAL, G-CKPT, G-P2GATE) **except G-SUITE**.
  - In G-SUITE, `test_p2rep_chain.py` returned rc 1 with `ZZTEST_PASS` NOT seen. `test_p2r_chain.py` and `test_launch_gate.py` passed.
  - Files: `launch_gate.log`, `evidence_G-SUITE.json`, `FAIL_867984990a45.json`, `chain_excerpt.log`.
- **The same test passes on the dev box.** At the shipped commit `8679849` it reads `ZZTEST_PASS 350` in 305 s.
- The gate captures no test output (`evidence_G-SUITE.json` keeps only rc and token), so the cause had to be reproduced.

## Root cause: the test's shims were not executable on Linux

**The defect.** `test_p2rep_chain.py`'s `write_sh()` wrote its `flock` and `nvidia-smi` shims with mode **0644**. The chain runs both DIRECTLY:
- `"$nv"`, through `P2REP_NVSMI`;
- `flock`, looked up on `PATH`.

**What follows on the pod.**
- **Every `nv.sh` call fails with rc 126.** `gpu_busy()` treats "cannot ask" as busy (`|| return 0`). So the `wait_clear` harness never clears, and the
  first behaviour check (l.588) raises `TimeoutExpired` after 60 s.
- **A non-executable `flock` on `PATH` is silently skipped** in favour of the real `/usr/bin/flock`.
- **Git Bash on the dev box needs no exec bit,** so the dev box never saw any of it.

**Evidence (MEASURED, pod).**
- `pod_prerun_test_p2rep_chain.py.log`: the traceback at l.588 / 585, `TimeoutExpired ... timed out after 60 seconds`.
- `wc_probe.py`, which rebuilds the harness and runs it under `bash -x`: `gpu_busy` returns 0 on every poll and `nvc` stays 0. That means the shim
  never ran.
- Run directly, the shim returns `Permission denied`, rc 126.
- `pod_shim_mode_probe.txt`: OLD `write_sh` gives mode `0o644`, executable False. NEW gives `0o755`, executable True.

**What I first thought (corrected here, before landing).** I first attributed the refusal to the token section, which re-measures the REAL
`navtrain_pixels` (413,184 files over NFS, **206 s** serial) on every mint and verify, under `bash_run`'s 300 s timeout. That cost is real. But the
`wait_clear` checks (l.588) run BEFORE the token section (about l.628), and the pod pre-run with only the timeout fix failed at l.588. So the exec
bit is the observed cause. The listing cost is a hazard that was never the observed failure. Both are fixed.

## Fix, shipped as its own branch so P2-s1's trainer bytes stay `8679849`'s

| file | change | why |
|---|---|---|
| `refe/test_p2rep_chain.py` | `write_sh()` chmods 0o755 | **the cause** |
| `refe/test_p2rep_chain.py` | `tok()`'s `bash_run` timeout 300 -> 1800 s | a verify is a real re-measure on the pod |
| `refe/gate_token.py` | the threaded listing of `../2026-10-15-s1-gate-gap/` (lever A), digests bit-identical | **206.5 s -> 49.1 s on the pod, `identical: true`** (`fp_pod.json`, all 413,184 files) |

**Ship branch `ship/p2rep-s1-20261015`:**
- `4d2103d` (listing and timeout) -> **`12b1328`** (plus the exec bit).
- Its parent chain is `8679849`, and only these two files differ.
- On the pod: `ZZSHIP_OK 235 files`, **235/235 sha256 verified, 0 bad**, and **0 files differ** from the scratch copy that passed below.
- Old trees are kept: `.old_8679849`, `.old_4d2103d`.

## Pod proof (MEASURED, gate environment: cwd `refe/`, `REFE_GATE_NESTED=1`, the real `/workspace/data`, nice 19 beside S1's training)

| suite | pod | dev box |
|---|---|---|
| `test_p2r_chain.py` | ZZTEST_PASS 139 (186 s) | 139 |
| `test_launch_gate.py --static --no-banked` | ZZLG_STATIC_PASS 538 (55 s) | 538 |
| `test_p2rep_chain.py` | **ZZTEST_PASS 348 (817 s)** | 350 |

**Why 348 vs 350.** An instrumented per-line count (`count_ok.py`; `pod_count_pod.json` vs `dev_count_ok.json`) shows that exactly **l.1117-1118** do not
run on the pod. Those are the hygiene checks on `eval/P2REP_POD_RUNBOOK.md`, guarded by `if RUNBOOK.is_file()`, because `eval/` is never shipped (the
test's own comment). They are not chain checks.

The chain restarted at 02:33:03Z on `12b1328`. `ZZP2REP_PRECHECK_OK` reads `commit 12b13288a134`. It then WAITS on S1's lock (N-s1 / N-s2 run first).

## Class

This is the third time on 2026-10-14/15 that a G-SUITE passed on the dev box and failed on the pod. The first two were the S1 E10 navtest path and S1
G12 under CUDA bf16. Each time, **a test touched something that differs between the hosts** (a path, a device, a file mode) and was proven only on
the cheap host.
- ⇒ **Pre-run each chain's G-SUITE ON THE POD, in the gate's environment and against the real data, before the GPU frees.** It was the pod pre-run
  that found this cause.
- ⇒ A sibling defect is latent: `write_sh` in `test_s1aux_chain.py` and `test_rls1_chain.py` has no chmod either. Those suites pass on the pod
  because their shims are invoked through `bash <file>`. They should get the same chmod at their next touch.
