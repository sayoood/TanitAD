# TEST-RED-1 -- the three "not clean on the tip" refe tests (tip 1a10b22), root-caused

**Headline: none of the three is a code defect. Two are ENVIRONMENT (one documented, one an export-layout artefact), one is a TEST-ROBUSTNESS gap. Two test files are edited (staged, local fixes).
INTEGRATION ESCALATION: both edited files are listed in `SHIP_MANIFEST.json` (S1c ship, commit 22b4068); editing them changes the manifest tree digest, so landing them means re-minting the manifest
before the next S1/S1c token. Land after the S1c ship, or re-ship.**

PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.  Runner: `C:/so/s4r_tools/queue.py` (one test at a time, BELOW_NORMAL, free RAM >= 8 GB, Git dirs forced onto PATH).  Python = `driverl-eval` (torch 2.7.1+cu128, RTX 4060).
All "tip" runs below are on a fresh `git archive 1a10b22` of the PK (+ `Project Steering/MODEL_REGISTRY.md`, + the one repo file the tests need, see 1).  Logs: `C:/so/tred_logs/q1,q3,q4/`.

| test | before (classification logs) | tip, correct environment (MEASURED) | after the staged fix (MEASURED) |
|---|---|---|---|
| `refe/test_s1aux_chain.py` | rc 1 (qfinal: FileNotFoundError :733; qbase: AssertionError :413) | **rc 0, ZZTEST_PASS 404, 0 skips** (376 s) | full layout **rc 0, ZZTEST_PASS 404, 0 skips** (374 s); no-FlyWheels layout: rc 0, `ZZTEST_PASS 256` + 1 loud `ZZTEST_SKIP` (was rc 1 FileNotFoundError, reproduced in the same tree, `nofw_orig_s1aux`) |
| `refe/test_aux_g12_bf16.py` | rc 1, 8 `h1cuda/*`+`cause/*` checks FAIL | no env: **rc 1, same 8 names (reproduced)**; `CUBLAS_WORKSPACE_CONFIG=:4096:8` exported: **rc 0, ZZTEST_PASS 7** (30 checks + 6 RED mutations) | no env exported: **rc 0, ZZTEST_PASS 7** |
| `refe/test_s1c.py` | rc 0 but 2 x ZZTEST_SKIP (not counted) | **rc 0, ZZTEST_PASS 242, 0 skips** (181 s) | not edited |

## 1. `test_s1aux_chain.py` and `test_s1c.py` -- the export had no `FlyWheels/` (ENVIRONMENT) + an unguarded copy (TEST GAP)
* **Cause of the s1c skips (legitimate on that host, not on a repo checkout).** `test_s1c.py:436-439` looks for the real navtest list at `PKGROOT.parents[3]/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-estimator-and-route-leak/raw/cluster_maps/navtest.json` (or `$AUX_NAVTEST`); `:604` needs the same file plus the real label store.
  The classification trees (`C:/so/s4r`, `s4r_p`) were PK-only exports with **no `FlyWheels/` at all** -> both print `ZZTEST_SKIP`.  The file IS in the repo at the tip (`git ls-tree 1a10b22` lists `.../raw/cluster_maps/navtest.json`, 753,185 B).
  With that one file present (and `D:/Projects/TanitAD-artifacts/aux_drivable/label_store` present) the same unedited `test_s1c.py` runs both cases: **242, 0 skips**.  Per `raw/2026-10-16-s1c-build/POD_GSUITE_FIX.md` (Part 2) the e2e case is dev-box-only by design (`--no-e2e` is the pod's declared scope).  **No change needed; a skip on a PK-only export is correct.**
* **Cause of the s1aux FileNotFoundError.** `test_s1aux_chain.py:733` (`Sandbox.__init__`) does `shutil.copy(NAVTEST, ...)` unconditionally.  Line 35 (fd7e175, 2026-10-16) was changed so a shallow tree gets a non-existent NAVTEST "so the sandbox cases SKIP there", but
  `sandbox_runs` (:791) had no matching guard (only `test_s1c.py:604` has one); fd7e175's proof (`raw/2026-10-16-s1c-build/logs/podfix/`) ran the shallow case for `test_s1c` only.  Result: FileNotFoundError instead of the intended loud skip, whenever `FlyWheels/` is absent.
  Breaking commits: 6f9756d (unconditional copy), fd7e175 (made the path non-existent without guarding the use).  **Not a code defect: a test-robustness gap.**
* **Fix (staged):** `sandbox_runs` prints `ZZTEST_SKIP sandbox_runs: the repo's navtest list is absent ...` and returns when `not NAVTEST.is_file()`.  Same convention as the existing fd-200 skip two lines above.  Nothing else changes: full layout still 404 / 0 skips.
* **The other baseline signature (`qbase`: bare `AssertionError` at :413, `ok(pin(...) == 0)`, 4 s).**  NOT reproducible on the tip: the identical file passes :413 and runs to 404.  `file_pin_ok` (`s1aux_pod.sh:114-119`) shells out to `stat`/`md5sum`/`sha256sum`; MEASURED (`C:/so/tred_work/path_probe.py`):
  with Git's dirs absent from PATH and `usr\bin\bash.exe` as the outer shell, the inner `bash` resolves to `C:\Windows\system32\bash` (WSL) and `stat`/`md5sum`/`sha256sum` resolve to nothing; the pin snippet returns rc 127 (`C:/so/tred_work/pin_sim.py`).  The harness itself recorded the same bare-AssertionError-in-seconds signature for `test_sft4r_chain`/`test_p2r_chain` in its first PATH-less attempt (`C:/so/s4r_tools/queue.py:24-26`, logs `C:/so/s4r_logs/qfinal_attempt1`).
  **Classification: ENVIRONMENT (process PATH), INFERRED -- I did not see the baseline run's PATH.**  I make no code claim from that log.

## 2. `test_aux_g12_bf16.py` -- torch 2.7.1 needs `CUBLAS_WORKSPACE_CONFIG` at process start; the test did not provide it (ENVIRONMENT, documented)
* **Cause.** Under strict `torch.use_deterministic_algorithms(True)` (the H1 identity probe, `scorer_finetune.py:1360`) torch 2.7.1 raises its cuBLAS determinism error unless `CUBLAS_WORKSPACE_CONFIG` is in the environment at process start.  The in-process `setdefault` (`scorer_finetune.py:1359`) is too late (the comment at :1354-1357 says so).
  The preflight dies, G12 is never written, `W.gate(...)` returns `None`, and every H1 check fails with `TypeError: 'NoneType' object is not subscriptable` (log lines), plus `cause/sdpa_...` with the cuBLAS RuntimeError.  Exactly the 8 names in the classification log.
* **It is a documented requirement.** `eval/PREREG_AUX_DRIVABLE_S1.md:352-358` (A6 addendum): "As written, it FAILS ... With CUBLAS_WORKSPACE_CONFIG=:4096:8 ... it PASSES: ZZTEST_PASS 7"; `raw/2026-10-15-r0c-build/MM_VERIFY.md:16` ran it with the variable.  The pod side was fixed in the same commit: `s1aux_pod.sh:419` exports it on the preflight line, pinned by `PREFLIGHT_NO_CUBLAS_ENV`.
  The test's own `Run:` line (docstring, :3) never got the variable.  **Breaking commit: d257647 (2026-10-15)** introduced the requirement and fixed the pod, not the test.  No gate suite runs this file (`grep` over `refe/*.py`, `eval/*.py`: no caller), so it stayed red on every bare dev-box run.
* **Not a lost property.** With the variable the real-source checks all hold (30 checks) and all 6 RED mutations are caught: the G12 code is fine.
* **Fix (staged):** `os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")` before `import torch` (an explicit value still wins).  This reproduces the pod's launch condition; no check is weakened or skipped.  Result: rc 0, ZZTEST_PASS 7 with nothing exported.
* **Minor stale comment (not edited, would change shipped code):** `scorer_finetune.py:1357` still says "s1aux_pod.sh does not today"; it has since d257647 (`s1aux_pod.sh:419`).  Other chains that run `scorer_finetune.py --preflight` (`rls1_pod.sh:525`, `sft*_pod.sh`) do not use the aux head, so they never reach the H1 branch.

## Harness notes (for whoever reads the logs)
* `rc=120` + `object refcount : 2` in `C:/so/tred_logs/q2` is NOT a test result: C: ran out of disk (`OSError: [Errno 28]`) while the other queue's p6 tests were writing temp; re-run as q3/q4.  C: free space swung 0 - 8 GB during this task.
* My two scratch copies (`C:/so/tred`, `C:/so/tred_nofw`, 386 MB each) were deleted after the last run; the staged files below are byte-identical to the ones that were tested (asserted).
